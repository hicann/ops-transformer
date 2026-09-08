# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch


SUPPORTED_RATIOS = (2, 4, 8, 16, 32, 64, 128)
DEFAULT_NORM_EPS = 1e-6
HEAD_DIM = 128
HIDDEN_SIZE = 4096
ROPE_DIM = 64


def _as_list(value: torch.Tensor) -> list[int]:
    return [int(item) for item in value.detach().cpu().reshape(-1).tolist()]


def _split_hidden(
    hidden_states: torch.Tensor,
    cu_seqlens: Optional[torch.Tensor],
    batch_size: int,
) -> Tuple[list[torch.Tensor], list[int]]:
    if hidden_states.dim() == 3:
        if cu_seqlens is not None:
            raise ValueError("cu_seqlens must be absent for BSH")
        return [hidden_states[index] for index in range(hidden_states.size(0))], [
            int(hidden_states.size(1))
        ] * hidden_states.size(0)
    if hidden_states.dim() != 2:
        raise ValueError("hidden_states rank must be 2 (TH) or 3 (BSH)")
    if cu_seqlens is None:
        raise ValueError("cu_seqlens is required for TH")
    if cu_seqlens.dtype != torch.int32 or cu_seqlens.dim() != 1:
        raise ValueError("cu_seqlens must be a rank-1 INT32 tensor")
    offsets = _as_list(cu_seqlens)
    if len(offsets) != batch_size + 1:
        raise ValueError("cu_seqlens shape must be [B+1]")
    if offsets[0] != 0:
        raise ValueError("cu_seqlens[0] must be 0")
    if any(next_offset < offset for offset, next_offset in zip(offsets, offsets[1:])):
        raise ValueError("cu_seqlens must be nondecreasing")
    if offsets[-1] != hidden_states.size(0):
        raise ValueError("cu_seqlens[-1] must equal hidden_states.shape[0]")
    lengths = [
        next_offset - offset for offset, next_offset in zip(offsets, offsets[1:])
    ]
    chunks = []
    offset = 0
    for length in lengths:
        chunks.append(hidden_states[offset : offset + length])
        offset += length
    return chunks, lengths


def validate_inputs(
    hidden_states: torch.Tensor,
    wk: torch.Tensor,
    gate_weight: torch.Tensor,
    ape: torch.Tensor,
    state_cache: torch.Tensor,
    cache_block_table: torch.Tensor,
    start_pos: torch.Tensor,
    norm_weight: Optional[torch.Tensor] = None,
    norm_bias: Optional[torch.Tensor] = None,
    cos: Optional[torch.Tensor] = None,
    sin: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    seqused: Optional[torch.Tensor] = None,
    cmp_ratio: int = 4,
    norm_eps: float = DEFAULT_NORM_EPS,
    rotary_mode: int = 1,
    enforce_model_dims: bool = True,
) -> None:
    if cmp_ratio not in SUPPORTED_RATIOS:
        raise ValueError(f"unsupported cmp_ratio={cmp_ratio}")
    if norm_eps <= 0:
        raise ValueError("norm_eps must be positive")
    if rotary_mode not in (0, 1):
        raise ValueError("rotary_mode must be 0 or 1")
    if hidden_states.dtype != torch.bfloat16:
        raise TypeError("hidden_states must be BF16")
    if wk.dtype != torch.bfloat16 or gate_weight.dtype != torch.bfloat16:
        raise TypeError("wk and gate_weight must be BF16")
    if ape.dtype != torch.float32:
        raise TypeError("ape must be FP32")
    if state_cache.dtype != torch.float32:
        raise TypeError("state_cache must be FP32")
    if cache_block_table.dtype != torch.int32 or start_pos.dtype != torch.int32:
        raise TypeError("cache_block_table and start_pos must be INT32")
    if hidden_states.dim() not in (2, 3):
        raise ValueError("hidden_states rank must be 2 or 3")
    if wk.dim() != 2 or gate_weight.shape != wk.shape:
        raise ValueError("wk and gate_weight must have the same rank-2 shape")
    d, h = wk.shape
    if hidden_states.size(-1) != h:
        raise ValueError("hidden_states last dimension must match wk.shape[1]")
    if ape.shape != (cmp_ratio, d):
        raise ValueError("ape shape must be [cmp_ratio, D]")
    if state_cache.dim() != 3 or state_cache.size(-1) != 2 * d:
        raise ValueError("state_cache shape must be [block_num, block_size, 2*D]")
    if cache_block_table.dim() != 2:
        raise ValueError("cache_block_table shape must be [B, max_block_num_per_batch]")
    if start_pos.dim() != 1 or start_pos.numel() != cache_block_table.size(0):
        raise ValueError("start_pos shape must be [B]")
    if enforce_model_dims and (h, d) != (HIDDEN_SIZE, HEAD_DIM):
        raise ValueError("first version requires H=4096 and D=128")
    if (norm_weight is None) != (norm_bias is None):
        raise ValueError("norm_weight and norm_bias must be passed as a pair")
    if norm_weight is not None:
        if norm_weight.dtype != torch.float32 or norm_bias.dtype != torch.float32:
            raise TypeError("norm parameters must be FP32")
        if norm_weight.shape != (d,) or norm_bias.shape != (d,):
            raise ValueError("norm parameters must have shape [D]")
    if (cos is None) != (sin is None):
        raise ValueError("cos and sin must be passed as a pair")
    if cos is not None:
        if cos.dtype != torch.bfloat16 or sin.dtype != torch.bfloat16:
            raise TypeError("cos and sin must be BF16")
        if cos.shape != sin.shape:
            raise ValueError("cos and sin must have the same shape")
        raise NotImplementedError("KeyPool RoPE is reserved but not implemented in v1")
    if seqused is not None:
        raise ValueError("seqused is reserved and must be None in this stage")
    batch_hidden, lengths = _split_hidden(
        hidden_states, cu_seqlens, cache_block_table.size(0)
    )
    if cache_block_table.size(0) != len(batch_hidden):
        raise ValueError("cache_block_table batch does not match hidden_states")
    if any(int(value) < 0 for value in _as_list(start_pos)):
        raise ValueError("start_pos must be non-negative")
    if any(int(value) < 0 for value in _as_list(cache_block_table)):
        raise ValueError("cache_block_table must be non-negative")
    block_num = state_cache.size(0)
    if block_num == 0 or state_cache.size(1) == 0:
        raise ValueError("state_cache must not be empty")
    if any(int(value) >= block_num for value in _as_list(cache_block_table)):
        raise ValueError("cache_block_table contains an out-of-range physical block")
    capacity = cache_block_table.size(1) * state_cache.size(1)
    for batch, length in enumerate(lengths):
        if int(start_pos[batch]) + length > capacity:
            raise ValueError("updated sequence length exceeds logical cache capacity")
        if int(start_pos[batch]) + length > 0:
            required_blocks = (
                int(start_pos[batch]) + length + state_cache.size(1) - 1
            ) // state_cache.size(1)
            if any(
                int(value) == 0 for value in cache_block_table[batch, :required_blocks]
            ):
                raise ValueError("physical block 0 is invalid for a used logical block")


def _project(
    hidden: torch.Tensor,
    wk: torch.Tensor,
    gate_weight: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    hidden_fp32 = hidden.float()
    k = torch.matmul(hidden_fp32, wk.float().transpose(-1, -2)).to(hidden.dtype)
    gate = torch.matmul(hidden_fp32, gate_weight.float().transpose(-1, -2)).to(
        hidden.dtype
    )
    return k, gate


def _normalize_key(
    key: torch.Tensor,
    norm_weight: Optional[torch.Tensor],
    norm_bias: Optional[torch.Tensor],
    norm_eps: float,
) -> torch.Tensor:
    if norm_weight is None:
        return key
    key_fp32 = key.float()
    mean = key_fp32.mean(dim=-1, keepdim=True)
    variance = (key_fp32 - mean).square().mean(dim=-1, keepdim=True)
    return (
        ((key_fp32 - mean) / torch.sqrt(variance + norm_eps)) * norm_weight + norm_bias
    ).to(key.dtype)


def _write_cache(
    state_cache: torch.Tensor,
    cache_block_table: torch.Tensor,
    start_pos: torch.Tensor,
    batch: int,
    k: torch.Tensor,
    gate: torch.Tensor,
    cmp_ratio: int,
) -> None:
    block_size = state_cache.size(1)
    end_pos = int(start_pos[batch]) + k.size(0)
    tail_len = end_pos % cmp_ratio
    if tail_len == 0:
        return
    tail_start = max(int(start_pos[batch]), end_pos - tail_len)
    for token in range(k.size(0)):
        logical_pos = int(start_pos[batch]) + token
        if logical_pos < tail_start:
            continue
        logical_block, offset = divmod(logical_pos, block_size)
        physical_block = int(cache_block_table[batch, logical_block])
        if physical_block == 0:
            raise ValueError("attempted to write through invalid physical block 0")
        state_cache[physical_block, offset, : k.size(-1)] = k[token].float()
        state_cache[physical_block, offset, k.size(-1) :] = gate[token].float()


def _read_cache(
    state_cache: torch.Tensor,
    cache_block_table: torch.Tensor,
    batch: int,
    logical_pos: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    block_size = state_cache.size(1)
    logical_block, offset = divmod(logical_pos, block_size)
    physical_block = int(cache_block_table[batch, logical_block])
    if physical_block == 0:
        raise ValueError("attempted to read through invalid physical block 0")
    width = state_cache.size(-1) // 2
    return state_cache[physical_block, offset, :width], state_cache[
        physical_block, offset, width:
    ]


def _pool_one_batch_loop(
    state_cache: torch.Tensor,
    cache_block_table: torch.Tensor,
    batch: int,
    start_pos: int,
    updated_end: int,
    current_k: torch.Tensor,
    current_gate: torch.Tensor,
    cmp_ratio: int,
    ape: torch.Tensor,
    output: torch.Tensor,
) -> None:
    width = state_cache.size(-1) // 2
    first_pool = start_pos // cmp_ratio
    valid_pool_count = updated_end // cmp_ratio
    for pool in range(first_pool, valid_pool_count):
        keys = []
        gates = []
        for relative in range(cmp_ratio):
            logical_pos = pool * cmp_ratio + relative
            if start_pos <= logical_pos < updated_end:
                current_index = logical_pos - start_pos
                key = current_k[current_index].float()
                gate = current_gate[current_index].float()
            else:
                key, gate = _read_cache(
                    state_cache, cache_block_table, batch, logical_pos
                )
            keys.append(key)
            gates.append(gate + ape[relative])
        key_tensor = torch.stack(keys, dim=0)
        logits = torch.stack(gates, dim=0)
        probabilities = torch.softmax(logits, dim=0)
        probabilities = probabilities.to(torch.bfloat16).float()
        output[batch, pool - first_pool] = (
            (probabilities * key_tensor).sum(dim=0).to(torch.bfloat16)
        )


def run_key_pool_golden(
    hidden_states: torch.Tensor,
    wk: torch.Tensor,
    gate_weight: torch.Tensor,
    ape: torch.Tensor,
    state_cache: torch.Tensor,
    cache_block_table: torch.Tensor,
    start_pos: torch.Tensor,
    norm_weight: Optional[torch.Tensor] = None,
    norm_bias: Optional[torch.Tensor] = None,
    cos: Optional[torch.Tensor] = None,
    sin: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    seqused: Optional[torch.Tensor] = None,
    cmp_ratio: int = 4,
    norm_eps: float = DEFAULT_NORM_EPS,
    rotary_mode: int = 1,
    enforce_model_dims: bool = True,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Reference implementation using scalar logical Cache traversal."""
    validate_inputs(
        hidden_states,
        wk,
        gate_weight,
        ape,
        state_cache,
        cache_block_table,
        start_pos,
        norm_weight,
        norm_bias,
        cos,
        sin,
        cu_seqlens,
        seqused,
        cmp_ratio,
        norm_eps,
        rotary_mode,
        enforce_model_dims,
    )
    batch_hidden, lengths = _split_hidden(
        hidden_states, cu_seqlens, cache_block_table.size(0)
    )
    k_parts = []
    gate_parts = []
    for hidden in batch_hidden:
        k, gate = _project(hidden, wk, gate_weight)
        k = _normalize_key(k.float(), norm_weight, norm_bias, norm_eps)
        k_parts.append(k)
        gate_parts.append(gate)
    for batch, (k, gate) in enumerate(zip(k_parts, gate_parts)):
        _write_cache(
            state_cache, cache_block_table, start_pos, batch, k, gate, cmp_ratio
        )

    pcap = (
        cache_block_table.size(1) * state_cache.size(1) + cmp_ratio - 1
    ) // cmp_ratio
    output = torch.zeros(
        (len(batch_hidden), pcap, wk.size(0)),
        dtype=torch.bfloat16,
        device=state_cache.device,
    )
    for batch, length in enumerate(lengths):
        updated_len = int(start_pos[batch]) + length
        _pool_one_batch_loop(
            state_cache,
            cache_block_table,
            batch,
            int(start_pos[batch]),
            updated_len,
            k_parts[batch],
            gate_parts[batch],
            cmp_ratio,
            ape,
            output,
        )
    intermediates = {
        "k_projection": torch.cat(k_parts, dim=0) if k_parts else torch.empty(0),
        "gate_projection": torch.cat(gate_parts, dim=0)
        if gate_parts
        else torch.empty(0),
        "state_cache": state_cache,
    }
    return output, intermediates


def run_key_pool_oracle(
    hidden_states: torch.Tensor,
    wk: torch.Tensor,
    gate_weight: torch.Tensor,
    ape: torch.Tensor,
    state_cache: torch.Tensor,
    cache_block_table: torch.Tensor,
    start_pos: torch.Tensor,
    norm_weight: Optional[torch.Tensor] = None,
    norm_bias: Optional[torch.Tensor] = None,
    cos: Optional[torch.Tensor] = None,
    sin: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    seqused: Optional[torch.Tensor] = None,
    cmp_ratio: int = 4,
    norm_eps: float = DEFAULT_NORM_EPS,
    rotary_mode: int = 1,
    enforce_model_dims: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Independent vectorized Pool read/softmax implementation."""
    validate_inputs(
        hidden_states,
        wk,
        gate_weight,
        ape,
        state_cache,
        cache_block_table,
        start_pos,
        norm_weight,
        norm_bias,
        cos,
        sin,
        cu_seqlens,
        seqused,
        cmp_ratio,
        norm_eps,
        rotary_mode,
        enforce_model_dims,
    )
    batch_hidden, lengths = _split_hidden(
        hidden_states, cu_seqlens, cache_block_table.size(0)
    )
    k_parts = []
    gate_parts = []
    for batch, hidden in enumerate(batch_hidden):
        k, gate = _project(hidden, wk, gate_weight)
        k = _normalize_key(k.float(), norm_weight, norm_bias, norm_eps)
        k_parts.append(k)
        gate_parts.append(gate)
        _write_cache(
            state_cache, cache_block_table, start_pos, batch, k, gate, cmp_ratio
        )

    batch_size = len(batch_hidden)
    block_size = state_cache.size(1)
    pcap = (cache_block_table.size(1) * block_size + cmp_ratio - 1) // cmp_ratio
    first_pool = torch.div(start_pos.to(torch.long), cmp_ratio, rounding_mode="floor")
    logical_positions = (
        first_pool[:, None] * cmp_ratio
        + torch.arange(pcap * cmp_ratio, device=state_cache.device)[None, :]
    )
    logical_capacity = cache_block_table.size(1) * block_size
    safe_positions = logical_positions.clamp(max=logical_capacity - 1)
    block_ids = safe_positions // block_size
    offsets = safe_positions % block_size
    physical = torch.gather(
        cache_block_table,
        1,
        block_ids.to(torch.long),
    ).to(torch.long)
    offsets = offsets.to(torch.long)
    gathered = state_cache[physical, offsets]
    gathered = gathered.view(batch_size, pcap, cmp_ratio, -1)
    updated_lens = start_pos.to(torch.long) + torch.tensor(
        lengths, device=start_pos.device
    )
    # Completed pools may contain current-call tokens that are intentionally
    # absent from the final tail-only state_cache. Overlay those tokens from
    # the current projection before pooling.
    logical_positions = logical_positions.view(batch_size, pcap, cmp_ratio)
    for batch, (k, gate) in enumerate(zip(k_parts, gate_parts)):
        current_start = int(start_pos[batch])
        current_end = int(updated_lens[batch])
        current_mask = (logical_positions[batch] >= current_start) & (
            logical_positions[batch] < current_end
        )
        current_index = (logical_positions[batch] - current_start).clamp(
            min=0, max=max(k.size(0) - 1, 0)
        )
        gathered[batch, ..., : wk.size(0)][current_mask] = k[
            current_index[current_mask]
        ].float()
        gathered[batch, ..., wk.size(0) :][current_mask] = gate[
            current_index[current_mask]
        ].float()
    keys = gathered[..., : wk.size(0)]
    gates = gathered[..., wk.size(0) :]
    valid = (
        logical_positions.view(batch_size, pcap, cmp_ratio)
        < updated_lens[:, None, None]
    )
    logits = gates + ape.view(1, 1, cmp_ratio, -1)
    logits = logits.masked_fill(~valid[..., None], float("-inf"))
    probabilities = (
        torch.nan_to_num(torch.softmax(logits, dim=2)).to(torch.bfloat16).float()
    )
    pooled = (probabilities * keys).sum(dim=2).to(torch.bfloat16)
    complete = valid.all(dim=2)
    pooled = pooled.masked_fill(~complete[..., None], 0)
    return pooled, state_cache


def compare_golden_and_oracle(
    golden_output: torch.Tensor,
    oracle_output: torch.Tensor,
    golden_cache: torch.Tensor,
    oracle_cache: torch.Tensor,
) -> None:
    torch.testing.assert_close(golden_output, oracle_output, rtol=0, atol=0)
    torch.testing.assert_close(golden_cache, oracle_cache, rtol=0, atol=0)
