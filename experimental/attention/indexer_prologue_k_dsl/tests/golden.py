# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""CPU golden for indexer_prologue_k cache updates."""

from __future__ import annotations

import math

import torch


def make_inputs(
    t: int,
    h: int,
    d: int,
    dr: int,
    *,
    block_num: int = 2,
    block_size: int = 64,
    storage_mode: int = 0,
    combined_block_size: int = -1,
    with_scale_cache: bool = True,
    seed: int = 0,
):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    latent = torch.randn(t, h, generator=generator).to(torch.bfloat16)
    wk = (
        torch.randn(d, h, generator=generator, dtype=torch.float32) / math.sqrt(h)
    ).to(torch.bfloat16)
    norm_weight = torch.randn(d, generator=generator, dtype=torch.float32)
    angles = torch.randn(t, dr, generator=generator, dtype=torch.float32)
    rope_sin, rope_cos = torch.sin(angles), torch.cos(angles)
    value_bytes, scale_bytes = d // 2, d // 32
    if storage_mode == 0:
        cache_shape = (block_num, block_size, 1, value_bytes)
    else:
        group = combined_block_size
        cache_shape = (
            block_num,
            block_size // group,
            1,
            group * (value_bytes + scale_bytes),
        )
    k_cache = torch.randint(0, 256, cache_shape, generator=generator, dtype=torch.uint8)
    k_scale_cache = None
    if storage_mode == 0 and with_scale_cache:
        k_scale_cache = torch.randint(
            0,
            256,
            (block_num, block_size, 1, scale_bytes),
            generator=generator,
            dtype=torch.uint8,
        )
    cache_index = torch.randperm(block_num * block_size, generator=generator)[:t]
    cache_index = cache_index.to(torch.int64)
    if t > 1:
        cache_index[-1] = -1
    return {
        "latent": latent,
        "wk": wk,
        "norm_weight": norm_weight,
        "rope_sin": rope_sin,
        "rope_cos": rope_cos,
        "k_cache": k_cache,
        "k_scale_cache": k_scale_cache,
        "cache_index": cache_index,
    }


def pack_mxfp4(key: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack signed E2M1 nibbles and produce one UE8M0 byte per 32 values."""
    t, d = key.shape
    groups = key.float().view(t, d // 32, 32)
    amax = groups.abs().amax(dim=-1).clamp_min(6.0 * (2.0**-126))
    scale = torch.exp2(torch.ceil(torch.log2(amax / 6.0)))
    magnitude = (groups.abs() / scale.unsqueeze(-1)).clamp_max(6.0)
    code = torch.full_like(magnitude, 7, dtype=torch.int64)
    for threshold, lower_code in (
        (5.0, 6),
        (3.5, 5),
        (2.5, 4),
        (1.75, 3),
        (1.25, 2),
        (0.75, 1),
        (0.25, 0),
    ):
        code = torch.where(magnitude <= threshold, lower_code, code)
    code = code + (groups < 0).to(torch.int64) * 8
    packed = code[..., 0::2] | (code[..., 1::2] << 4)
    data = packed.flatten(1).to(torch.uint8)
    scales = (torch.log2(scale).to(torch.int64) + 127).to(torch.uint8)
    return data, scales


def indexer_prologue_k_golden(
    inputs: dict[str, torch.Tensor | None],
    *,
    storage_mode: int,
    norm_eps: float,
    combined_block_size: int = -1,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    latent = inputs["latent"]
    wk = inputs["wk"]
    norm_weight = inputs["norm_weight"]
    rope_sin = inputs["rope_sin"]
    rope_cos = inputs["rope_cos"]
    cache_index = inputs["cache_index"]
    assert all(
        x is not None
        for x in (latent, wk, norm_weight, rope_sin, rope_cos, cache_index)
    )

    projected = torch.matmul(latent, wk.T).float()
    inverse_rms = torch.rsqrt(projected.square().mean(dim=-1, keepdim=True) + norm_eps)
    key = (projected * inverse_rms * norm_weight).to(torch.bfloat16)
    dr = rope_sin.shape[-1]
    tail = key[:, -dr:].float()
    even, odd = tail[:, 0::2], tail[:, 1::2]
    rotated = torch.empty_like(tail)
    rotated[:, 0::2] = even * rope_cos[:, 0::2] - odd * rope_sin[:, 0::2]
    rotated[:, 1::2] = odd * rope_cos[:, 1::2] + even * rope_sin[:, 1::2]
    key[:, -dr:] = rotated.to(torch.bfloat16)
    packed_data, packed_scale = pack_mxfp4(key)

    expected_cache = inputs["k_cache"].clone()
    scale_cache = inputs["k_scale_cache"]
    expected_scale = None if scale_cache is None else scale_cache.clone()
    if storage_mode == 0:
        cache_flat = expected_cache.view(-1, packed_data.shape[-1])
        scale_flat = (
            None
            if expected_scale is None
            else expected_scale.view(-1, packed_scale.shape[-1])
        )
        for row, slot in enumerate(cache_index.tolist()):
            if slot >= 0:
                cache_flat[slot] = packed_data[row]
                if scale_flat is not None:
                    scale_flat[slot] = packed_scale[row]
    else:
        group = combined_block_size
        block_size = expected_cache.shape[1] * group
        value_bytes, scale_bytes = packed_data.shape[-1], packed_scale.shape[-1]
        for row, flat_slot in enumerate(cache_index.tolist()):
            if flat_slot < 0:
                continue
            block_id, in_block = divmod(flat_slot, block_size)
            combined_row, sub_slot = divmod(in_block, group)
            data_start = sub_slot * value_bytes
            scale_start = group * value_bytes + sub_slot * scale_bytes
            expected_cache[
                block_id, combined_row, 0, data_start : data_start + value_bytes
            ] = packed_data[row]
            expected_cache[
                block_id, combined_row, 0, scale_start : scale_start + scale_bytes
            ] = packed_scale[row]
    return expected_cache, expected_scale
