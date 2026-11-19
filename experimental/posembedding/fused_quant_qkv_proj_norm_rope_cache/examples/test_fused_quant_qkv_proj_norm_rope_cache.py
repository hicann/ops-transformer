# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Run a deterministic installation smoke test through the public Torch API."""

import argparse

import torch
import torch_npu  # noqa: F401

# Register the installed operator through the standard package.
import cann_ops_transformer  # noqa: F401


def run(mirror=False, chunk=False, contiguous=False, device="npu:0"):
    m, hidden_size, hq, hkv, cache_rows, table_rows = 4, 256, 2, 1, 16, 8
    head_dim = 256
    qw, kvw = hq * head_dim, hkv * head_dim
    n = qw + (4 if mirror else 2) * kvw
    rank_tokens, chunk_tokens, chunk_offset = 3, 2, 1
    t = (m // chunk_tokens) * rank_tokens if chunk else m
    local_rows = torch.arange(m, dtype=torch.int64)
    rows = (
        local_rows // chunk_tokens * rank_tokens
        + chunk_offset
        + local_rows % chunk_tokens
        if chunk
        else local_rows
    )

    # Payload=1, scale=1 gives an exactly representable projection of K=256.
    def payload(shape):
        return torch.ones(shape).to(torch.float8_e4m3fn).to(device)

    def scales(shape):
        return (
            torch.full(shape, 127, dtype=torch.uint8)
            .view(torch.float8_e8m0fnu)
            .to(device)
        )

    hidden_quant = payload((m, hidden_size))
    hidden_scale = scales((m, hidden_size // 64, 2))
    weight_quant = payload((n, hidden_size))
    weight_scale = scales((n, hidden_size // 64, 2))
    gamma = torch.ones(256, dtype=torch.bfloat16, device=device)
    positions = torch.zeros(t, dtype=torch.int64)
    positions[rows] = local_rows if contiguous else local_rows.flip(0)
    # Identity RoPE makes the smoke-test result analytically checkable.
    cos_sin = torch.cat(
        (torch.ones(table_rows, 32), torch.zeros(table_rows, 32)), dim=1
    ).to(device)
    slots = torch.full((t,), -1, dtype=torch.int64)
    slots[rows] = local_rows

    def buffer(shape, value):
        return torch.full(shape, value, dtype=torch.bfloat16, device=device)

    q, k, v = buffer((t, qw), 7), buffer((t, kvw), 8), buffer((t, kvw), 9)
    kc, vc = buffer((cache_rows, kvw), 10), buffer((cache_rows, kvw), 11)
    kwargs = {"epsilon": 1e-6, "positions_contiguous": contiguous}
    mk = buffer((t, kvw), 12) if mirror else None
    mv = buffer((t, kvw), 13) if mirror else None
    if mirror:
        kwargs.update(mirror_k_out=mk, mirror_v_out=mv)
    if chunk:
        kwargs.update(
            tokens_per_rank=rank_tokens,
            chunk_tokens_per_rank=chunk_tokens,
            chunk_token_offset=chunk_offset,
        )

    result = torch.ops.cann_ops_transformer.fused_quant_qkv_proj_norm_rope_cache(
        hidden_quant,
        hidden_scale,
        weight_quant,
        weight_scale,
        gamma,
        positions.to(device),
        cos_sin,
        slots.to(device),
        q,
        k,
        v,
        kc,
        vc,
        **kwargs,
    )
    torch.npu.synchronize(device)
    assert result is None
    inactive = torch.ones(t, dtype=torch.bool)
    inactive[rows] = False
    for tensor, expected, initial in [(q, 256, 7), (k, 1, 8), (v, 256, 9)]:
        actual = tensor.cpu()
        assert bool((actual[rows] == expected).all())
        assert bool((actual[inactive] == initial).all())
    assert bool((kc.cpu()[:m] == 1).all())
    assert bool((vc.cpu()[:m] == 256).all())
    assert bool((kc.cpu()[m:] == 10).all())
    assert bool((vc.cpu()[m:] == 11).all())
    if mirror:
        for tensor, initial in [(mk, 12), (mv, 13)]:
            actual = tensor.cpu()
            assert bool((actual[rows] == 256).all())
            assert bool((actual[inactive] == initial).all())
    print(
        f"PASS: M={m}, K={hidden_size}, Hq={hq}, Hkv={hkv}, N={n}, "
        f"T={t}, S={cache_rows}, P={table_rows}, mirror={mirror}, chunk={chunk}, "
        f"positions_contiguous={contiguous}; returned None and updated buffers in place"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mirror", action="store_true")
    parser.add_argument("--chunk", action="store_true")
    parser.add_argument("--contiguous", action="store_true")
    parser.add_argument("--device", default="npu:0")
    args = parser.parse_args()
    run(args.mirror, args.chunk, args.contiguous, args.device)
