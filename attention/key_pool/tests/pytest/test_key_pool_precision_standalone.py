# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch_npu

sys.path.insert(0, str(Path(__file__).resolve().parent))

from key_pool_golden import run_key_pool_golden
from key_pool_public_loader import load_key_pool_public_api


def generate_data():
    """Generate all KeyPool tensors and attributes for representative cases."""
    torch.manual_seed(20260823)
    hidden_size = 4096
    head_dim = 128
    cases = []

    # Each case owns its cache.  Physical block 0 is reserved, as in the
    # production block-table contract; rows may point to blocks in any order.
    case_specs = (
        ("bsh_norm_off_history", "BSH", (9, 9), (3, 0), 4, 4, False, False),
        ("th_norm_on_zero_batch", "TH", (6, 3, 0), (5, 2, 4), 8, 2, True, True),
        ("bsh_norm_on_long_history", "BSH", (17,), (16,), 16, 8, False, True),
        (
            "th_norm_off_small_blocks",
            "TH",
            (4, 4, 4, 4),
            (2, 2, 2, 2),
            2,
            1,
            True,
            False,
        ),
    )

    for (
        name,
        layout,
        lengths,
        starts,
        cmp_ratio,
        block_size,
        scramble,
        norm_on,
    ) in case_specs:
        max_updated = max(start + length for start, length in zip(starts, lengths))
        max_blocks = max(1, (max_updated + block_size - 1) // block_size)
        block_num = 1 + len(lengths) * max_blocks
        cache_block_table = torch.empty((len(lengths), max_blocks), dtype=torch.int32)
        physical_block = 1
        for batch in range(len(lengths)):
            for logical_block in range(max_blocks):
                cache_block_table[batch, logical_block] = physical_block
                physical_block += 1
            if scramble:
                cache_block_table[batch] = cache_block_table[batch].flip(0)

        if layout == "BSH":
            hidden_states = torch.randn(
                (len(lengths), lengths[0], hidden_size), dtype=torch.bfloat16
            )
            cu_seqlens = None
        else:
            hidden_states = torch.randn(
                (sum(lengths), hidden_size), dtype=torch.bfloat16
            )
            cu_seqlens = torch.tensor(
                [0, *torch.tensor(lengths, dtype=torch.int32).cumsum(0).tolist()],
                dtype=torch.int32,
            )

        wk = (torch.randn(head_dim, hidden_size) * 0.02).to(torch.bfloat16)
        gate_weight = (torch.randn(head_dim, hidden_size) * 0.02).to(torch.bfloat16)
        ape = torch.randn(cmp_ratio, head_dim, dtype=torch.float32) * 0.1
        state_cache = (
            torch.randn((block_num, block_size, 2 * head_dim), dtype=torch.float32)
            * 0.1
        )
        start_pos = torch.tensor(starts, dtype=torch.int32)
        norm_weight = torch.randn(head_dim, dtype=torch.float32) if norm_on else None
        norm_bias = torch.randn(head_dim, dtype=torch.float32) if norm_on else None

        cases.append(
            {
                "case": name,
                "hidden_states": hidden_states,
                "wk": wk,
                "gate_weight": gate_weight,
                "ape": ape,
                "state_cache": state_cache,
                "cache_block_table": cache_block_table,
                "start_pos": start_pos,
                "norm_weight": norm_weight,
                "norm_bias": norm_bias,
                # RoPE is an interface placeholder in the current version.
                "cos": None,
                "sin": None,
                "cu_seqlens": cu_seqlens,
                # seqused is reserved and must remain empty in this version.
                "seqused": None,
                "cmp_ratio": cmp_ratio,
                "norm_eps": 1e-6,
                "rotary_mode": 1,
            }
        )
    return cases


def call_operator(
    hidden_states: torch.Tensor,
    wk: torch.Tensor,
    gate_weight: torch.Tensor,
    ape: torch.Tensor,
    state_cache: torch.Tensor,
    cache_block_table: torch.Tensor,
    start_pos: torch.Tensor,
    *,
    norm_weight: torch.Tensor | None = None,
    norm_bias: torch.Tensor | None = None,
    cos: torch.Tensor | None = None,
    sin: torch.Tensor | None = None,
    cu_seqlens: torch.Tensor | None = None,
    seqused: torch.Tensor | None = None,
    cmp_ratio: int = 4,
    norm_eps: float = 1e-6,
    rotary_mode: int = 1,
) -> torch.Tensor:
    """Call the production PyTorch KeyPool interface."""
    load_key_pool_public_api()
    torch_npu.npu.set_device(0)
    npu_state_cache = state_cache.npu()
    output = torch_npu.key_pool(
        hidden_states.npu(),
        wk.npu(),
        gate_weight.npu(),
        ape.npu(),
        npu_state_cache,
        cache_block_table.npu(),
        start_pos.npu(),
        norm_weight=None if norm_weight is None else norm_weight.npu(),
        norm_bias=None if norm_bias is None else norm_bias.npu(),
        cos=None if cos is None else cos.npu(),
        sin=None if sin is None else sin.npu(),
        cu_seqlens=None if cu_seqlens is None else cu_seqlens.npu(),
        seqused=None if seqused is None else seqused.npu(),
        cmp_ratio=cmp_ratio,
        norm_eps=norm_eps,
        rotary_mode=rotary_mode,
    )
    torch_npu.npu.synchronize()
    # state_cache is an in-place operator input.  Copy the device-side update
    # back so the caller can compare the complete operator state transition.
    state_cache.copy_(npu_state_cache.cpu())
    return output


def golden_call(
    hidden_states: torch.Tensor,
    wk: torch.Tensor,
    gate_weight: torch.Tensor,
    ape: torch.Tensor,
    state_cache: torch.Tensor,
    cache_block_table: torch.Tensor,
    start_pos: torch.Tensor,
    *,
    norm_weight: torch.Tensor | None = None,
    norm_bias: torch.Tensor | None = None,
    cos: torch.Tensor | None = None,
    sin: torch.Tensor | None = None,
    cu_seqlens: torch.Tensor | None = None,
    seqused: torch.Tensor | None = None,
    cmp_ratio: int = 4,
    norm_eps: float = 1e-6,
    rotary_mode: int = 1,
) -> torch.Tensor:
    """Run the CPU Golden with the same signature and output as KeyPool."""
    # Golden projects K and gate for every current token, writes the unpooled
    # tail to the paged state cache, then reads each logical pool through the
    # block table.  For a complete pool it applies:
    #   softmax(gate + ape) @ LayerNorm(key)
    output, _ = run_key_pool_golden(
        hidden_states,
        wk,
        gate_weight,
        ape,
        state_cache,
        cache_block_table,
        start_pos,
        norm_weight=norm_weight,
        norm_bias=norm_bias,
        cos=cos,
        sin=sin,
        cu_seqlens=cu_seqlens,
        seqused=seqused,
        cmp_ratio=cmp_ratio,
        norm_eps=norm_eps,
        rotary_mode=rotary_mode,
        enforce_model_dims=False,
    )
    return output


def compare_precision(
    operator_output: torch.Tensor,
    golden_output: torch.Tensor,
    operator_state_cache: torch.Tensor,
    golden_state_cache: torch.Tensor,
    case_name: str,
) -> None:
    """Compare pooled output and the in-place state-cache result."""
    actual = operator_output.detach().cpu().float()
    expected = golden_output.detach().cpu().float()
    actual_cache = operator_state_cache.detach().cpu().float()
    expected_cache = golden_state_cache.detach().cpu().float()
    output_close = torch.isclose(
        actual, expected, rtol=0.0078125, atol=0.0001, equal_nan=True
    )
    cache_close = torch.isclose(
        actual_cache, expected_cache, rtol=0.0078125, atol=0.0001, equal_nan=True
    )
    output_percent = float(output_close.float().mean()) * 100.0
    cache_percent = float(cache_close.float().mean()) * 100.0
    output_max = float((actual - expected).abs().max()) if actual.numel() else 0.0
    cache_max = (
        float((actual_cache - expected_cache).abs().max())
        if actual_cache.numel()
        else 0.0
    )
    passed = output_percent >= 99.5 and cache_percent >= 99.5
    print(
        f"{case_name}: {'PASS' if passed else 'FAIL'}; "
        f"pooled_key={output_percent:.4f}% max_abs={output_max:.6g}; "
        f"state_cache={cache_percent:.4f}% max_abs={cache_max:.6g}"
    )
    if not passed:
        raise AssertionError(f"KeyPool precision check failed: {case_name}")


if __name__ == "__main__":
    failures = 0
    for case in generate_data():
        actual_inputs = {
            key: value.clone() if isinstance(value, torch.Tensor) else value
            for key, value in case.items()
            if key != "case"
        }
        golden_inputs = {
            key: value.clone() if isinstance(value, torch.Tensor) else value
            for key, value in actual_inputs.items()
        }
        try:
            actual_output = call_operator(**actual_inputs)
            torch_npu.npu.synchronize()
            golden_output = golden_call(**golden_inputs)
            compare_precision(
                actual_output,
                golden_output,
                actual_inputs["state_cache"],
                golden_inputs["state_cache"],
                case["case"],
            )
        except Exception as error:
            failures += 1
            print(f"{case['case']}: FAIL; {error}")
    if failures:
        raise SystemExit(1)
    print("KeyPool standalone precision check: all cases passed")
