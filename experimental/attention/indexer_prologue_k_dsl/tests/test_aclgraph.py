# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""ACLGraph capture/replay coverage for the installed ds41 operator."""

import math
import random

import pytest
import torch

from golden import indexer_prologue_k_golden, make_inputs

torch_npu = pytest.importorskip("torch_npu", reason="torch_npu not installed")
if not torch.npu.is_available():  # pragma: no cover - host dependent
    pytest.skip("no NPU visible", allow_module_level=True)

try:
    import torchair
    from torchair.configs.compiler_config import CompilerConfig
except Exception as error:  # pragma: no cover - environment dependent
    pytest.skip(f"torchair is unavailable: {error}", allow_module_level=True)


class _IndexerPrologueKGraph(torch.nn.Module):
    def __init__(
        self, storage_mode: int, combined_block_size: int, norm_eps: float = 1e-6
    ):
        super().__init__()
        self.storage_mode = storage_mode
        self.combined_block_size = combined_block_size
        self.norm_eps = norm_eps

    def forward(
        self,
        latent,
        wk,
        norm_weight,
        rope_sin,
        rope_cos,
        k_cache,
        k_scale_cache,
        cache_index,
    ):
        from cann_ops_transformer.ops.ds41 import indexer_prologue_k

        return indexer_prologue_k(
            latent,
            wk,
            norm_weight,
            rope_sin,
            rope_cos,
            k_cache,
            k_scale_cache,
            cache_index=cache_index,
            storage_mode=self.storage_mode,
            norm_eps=self.norm_eps,
            combined_block_size=self.combined_block_size,
        )


def _to_npu(inputs):
    scale_cache = inputs["k_scale_cache"]
    return (
        inputs["latent"].npu(),
        torch_npu.npu_format_cast(inputs["wk"].npu(), torch_npu.Format.FRACTAL_NZ),
        inputs["norm_weight"].npu(),
        inputs["rope_sin"].npu(),
        inputs["rope_cos"].npu(),
        inputs["k_cache"].npu(),
        None if scale_cache is None else scale_cache.npu(),
        inputs["cache_index"].npu(),
    )


@pytest.mark.npu
@pytest.mark.slow
@pytest.mark.parametrize(
    "storage_mode,group,with_scale",
    [(0, -1, True), (0, -1, False), (1, 4, False), (1, 8, False)],
)
def test_indexer_prologue_k_aclgraph_capture_and_replay(
    storage_mode, group, with_scale
):
    warmup_inputs = make_inputs(
        72,
        512,
        128,
        64,
        storage_mode=storage_mode,
        combined_block_size=group,
        with_scale_cache=with_scale,
        seed=700 + storage_mode * 10 + max(group, 0),
    )
    warmup_args = _to_npu(warmup_inputs)
    eager_model = _IndexerPrologueKGraph(storage_mode, group).npu()
    eager_model(*warmup_args)
    torch.npu.synchronize()

    inputs = make_inputs(
        72,
        512,
        128,
        64,
        storage_mode=storage_mode,
        combined_block_size=group,
        with_scale_cache=with_scale,
        seed=800 + storage_mode * 10 + max(group, 0),
    )
    expected_cache, expected_scale = indexer_prologue_k_golden(
        inputs,
        storage_mode=storage_mode,
        norm_eps=1e-6,
        combined_block_size=group,
    )
    graph_args = _to_npu(inputs)
    for tensor in graph_args:
        if tensor is not None:
            torch._dynamo.mark_static(tensor)

    config = CompilerConfig()
    config.mode = "npugraph_ex"
    # FRACTAL_NZ is private storage metadata. The default input clone uses
    # empty_like and degrades it to ND, which the DSL provider must reject.
    config.debug.aclgraph.clone_input = False
    backend = torchair.get_npu_backend(compiler_config=config)
    torch._dynamo.reset()
    compiled_model = torch.compile(
        _IndexerPrologueKGraph(storage_mode, group).npu(),
        fullgraph=True,
        backend=backend,
        dynamic=False,
    )

    result = compiled_model(*graph_args)
    torch.npu.synchronize()
    assert result is graph_args[5]
    assert torch.equal(result.cpu(), expected_cache)
    if graph_args[6] is not None:
        assert torch.equal(graph_args[6].cpu(), expected_scale)

    # Restore the output buffers. The second call must replay the captured
    # kernels and mutate them back to the golden values.
    graph_args[5].copy_(inputs["k_cache"].npu())
    if graph_args[6] is not None:
        graph_args[6].copy_(inputs["k_scale_cache"].npu())
    torch.npu.synchronize()
    replay_result = compiled_model(*graph_args)
    torch.npu.synchronize()
    assert replay_result is graph_args[5]
    assert torch.equal(replay_result.cpu(), expected_cache)
    if graph_args[6] is not None:
        assert torch.equal(graph_args[6].cpu(), expected_scale)


def _make_deployment_cases():
    """Fixed H/D/Dr deployment matrix, including a non-aligned T=7777.

    Mirrors the arena deployment suite so both verification layers cover
    the same 50 shapes.
    """
    rng = random.Random(20260920)
    token_counts = [
        1,
        2,
        15,
        16,
        17,
        31,
        32,
        33,
        63,
        64,
        65,
        127,
        128,
        129,
        255,
        256,
        257,
        511,
        512,
        513,
        1023,
        1024,
        1025,
        2048,
        4096,
        7777,
        8192,
    ]
    while len(token_counts) < 50:
        candidate = rng.randint(1, 8192)
        if candidate not in token_counts:
            token_counts.append(candidate)

    cases = []
    groups = (2, 4, 8, 16)
    epsilons = (1e-6, 5e-7, 1e-5)
    for case_idx, t in enumerate(token_counts):
        block_size = 64 if case_idx % 2 == 0 else 128
        block_num = math.ceil(t / block_size) + case_idx % 3
        storage_mode = 0 if case_idx % 3 != 2 else 1
        group_size = -1 if storage_mode == 0 else groups[case_idx % len(groups)]
        with_scale = storage_mode == 0 and case_idx % 2 == 0
        block_axis_step = 1 if case_idx % 4 == 0 else 2 + case_idx % 2
        block_axis_start = 0 if block_axis_step == 1 else case_idx % 2
        duplicate_indices = t >= 4 and case_idx % 7 == 0
        cases.append(
            pytest.param(
                t,
                block_num,
                block_size,
                epsilons[case_idx % len(epsilons)],
                storage_mode,
                group_size,
                with_scale,
                duplicate_indices,
                block_axis_step,
                block_axis_start,
                50_000 + case_idx,
                id=(
                    f"{case_idx:02d}_t{t}_bs{block_size}_mode{storage_mode}_"
                    f"g{group_size}_stride{block_axis_step}"
                ),
            )
        )
    return cases


def _is_acceptable_quantization_tie(
    actual_cache,
    expected_cache,
    mismatch,
    *,
    total_rows,
    index_head_dim,
    storage_mode,
    combined_block_size,
):
    """Allow only sparse, same-sign adjacent E2M1 choices at exact ties.

    The CPU and vector-core FP32 paths can place the BF16 value on opposite
    sides of an E2M1 midpoint by one ULP. Both adjacent codes are then nearest
    for their respective intermediate. The two zero encodings (+0.0/-0.0) are
    also equivalent. Scale bytes and non-data cache regions remain bit-exact.

    The sparsity cap is looser than the arena suite's 1-per-million because
    this suite's randn-angle RoPE coefficients hit exact quantization
    boundaries several times more often (about 4 per million is observed).
    """
    max_mismatched_bytes = max(1, math.ceil(total_rows * index_head_dim / 100_000))
    if mismatch.shape[0] > max_mismatched_bytes:
        return False

    value_bytes = index_head_dim // 2
    data_region_bytes = (
        value_bytes if storage_mode == 0 else combined_block_size * value_bytes
    )
    for coordinate in mismatch:
        index = tuple(coordinate.tolist())
        if index[-1] >= data_region_bytes:
            return False
        actual_byte = int(actual_cache[index])
        expected_byte = int(expected_cache[index])
        for shift in (0, 4):
            actual_code = (actual_byte >> shift) & 0xF
            expected_code = (expected_byte >> shift) & 0xF
            if actual_code == expected_code:
                continue
            if (actual_code & 0x8) != (expected_code & 0x8):
                # E2M1 encodes both +0.0 and -0.0; the two zero encodings
                # are value-identical, so a sign-only difference on a zero
                # magnitude is acceptable.
                if (actual_code & 0x7) == 0 and (expected_code & 0x7) == 0:
                    continue
                return False
            if abs((actual_code & 0x7) - (expected_code & 0x7)) != 1:
                return False
    return True


def _assert_cache_matches(
    actual,
    expected,
    *,
    total_rows,
    storage_mode,
    combined_block_size,
):
    if torch.equal(actual, expected):
        return
    mismatch = (actual != expected).nonzero()
    if _is_acceptable_quantization_tie(
        actual,
        expected,
        mismatch,
        total_rows=total_rows,
        index_head_dim=128,
        storage_mode=storage_mode,
        combined_block_size=combined_block_size,
    ):
        return
    details = []
    for coordinate in mismatch[:8]:
        index = tuple(coordinate.tolist())
        details.append((index, int(actual[index]), int(expected[index])))
    pytest.fail(
        f"k_cache mismatch_count={mismatch.shape[0]}, first_mismatches={details}"
    )


class _StridedCache:
    """Dim-0 strided view over a 0xA5-filled backing, like the arena suite."""

    def __init__(self, tensor, step, start):
        if step == 1 and start == 0:
            self.backing_npu = tensor.npu()
            self.view_npu = self.backing_npu
            self.before = None
            self.selected = None
            return
        stop = start + int(tensor.shape[0]) * step
        backing_shape = (stop + 1, *tensor.shape[1:])
        backing = torch.full(backing_shape, 0xA5, dtype=torch.uint8)
        backing[start:stop:step].copy_(tensor)
        self.before = backing.clone()
        self.backing_npu = backing.npu()
        self.view_npu = self.backing_npu[start:stop:step]
        self.selected = set(range(start, stop, step))

    def assert_untouched_rows(self):
        if self.before is None:
            return
        after = self.backing_npu.cpu()
        for block_index in range(after.shape[0]):
            if block_index not in self.selected:
                assert torch.equal(after[block_index], self.before[block_index])


def _build_deployment_args(
    t,
    block_num,
    block_size,
    storage_mode,
    group,
    with_scale,
    duplicate_indices,
    block_axis_step,
    block_axis_start,
    seed,
):
    inputs = make_inputs(
        t,
        512,
        128,
        64,
        block_num=block_num,
        block_size=block_size,
        storage_mode=storage_mode,
        combined_block_size=group,
        with_scale_cache=with_scale,
        seed=seed,
    )
    if duplicate_indices:
        cache_index = inputs["cache_index"]
        cache_index[1] = cache_index[0]
        if storage_mode == 1:
            # Exercise two sub-slots in one combined row as well as an exact
            # duplicate.  Destination-row hashing must keep both ordered.
            cache_index[2] = 0
            cache_index[3] = 1
    k_cache = _StridedCache(inputs["k_cache"], block_axis_step, block_axis_start)
    scale_cache = (
        None
        if inputs["k_scale_cache"] is None
        else _StridedCache(inputs["k_scale_cache"], block_axis_step, block_axis_start)
    )
    args = (
        inputs["latent"].npu(),
        torch_npu.npu_format_cast(inputs["wk"].npu(), torch_npu.Format.FRACTAL_NZ),
        inputs["norm_weight"].npu(),
        inputs["rope_sin"].npu(),
        inputs["rope_cos"].npu(),
        k_cache.view_npu,
        None if scale_cache is None else scale_cache.view_npu,
        inputs["cache_index"].npu(),
    )
    return inputs, args, k_cache, scale_cache


@pytest.mark.npu
@pytest.mark.slow
@pytest.mark.parametrize(
    (
        "t,block_num,block_size,norm_eps,storage_mode,group_size,"
        "with_scale,duplicate_indices,block_axis_step,block_axis_start,seed"
    ),
    _make_deployment_cases(),
)
def test_indexer_prologue_k_aclgraph_deployment_50(
    t,
    block_num,
    block_size,
    norm_eps,
    storage_mode,
    group_size,
    with_scale,
    duplicate_indices,
    block_axis_step,
    block_axis_start,
    seed,
):
    # Warm up the eager path so the Native provider JIT-compiles this
    # shape/stride signature before ACLGraph capture starts.
    _, warmup_args, warmup_cache, warmup_scale = _build_deployment_args(
        t,
        block_num,
        block_size,
        storage_mode,
        group_size,
        with_scale,
        duplicate_indices,
        block_axis_step,
        block_axis_start,
        seed + 100_000,
    )
    eager_model = _IndexerPrologueKGraph(storage_mode, group_size, norm_eps).npu()
    eager_model(*warmup_args)
    torch.npu.synchronize()

    inputs, graph_args, k_cache, scale_cache = _build_deployment_args(
        t,
        block_num,
        block_size,
        storage_mode,
        group_size,
        with_scale,
        duplicate_indices,
        block_axis_step,
        block_axis_start,
        seed,
    )
    expected_cache, expected_scale = indexer_prologue_k_golden(
        inputs,
        storage_mode=storage_mode,
        norm_eps=norm_eps,
        combined_block_size=group_size,
    )
    for tensor in graph_args:
        if tensor is not None:
            torch._dynamo.mark_static(tensor)

    config = CompilerConfig()
    config.mode = "npugraph_ex"
    # FRACTAL_NZ is private storage metadata. The default input clone uses
    # empty_like and degrades it to ND, which the DSL provider must reject.
    config.debug.aclgraph.clone_input = False
    backend = torchair.get_npu_backend(compiler_config=config)
    torch._dynamo.reset()
    compiled_model = torch.compile(
        _IndexerPrologueKGraph(storage_mode, group_size, norm_eps).npu(),
        fullgraph=True,
        backend=backend,
        dynamic=False,
    )

    result = compiled_model(*graph_args)
    torch.npu.synchronize()
    assert result is graph_args[5]
    _assert_cache_matches(
        result.cpu(),
        expected_cache,
        total_rows=t,
        storage_mode=storage_mode,
        combined_block_size=group_size,
    )
    if graph_args[6] is not None:
        assert torch.equal(graph_args[6].cpu(), expected_scale)
    k_cache.assert_untouched_rows()
    if scale_cache is not None:
        scale_cache.assert_untouched_rows()

    # Restore the output buffers. The second call must replay the captured
    # kernels and mutate them back to the golden values.
    graph_args[5].copy_(inputs["k_cache"].npu())
    if graph_args[6] is not None:
        graph_args[6].copy_(inputs["k_scale_cache"].npu())
    torch.npu.synchronize()
    replay_result = compiled_model(*graph_args)
    torch.npu.synchronize()
    assert replay_result is graph_args[5]
    _assert_cache_matches(
        replay_result.cpu(),
        expected_cache,
        total_rows=t,
        storage_mode=storage_mode,
        combined_block_size=group_size,
    )
    if graph_args[6] is not None:
        assert torch.equal(graph_args[6].cpu(), expected_scale)
    k_cache.assert_untouched_rows()
    if scale_cache is not None:
        scale_cache.assert_untouched_rows()
