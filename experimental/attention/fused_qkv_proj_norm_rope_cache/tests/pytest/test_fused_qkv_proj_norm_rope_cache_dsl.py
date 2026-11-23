# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""DSL precision regression for fused_qkv_proj_norm_rope_cache."""

import importlib
import os
import sys
from pathlib import Path

import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")

pytestmark = pytest.mark.npu

HEAD_DIM = 256
ROPE_DIM = 64
NOPE_DIM = HEAD_DIM - ROPE_DIM
HALF_ROPE_DIM = ROPE_DIM // 2
NUM_Q_HEADS = 6
Q_WIDTH = NUM_Q_HEADS * HEAD_DIM
UNIT_K = NUM_Q_HEADS
UNIT_V = NUM_Q_HEADS + 1
UNIT_MK = NUM_Q_HEADS + 2
UNIT_MV = NUM_Q_HEADS + 3
HIDDEN = 2048

_OP_DIR = Path(__file__).resolve().parents[2]
_OP_DSL = _OP_DIR / "op_dsl"
_OP_MODULE = None


def _module():
    global _OP_MODULE
    if _OP_MODULE is None:
        _torch_extension_module()
        _OP_MODULE = sys.modules[
            "fused_qkv_torch_extension_test.op_dsl.fused_qkv_proj_norm_rope_cache"
        ]
    return _OP_MODULE


def _implementation():
    return _torch_extension_module()._fused_qkv_proj_norm_rope_cache_impl


def _device():
    raw = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "0").split(",")[0].strip()
    return int(raw or "0")


def _run_operator(inputs, case):
    """Invoke the DSL entry and normalize the optional outputs."""
    return_v = case.get("return_v", True)
    eps = case.get("eps", 1e-6)
    npu_inputs = tuple(tensor.npu() for tensor in inputs)
    k_cache_npu = npu_inputs[6].clone()
    v_cache_npu = npu_inputs[7].clone()
    outputs = _implementation()(
        npu_inputs[0],
        npu_inputs[1],
        npu_inputs[2],
        npu_inputs[3],
        npu_inputs[4],
        npu_inputs[5],
        k_cache_npu,
        v_cache_npu,
        eps=float(eps),
        return_v=bool(return_v),
        positions_contiguous=case.get("contiguous", True),
    )
    torch.npu.synchronize()
    if int(npu_inputs[1].shape[0]) // HEAD_DIM == 10:
        if return_v:
            q, k, v, mirror_k, mirror_v = outputs
        else:
            q, k, mirror_k, mirror_v = outputs
            v = torch.empty(0, dtype=q.dtype, device=q.device)
    else:
        if return_v:
            q, k, v = outputs
        else:
            q, k = outputs
            v = torch.empty(0, dtype=q.dtype, device=q.device)
        mirror_k = torch.empty(0, dtype=q.dtype, device=q.device)
        mirror_v = torch.empty(0, dtype=q.dtype, device=q.device)
    return q, k, v, mirror_k, mirror_v, k_cache_npu, v_cache_npu


def _rope_tail(values, cos, sin):
    x1 = values[:, NOPE_DIM : NOPE_DIM + HALF_ROPE_DIM]
    x2 = values[:, NOPE_DIM + HALF_ROPE_DIM : HEAD_DIM]
    out = values.clone()
    out[:, NOPE_DIM : NOPE_DIM + HALF_ROPE_DIM] = x1 * cos - x2 * sin
    out[:, NOPE_DIM + HALF_ROPE_DIM : HEAD_DIM] = x1 * sin + x2 * cos
    return out


def golden(
    hidden,
    weight,
    k_gamma,
    positions,
    cos_sin_cache,
    slot_mapping,
    k_cache,
    v_cache,
    eps,
    return_v,
):
    """CPU FP32 reference for the fused QKV projection."""
    n_units = int(weight.shape[0]) // HEAD_DIM
    hidden_cpu = hidden.detach().float().cpu()
    weight_cpu = weight.detach().float().cpu()
    gamma_cpu = k_gamma.detach().float().cpu()
    positions_cpu = positions.detach().cpu().to(torch.int64)
    table_cpu = cos_sin_cache.detach().float().cpu()
    slot_cpu = slot_mapping.detach().cpu().to(torch.int64)

    qkv = (hidden_cpu @ weight_cpu.T).bfloat16().float()
    cos = table_cpu[positions_cpu, :HALF_ROPE_DIM]
    sin = table_cpu[positions_cpu, HALF_ROPE_DIM:]

    q_fp32 = qkv[:, :Q_WIDTH].clone()
    for head in range(NUM_Q_HEADS):
        base = head * HEAD_DIM
        q_fp32[:, base : base + HEAD_DIM] = _rope_tail(
            qkv[:, base : base + HEAD_DIM], cos, sin
        )
    q = q_fp32.bfloat16()

    k_fp32 = qkv[:, UNIT_K * HEAD_DIM : (UNIT_K + 1) * HEAD_DIM]
    var = (k_fp32 * k_fp32).mean(dim=-1, keepdim=True) + float(eps)
    k_norm = (k_fp32 * torch.rsqrt(var) * gamma_cpu).bfloat16().float()
    k = _rope_tail(k_norm, cos, sin).bfloat16()
    v = qkv[:, UNIT_V * HEAD_DIM : (UNIT_V + 1) * HEAD_DIM].bfloat16()

    if n_units == 10:
        mirror_k = qkv[:, UNIT_MK * HEAD_DIM : (UNIT_MK + 1) * HEAD_DIM].bfloat16()
        mirror_v = qkv[:, UNIT_MV * HEAD_DIM : (UNIT_MV + 1) * HEAD_DIM].bfloat16()
    else:
        mirror_k = torch.empty(0, dtype=q.dtype)
        mirror_v = torch.empty(0, dtype=q.dtype)

    cache_shape = tuple(k_cache.shape)
    k_cache_out = k_cache.detach().float().cpu().reshape(-1, HEAD_DIM).clone()
    v_cache_out = v_cache.detach().float().cpu().reshape(-1, HEAD_DIM).clone()
    for row, slot in enumerate(slot_cpu.tolist()):
        if slot != -1:
            k_cache_out[slot] = k[row].float()
            v_cache_out[slot] = v[row].float()
    k_cache_out = k_cache_out.bfloat16().reshape(cache_shape)
    v_cache_out = v_cache_out.bfloat16().reshape(cache_shape)

    if not return_v:
        v = torch.empty(0, dtype=q.dtype)
    return q, k, v, mirror_k, mirror_v, k_cache_out, v_cache_out


def make_inputs(case):
    """Deterministic inputs covering the case's shape and flag combination."""
    tokens = case["tokens"]
    units = case.get("units", 10)
    max_pos = case.get("max_pos", 4096)
    slots = case.get("slots", 1024)
    index_dtype = getattr(torch, case.get("index_dtype", "int64"))
    gen = torch.Generator().manual_seed(case.get("seed", tokens * 7 + units))
    hidden = (torch.randn(tokens, HIDDEN, generator=gen) * 0.1).bfloat16()
    weight = (torch.randn(units * HEAD_DIM, HIDDEN, generator=gen) * 0.05).bfloat16()
    k_gamma = (torch.randn(HEAD_DIM, generator=gen) * 0.1 + 1.0).bfloat16()
    cos_sin = torch.randn(max_pos, ROPE_DIM, generator=gen)
    offset = case.get("offset", 0)
    contiguous = case.get("contiguous", True)
    if contiguous is None:
        contiguous = True
    if contiguous:
        positions = (torch.arange(tokens, dtype=index_dtype) + offset) % max_pos
    else:
        positions = torch.randint(
            offset, max_pos, (tokens,), generator=gen, dtype=index_dtype
        )
    slot_mapping = torch.arange(tokens, dtype=index_dtype) % slots
    skip_mod = int(case.get("skip_mod", 0))
    if skip_mod:
        slot_mapping[::skip_mod] = -1
    page_size = case.get("page_size")
    if page_size:
        cache_shape = (slots // page_size, page_size, HEAD_DIM)
    else:
        cache_shape = (slots, HEAD_DIM)
    k_cache = torch.zeros(cache_shape, dtype=torch.bfloat16)
    v_cache = torch.zeros(cache_shape, dtype=torch.bfloat16)
    return (
        hidden,
        weight,
        k_gamma,
        positions,
        cos_sin,
        slot_mapping,
        k_cache,
        v_cache,
    )


CASES = [
    dict(name="prefill_m1", tokens=1),
    dict(name="prefill_m2", tokens=2),
    dict(name="prefill_m7", tokens=7),
    dict(name="prefill_m8", tokens=8),
    dict(name="prefill_m16", tokens=16),
    dict(name="prefill_m31", tokens=31),
    dict(name="prefill_m33", tokens=33),
    dict(name="prefill_m63", tokens=63),
    dict(name="prefill_m64", tokens=64),
    dict(name="prefill_m65", tokens=65),
    dict(name="prefill_m72", tokens=72),
    dict(name="prefill_m100", tokens=100),
    dict(name="prefill_m127", tokens=127),
    dict(name="prefill_m128", tokens=128),
    dict(name="prefill_m200", tokens=200),
    dict(name="prefill_m255", tokens=255),
    dict(name="prefill_m256", tokens=256),
    dict(name="prefill_m257", tokens=257, seed=3),
    dict(name="prefill_m300", tokens=300, seed=5),
    dict(name="prefill_m511", tokens=511, seed=7),
    dict(name="prefill_m512", tokens=512, seed=11),
    dict(name="prefill_m513", tokens=513, seed=12),
    dict(name="prefill_m1000", tokens=1000, seed=13),
    dict(name="decode_m8", tokens=8, contiguous=False, seed=17),
    dict(name="decode_m31", tokens=31, contiguous=False, seed=18),
    dict(name="decode_m63", tokens=63, contiguous=False, seed=21),
    dict(name="decode_m257", tokens=257, contiguous=False, seed=19),
    dict(name="decode_m513", tokens=513, contiguous=False, seed=22),
    dict(name="plain_m72", tokens=72, units=8, seed=23),
    dict(name="plain_m300", tokens=300, units=8, seed=24),
    dict(name="plain_decode_m64", tokens=64, units=8, contiguous=False, seed=25),
    dict(name="plain_decode_m257", tokens=257, units=8, contiguous=False, seed=26),
    dict(name="plain_skip_m72", tokens=72, units=8, skip_mod=4, seed=27),
    dict(name="skip_slots_m128_step2", tokens=128, skip_mod=2, seed=30),
    dict(name="skip_slots_m128_step3", tokens=128, skip_mod=3, seed=31),
    dict(name="skip_slots_m64_step5", tokens=64, skip_mod=5, seed=32),
    dict(name="offset_m128", tokens=128, offset=1024, seed=37),
    dict(name="offset_m72", tokens=72, offset=7, seed=38),
    dict(name="auto_detect_m72", tokens=72, offset=9, contiguous=None, seed=39),
    dict(name="eps1e5_m64", tokens=64, eps=1e-5, seed=40),
    dict(name="eps1e3_m64", tokens=64, eps=1e-3, seed=41),
    dict(name="eps1e1_m33", tokens=33, eps=1e-1, seed=42),
    dict(name="return_v_false_m72", tokens=72, return_v=False, seed=43),
    dict(name="small_cache_m32", tokens=32, slots=64, skip_mod=5, seed=47),
    dict(name="small_table_m256", tokens=256, max_pos=256, seed=53),
    dict(name="paged_cache_m72", tokens=72, slots=96, page_size=16, seed=57),
    dict(name="int32_index_m72", tokens=72, index_dtype="int32", seed=59),
    dict(name="large_table_m300", tokens=300, max_pos=8192, offset=4096, seed=61),
]


@pytest.mark.parametrize("case", CASES, ids=[case["name"] for case in CASES])
def test_dsl_matches_cpu_golden(case):
    torch.npu.set_device(_device())
    return_v = case.get("return_v", True)
    eps = case.get("eps", 1e-6)
    inputs = make_inputs(case)
    expected = golden(*inputs, eps, return_v)
    actual = _run_operator(inputs, case)
    for name, got, ref in zip(
        ("q", "k", "v", "mirror_k", "mirror_v", "k_cache", "v_cache"),
        actual,
        expected,
    ):
        got_cpu = got.detach().cpu().float()
        ref_cpu = ref.detach().cpu().float()
        assert got_cpu.shape == ref_cpu.shape, f"{name} shape mismatch"
        torch.testing.assert_close(
            got_cpu, ref_cpu, rtol=2e-2, atol=2e-2, msg=f"{name} mismatch"
        )


def test_rejects_duplicate_slots():
    """Non-negative slots must be unique; conflicting scatter is rejected."""
    torch.npu.set_device(_device())
    case = dict(tokens=72, slots=64, page_size=16, seed=57)
    inputs = make_inputs(case)
    with pytest.raises(AssertionError, match="duplicate non-negative slots"):
        _run_operator(inputs, case)


def test_stream_quota_change_recompiles():
    """A lower stream quota must not reuse a larger over-subscribed launch."""
    torch.npu.set_device(_device())
    case = dict(name="quota_m777", tokens=777, seed=211)
    inputs = make_inputs(case)
    expected = golden(*inputs, 1e-6, True)
    stream = torch.npu.current_stream()
    factory = _module()._get_compiled

    def cached_quotas():
        quotas = set()
        for key in factory._cache:
            config = {name: value for name, _kind, value in key[0]}
            quotas.add(config["block_num"])
        return quotas

    try:
        torch.npu.set_stream_limit(stream, 32, 64)
        wide = _run_operator(inputs, case)
        assert 32 in cached_quotas(), (
            "32-AIC quota must be part of the compile cache key"
        )

        torch.npu.set_stream_limit(stream, 8, 16)
        narrow = _run_operator(inputs, case)
        quotas = cached_quotas()
        assert 8 in quotas, "8-AIC quota must compile a separate program"
        assert 32 in quotas, "the wider program must remain independently cached"

        for got, ref in zip(wide, expected):
            torch.testing.assert_close(
                got.detach().cpu().float(),
                ref.detach().cpu().float(),
                rtol=2e-2,
                atol=2e-2,
            )
        for got, ref in zip(narrow, expected):
            torch.testing.assert_close(
                got.detach().cpu().float(),
                ref.detach().cpu().float(),
                rtol=2e-2,
                atol=2e-2,
            )
    finally:
        torch.npu.reset_stream_limit(stream)


_TORCH_EXTENSION = None


def _torch_extension_module():
    """Load the shipped Torch adapter once, stubbing op_builder when unbuilt."""
    global _TORCH_EXTENSION
    if _TORCH_EXTENSION is not None:
        return _TORCH_EXTENSION

    import types

    try:
        from cann_ops_transformer.op_builder import get_as_library
    except ImportError:
        from torch.library import Library

        library = None

        def get_as_library():
            nonlocal library
            if library is None:
                try:
                    library = Library("cann_ops_transformer", "DEF")
                except RuntimeError:
                    library = Library("cann_ops_transformer", "FRAGMENT")
            return library

        package = types.ModuleType("cann_ops_transformer")
        package.__path__ = []
        op_builder = types.ModuleType("cann_ops_transformer.op_builder")
        op_builder.get_as_library = get_as_library
        package.op_builder = op_builder
        sys.modules["cann_ops_transformer"] = package
        sys.modules["cann_ops_transformer.op_builder"] = op_builder

    package_name = "fused_qkv_torch_extension_test"
    torch_extension = _OP_DIR / "torch_extension"
    system_package = types.ModuleType(package_name)
    system_package.__path__ = [str(torch_extension)]
    sys.modules[package_name] = system_package
    op_dsl_package = types.ModuleType(package_name + ".op_dsl")
    op_dsl_package.__path__ = [str(_OP_DSL)]
    sys.modules[package_name + ".op_dsl"] = op_dsl_package
    spec = importlib.util.spec_from_file_location(
        package_name + ".fused_qkv_proj_norm_rope_cache",
        torch_extension / "fused_qkv_proj_norm_rope_cache.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    _TORCH_EXTENSION = module
    return module


def _graph_entry(
    hidden,
    weight,
    k_gamma,
    positions,
    cos_sin_cache,
    slot_mapping,
    k_cache,
    v_cache,
    *,
    positions_contiguous=True,
):
    """Call the registered dispatcher op so Dynamo captures one custom op."""
    return torch.ops.cann_ops_transformer.fused_qkv_proj_norm_rope_cache(
        hidden,
        weight,
        k_gamma,
        positions,
        cos_sin_cache,
        slot_mapping,
        k_cache,
        v_cache,
        positions_contiguous=positions_contiguous,
    )


def _npu_args(inputs):
    return [tensor.npu() for tensor in inputs]


def _assert_matches_golden(outputs, args, expected):
    for name, got, ref in zip(
        ("q", "k", "v", "mirror_k", "mirror_v", "k_cache", "v_cache"),
        (*outputs, args[6], args[7]),
        expected,
    ):
        torch.testing.assert_close(
            got.detach().cpu().float(),
            ref.detach().cpu().float(),
            rtol=2e-2,
            atol=2e-2,
            msg=f"{name} mismatch",
        )


GRAPH_CASES = [
    dict(name="graph_m1", tokens=1, seed=301),
    dict(name="graph_prefill_m72", tokens=72, seed=302),
    dict(name="graph_decode_m8", tokens=8, contiguous=False, seed=303),
]


@pytest.mark.parametrize(
    "case", GRAPH_CASES, ids=[case["name"] for case in GRAPH_CASES]
)
def test_torch_entry_aclgraph_capture_and_replay(case):
    """The registered entry must capture and replay without host readbacks."""
    torch.npu.set_device(_device())
    _torch_extension_module()
    op = torch.ops.cann_ops_transformer.fused_qkv_proj_norm_rope_cache
    inputs = make_inputs(case)
    expected = golden(*inputs, 1e-6, True)

    warmup = _npu_args(inputs)
    op(*warmup, positions_contiguous=case.get("contiguous", True))
    torch.npu.synchronize()

    args = _npu_args(inputs)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        captured = op(*args, positions_contiguous=case.get("contiguous", True))
    graph.replay()
    torch.npu.synchronize()
    _assert_matches_golden(captured, args, expected)

    args[6].zero_()
    args[7].zero_()
    torch.npu.synchronize()
    graph.replay()
    torch.npu.synchronize()
    _assert_matches_golden(captured, args, expected)


@pytest.mark.parametrize(
    "case", GRAPH_CASES, ids=[case["name"] for case in GRAPH_CASES]
)
def test_torch_entry_npugraph_ex_capture_and_replay(case):
    """npugraph_ex must capture the public entry with in-place cache writes."""
    torch.npu.set_device(_device())
    _torch_extension_module()
    contiguous = case.get("contiguous", True)
    op = torch.ops.cann_ops_transformer.fused_qkv_proj_norm_rope_cache
    inputs = make_inputs(case)
    expected = golden(*inputs, 1e-6, True)

    warmup = _npu_args(inputs)
    op(*warmup, positions_contiguous=contiguous)
    torch.npu.synchronize()

    compiled = torch.compile(
        _graph_entry,
        backend="npugraph_ex",
        fullgraph=True,
        dynamic=False,
        options={"clone_input": False},
    )
    args = _npu_args(inputs)
    for _ in range(2):
        args[6].zero_()
        args[7].zero_()
        torch.npu.synchronize()
        outputs = compiled(*args, positions_contiguous=contiguous)
        torch.npu.synchronize()
        _assert_matches_golden(outputs, args, expected)


def test_torch_registration_meta_matches_actual_for_plain_layer():
    """The formal Torch entry must expose identical metadata for 8-unit layers."""
    torch.npu.set_device(_device())
    _torch_extension_module()
    op = torch.ops.cann_ops_transformer.fused_qkv_proj_norm_rope_cache
    tokens, slots = 8, 512
    meta_args = (
        torch.empty((tokens, HIDDEN), dtype=torch.bfloat16, device="meta"),
        torch.empty((8 * HEAD_DIM, HIDDEN), dtype=torch.bfloat16, device="meta"),
        torch.empty((HEAD_DIM,), dtype=torch.bfloat16, device="meta"),
        torch.empty((tokens,), dtype=torch.int64, device="meta"),
        torch.empty((512, ROPE_DIM), dtype=torch.float32, device="meta"),
        torch.empty((tokens,), dtype=torch.int64, device="meta"),
        torch.empty((slots, HEAD_DIM), dtype=torch.bfloat16, device="meta"),
        torch.empty((slots, HEAD_DIM), dtype=torch.bfloat16, device="meta"),
    )
    case = dict(name="plain_meta_m8", tokens=tokens, units=8, slots=slots, seed=223)
    inputs = make_inputs(case)
    real_args = []
    for index, tensor in enumerate(inputs):
        if index == 6 or index == 7:
            real_args.append(torch.zeros_like(tensor).npu())
        else:
            real_args.append(tensor.npu())

    meta_outputs = op(*meta_args)
    real_outputs = op(*real_args)
    torch.npu.synchronize()
    assert len(meta_outputs) == len(real_outputs) == 5
    for meta, real in zip(meta_outputs, real_outputs):
        assert meta.dim() == real.dim()
        assert tuple(meta.shape) == tuple(real.shape)
        assert meta.dtype == real.dtype
    assert tuple(real_outputs[3].shape) == (0, HEAD_DIM)
    assert tuple(real_outputs[4].shape) == (0, HEAD_DIM)
