# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Torch/DSL regression with local input builders and independent CPU golden."""

import builtins
from contextlib import contextmanager
from dataclasses import dataclass
import importlib
import importlib.util
import os
from pathlib import Path
import sys
import types
from typing import Optional

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

OP_ROOT = Path(__file__).resolve().parents[2]


def _selected_backend():
    backend = os.environ.get("QKV_TEST_BACKEND")
    if backend is None:
        # Keep the previous explicit OPP command working without gating normal DSL tests.
        backend = "opp" if os.environ.get("QKV_RUN_OPP") == "1" else "source"
    if backend not in ("source", "opp"):
        raise pytest.UsageError("QKV_TEST_BACKEND must be 'source' or 'opp'")
    return backend


@pytest.fixture(scope="session")
def adapter():
    pytest.importorskip(
        "torch_npu", reason="The Torch adapter imports the CANN Python package"
    )
    if _selected_backend() == "opp":
        try:
            importlib.import_module("cann_ops_transformer")
            ops = importlib.import_module("cann_ops_transformer.ops")
            entry = ops.fused_quant_qkv_proj_norm_rope_cache
            return importlib.import_module(entry.__module__)
        except (ImportError, AttributeError) as error:
            pytest.fail(
                f"OPP backend requires the matching installed RUN/wheel: {error}. "
                "Use QKV_TEST_BACKEND=source for source DSL regression.",
                pytrace=False,
            )
    # Source mode owns dispatcher registration. Load the repository OpBuilder
    # without running package entry-point discovery for installed operators.
    if hasattr(torch.ops.cann_ops_transformer, "fused_quant_qkv_proj_norm_rope_cache"):
        pytest.fail(
            "Run source regression in a fresh process before importing installed operators"
        )
    base_root = OP_ROOT.parents[2] / "torch_extension" / "cann_ops_transformer"
    base_package = types.ModuleType("cann_ops_transformer")
    base_package.__path__ = [str(base_root)]
    sys.modules["cann_ops_transformer"] = base_package
    # Reproduce wheel layout for the operator module and its DSL sibling.
    name = "qkv_delivery_test.ops.posembedding.fused_quant_qkv_proj_norm_rope_cache"
    package = types.ModuleType(name)
    package.__path__ = [str(OP_ROOT / "torch_extension"), str(OP_ROOT)]
    sys.modules[name] = package
    module_name = name + ".fused_quant_qkv_proj_norm_rope_cache"
    spec = importlib.util.spec_from_file_location(
        module_name, OP_ROOT / "torch_extension/fused_quant_qkv_proj_norm_rope_cache.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def inputs():
    def make(m=3, n=2048, t=None, s=16, p=32, device="cpu"):
        t = m if t is None else t

        def empty(shape, dtype):
            return torch.empty(shape, dtype=dtype, device=device)

        return [
            empty((m, 2048), torch.float8_e4m3fn),
            empty((m, 32, 2), torch.float8_e8m0fnu),
            empty((n, 2048), torch.float8_e4m3fn),
            empty((n, 32, 2), torch.float8_e8m0fnu),
            empty((256,), torch.bfloat16),
            empty((t,), torch.int64),
            empty((p, 64), torch.float32),
            empty((t,), torch.int64),
            empty((t, 1536), torch.bfloat16),
            empty((t, 256), torch.bfloat16),
            empty((t, 256), torch.bfloat16),
            empty((s, 256), torch.bfloat16),
            empty((s, 256), torch.bfloat16),
        ]

    return make


@pytest.fixture(scope="session", params=[_selected_backend()])
def npu_runner(request, adapter):
    """Run source regressions by default; OPP mode never compiles a fallback."""
    pytest.importorskip(
        "torch_npu", reason="NPU tests require torch_npu and its CANN runtime"
    )
    if not torch.npu.is_available():
        pytest.skip("NPU unavailable")
    torch.npu.set_device(os.environ.get("QKV_DEVICE", "npu:0"))
    if request.param == "opp":
        if os.environ.get("CANNBOTDSL_NATIVE_BINARY_MODE") != "require":
            pytest.fail(
                "OPP validation requires CANNBOTDSL_NATIVE_BINARY_MODE=require; no source fallback",
                pytrace=False,
            )
        return torch.ops.cann_ops_transformer.fused_quant_qkv_proj_norm_rope_cache

    try:
        factory = importlib.import_module(
            adapter.__package__ + ".op_dsl.fused_quant_qkv_proj_norm_rope_cache"
        )
    except ImportError as error:
        pytest.fail(
            f"Source backend requires cannbot-dsl with the @host compiler: {error}",
            pytrace=False,
        )
    programs = {}

    def invoke(*args, **kwargs):
        has_mirror, has_chunk, chunk = adapter._validate_metadata(*args, **kwargs)
        (
            qx,
            sa,
            weight,
            sb,
            gamma,
            positions,
            cos_sin,
            slots,
            q,
            k,
            v,
            k_cache,
            v_cache,
        ) = args
        contiguous = kwargs.get("positions_contiguous", False)
        act_dtype = (
            factory.FP8_E5M2 if qx.dtype == torch.float8_e5m2 else factory.FP8_E4M3
        )
        weight_dtype = (
            factory.FP8_E5M2 if weight.dtype == torch.float8_e5m2 else factory.FP8_E4M3
        )
        choice = (
            has_mirror,
            contiguous,
            has_chunk,
            act_dtype,
            weight_dtype,
        )
        with torch.npu.device(qx.device):
            if choice not in programs:
                programs[choice] = factory._compile_source(*choice)
            if has_mirror:
                mk, mv = kwargs["mirror_k_out"], kwargs["mirror_v_out"]
            else:
                mk = torch.empty(
                    (1, k.shape[1]), dtype=torch.bfloat16, device=qx.device
                )
                mv = torch.empty_like(mk)
            cores = int(torch.npu.get_device_properties(qx.device).cube_core_num)
            if cores <= 0:
                raise RuntimeError("cannot determine a positive cube core count")
            programs[choice](
                qx,
                sa,
                weight,
                sb,
                gamma,
                positions,
                cos_sin,
                slots,
                q,
                k,
                v,
                mk,
                mv,
                k_cache,
                v_cache,
                float(kwargs.get("epsilon", 1e-6)),
                *chunk,
                cores,
            )

    return invoke


def output_rows(m, chunk=None):
    rows = torch.arange(m, dtype=torch.int64)
    if chunk is None:
        return rows
    length, count, offset = chunk
    return (rows // count) * length + offset + rows % count


def dequant(payload, scale):
    values = payload.cpu().contiguous().float()
    exponents = (
        scale.cpu()
        .contiguous()
        .view(torch.uint8)
        .reshape(values.shape[0], values.shape[1] // 32)
    )
    factors = torch.pow(2.0, exponents.to(torch.float64) - 127)
    return values.double() * factors.repeat_interleave(32, dim=1)


@dataclass
class KernelCase:
    hidden_quant: torch.Tensor
    hidden_scale: torch.Tensor
    weight_quant: torch.Tensor
    weight_scale: torch.Tensor
    k_gamma: torch.Tensor
    positions: torch.Tensor
    cos_sin: torch.Tensor
    slot_mapping: torch.Tensor
    q_out: torch.Tensor
    k_out: torch.Tensor
    v_out: torch.Tensor
    k_cache: torch.Tensor
    v_cache: torch.Tensor
    rows: torch.Tensor
    chunk: Optional[tuple] = None
    positions_contiguous: bool = False
    mirror_k_out: Optional[torch.Tensor] = None
    mirror_v_out: Optional[torch.Tensor] = None

    @property
    def args(self):
        return [
            self.hidden_quant,
            self.hidden_scale,
            self.weight_quant,
            self.weight_scale,
            self.k_gamma,
            self.positions,
            self.cos_sin,
            self.slot_mapping,
            self.q_out,
            self.k_out,
            self.v_out,
            self.k_cache,
            self.v_cache,
        ]

    @property
    def kwargs(self):
        result = {"positions_contiguous": self.positions_contiguous}
        if self.mirror_k_out is not None:
            result.update(
                mirror_k_out=self.mirror_k_out,
                mirror_v_out=self.mirror_v_out,
            )
        if self.chunk is not None:
            result.update(
                tokens_per_rank=self.chunk[0],
                chunk_tokens_per_rank=self.chunk[1],
                chunk_token_offset=self.chunk[2],
            )
        return result

    @property
    def outputs(self):
        return [self.q_out, self.k_out, self.v_out]

    @property
    def caches(self):
        return [self.k_cache, self.v_cache]

    @property
    def mirrors(self):
        if self.mirror_k_out is None:
            return None
        return (self.mirror_k_out, self.mirror_v_out)


def make_case(
    m,
    mirror,
    contiguous,
    chunk,
    hidden_size=2048,
    q_heads=6,
    kv_heads=1,
    chunk_config=None,
    act_dtype=torch.float8_e4m3fn,
    weight_dtype=torch.float8_e4m3fn,
):
    import torch_npu  # noqa: F401

    if chunk_config is None and chunk:
        chunk_config = (m + 3, m // 2, 1)
    chunk = chunk_config is not None
    t = (m // chunk_config[1]) * chunk_config[0] if chunk else m
    n = (q_heads + (4 if mirror else 2) * kv_heads) * 256
    qw, kvw = q_heads * 256, kv_heads * 256
    s, p = t + 17, t + 32
    device = os.environ.get("QKV_DEVICE", "npu:0")
    gen = torch.Generator().manual_seed(17 + m)

    hidden = torch.randn((m, hidden_size), generator=gen).bfloat16().to(device)
    hidden_quant, hidden_scale = torch.ops.npu.npu_dynamic_mx_quant(
        hidden, dst_type=act_dtype
    )
    weight_bf16 = (
        (torch.randn((n, hidden_size), generator=gen) * 0.02).bfloat16().to(device)
    )
    weight_quant, weight_scale = torch.ops.npu.npu_dynamic_mx_quant(
        weight_bf16, dst_type=weight_dtype
    )
    k_gamma = torch.randn((256,), generator=gen).bfloat16().to(device)

    rows = output_rows(m, chunk_config)
    positions = torch.zeros(t, dtype=torch.int64)
    positions[rows] = torch.arange(m) + 2 if contiguous else (torch.arange(m) * 7) % p
    angles = (
        torch.arange(p).float()[:, None]
        / (10000 ** (torch.arange(32).float() / 32))[None, :]
    )
    cos_sin = torch.cat((angles.cos(), angles.sin()), dim=1).to(device)
    slots = torch.full((t,), -1, dtype=torch.int64)
    slots[rows] = torch.arange(m)
    slots[rows[::3]] = -1

    def out(shape, value):
        return torch.full(shape, value, dtype=torch.bfloat16, device=device)

    return KernelCase(
        hidden_quant=hidden_quant,
        hidden_scale=hidden_scale.view(torch.float8_e8m0fnu),
        weight_quant=weight_quant,
        weight_scale=weight_scale.view(torch.float8_e8m0fnu),
        k_gamma=k_gamma,
        positions=positions.to(device),
        cos_sin=cos_sin,
        slot_mapping=slots.to(device),
        q_out=out((t, qw), 7),
        k_out=out((t, kvw), 8),
        v_out=out((t, kvw), 9),
        k_cache=out((s, kvw), 10),
        v_cache=out((s, kvw), 11),
        rows=rows,
        chunk=chunk_config,
        positions_contiguous=contiguous,
        mirror_k_out=out((t, kvw), 12) if mirror else None,
        mirror_v_out=out((t, kvw), 13) if mirror else None,
    )


def reference(case):
    m = case.hidden_quant.shape[0]
    rows = case.rows
    projected = (
        dequant(case.hidden_quant, case.hidden_scale)
        @ dequant(case.weight_quant, case.weight_scale).T
    ).to(torch.bfloat16)

    qw, kvw = case.q_out.shape[1], case.k_out.shape[1]
    q0 = projected[:, :qw].reshape(m, -1, 256)
    k0 = projected[:, qw : qw + kvw].reshape(m, -1, 256).float()
    k0 = (
        k0
        * torch.rsqrt(k0.square().mean(-1, keepdim=True) + 1e-6)
        * case.k_gamma.cpu().float()
    ).to(torch.bfloat16)
    cos_sin_rows = case.cos_sin.cpu()[case.positions.cpu()[rows]]

    def rope(values):
        x = values.reshape(m, -1, 256)
        result = x.clone()
        left = x[..., 192:224].float()
        right = x[..., 224:256].float()
        cos = cos_sin_rows[:, None, :32]
        sin = cos_sin_rows[:, None, 32:]
        result[..., 192:224] = (left * cos - right * sin).bfloat16()
        result[..., 224:256] = (left * sin + right * cos).bfloat16()
        return result.reshape(m, -1)

    q = case.q_out.cpu().clone()
    k = case.k_out.cpu().clone()
    v = case.v_out.cpu().clone()
    q[rows], k[rows], v[rows] = (
        rope(q0),
        rope(k0),
        projected[:, qw + kvw : qw + 2 * kvw],
    )
    k_cache = case.k_cache.cpu().clone()
    v_cache = case.v_cache.cpu().clone()
    selected_slots = case.slot_mapping.cpu()[rows]
    valid = selected_slots >= 0
    k_cache[selected_slots[valid]] = k[rows[valid]]
    v_cache[selected_slots[valid]] = v[rows[valid]]

    mirrors = None
    if case.mirrors is not None:
        mirror_k, mirror_v = (value.cpu().clone() for value in case.mirrors)
        mirror_k[rows] = projected[:, qw + 2 * kvw : qw + 3 * kvw]
        mirror_v[rows] = projected[:, qw + 3 * kvw : qw + 4 * kvw]
        mirrors = (mirror_k, mirror_v)
    return q, k, v, k_cache, v_cache, mirrors


def compare_case(invoke, case):
    expected = reference(case)
    old_outputs = [tensor.cpu().clone() for tensor in case.outputs + case.caches]
    assert invoke(*case.args, **case.kwargs) is None
    torch.npu.synchronize(case.hidden_quant.device)

    comparisons = [
        ("q", case.q_out, expected[0], False),
        ("k", case.k_out, expected[1], False),
        ("v", case.v_out, expected[2], False),
        ("k_cache", case.k_cache, expected[3], True),
        ("v_cache", case.v_cache, expected[4], True),
    ]
    if case.mirrors is not None:
        comparisons.extend(
            (name, actual, target, False)
            for name, actual, target in zip(
                ("mirror_k", "mirror_v"), case.mirrors, expected[5]
            )
        )
    for name, actual, target, is_cache in comparisons:
        actual_cpu = actual.cpu()
        close = torch.isclose(actual_cpu.float(), target.float(), rtol=1e-3, atol=1e-3)
        assert torch.isfinite(actual_cpu).all(), f"{name}: non-finite output"
        selected = case.slot_mapping.cpu()[case.rows] if is_cache else case.rows
        selected = selected[selected >= 0]
        if selected.numel():
            assert (1 - close[selected].float().mean()).item() <= 0.001, (
                f"{name}: selected rows exceed the allowed error rate"
            )

    selected_slots = case.slot_mapping.cpu()[case.rows]
    valid = selected_slots >= 0
    selected_slots = selected_slots[valid]
    assert torch.equal(
        case.k_cache.cpu()[selected_slots], case.k_out.cpu()[case.rows[valid]]
    )
    assert torch.equal(
        case.v_cache.cpu()[selected_slots], case.v_out.cpu()[case.rows[valid]]
    )

    untouched = torch.ones(case.q_out.shape[0], dtype=torch.bool)
    untouched[case.rows] = False
    for actual, old in zip(case.outputs, old_outputs[:3]):
        assert torch.equal(actual.cpu()[untouched], old[untouched])
    if case.mirrors is not None:
        for actual, target in zip(case.mirrors, expected[5]):
            assert torch.equal(actual.cpu()[untouched], target[untouched])

    untouched_cache = torch.ones(case.k_cache.shape[0], dtype=torch.bool)
    untouched_cache[selected_slots] = False
    for actual, old in zip(case.caches, old_outputs[3:]):
        assert torch.equal(actual.cpu()[untouched_cache], old[untouched_cache])


@pytest.mark.parametrize("mirror", [False, True])
@pytest.mark.parametrize("chunk", [False, True])
@pytest.mark.parametrize("contiguous", [False, True])
def test_metadata_modes_are_independent(adapter, inputs, mirror, chunk, contiguous):
    args = inputs(m=6, n=2560 if mirror else 2048, t=16 if chunk else 6)
    kwargs = dict(positions_contiguous=contiguous)
    if mirror:
        kwargs.update(
            mirror_k_out=torch.empty((args[8].shape[0], 256), dtype=torch.bfloat16),
            mirror_v_out=torch.empty((args[8].shape[0], 256), dtype=torch.bfloat16),
        )
    if chunk:
        kwargs.update(tokens_per_rank=8, chunk_tokens_per_rank=3, chunk_token_offset=2)
    result = adapter._validate_metadata(*args, **kwargs, require_npu=False)
    assert result == (mirror, chunk, (8, 3, 2) if chunk else (1, 1, 0))


@pytest.mark.parametrize(
    "changes,error",
    [
        ({"epsilon": 0.0}, ValueError),
        ({"epsilon": float("nan")}, ValueError),
        ({"tokens_per_rank": 8}, ValueError),
        (
            {"tokens_per_rank": 8, "chunk_tokens_per_rank": 0, "chunk_token_offset": 0},
            ValueError,
        ),
        (
            {"tokens_per_rank": 8, "chunk_tokens_per_rank": 2, "chunk_token_offset": 0},
            ValueError,
        ),
        (
            {"tokens_per_rank": 8, "chunk_tokens_per_rank": 3, "chunk_token_offset": 6},
            ValueError,
        ),
        ({"positions_contiguous": None}, TypeError),
    ],
)
def test_invalid_attributes(adapter, inputs, changes, error):
    with pytest.raises(error):
        adapter._validate_metadata(*inputs(), **changes, require_npu=False)


@pytest.mark.parametrize(
    "index,shape,dtype",
    [
        (0, (3, 2047), torch.float8_e4m3fn),
        (1, (3, 64), torch.float8_e8m0fnu),
        (2, (2304, 2048), torch.float8_e5m2),
        (3, (2048, 32, 2), torch.float32),
        (4, (1, 256), torch.bfloat16),
        (4, (256, 1), torch.bfloat16),
        (4, (255,), torch.bfloat16),
        (4, (256,), torch.float32),
        (5, (3,), torch.int32),
        (6, (32, 64), torch.bfloat16),
        (8, (3, 1535), torch.bfloat16),
        (0, (3, 2048), torch.int8),
        (1, (3, 32, 2), torch.float16),
    ],
)
def test_invalid_tensor_contract(adapter, inputs, index, shape, dtype):
    args = inputs()
    args[index] = torch.empty(shape, dtype=dtype)
    with pytest.raises((ValueError, TypeError, RuntimeError)):
        adapter._validate_metadata(*args, require_npu=False)


def test_overlap_and_strided_output_rejected(adapter, inputs):
    args = inputs()
    args[10] = args[9]
    with pytest.raises(ValueError, match="overlap"):
        adapter._validate_metadata(*args, require_npu=False)
    args = inputs()
    args[9] = torch.empty((3, 512), dtype=torch.bfloat16)[:, ::2]
    with pytest.raises(ValueError, match="contiguous"):
        adapter._validate_metadata(*args, require_npu=False)


def test_mirror_and_zero_tokens(adapter, inputs):
    with pytest.raises(ValueError, match="mirror"):
        adapter._validate_metadata(*inputs(n=2560), require_npu=False)
    args = inputs(n=2560)
    with pytest.raises(ValueError, match="mirror"):
        adapter._validate_metadata(
            *args,
            mirror_k_out=torch.empty((3, 256), dtype=torch.bfloat16),
            require_npu=False,
        )
    adapter._validate_metadata(*inputs(m=0), require_npu=False)
    adapter._validate_metadata(
        *inputs(m=0, t=0),
        tokens_per_rank=8,
        chunk_tokens_per_rank=2,
        chunk_token_offset=0,
        require_npu=False,
    )


@pytest.mark.parametrize("fake", [False, True])
@pytest.mark.parametrize("scale_dtype", [torch.float8_e8m0fnu, torch.int8, torch.uint8])
def test_registered_meta_and_fake_are_dsl_independent(
    adapter, inputs, monkeypatch, fake, scale_dtype
):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.startswith(("cannbotdsl", "opkit")):
            raise AssertionError("Meta/Fake imported " + name)
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)

    def invoke(device):
        args = inputs(device=device)
        args[1] = args[1].view(scale_dtype)
        args[3] = args[3].view(scale_dtype)
        return adapter.fused_quant_qkv_proj_norm_rope_cache(*args)

    if fake:
        with FakeTensorMode():
            result = invoke("cpu")
    else:
        result = invoke("meta")
    assert result is None
    schema = getattr(
        getattr(torch.ops, adapter._NAMESPACE), adapter._OP
    ).default._schema
    writes = [
        arg.name
        for arg in schema.arguments
        if arg.alias_info and arg.alias_info.is_write
    ]
    assert writes == [
        "q_out",
        "k_out",
        "v_out",
        "k_cache",
        "v_cache",
        "mirror_k_out",
        "mirror_v_out",
    ]


@pytest.mark.parametrize("scale_dtype", [torch.float8_e8m0fnu, torch.int8, torch.uint8])
def test_compile_factory_and_provider_abi(adapter, inputs, monkeypatch, scale_dtype):
    args = inputs(m=6, t=16)
    for index in (1, 3):
        args[index] = args[index].view(scale_dtype)
        args[index].view(torch.uint8).fill_(127)
    calls = []

    def factory(*config):
        from cannbotdsl import dtypes

        assert config == (
            False,
            True,
            True,
            dtypes.float8_e4m3fn,
            dtypes.float8_e4m3fn,
        )

        def provider(*values):
            calls.append(values)

        return provider

    monkeypatch.setattr(
        adapter, "_validate_metadata", lambda *a, **kw: (False, True, (8, 3, 2))
    )
    name = adapter.__package__ + ".op_dsl.fused_quant_qkv_proj_norm_rope_cache"
    from cannbotdsl import dtypes

    monkeypatch.setitem(
        sys.modules,
        name,
        types.SimpleNamespace(
            FP8_E4M3=dtypes.float8_e4m3fn,
            FP8_E5M2=dtypes.float8_e5m2,
            compile_fused_quant_qkv=factory,
        ),
    )

    @contextmanager
    def device(_):
        yield

    monkeypatch.setattr(
        torch,
        "npu",
        types.SimpleNamespace(
            device=device,
            get_device_properties=lambda _: types.SimpleNamespace(cube_core_num=28),
        ),
        raising=False,
    )
    adapter._execute(*args, positions_contiguous=True)
    assert len(calls) == 1
    assert calls[0][-5:] == (1e-6, 8, 3, 2, 28)
    assert calls[0][8] is args[8] and calls[0][13] is args[11]
    for index in (1, 3):
        assert calls[0][index].dtype == torch.float8_e8m0fnu
        assert calls[0][index].data_ptr() == args[index].data_ptr()
        assert torch.equal(
            calls[0][index].view(torch.uint8), args[index].view(torch.uint8)
        )
    assert len(calls[0]) == 20


def test_missing_binary_error_propagates(adapter, inputs, monkeypatch):
    monkeypatch.setattr(
        adapter, "_validate_metadata", lambda *a, **kw: (False, False, (1, 1, 0))
    )

    def missing(*choices):
        raise RuntimeError("required Native binary is missing")

    name = adapter.__package__ + ".op_dsl.fused_quant_qkv_proj_norm_rope_cache"
    from cannbotdsl import dtypes

    monkeypatch.setitem(
        sys.modules,
        name,
        types.SimpleNamespace(
            FP8_E4M3=dtypes.float8_e4m3fn,
            FP8_E5M2=dtypes.float8_e5m2,
            compile_fused_quant_qkv=missing,
        ),
    )

    @contextmanager
    def device(_):
        yield

    monkeypatch.setattr(
        torch,
        "npu",
        types.SimpleNamespace(
            device=device,
            get_device_properties=lambda _: types.SimpleNamespace(cube_core_num=28),
        ),
        raising=False,
    )
    with pytest.raises(RuntimeError, match="required Native binary"):
        adapter._execute(*inputs())


@pytest.mark.parametrize("mirror", [False, True])
@pytest.mark.parametrize("contiguous", [False, True])
@pytest.mark.parametrize("chunk", [False, True])
def test_dsl_modes(npu_runner, mirror, contiguous, chunk):
    compare_case(npu_runner, make_case(6, mirror, contiguous, chunk))


TAIL_REUSE_CASES = [
    pytest.param(1, True, 2048, 6, 1, id="m1"),
    pytest.param(127, True, 2048, 6, 1, id="m127"),
    pytest.param(128, True, 2048, 6, 1, id="m128"),
    pytest.param(129, True, 2048, 6, 1, id="m129"),
    pytest.param(255, True, 2048, 6, 1, id="m255"),
    pytest.param(256, True, 2048, 6, 1, id="m256"),
    pytest.param(257, True, 2048, 6, 1, id="m257"),
    pytest.param(2048, True, 2048, 6, 1, id="m2048"),
    pytest.param(129, False, 256, 2, 2, id="reuse-h256"),
    pytest.param(257, False, 512, 4, 2, id="reuse-h512"),
]


@pytest.mark.parametrize("m,mirror,hidden_size,q_heads,kv_heads", TAIL_REUSE_CASES)
def test_dsl_tails_and_reuse(npu_runner, m, mirror, hidden_size, q_heads, kv_heads):
    compare_case(
        npu_runner,
        make_case(m, mirror, False, False, hidden_size, q_heads, kv_heads),
    )


@pytest.mark.parametrize(
    "hidden_size,q_heads,kv_heads,mirror,chunk,contiguous",
    [
        (256, 1, 1, False, False, False),
        (512, 4, 2, True, True, True),
        (1024, 3, 2, False, True, False),
        (4096, 8, 2, True, False, False),
    ],
)
def test_generalized_shapes(
    npu_runner, hidden_size, q_heads, kv_heads, mirror, chunk, contiguous
):
    compare_case(
        npu_runner,
        make_case(6, mirror, contiguous, chunk, hidden_size, q_heads, kv_heads),
    )


def test_empty_m_is_noop(adapter, inputs, monkeypatch):
    args = inputs(m=0)
    for cache in args[11:]:
        cache.fill_(3)
    caches = [x.clone() for x in args[11:]]
    monkeypatch.setattr(
        adapter, "_validate_metadata", lambda *a, **kw: (False, False, (1, 1, 0))
    )
    assert adapter._execute(*args) is None
    assert all(torch.equal(a, b) for a, b in zip(args[11:], caches))


def test_empty_m_npu(adapter, inputs):
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("NPU unavailable")
    args = inputs(m=0, device=os.environ.get("QKV_DEVICE", "npu:0"))
    for cache in args[11:]:
        cache.fill_(3)
    before = [cache.clone() for cache in args[11:]]
    assert adapter.fused_quant_qkv_proj_norm_rope_cache(*args) is None
    assert all(torch.equal(a, b) for a, b in zip(args[11:], before))


CHUNK_EDGE_CASES = [
    pytest.param(2, True, True, None, id="cross-tile-m2"),
    pytest.param(258, True, True, None, id="cross-tile-m258"),
    pytest.param(514, True, True, None, id="cross-tile-m514"),
    pytest.param(1026, True, True, None, id="cross-tile-m1026"),
    pytest.param(512, False, True, (512, 256, 0), id="bulk-plain-offset0"),
    pytest.param(512, False, True, (512, 256, 1), id="bulk-plain-offset1"),
    pytest.param(512, True, True, (512, 256, 0), id="bulk-mirror-offset0"),
    pytest.param(512, True, True, (512, 256, 1), id="bulk-mirror-offset1"),
    pytest.param(384, False, False, (384, 128, 0), id="multi-plain"),
    pytest.param(384, True, False, (384, 128, 0), id="multi-mirror"),
    pytest.param(1, False, False, (2, 1, 0), id="single-row"),
]


@pytest.mark.parametrize("m,mirror,contiguous,chunk_config", CHUNK_EDGE_CASES)
def test_dsl_chunk_edges(npu_runner, m, mirror, contiguous, chunk_config):
    compare_case(
        npu_runner,
        make_case(m, mirror, contiguous, True, chunk_config=chunk_config),
    )


@pytest.mark.parametrize(
    "act_dtype,weight_dtype",
    [
        (torch.float8_e4m3fn, torch.float8_e4m3fn),
        (torch.float8_e5m2, torch.float8_e5m2),
        (torch.float8_e4m3fn, torch.float8_e5m2),
        (torch.float8_e5m2, torch.float8_e4m3fn),
    ],
)
def test_dsl_fp8_formats(npu_runner, act_dtype, weight_dtype):
    compare_case(
        npu_runner,
        make_case(
            6,
            True,
            True,
            False,
            act_dtype=act_dtype,
            weight_dtype=weight_dtype,
        ),
    )


def test_dsl_profile_switch_reuses_provider(npu_runner, adapter):
    factory = importlib.import_module(
        adapter.__package__ + ".op_dsl.fused_quant_qkv_proj_norm_rope_cache"
    )
    # The same feature/dtype contract must serve both sides of the Host branch.
    choice = (False, False, False, factory.FP8_E4M3, factory.FP8_E4M3)
    provider = factory.compile_fused_quant_qkv(*choice)
    for m in (3072, 3073, 1, 3073):
        assert factory.compile_fused_quant_qkv(*choice) is provider
        compare_case(
            npu_runner,
            make_case(m, False, False, False, hidden_size=256, q_heads=1),
        )


@pytest.mark.parametrize(
    "m,mirror,contiguous,chunk",
    [
        pytest.param(513, False, False, False, id="base128-window-tail"),
        pytest.param(1153, True, True, True, id="base128-snake-and-tail"),
        pytest.param(3329, False, False, True, id="base256-window-tail"),
        pytest.param(4353, True, True, False, id="base256-multiple-windows"),
    ],
)
def test_dsl_sliding_window(npu_runner, m, mirror, contiguous, chunk):
    compare_case(
        npu_runner,
        make_case(
            m,
            mirror,
            contiguous,
            chunk,
            hidden_size=256,
            q_heads=2,
            kv_heads=2,
            chunk_config=(3, 1, 1) if chunk else None,
        ),
    )


def test_dsl_large_prefill(npu_runner):
    compare_case(npu_runner, make_case(16500, True, True, False))


def test_dsl_repeated_rope_rows(npu_runner):
    case = make_case(2048, True, True, False)
    expected = reference(case)
    references = [
        expected[0].view(2048, -1, 256)[..., 192:].float(),
        expected[1].view(2048, -1, 256)[..., 192:].float(),
    ]
    for launch in range(20):
        assert npu_runner(*case.args, **case.kwargs) is None
        torch.npu.synchronize(case.hidden_quant.device)
        for name, actual, target in zip(("q", "k"), case.outputs[:2], references):
            rope = actual.cpu().view(2048, -1, 256)[..., 192:].float()
            bad_rows = ((rope - target).abs() > 0.05).any(dim=-1).sum().item()
            assert bad_rows == 0, (
                f"{name}: launch {launch} has {bad_rows} stale RoPE rows"
            )
        selected_slots = case.slot_mapping.cpu()[case.rows]
        valid = selected_slots >= 0
        selected_slots = selected_slots[valid]
        assert torch.equal(
            case.k_cache.cpu()[selected_slots], case.k_out.cpu()[case.rows[valid]]
        )
        assert torch.equal(
            case.v_cache.cpu()[selected_slots], case.v_out.cpu()[case.rows[valid]]
        )


@pytest.mark.parametrize("contiguous", [False, True])
def test_rank_one_gamma_graph_replay(npu_runner, contiguous):
    case = make_case(6, True, contiguous, True, hidden_size=256, q_heads=1, kv_heads=2)
    assert case.k_gamma.shape == (256,)
    npu_runner(*case.args, **case.kwargs)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        npu_runner(*case.args, **case.kwargs)
    torch.npu.synchronize()

    rows = output_rows(case.hidden_quant.shape[0], case.chunk).to(case.positions.device)
    local = torch.arange(rows.numel(), dtype=torch.int64, device=case.positions.device)
    case.positions[rows] = local + 3 if contiguous else local * 3 + 1
    case.slot_mapping[rows] = local + 5
    case.k_gamma.add_(0.125)

    compare_case(lambda *args, **kwargs: graph.replay(), case)
    for actual, target in zip(case.mirrors, reference(case)[5]):
        close = torch.isclose(
            actual.cpu().float(), target.float(), rtol=1e-3, atol=1e-3
        )
        assert (1 - close.float().mean()).item() <= 0.001


@pytest.fixture(scope="module")
def qkv_assets():
    spec = importlib.util.spec_from_file_location(
        "qkv_graph_assets", OP_ROOT / "tests/assets/golden.py"
    )
    assets = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(assets)
    return assets


@pytest.mark.parametrize("mirror_name", ["mirror_k_out", "mirror_v_out"])
def test_ttk_golden_rejects_single_mirror(qkv_assets, inputs, mirror_name):
    args = inputs(m=0)
    with pytest.raises(
        ValueError, match="mirror outputs must both be present or both absent"
    ):
        qkv_assets.golden_all(*args, **{mirror_name: torch.empty_like(args[9])})


@pytest.mark.parametrize(
    "m,mirror,contiguous,chunk_config,scale_dtype",
    [
        (6, mirror, contiguous, chunk_config, scale_dtype)
        for mirror in (False, True)
        for contiguous, chunk_config in ((False, None), (True, (9, 3, 1)))
        for scale_dtype in (torch.float8_e8m0fnu, torch.int8, torch.uint8)
    ]
    + [
        pytest.param(
            512, True, True, (512, 256, 1), torch.float8_e8m0fnu, id="aligned-chunk"
        )
    ],
)
def test_aclgraph_optional_outputs(
    npu_runner, qkv_assets, m, mirror, contiguous, chunk_config, scale_dtype
):
    case = make_case(
        m,
        mirror,
        contiguous,
        chunk_config is not None,
        hidden_size=256,
        q_heads=1,
        chunk_config=chunk_config,
    )
    aligned_chunk = chunk_config == (512, 256, 1)
    if aligned_chunk:
        # As in WeLM aligned prefill, each segment is internally continuous
        # and tile boundaries allow adjacent segments to start at different positions.
        case.positions[case.rows[256:]] += 256
    outputs = case.outputs + case.caches + list(case.mirrors or ())
    initial = [tensor.clone() for tensor in outputs]
    # Prepare the provider before graph capture using the selected test backend.
    npu_runner(*case.args, **case.kwargs)
    case.hidden_scale = case.hidden_scale.view(scale_dtype)
    case.weight_scale = case.weight_scale.view(scale_dtype)
    spec = qkv_assets.FusedQuantQkvProjNormRopeCacheE2eTestSpec

    def golden_outputs():
        cpu_args = [value.cpu() for value in case.args]
        cpu_kwargs = {
            name: value.cpu() if isinstance(value, torch.Tensor) else value
            for name, value in case.kwargs.items()
        }
        return spec.golden(*cpu_args, **cpu_kwargs)

    model = spec.torch_graph()
    torch._dynamo.reset()
    compiled = torch.compile(
        model, backend="npugraph_ex", dynamic=False, fullgraph=True
    )

    def reset_outputs():
        for tensor, value in zip(outputs, initial):
            tensor.copy_(value)

    try:
        for update_metadata in (False, True):
            if update_metadata:
                rows = case.rows.to(case.positions.device)
                local = torch.arange(rows.numel(), device=case.positions.device)
                case.k_gamma.add_(0.125)
                if contiguous:
                    case.positions[rows] += 1
                else:
                    case.positions[rows] = local * 3 + 1
                case.slot_mapping[rows] = local + 5
            reset_outputs()
            golden = golden_outputs()
            assert len(golden) == len(outputs)
            torch.ops.cann_ops_transformer.fused_quant_qkv_proj_norm_rope_cache(
                *case.args, **case.kwargs
            )
            expected = [tensor.cpu().clone() for tensor in outputs]
            for replay in range(2):
                reset_outputs()
                actual = compiled(*case.args, **case.kwargs)
                torch.npu.synchronize()
                assert len(actual) == len(outputs)
                for result, buffer, target in zip(actual, outputs, expected):
                    assert result.data_ptr() == buffer.data_ptr()
                    assert torch.equal(result.cpu(), target)
                if aligned_chunk and replay == 0:
                    for name, result, target in zip(("q", "k"), actual[:2], golden[:2]):
                        rope = result.cpu()[case.rows, 192:].float()
                        reference_rope = target[case.rows, 192:].float()
                        close = torch.isclose(
                            rope, reference_rope, rtol=1e-3, atol=1e-3
                        )
                        assert (1 - close.float().mean()).item() <= 0.001, name
    finally:
        torch._dynamo.reset()
