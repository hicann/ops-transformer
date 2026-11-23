# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Bounded CI coverage: six precision cases and two metadata-only cases.

The broader token sweep, intermediate-byte checks, graph stress and profiler
utilities are maintained outside the repository under local validation/.
"""

import importlib
import importlib.util
import sys
from functools import lru_cache
from pathlib import Path

import pytest
import torch


torch_npu = pytest.importorskip("torch_npu")

_OP_DIR = Path(__file__).resolve().parents[2]
_OP_DSL = _OP_DIR / "op_dsl"


def _torch_extension_module():
    """Return the installed Torch op, loading the shipped adapter only if needed."""
    import types

    try:
        import cann_ops_transformer  # noqa: F401

        installed = getattr(torch.ops.cann_ops_transformer, "attn_epilogue", None)
        if installed is not None:
            return installed
        from cann_ops_transformer.op_builder import get_as_library  # noqa: F401
    except (ImportError, AttributeError):
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

    package_name = "attn_epilogue_torch_extension_test"
    torch_extension = _OP_DIR / "torch_extension"
    system_package = types.ModuleType(package_name)
    system_package.__path__ = [str(torch_extension)]
    sys.modules[package_name] = system_package
    op_dsl_package = types.ModuleType(package_name + ".op_dsl")
    op_dsl_package.__path__ = [str(_OP_DSL)]
    sys.modules[package_name + ".op_dsl"] = op_dsl_package
    spec = importlib.util.spec_from_file_location(
        package_name + ".attn_epilogue",
        torch_extension / "attn_epilogue.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.attn_epilogue


op = _torch_extension_module()


def _graph_entry(*args):
    """Call the registered dispatcher op so Dynamo captures a single custom op."""
    return torch.ops.cann_ops_transformer.attn_epilogue(*args)


@lru_cache(maxsize=3)
def _weights(ng):
    """Quantize one deterministic ND weight pair per local group count."""
    rng = torch.Generator().manual_seed(2026)
    wa = (torch.randn(ng, 1024, 4096, generator=rng) * 0.3).bfloat16().npu()
    wb = (torch.randn(5120, ng * 1024, generator=rng) * 0.2).bfloat16().npu()
    qa, sa = torch_npu.npu_dynamic_mx_quant(wa.reshape(-1, 4096), dst_type=292)
    qb, sb = torch_npu.npu_dynamic_mx_quant(wb, dst_type=292)
    e8 = torch.float8_e8m0fnu
    return (
        qa.reshape(ng, 1024, 4096),
        qb,
        sa.view(e8).reshape(ng, 1024, 64, 2),
        sb.view(e8),
    )


def _inputs(t, ng):
    """Build BF16 activations and nonzero-position interleaved RoPE tables."""
    rng = torch.Generator().manual_seed(2027)
    x = (torch.randn(t, ng * 8, 512, generator=rng) * 0.3).bfloat16()
    angles = torch.outer(
        torch.arange(t).float() + 3, 10000.0 ** (-torch.arange(0, 64, 2).float() / 64)
    )
    return (
        x,
        angles.sin().repeat_interleave(2, -1),
        angles.cos().repeat_interleave(2, -1),
    )


def _native(x, sin, cos, weights):
    """Compute inverse RoPE and both native MX projections with BF16 MM1 rounding."""
    wa, wb, sa, sb = weights
    t, ng = x.shape[0], wa.shape[0]
    rotated = x.float()
    even, odd = rotated[..., 448::2].clone(), rotated[..., 449::2].clone()
    c, s = cos[:, None, ::2], sin[:, None, ::2]
    rotated[..., 448::2] = even * c + odd * s
    rotated[..., 449::2] = odd * c - even * s
    aq, asc = torch_npu.npu_dynamic_mx_quant(
        rotated.bfloat16().npu().reshape(t * ng, 4096), dst_type=292
    )
    e8 = torch.float8_e8m0fnu
    y = torch_npu.npu_transpose_quant_batchmatmul(
        aq.reshape(t, ng, 4096),
        wa,
        15,
        x1_scale=asc.view(e8).reshape(t, ng, 64, 2),
        x2_scale=sa,
        group_sizes=[1, 1, 32],
        perm_x1=[1, 0, 2],
        perm_x2=[0, 2, 1],
        perm_y=[1, 0, 2],
    ).reshape(t, ng * 1024)
    yq, ys = torch_npu.npu_dynamic_mx_quant(y, dst_type=292)
    return torch_npu.npu_quant_matmul(
        yq,
        wb.t(),
        sb.transpose(0, 1),
        pertoken_scale=ys.view(e8),
        output_dtype=torch.bfloat16,
        group_sizes=[1, 1, 32],
    )


def _as_nz(weights):
    """Prepack FP8 bytes into private NZ and transpose their ND scale axes."""
    wa, wb, sa, sb = weights
    e8 = torch.float8_e8m0fnu
    packed = []
    for weight in (wa, wb):
        kn = weight.view(torch.uint8).cpu().transpose(-2, -1).contiguous().npu()
        packed.append(torch_npu.npu_format_cast(kn.view(torch.float8_e4m3fn), 29))
    sa = sa.view(torch.uint8).cpu().permute(0, 2, 1, 3).contiguous().npu().view(e8)
    sb = sb.view(torch.uint8).cpu().permute(1, 0, 2).contiguous().npu().view(e8)
    return *packed, sa, sb


@pytest.mark.npu
@pytest.mark.parametrize(
    "t,ng,nz",
    [
        pytest.param(1, 1, False, id="ng1_nd_t1"),
        pytest.param(16, 1, True, id="ng1_nz_t16"),
        pytest.param(17, 2, False, id="ng2_nd_tail"),
        pytest.param(1, 2, True, id="ng2_nz_t1"),
        pytest.param(72, 8, False, id="ng8_nd_t72"),
        pytest.param(17, 8, True, id="ng8_nz_tail"),
    ],
)
def test_attn_epilogue(t, ng, nz):
    """Check each group/layout once; NG=1 NZ also covers both graph interfaces."""
    torch.npu.set_device(0)
    x, sin, cos = _inputs(t, ng)
    weights = _weights(ng)
    expected = _native(x, sin, cos, weights)
    args = (x.npu(), *(_as_nz(weights) if nz else weights), sin.npu(), cos.npu())
    actual = op(*args)
    torch.testing.assert_close(
        actual.view(torch.int16).cpu(), expected.view(torch.int16).cpu(), rtol=0, atol=0
    )
    if ng == 1 and nz:
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            captured = op(*args)
        graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(
            captured.view(torch.int16).cpu(),
            expected.view(torch.int16).cpu(),
            rtol=0,
            atol=0,
        )
        compiled = torch.compile(
            _graph_entry,
            backend="npugraph_ex",
            fullgraph=True,
            dynamic=False,
            options={"clone_input": False},
        )
        for _ in range(3):
            result = compiled(*args)
            torch.testing.assert_close(
                result.view(torch.int16).cpu(),
                expected.view(torch.int16).cpu(),
                rtol=0,
                atol=0,
            )


@pytest.mark.parametrize("ng,nz", [(8, False), (2, True)])
def test_meta(ng, nz):
    """Check output metadata using logical ND/NZ shapes without NPU access."""
    t = 17
    dtype, e8 = torch.float8_e4m3fn, torch.float8_e8m0fnu
    shapes = [
        ((t, ng * 8, 512), torch.bfloat16),
        ((ng, 4096, 1024) if nz else (ng, 1024, 4096), dtype),
        ((ng * 1024, 5120) if nz else (5120, ng * 1024), dtype),
        ((ng, 64, 1024, 2) if nz else (ng, 1024, 64, 2), e8),
        ((ng * 16, 5120, 2) if nz else (5120, ng * 16, 2), e8),
        ((t, 64), torch.float32),
        ((t, 64), torch.float32),
    ]
    args = [torch.empty(shape, dtype=dt, device="meta") for shape, dt in shapes]
    result = torch.ops.cann_ops_transformer.attn_epilogue(*args)
    assert result.shape == (t, 5120) and result.dtype == torch.bfloat16
