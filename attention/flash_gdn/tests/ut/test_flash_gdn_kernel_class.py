# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Review regressions for the one-launch class kernel and scalar/VF boundaries."""

import ast
import importlib
import os
from pathlib import Path
import sys

import pytest
import torch
from cann_ops_transformer import flash_gdn_fwd, flash_gdn_metadata

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "assets" / "impl"))
import gdn_ttk_ops


def _kernel_tree():
    module = importlib.import_module(
        "cann_ops_transformer.ops.attention.flash_gdn.op_dsl.flash_gdn_kernel"
    )
    return ast.parse(Path(module.__file__).read_text())


def test_single_class_entry_preserves_global_phase_boundary():
    tree = _kernel_tree()
    operator = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "FlashGdnKernel"
    )
    entry = next(
        node
        for node in operator.body
        if isinstance(node, ast.FunctionDef) and node.name == "__call__"
    )
    calls = [
        node.value.func
        for node in entry.body
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
    ]
    assert [
        call.attr if isinstance(call, ast.Attribute) else call.id for call in calls
    ] == ["_chunks", "global_sync_all", "onchip_rewind", "_state"]
    host = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_entry"
    )
    launches = [
        node
        for node in ast.walk(host)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Subscript)
    ]
    assert len(launches) == 1
    assert not any(
        isinstance(node, ast.FunctionDef) and node.name in ("_chunks", "_state")
        for node in tree.body
    )


def test_runtime_decisions_are_outside_vf_regions():
    for node in ast.walk(_kernel_tree()):
        if not isinstance(node, ast.With):
            continue
        if any(
            isinstance(item.context_expr, ast.Call)
            and isinstance(item.context_expr.func, ast.Name)
            and item.context_expr.func.id == "vf"
            for item in node.items
        ):
            assert not any(
                isinstance(child, (ast.If, ast.IfExp)) for child in ast.walk(node)
            )


def test_full_constant_names_and_compile_time_vector_halves():
    tree = _kernel_tree()
    assert not any(
        isinstance(node, ast.Name) and node.id in ("D", "C", "R")
        for node in ast.walk(tree)
    )
    loops = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.For)
        and isinstance(node.iter, ast.Call)
        and isinstance(node.iter.func, ast.Name)
        and node.iter.func.id == "range_constexpr"
    ]
    assert len(loops) == 3


def test_run_device_import_is_module_level():
    module = importlib.import_module(
        "cann_ops_transformer.ops.attention.flash_gdn.op_dsl.flash_gdn"
    )
    tree = ast.parse(Path(module.__file__).read_text())
    imports = [
        node
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module == "flash_gdn_kernel"
    ]
    assert len(imports) == 1
    assert {item.name for item in imports[0].names} == {"compile_kernel", "run_device"}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            assert not any(
                isinstance(child, ast.ImportFrom) and child.module == "flash_gdn_kernel"
                for child in ast.walk(node)
            )


@pytest.mark.npu
@pytest.mark.skipif(
    os.environ.get("RUN_GDN_NPU") != "1", reason="requires an idle Ascend NPU"
)
@pytest.mark.parametrize("layout", ["TND", "BSND", "BNSD"])
def test_single_launch_unbalanced_multicore_chunks(layout):
    import torch_npu  # noqa: F401

    torch.npu.set_device(os.environ.get("GDN_DEVICE", "npu:0"))
    generator = torch.Generator().manual_seed(1706)
    lengths = [129, 7]

    def random(shape):
        return torch.randn(shape, generator=generator)

    def pack(value):
        if layout == "TND":
            return torch.cat(
                [value[b, :length] for b, length in enumerate(lengths)]
            ).contiguous()
        return value.transpose(1, 2).contiguous() if layout == "BNSD" else value

    q, k = [(random((2, 129, 2, 128)) * 0.08).bfloat16() for _ in range(2)]
    v = random((2, 129, 4, 128)).bfloat16()
    g = -random((2, 129, 4)).abs() * 0.1
    beta = random((2, 129, 4)).sigmoid()
    initial_state = random((2, 4, 128, 128)) * 0.02
    cu_seqlens = torch.tensor([0, 129, 136], dtype=torch.int32)
    seqused = torch.tensor(lengths, dtype=torch.int32)
    values = [pack(value) for value in (q, k, v, g, beta)] + [
        None,
        initial_state,
        None,
        None,
        cu_seqlens,
        seqused,
    ]
    attributes = dict(
        layout=layout,
        valid_lengths=lengths,
        initial_state_mode="random",
        output_final_state=True,
        return_intermediate_states=True,
        scale=0.25,
    )
    expected = gdn_ttk_ops.load_asset("golden").reference(values, attributes)
    device = os.environ.get("GDN_DEVICE", "npu:0")
    tensors = [
        value.to(device) if isinstance(value, torch.Tensor) else value
        for value in values
    ]
    metadata = flash_gdn_metadata(
        tensors[9],
        tensors[10],
        batch_size=2,
        nheads_qk=2,
        nheads_v=4,
        seqlens=136 if layout == "TND" else 129,
        layout=layout,
    )
    actual = flash_gdn_fwd(
        *tensors[:5],
        metadata,
        initial_state=tensors[6],
        cu_seqlens=tensors[9],
        seqused=tensors[10],
        **{
            key: attributes[key]
            for key in (
                "layout",
                "scale",
                "output_final_state",
                "return_intermediate_states",
            )
        },
    )
    torch.npu.synchronize()
    for index, (got, want) in enumerate(zip(actual[:3], expected[:3])):
        torch.testing.assert_close(
            got.cpu(),
            want,
            atol=0.01 if index == 0 else 1e-4,
            rtol=0.01 if index == 0 else 1e-3,
        )
    torch.testing.assert_close(tensors[6].cpu(), initial_state, atol=0, rtol=0)
