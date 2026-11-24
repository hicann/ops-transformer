# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""CPU tests for the Python schema, Meta contract and extension forwarding."""

import importlib.util
from pathlib import Path
import sys
import types

import pytest
import torch


@pytest.fixture(scope="module")
def modules():
    # Register the production schemas and Meta kernels without importing torch_npu
    # or compiling the device extension. These tests require only CPU PyTorch.
    lib = torch.library.Library("cann_ops_transformer", "FRAGMENT")
    builder_module = types.ModuleType("cann_ops_transformer.op_builder")

    class OpBuilder:
        def __init__(self, name, category):
            self.name = name

        def _ensure_initialized(self):
            lib.define(self.schema())
            self.register_meta()

    builder_module.OpBuilder = OpBuilder
    builder_module.get_as_library = lambda: lib
    previous = sys.modules.get(builder_module.__name__)
    sys.modules[builder_module.__name__] = builder_module
    # Master shares layout helpers through a relative import from V1 to V2.
    package_name = "quant_contract_extension"
    package = types.ModuleType(package_name)
    package.__path__ = [str(Path(__file__).parents[2] / "torch_extension")]
    sys.modules[package_name] = package
    loaded = {}
    try:
        for name in ("msa_index_score_v2", "msa_index_score"):
            path = Path(__file__).parents[2] / "torch_extension" / f"{name}.py"
            spec = importlib.util.spec_from_file_location(
                f"{package_name}.{name}", path
            )
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            loaded[name] = module
        yield loaded
    finally:
        for name in ("msa_index_score", "msa_index_score_v2"):
            sys.modules.pop(f"{package_name}.{name}", None)
        sys.modules.pop(package_name, None)
        if previous is None:
            del sys.modules[builder_module.__name__]
        else:
            sys.modules[builder_module.__name__] = previous


def inputs(total_q=3, dtype=torch.int8):
    return dict(
        query=torch.empty((total_q, 2, 128), dtype=dtype, device="meta"),
        key=torch.empty((2, 128, 1, 128), dtype=dtype, device="meta"),
        # Token position 129 is in logical block 1 when the block size is 128.
        start_loc=torch.tensor([1], dtype=torch.int32),
        block_table=torch.empty((1, 2), dtype=torch.int32, device="meta"),
        key_scale=torch.ones((2, 1, 128), device="meta")
        if dtype == torch.int8
        else None,
        atten_mask=None,
        actual_seq_qlen=torch.tensor([0, total_q], dtype=torch.int32),
        actual_seq_klen=torch.tensor([256], dtype=torch.int32),
        sparse_mode=0,
        query_scale=torch.ones((total_q, 2), device="meta")
        if dtype == torch.int8
        else None,
    )


@pytest.mark.parametrize("name", ["msa_index_score", "msa_index_score_v2"])
def test_pa_meta_a8c8_and_existing_bf16(modules, name):
    api = getattr(modules[name], name)
    dtypes = (
        (torch.int8, torch.float16, torch.bfloat16)
        if name == "msa_index_score_v2"
        else (torch.float16, torch.bfloat16)
    )
    for dtype in dtypes:
        output = api(**inputs(dtype=dtype))
        assert output.shape == (2, 3, 16)
        assert output.dtype == torch.float32


@pytest.mark.parametrize("name", ["msa_index_score_v2"])
def test_empty_tnd_scales_are_present(modules, name):
    args = inputs(total_q=0)
    args.update(
        key=torch.empty((0, 1, 128), dtype=torch.int8, device="meta"),
        block_table=None,
        key_scale=torch.empty((0,), device="meta"),
        actual_seq_klen=torch.tensor([0, 0], dtype=torch.int32),
        layout_key="TND",
    )
    output = getattr(modules[name], name)(**args)
    assert output.shape == (2, 0, 16)


def test_v2_requires_explicit_key_scale_parameter(modules):
    api = modules["msa_index_score_v2"].msa_index_score_v2
    args = inputs()
    assert api(**args).shape == (2, 3, 16)
    with pytest.raises(TypeError, match="unexpected keyword argument 'scale'"):
        api(**args, scale=args["key_scale"])


@pytest.mark.parametrize("name", ["msa_index_score_v2"])
def test_extension_receives_query_scale(modules, name):
    module = modules[name]
    previous = module._op_module
    module._op_module = types.SimpleNamespace(
        **{name: lambda *args: (args, None, None)}
    )
    try:
        args = inputs()
        forwarded = getattr(module, f"_{name}")(**args)
        assert forwarded[4] is args["query_scale"]
        assert forwarded[5] is args["key_scale"]
        assert forwarded[0] is args["query"]
        assert forwarded[2] is args["start_loc"]
        assert forwarded[2].item() == 1
        assert forwarded[-1] == 1
        args["query_scale"] = None
        args["key_scale"] = torch.ones((2, 1, 127), dtype=torch.float16, device="meta")
        forwarded = getattr(module, f"_{name}")(**args)
        assert forwarded[4] is None
        assert forwarded[5] is args["key_scale"]
    finally:
        module._op_module = previous


@pytest.mark.parametrize("name", ["msa_index_score", "msa_index_score_v2"])
def test_torch_schema_preserves_master_positional_order(modules, name):
    operation = getattr(torch.ops.cann_ops_transformer, f"_{name}")
    args = inputs(dtype=torch.int8 if name == "msa_index_score_v2" else torch.bfloat16)
    output = operation(
        *(
            args[key]
            for key in (
                "query",
                "key",
                "start_loc",
                "block_table",
                "query_scale",
                "key_scale",
                "atten_mask",
                "actual_seq_qlen",
                "actual_seq_klen",
            )
        ),
        sparse_mode=0,
    )
    assert output.shape == (2, 3, 16)


def test_sparse_public_api_preserves_master_topk_arguments(modules):
    module = modules["msa_index_score"]
    previous = module._op_module
    captured = []
    module._op_module = types.SimpleNamespace(
        msa_index_score_sparse=lambda *args: captured.append(args)
        or ("indices", "values")
    )
    try:
        args = inputs(dtype=torch.bfloat16)
        assert module.msa_index_score(**args, return_choice=1, topk=32) == "indices"
        assert module.msa_index_score(**args, return_choice=2, topk=32) == (
            "indices",
            "values",
        )
        assert captured[-1][2] is args["start_loc"]
        assert captured[-1][-3:] == (2, 32, 1)
    finally:
        module._op_module = previous


def test_v1_rejects_a8c8_at_public_and_registered_entries(modules):
    module = modules["msa_index_score"]
    for api in (
        module.msa_index_score,
        module._msa_index_score,
        torch.ops.cann_ops_transformer._msa_index_score,
    ):
        with pytest.raises(RuntimeError, match="requires msa_index_score_v2"):
            api(**inputs())
    with pytest.raises(RuntimeError, match="requires msa_index_score_v2"):
        module.msa_index_score(**inputs(), return_choice=1)


@pytest.mark.parametrize("name", ["msa_index_score_v2"])
@pytest.mark.parametrize("layout", ["TND", "BBND", "BNBD"])
def test_a8c8_key_layout_scales(modules, name, layout):
    args = inputs()
    args["layout_key"] = layout
    if layout == "TND":
        args.update(
            key=torch.empty((256, 1, 128), dtype=torch.int8, device="meta"),
            key_scale=torch.empty((256, 1), device="meta"),
            block_table=None,
            actual_seq_klen=torch.tensor([0, 256], dtype=torch.int32),
        )
    elif layout == "BNBD":
        args["key"] = args["key"].transpose(1, 2)
    assert getattr(modules[name], name)(**args).shape == (2, 3, 16)


@pytest.mark.parametrize("name", ["msa_index_score", "msa_index_score_v2"])
def test_graph_converter_preserves_master_ir_and_native_scales(
    modules, monkeypatch, name
):
    captured = {}

    class IrDef:
        def __init__(self, name):
            self.name = name

        def __getattr__(self, name):
            def record(*args):
                captured.setdefault("ir_entries", []).append((name, args))
                return self

            return record

    def ge_op(**kwargs):
        captured.update(kwargs)
        return ["score"] * len(kwargs["outputs"])

    def get_default_ge_graph():
        return None

    def ge_dtype_to_ge_proto_dtype(dtype):
        return dtype

    stubs = {
        key: types.ModuleType(key)
        for key in (
            "torch_npu",
            "torchair",
            "torchair.ge",
            "torchair.ge._ge_graph",
            "torchair._ge_concrete_graph",
            "torchair._ge_concrete_graph.compat_ir",
            "torchair._ge_concrete_graph.fx2ge_converter",
        )
    }
    stubs["torchair.ge"].attr = types.SimpleNamespace(
        Str=lambda value: value, Int=lambda value: value
    )
    graph = stubs["torchair.ge._ge_graph"]
    graph.Tensor = object
    graph.TensorSpec = object
    graph.DataType = types.SimpleNamespace(DT_INT8=2, DT_UINT8=4)
    graph.auto_convert_to_tensor = lambda *flags: lambda function: function
    graph.get_default_ge_graph = get_default_ge_graph
    graph._ge_dtype_to_ge_proto_dtype = ge_dtype_to_ge_proto_dtype
    stubs["torchair._ge_concrete_graph"].ge_apis = types.SimpleNamespace()
    ir = stubs["torchair._ge_concrete_graph.compat_ir"]
    ir.ge_op, ir.IrDef = ge_op, IrDef
    stubs[
        "torchair._ge_concrete_graph.fx2ge_converter"
    ].register_fx_node_ge_converter = lambda operation: lambda function: function
    for key, value in stubs.items():
        monkeypatch.setitem(sys.modules, key, value)
    extension = Path(__file__).parents[2] / "torch_extension"
    package = modules[name].__package__
    for converter_name in ("msa_index_score", "msa_index_score_v2"):
        filename = f"graph_convert_{converter_name}"
        spec = importlib.util.spec_from_file_location(
            f"{package}.{filename}", extension / f"{filename}.py"
        )
        converter = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, spec.name, converter)
        spec.loader.exec_module(converter)
        if converter_name == name:
            break
    args = inputs(dtype=torch.int8 if name == "msa_index_score_v2" else torch.bfloat16)
    convert = getattr(converter, f"convert_{name}")
    if name == "msa_index_score":
        with pytest.raises(ValueError, match="requires msa_index_score_v2"):
            convert(**inputs())
    convert(**args)
    assert captured["inputs"]["query_scale"] is args["query_scale"]
    assert captured["inputs"]["key_scale"] is args["key_scale"]
    assert captured["inputs"]["start_loc"] is args["start_loc"]
    assert "scale" not in captured["inputs"]
    assert captured["attrs"]["compress_factor"] == 1
    assert [
        entry[1][0] for entry in captured["ir_entries"] if entry[0] == "input"
    ].count("start_loc") == 1
    assert any(
        entry[0] == "input"
        and entry[1][0] == "query"
        and "DT_FLOAT4_E2M1" in entry[1][1]
        for entry in captured["ir_entries"]
    )
    (query_types,) = [
        entry[1][1]
        for entry in captured["ir_entries"]
        if entry[0] == "input" and entry[1][0] == "query"
    ]
    assert ("DT_INT8" in query_types) == (name == "msa_index_score_v2")
    assert ("optional_input", ("query_scale", "DT_FLOAT, DT_FLOAT8_E8M0")) in captured[
        "ir_entries"
    ]
    if name == "msa_index_score_v2":
        convert(**args, compress_factor=4)
        assert captured["attrs"]["compress_factor"] == 4
    else:
        with pytest.raises(ValueError, match="compress_factor only supports 1"):
            convert(**args, compress_factor=4)
