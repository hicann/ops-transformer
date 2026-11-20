# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Public contracts not covered by the shared numerical and performance runner."""

import ctypes
import importlib
import importlib.util
import inspect
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from .common import test_runner as runner
from .common.quant_flash_attn_int8_golden import golden
from .quant_flash_attn_int8_paramset_common import expand_paramset_to_cases

pytestmark = [pytest.mark.contract, pytest.mark.usefixtures("int8_npu")]


def _case(layout="BNSD", dim=64, **attributes):
    values = dict(layout=layout, D=dim, enable_lse=True, **attributes)
    params = expand_paramset_to_cases(
        {"Contract": {name: [value] for name, value in values.items()}}
    )[0]
    batches, vs = golden._inputs(
        params["actual_seq_q"],
        params["actual_seq_kv"],
        params["N_q"],
        params["N_kv"],
        dim,
    )
    qfa, args, options = runner.prepare_npu(params, batches, vs)
    return params, batches, vs, qfa, args, options


@pytest.mark.parametrize("layout", ["BNSD", "BSND", "TND"])
@pytest.mark.parametrize("mask", [0, 3])
def test_dense_prefill_contiguous_inputs(layout, mask):
    params, batches, vs, qfa, args, options = _case(layout, mask_mode=mask)
    expected = tuple(t.cpu() for t in qfa(*args, **options))
    runner.compare(params, runner.reference(params, batches, vs), expected)
    offset_args = list(args)
    for index in (1, 2, 4, 5):
        tensor = args[index]
        storage = torch.empty(
            (*tensor.shape[:-1], 2 * tensor.shape[-1] + 1),
            dtype=tensor.dtype,
            device=tensor.device,
        )
        strided = storage[..., 1::2]
        strided.copy_(tensor)
        assert not strided.is_contiguous()
        changed = list(args)
        changed[index] = strided
        with pytest.raises(RuntimeError, match="contiguous"):
            qfa(*changed, **options)
        storage = torch.empty(
            tensor.numel() + 1, dtype=tensor.dtype, device=tensor.device
        )
        offset_args[index] = storage[1:].view(tensor.shape)
        offset_args[index].copy_(tensor)
    actual = qfa(*offset_args, **options)
    assert all(torch.equal(a, b.cpu()) for a, b in zip(expected, actual))


@pytest.mark.parametrize("layout", ["BNSD", "BSND"])
@pytest.mark.parametrize("mask", [0, 3])
def test_d72_automatic_padding(layout, mask):
    params, batches, vs, qfa, args, options = _case(
        layout,
        dim=72,
        mask_mode=mask,
        N_q=4,
        N_kv=4,
        actual_seq_q=(2048, 2048),
        actual_seq_kv=(2048, 2048),
        sample_rows=(0, 127, 511, -1),
    )
    assert "fast_d72" not in inspect.signature(qfa).parameters
    module = importlib.import_module(qfa.__module__)
    with patch.object(
        module,
        "_quant_flash_attn_aligned_padded",
        wraps=module._quant_flash_attn_aligned_padded,
    ) as pad:
        outputs = tuple(t.cpu() for t in qfa(*args, **options))
        pad.assert_called_once()
    runner.compare(params, runner.reference(params, batches, vs), outputs)
    for index in (1, 2):
        tensor = args[index]
        storage = torch.empty(
            (*tensor.shape[:-1], 2 * tensor.shape[-1]),
            dtype=tensor.dtype,
            device=tensor.device,
        )
        strided = storage[..., ::2]
        strided.copy_(tensor)
        changed = list(args)
        changed[index] = strided
        with patch.object(module, "_quant_flash_attn_aligned_padded") as pad:
            with pytest.raises(RuntimeError, match="contiguous"):
                qfa(*changed, **options)
            pad.assert_not_called()


@pytest.mark.parametrize(
    "index", [3, 4, 5, 7], ids=["q_descale", "k_descale", "v_descale", "p_scale"]
)
@pytest.mark.parametrize("invalid", ["missing", "shape", "dtype"])
def test_required_scales(index, invalid):
    _, _, _, qfa, args, options = _case()
    values = [*args[:6], args[6], options.pop("p_scale")]
    tensor = values[index]
    values[index] = (
        None
        if invalid == "missing"
        else tensor.half()
        if invalid == "dtype"
        else torch.ones(2 if index == 7 else 1, device=tensor.device)
    )
    with pytest.raises((RuntimeError, TypeError)):
        qfa(*values[:7], p_scale=values[7], **options)


@pytest.mark.parametrize("layout", ["BNSD", "BSND", "TND"])
@pytest.mark.parametrize("index", [0, 1, 2], ids=["q", "k", "v"])
@pytest.mark.parametrize("device", ["npu", "meta"])
def test_required_qkv(layout, index, device):
    _, _, _, qfa, args, options = _case(layout)
    values = list(args)
    call = qfa
    if device == "meta":
        values = [x.to("meta") if isinstance(x, torch.Tensor) else x for x in values]
        options = {
            name: x.to("meta") if isinstance(x, torch.Tensor) else x
            for name, x in options.items()
        }
        call = torch.ops.cann_ops_transformer.quant_flash_attn
    values[index] = None
    with pytest.raises(RuntimeError, match=f"{'qkv'[index]} must not be None"):
        call(*values, **options)


def test_int8_metadata_generation_unsupported():
    qfa = runner.load_qfa()
    module = importlib.import_module(qfa.__module__)
    with pytest.raises(
        (RuntimeError, ValueError),
        match="quant_flash_attn_metadata does not support A2/A3 INT8 Prefill",
    ):
        module.quant_flash_attn_metadata(
            num_heads_q=4,
            num_heads_kv=2,
            head_dim=64,
            quant_mode=12,
            batch_size=1,
            max_seqlen_q=17,
            max_seqlen_kv=33,
            layout_q="BNSD",
            layout_kv="BNSD",
            layout_out="BNSD",
        )


def test_int8_metadata_must_be_none():
    _, _, _, qfa, args, options = _case()
    metadata = torch.zeros(8192, dtype=torch.int32, device=args[0].device)
    with pytest.raises(
        (RuntimeError, ValueError), match="A2/A3 INT8 Prefill requires metadata=None"
    ):
        qfa(*args, **options, metadata=metadata)


@pytest.mark.parametrize("layout", ["BNSD", "BSND", "TND"])
@pytest.mark.parametrize("lse", [False, True])
def test_fake_output(layout, lse):
    runner.load_qfa()
    shape = (
        (2, 4, 17, 72)
        if layout == "BNSD"
        else (2, 17, 4, 72)
        if layout == "BSND"
        else (34, 4, 72)
    )
    q = torch.empty(shape, dtype=torch.int8, device="meta")
    scale = torch.empty(shape[:-1], device="meta")
    extra = (
        {
            name: torch.empty(3, dtype=torch.int32, device="meta")
            for name in ("cu_seqlens_q", "cu_seqlens_kv")
        }
        if layout == "TND"
        else {}
    )
    output, actual_lse = torch.ops.cann_ops_transformer.quant_flash_attn(
        q,
        q,
        q,
        scale,
        scale,
        torch.empty(4, device="meta"),
        12,
        p_scale=torch.empty(1, device="meta"),
        layout_q=layout,
        layout_kv=layout,
        layout_out=layout,
        return_softmax_lse=lse,
        **extra,
    )
    assert output.shape == q.shape and output.dtype == torch.bfloat16
    assert actual_lse.shape == (
        ((4, 34) if layout == "TND" else (2, 4, 17)) if lse else (0,)
    )


@pytest.mark.parametrize("layout", ["BNSD", "BSND", "TND"])
def test_pv_fixpipe_head_scales(layout):
    # Uniform scores have an exact mean. This also detects intermediate FP16
    # overflow which the general FP16 numerical benchmark cannot represent.
    _, batches, _, qfa, args, options = _case(
        layout, actual_seq_q=(16, 16), actual_seq_kv=(513, 513)
    )
    args[0].zero_()
    args[1].zero_()
    for batch in batches:
        batch[2][:, 0].fill_(-128)
        batch[2][:, 1].fill_(127)
    args[2].copy_(golden._pack(batches, layout, 2).npu())
    args[3].fill_(1.0)
    args[4].fill_(1.0)
    args[5].copy_(torch.tensor([1.0, 2.0], device=args[5].device))
    options["p_scale"].fill_(127.0)
    output, lse = qfa(*args, **options)
    expected = (
        torch.tensor([-128.0, -128.0, 254.0, 254.0])
        .bfloat16()
        .reshape(1, 4, 1, 1)
        .expand(2, 4, 16, 64)
    )
    expected = golden._pack([(part,) for part in expected.split(1)], layout, 0)
    assert torch.isfinite(output).all()
    torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0.25)
    torch.testing.assert_close(
        lse.cpu(),
        torch.full_like(lse.cpu(), torch.tensor(513.0).log()),
        rtol=5e-4,
        atol=3e-3,
    )


@pytest.mark.parametrize(
    "layout,dim", [("TND", 72), ("TND", 128), ("BNSD", 72), ("BSND", 72)]
)
def test_graph_replay_and_no_host_transfers(layout, dim):
    tnd = layout == "TND"
    lengths = (
        dict(
            actual_seq_q=(0, 17, 65),
            actual_seq_kv=(7, 33, 81),
            max_seqlen_q=82,
            max_seqlen_kv=121,
        )
        if tnd
        else dict(
            N_q=2,
            N_kv=2,
            actual_seq_q=(6144,),
            actual_seq_kv=(6144,),
            sample_rows=(0, 127, 511, -1),
        )
    )
    params, batches, vs, qfa, args, options = _case(layout, dim, mask_mode=3, **lengths)

    def run():
        return qfa(*args, **options)

    for _ in range(3):
        run()
    torch.npu.synchronize()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU]
    ) as prof:
        run()
    torch.npu.synchronize()
    assert not any(event.key == "aten::_to_copy" for event in prof.key_averages())
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        actual = run()
    variants = (
        [((19, 31, 32), (9, 45, 67)), ((30, 0, 52), (7, 0, 114))]
        if tnd
        else [(params["actual_seq_q"], params["actual_seq_kv"])]
    )
    for qlens, klens in variants:
        if tnd:
            for name, values in (("q", qlens), ("kv", klens)):
                options[f"cu_seqlens_{name}"].copy_(
                    torch.tensor(
                        [0, *torch.tensor(values).cumsum(0)],
                        dtype=torch.int32,
                        device="npu",
                    )
                )
            parts = [
                golden.unpack(tensor.cpu(), "TND", qlens if i in (0, 3) else klens)
                for i, tensor in enumerate(args[:5])
            ]
            batches = list(zip(*parts))
        graph.replay()
        torch.npu.synchronize()
        params.update(actual_seq_q=qlens, actual_seq_kv=klens)
        runner.compare(
            params,
            runner.reference(params, batches, vs),
            tuple(value.cpu() for value in actual),
        )


def test_int8_under_stream_core_limit():
    params, batches, vs, qfa, args, options = _case("TND")
    acl = ctypes.CDLL("libascendcl.so")
    stream = ctypes.c_void_p(torch.npu.current_stream().npu_stream)
    acl.aclrtSetStreamResLimit.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_uint]
    for name in (
        "aclrtUseStreamResInCurrentThread",
        "aclrtUnuseStreamResInCurrentThread",
        "aclrtResetStreamResLimit",
    ):
        getattr(acl, name).argtypes = [ctypes.c_void_p]
    try:
        assert acl.aclrtSetStreamResLimit(stream, 0, 8) == 0
        assert acl.aclrtSetStreamResLimit(stream, 1, 16) == 0
        assert acl.aclrtUseStreamResInCurrentThread(stream) == 0
        actual = qfa(*args, **options)
        runner.compare(
            params,
            runner.reference(params, batches, vs),
            tuple(value.cpu() for value in actual),
        )
    finally:
        assert acl.aclrtUnuseStreamResInCurrentThread(stream) == 0
        assert acl.aclrtResetStreamResLimit(stream) == 0


@pytest.fixture(scope="module")
def asset_spec():
    path = Path(__file__).resolve().parents[2] / "assets/spec.py"
    spec = importlib.util.spec_from_file_location("int8_test_asset_spec", path)
    module = importlib.util.module_from_spec(spec)
    registration = SimpleNamespace(
        quant_flash_attn=None, quant_flash_attn_metadata=None
    )
    with patch.dict(sys.modules, {"cann_ops_transformer.ops": registration}):
        spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("layout", ["BNSD", "BSND", "TND"])
@pytest.mark.parametrize("numpy", [False, True])
def test_assets_round_trip(asset_spec, layout, numpy):
    params, batches, vs, _, _, _ = _case(layout, 72)
    args = [golden._pack(batches, layout, index) for index in range(5)] + [vs, 12]
    options = dict(
        p_scale=torch.tensor([127.0]),
        layout_q=layout,
        layout_q_descale=layout,
        layout_kv=layout,
        layout_out=layout,
        max_seqlen_q=17,
        max_seqlen_kv=33,
        softmax_scale=72**-0.5,
        mask_mode=0 if numpy else 3,
        win_left=7,
        win_right=3,
        return_softmax_lse=not numpy,
    )
    if layout == "TND":
        for name, lengths in (
            ("q", params["actual_seq_q"]),
            ("kv", params["actual_seq_kv"]),
        ):
            options[f"cu_seqlens_{name}"] = torch.tensor(
                [0, *torch.tensor(lengths).cumsum(0)], dtype=torch.int32
            )
            options[f"cu_seqlens_{name}_values"] = options[
                f"cu_seqlens_{name}"
            ].tolist()
    if numpy:
        args = [value.numpy() if torch.is_tensor(value) else value for value in args]
        options = {
            name: value.numpy() if torch.is_tensor(value) else value
            for name, value in options.items()
        }
    name = f"asset_{layout}_{numpy}"
    asset_spec._inputs_dispatch(*args, **options, seed=902, p_scale_value=127.0)
    expected = asset_spec._golden_dispatch(*args, **options, testcase_name=name)
    args = [
        torch.as_tensor(value).npu() if not isinstance(value, int) else value
        for value in args
    ]
    options = {
        key: torch.as_tensor(value).npu()
        if isinstance(value, (torch.Tensor, np.ndarray))
        else value
        for key, value in options.items()
        if not key.endswith("_values")
    }
    preprocessed = asset_spec.QuantFlashAttnSpec.npu_preprocess(*args, **options)
    assert preprocessed == {"metadata": None}
    options.update(preprocessed)
    if not numpy:
        for key in (
            "block_table",
            "cu_seqlens_q",
            "cu_seqlens_kv",
            "seqused_q",
            "seqused_kv",
            "sinks",
            "attn_mask",
        ):
            options.setdefault(key, None)
        actual = asset_spec.QuantFlashAttnSpec.torch_graph(*args, **options)()
    else:
        actual = torch.ops.cann_ops_transformer.quant_flash_attn(*args, **options)
    actual = [value.cpu() for value in actual if value is not None and value.numel()]
    if numpy:
        actual = [actual[0].view(torch.uint16).numpy(), *actual[1:]]
    context = SimpleNamespace(testcase_name=name)
    result = asset_spec.QuantFlashAttnSpec.compare(
        *actual, *expected, compare_context=context
    )
    assert result["pass"], result
    wrong = [torch.full_like(expected[0], 1000.0), *actual[1:]]
    assert not asset_spec.QuantFlashAttnSpec.compare(
        *wrong, *expected, compare_context=context
    )["pass"]
