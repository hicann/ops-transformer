# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""External dispatcher/TTK adaptation checks; opt in to real NPU execution."""

import ast
import csv
import itertools
import os
from pathlib import Path
import sys

import pytest
import torch

TESTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TESTS / "assets" / "impl"))
import gdn_ttk_ops as adapter


def fixtures(layout, *, length=65, flags=(True, True, True), outputs=True):
    b, hq, hv = 2, 1, 2
    lengths = [length, max(1, length - 2)]
    if layout == "TND":
        qs, vs = (sum(lengths), hq, 128), (sum(lengths), hv, 128)
    elif layout == "BSND":
        qs, vs = (b, length, hq, 128), (b, length, hv, 128)
    else:
        qs, vs = (b, hq, length, 128), (b, hv, length, 128)
    shapes = [
        qs,
        qs,
        vs,
        vs[:-1],
        vs[:-1],
        (12 + b,),
        (b, hv, 128, 128),
        (hv,),
        (hv,),
        (b + 1,),
        (b,),
    ]
    dtypes = (
        [torch.bfloat16] * 3
        + [torch.float32] * 2
        + [torch.int32]
        + [torch.float32] * 3
        + [torch.int32] * 2
    )
    templates = [
        torch.empty(shape, dtype=dtype) for shape, dtype in zip(shapes, dtypes)
    ]
    attributes = dict(
        layout=layout,
        valid_lengths=lengths,
        initial_state_mode="random",
        dt_bias_mode="random",
        inference=True,
        output_final_state=outputs,
        return_intermediate_states=outputs,
        use_qk_l2norm_in_kernel=flags[0],
        use_gate_in_kernel=flags[1],
        use_beta_sigmoid_in_kernel=flags[2],
        allow_neg_eigval=True,
    )
    return adapter.load_asset("inputs").make(
        templates, attributes, host_fixture=True
    ), attributes


@pytest.mark.parametrize("layout", ["TND", "BSND", "BNSD"])
@pytest.mark.parametrize("flags", list(itertools.product([False, True], repeat=3)))
@pytest.mark.parametrize("entry", ["flash_gdn", "flash_gdn_fwd"])
def test_public_meta_contract(layout, flags, entry):
    inputs, attributes = fixtures(layout, flags=flags)
    # The host metadata and length descriptions are real; only compute tensors are Meta.
    inputs = [
        value.to("meta") if i not in (5, 9, 10) else value
        for i, value in enumerate(inputs)
    ]
    result = adapter.invoke(inputs, attributes, entry=entry)
    assert len(result) == (3 if entry == "flash_gdn" else 16)
    assert result[0].shape == inputs[2].shape


def test_independent_golden_two_token_recurrence():
    inputs, attributes = fixtures("TND", length=2, flags=(False, False, False))
    # Two sequences of lengths 2 and 1. Head group shares the same unit Q/K.
    for tensor in inputs[:3]:
        tensor.zero_()
    inputs[0][..., 0] = inputs[1][..., 0] = 1
    inputs[2][:, :, 0] = torch.tensor([1.0, 4.0, 7.0])[:, None]
    inputs[3].zero_()
    inputs[4].fill_(0.5)
    attributes.update(initial_state_mode="none", scale=1.0)
    result = adapter.load_asset("golden").reference(inputs, attributes)
    assert result[0][:, 0, 0].tolist() == [0.5, 2.25, 3.5]
    assert result[1][:, 0, 0, 0].tolist() == [2.25, 3.5]
    assert torch.count_nonzero(result[2]) == 0


def test_csv_fixtures_and_test_spec_registration():
    spec = adapter.load_asset("spec")
    assert set(spec.__spec__) == {"gdn_ttk_ops.flash_gdn", "gdn_ttk_ops.flash_gdn_fwd"}
    cases = sorted((TESTS / "assets/cases").glob("*.csv"))
    assert {path.name for path in cases} == {
        "flash_gdn_smoke.csv",
        "flash_gdn_a5_redline.csv",
    }
    for path in cases:
        with path.open(newline="") as stream:
            for row in csv.DictReader(stream):
                attributes = ast.literal_eval(row["attributes"])
                shapes = ast.literal_eval(row["tensor_view_shapes"])
                dtypes = ast.literal_eval(row["tensor_dtypes"])
                templates = [
                    torch.empty(shape, device="meta", dtype=getattr(torch, dtype))
                    for shape, dtype in zip(shapes, dtypes)
                ]
                c = adapter.load_asset("inputs").configuration(templates, attributes)
                assert shapes[3] == shapes[4] == shapes[2][:-1]
                assert shapes[5] == (12 + c["batch"],)
                assert shapes[7] == shapes[8] == (c["hv"],)
                assert shapes[6] == (c["batch"], c["hv"], 128, 128)
                assert row["api_name"] in spec.__spec__


@pytest.mark.parametrize("layout", ["TND", "BSND", "BNSD"])
def test_ttk_customize_and_omitted_output_flags(layout, monkeypatch):
    inputs, attributes = fixtures(layout, length=3, outputs=False)
    helper = adapter.load_asset("inputs")
    original = helper.make_metadata
    monkeypatch.setattr(
        helper, "make_metadata", lambda c, **kw: original(c, host_fixture=True)
    )
    target = [torch.empty_like(x) for x in inputs]
    result = adapter.load_asset("spec").FlashGdnFwdTestSpec.customize_inputs(
        *target, **attributes
    )
    for actual, expected in zip(result, inputs):
        assert torch.equal(actual, expected)
    meta = [x.to("meta") if i not in (5, 9, 10) else x for i, x in enumerate(result)]
    for entry in (adapter.flash_gdn_fwd, adapter.flash_gdn):
        assert len(entry(*meta, **attributes)) == 1
    attributes.update(
        output_final_state=True,
        return_intermediate_states=True,
        omitted_attributes="output_final_state,return_intermediate_states",
    )
    assert len(adapter.flash_gdn_fwd(*meta, **attributes)) == 1


NPU = pytest.mark.skipif(
    os.environ.get("RUN_GDN_NPU") != "1",
    reason="set RUN_GDN_NPU=1 for real NPU execution",
)


@pytest.mark.npu
@NPU
@pytest.mark.parametrize("layout", ["TND", "BSND", "BNSD"])
@pytest.mark.parametrize("flags", [(False, False, False), (True, True, True)])
def test_external_npu_forward(layout, flags):
    import torch_npu  # noqa: F401

    torch.npu.set_device(os.environ.get("GDN_DEVICE", "npu:0"))
    inputs, attributes = fixtures(layout, flags=flags)
    before = [x.clone() for x in inputs]
    expected = adapter.load_asset("golden").reference(inputs, attributes)
    device = [x.to("npu") for x in inputs]  # Includes TTK's metadata transfer behavior.
    actual = adapter.invoke(device, attributes)
    torch.npu.synchronize()
    for got, want in zip(
        adapter.load_asset("compare").pack(actual),
        adapter.load_asset("compare").pack(expected),
    ):
        torch.testing.assert_close(
            got.cpu(),
            want,
            atol=0.01 if got.dtype == torch.bfloat16 else 1e-4,
            rtol=0.01 if got.dtype == torch.bfloat16 else 1e-3,
        )
    for got, want in zip(device, before):
        assert torch.equal(got.cpu(), want)
    selected = adapter.flash_gdn_fwd(*device, **attributes)
    assert len(selected) == 3
    for got, want in zip(selected, actual[:3]):
        assert torch.equal(got, want)
