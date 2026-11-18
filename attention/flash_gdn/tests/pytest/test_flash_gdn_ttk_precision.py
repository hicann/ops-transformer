# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""CPU-only precision-policy and independent-reference checks; no DUT launch."""

import itertools
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "assets" / "impl"))
import gdn_ttk_ops as adapter


def _case(layout, flags, length=65):
    gen = torch.Generator().manual_seed(71)
    lengths = [length, max(1, length - 2)]
    attrs = dict(
        layout=layout,
        valid_lengths=lengths,
        initial_state_mode="random",
        dt_bias_mode="random",
        output_final_state=True,
        return_intermediate_states=True,
        use_qk_l2norm_in_kernel=flags[0],
        use_gate_in_kernel=flags[1],
        use_beta_sigmoid_in_kernel=flags[2],
        allow_neg_eigval=True,
    )

    def rand(shape):
        return torch.randn(shape, generator=gen)

    def pack(value):
        if layout == "TND":
            return torch.cat([value[b, :n] for b, n in enumerate(lengths)]).contiguous()
        if layout == "BNSD":
            return value.transpose(1, 2).contiguous()
        return value

    q, k = [(rand((2, length, 1, 128)) * 0.08).bfloat16() for _ in range(2)]
    v = rand((2, length, 2, 128)).bfloat16()
    g = rand((2, length, 2)) if flags[1] else -rand((2, length, 2)).abs() * 0.1
    beta = rand((2, length, 2)) if flags[2] else rand((2, length, 2)).sigmoid()
    # Neither CPU reference reads metadata content or launches its producer.
    inputs = [pack(x) for x in (q, k, v, g, beta)] + [
        torch.zeros(14, dtype=torch.int32),
        rand((2, 2, 128, 128)) * 0.02,
        torch.tensor([-1.2, -0.7]),
        torch.tensor([0.1, -0.2]),
        torch.tensor([0, lengths[0], sum(lengths)], dtype=torch.int32),
        torch.tensor(lengths, dtype=torch.int32),
    ]
    return inputs, attrs


@pytest.mark.parametrize("layout", ["TND", "BSND", "BNSD"])
@pytest.mark.parametrize("flags", list(itertools.product([False, True], repeat=3)))
def test_cpu_chunk_algebra_matches_token_recurrence(layout, flags):
    inputs, attrs = _case(layout, flags)
    before = [x.clone() for x in inputs]
    high = adapter.load_asset("golden").reference(inputs, attrs)
    peer = adapter.load_asset("peer").reference(inputs, attrs)
    assert len(peer) == 16 and all(x is None for x in peer[3:])
    for index, (actual, expected) in enumerate(zip(peer[:3], high[:3])):
        assert actual.shape == expected.shape and actual.dtype == expected.dtype
        torch.testing.assert_close(
            actual,
            expected,
            atol=0.01 if index == 0 else 1e-4,
            rtol=0.01 if index == 0 else 1e-3,
        )
    for actual, expected in zip(inputs, before):
        assert torch.equal(actual, expected)


def test_cpu_peer_analytic_two_token_case():
    inputs, attrs = _case("TND", (False, False, False), length=2)
    for x in inputs[:3]:
        x.zero_()
    inputs[0][..., 0] = inputs[1][..., 0] = 1
    inputs[2][:, :, 0] = torch.tensor([1.0, 4.0, 7.0])[:, None]
    inputs[3].zero_()
    inputs[4].fill_(0.5)
    attrs.update(initial_state_mode="none", scale=1.0)
    result = adapter.load_asset("peer").reference(inputs, attrs)
    assert result[0][:, 0, 0].tolist() == [0.5, 2.25, 3.5]
    assert result[1][:, 0, 0, 0].tolist() == [2.25, 3.5]
    assert torch.count_nonzero(result[2]) == 0


@pytest.mark.parametrize(
    "state,chunk", list(itertools.product([False, True], repeat=2))
)
def test_golden_promote_and_peer_output_contract(state, chunk):
    inputs, attrs = _case("BSND", (True, True, True), length=3)
    attrs.update(output_final_state=state, return_intermediate_states=chunk)
    registration = adapter.load_asset("spec")
    spec = registration.FlashGdnFwdTestSpec
    assert spec.tolerance == {
        dtype: {"standard": "cross_check", "level": "L2"}
        for dtype in ("bfloat16", "float32")
    }
    assert spec.third_party == {"torch": registration.CpuChunkSimulation}
    promoted = [
        x.float()
        if x.dtype == torch.bfloat16
        else x.double()
        if x.dtype == torch.float32
        else x
        for x in inputs
    ]
    high = spec.golden(*promoted, **attrs, golden_mode="Promote")
    peer = spec.third_party["torch"]()(*inputs, **attrs)
    assert len(high) == len(peer) == 1 + state + chunk
    assert high[0].dtype == torch.float32 and peer[0].dtype == torch.bfloat16
    assert all(x.dtype == torch.float64 for x in high[1:])
    assert all(x.dtype == torch.float32 for x in peer[1:])
    assert all(x.device.type == "cpu" for x in high + peer)


def test_native_ttk_l2_resolution_and_reject_missing_peer():
    resolve = pytest.importorskip("ttk.core_modules.common.comparison.resolve")
    cross = pytest.importorskip("ttk.core_modules.common.comparison.cross_check")
    import numpy as np

    tolerance = adapter.load_asset("spec").FlashGdnFwdTestSpec.tolerance
    standards = resolve.resolve_tolerance(
        tolerance, None, None, ["bfloat16", "float32"], None
    )
    assert resolve.needs_promote_golden(standards)
    for standard, small, atol in zip(standards, [2**-8, 2**-14], [2**-16, 2**-30]):
        assert standard.token == "cross_check"
        options = standard.params
        assert options["level"] == "L2"
        assert [options[k] for k in ("mare_ratio", "mere_ratio", "rmse_ratio")] == [
            2.0,
            1.2,
            1.2,
        ]
        assert options["small_value"] == small and options["small_value_atol"] == atol
        kwargs = dict(output_idx=0, output_dtype="float32", options=options)
        golden = np.ones(8)
        peer = golden + 0.01
        good = cross.CrossCheckComparison(
            peer, golden, third_party=peer, **kwargs
        ).compare_impl()
        bad = cross.CrossCheckComparison(
            golden + 0.03, golden, third_party=peer, **kwargs
        ).compare_impl()
        missing = cross.CrossCheckComparison(
            peer, golden, third_party=None, **kwargs
        ).compare_impl()
        assert good.is_pass and not bad.is_pass and not missing.is_pass
