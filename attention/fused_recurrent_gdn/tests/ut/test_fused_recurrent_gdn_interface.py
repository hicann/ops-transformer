# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Metadata contract, layout transforms and pre-mutation descriptor checks."""

import importlib
import pytest
import torch

module = importlib.import_module(
    "cann_ops_transformer_custom.ops.attention.fused_recurrent_gdn.fused_recurrent_gdn"
)
fused_recurrent_gdn = module.fused_recurrent_gdn


def inputs(layout="TND", dk=64, dv=96, beta_raw=False, gate_raw=False, device="meta"):
    shapes = {
        "TND": ((6, 1, dk), (6, 2, dv)),
        "NTD": ((1, 6, dk), (2, 6, dv)),
        "BSND": ((2, 3, 1, dk), (2, 3, 2, dv)),
        "BNSD": ((2, 1, 3, dk), (2, 2, 3, dv)),
    }
    qs, vs = shapes[layout]
    q = torch.empty(qs, dtype=torch.bfloat16, device=device)
    v = torch.empty(vs, dtype=torch.bfloat16, device=device)
    beta = torch.empty(
        vs[:-1], dtype=torch.bfloat16 if beta_raw else torch.float32, device=device
    )
    g = torch.empty(
        vs[:-1], dtype=torch.bfloat16 if gate_raw else torch.float32, device=device
    )
    options = dict(
        layout=layout, use_beta_sigmoid_in_kernel=beta_raw, use_gate_in_kernel=gate_raw
    )
    if gate_raw:
        options["A_log"] = torch.zeros(2, dtype=torch.float32, device=device)
    if layout in ("TND", "NTD"):
        options["cu_seqlens"] = torch.tensor(
            [0, 3, 6], dtype=torch.int32, device=device
        )
    return (q, q.clone(), v, beta, g), options


@pytest.mark.parametrize("layout", ("TND", "NTD", "BSND", "BNSD"))
@pytest.mark.parametrize("dim", (64, 65, 96, 128, 192, 256, 384, 511, 512))
@pytest.mark.parametrize(
    "raw", ((False, False), (True, False), (False, True), (True, True))
)
@pytest.mark.parametrize("state_dtype", (torch.bfloat16, torch.float32))
def test_meta_grid(layout, dim, raw, state_dtype):
    data, options = inputs(layout, dim, 576 - dim, *raw)
    state = torch.empty((2, 2, dim, 576 - dim), dtype=state_dtype, device="meta")
    out, final = fused_recurrent_gdn(*data, state, output_final_state=True, **options)
    assert out.shape == data[2].shape and out.dtype == torch.bfloat16
    assert final.shape == state.shape and final.dtype == state_dtype
    assert fused_recurrent_gdn(*data, state, **options)[1] is None


@pytest.mark.parametrize("index_rank", (1, 2))
@pytest.mark.parametrize("state_dtype", (torch.bfloat16, torch.float32))
def test_pool_meta(index_rank, state_dtype):
    data, options = inputs()
    state = torch.empty((8, 2, 64, 96), dtype=state_dtype, device="meta")
    indices = torch.empty(
        (2,) if index_rank == 1 else (2, 3), dtype=torch.int32, device="meta"
    )
    if index_rank == 2:
        options["num_accepted_tokens"] = torch.empty(
            2, dtype=torch.int32, device="meta"
        )
    out, final = fused_recurrent_gdn(*data, state, ssm_state_indices=indices, **options)
    assert out.shape == data[2].shape and final is None


@pytest.mark.parametrize("name", ("beta", "g"))
@pytest.mark.parametrize("raw", (False, True))
def test_dtype_activation_mismatch(name, raw):
    data, options = inputs(
        beta_raw=raw if name == "beta" else False,
        gate_raw=raw if name == "g" else False,
    )
    data = list(data)
    index = 3 if name == "beta" else 4
    data[index] = data[index].to(torch.float32 if raw else torch.bfloat16)
    with pytest.raises(ValueError, match=name):
        fused_recurrent_gdn(*data, **options)


@pytest.mark.parametrize("dim", (0, 63, 513))
def test_invalid_head_dimension(dim):
    data, options = inputs(dk=dim)
    with pytest.raises(ValueError, match="64, 512"):
        fused_recurrent_gdn(*data, **options)


def test_required_dependencies():
    data, options = inputs(gate_raw=True)
    del options["A_log"]
    with pytest.raises(ValueError, match="A_log"):
        fused_recurrent_gdn(*data, **options)
    data, options = inputs()
    indices = torch.empty((2, 3), dtype=torch.int32, device="meta")
    state = torch.empty((8, 2, 64, 96), dtype=torch.float32, device="meta")
    with pytest.raises(ValueError, match="initial_state"):
        fused_recurrent_gdn(*data, ssm_state_indices=indices, **options)
    with pytest.raises(ValueError, match="num_accepted_tokens"):
        fused_recurrent_gdn(*data, state, ssm_state_indices=indices, **options)
    with pytest.raises(ValueError, match="output_final_state=False"):
        fused_recurrent_gdn(
            *data, state, ssm_state_indices=indices, output_final_state=True, **options
        )


def test_sequence_plan_restore_and_destinations():
    q = torch.empty(5, 1, 64)
    state = torch.randn(8, 2, 64, 96)
    before = state.clone()
    cu = torch.tensor([0, 2, 5], dtype=torch.int32)
    indices = torch.tensor([[2, 3, 4], [5, 6, 7]], dtype=torch.int32)
    accepted = torch.tensor([2, 1], dtype=torch.int32)
    spans, reads, writes = module._sequence_plan(
        q, 2, "TND", cu, None, state, indices, accepted
    )
    assert spans == [[0, 2], [2, 5]] and reads == [3, 5] and writes == [2, 3, 5, 6, 7]
    assert torch.equal(before, state)


@pytest.mark.parametrize(
    "case",
    (
        "offsets",
        "length",
        "negative",
        "overflow",
        "overlap",
        "accepted",
        "width",
        "duplicate",
    ),
)
def test_descriptor_rejection_is_atomic(case):
    q = torch.empty(6, 1, 64)
    state = torch.randn(8, 2, 64, 96)
    before = state.clone()
    cu = torch.tensor([0, 3, 6], dtype=torch.int32)
    used = None
    indices = torch.tensor([[0, 1, 2], [4, 5, 6]], dtype=torch.int32)
    accepted = torch.tensor([1, 1], dtype=torch.int32)
    if case == "offsets":
        cu[1] = 7
    if case == "length":
        used = torch.tensor([4, 3], dtype=torch.int32)
    if case == "negative":
        indices[0, 0] = -1
    if case == "overflow":
        indices[0, 0] = 8
    if case == "overlap":
        indices[1, 0] = 1
    if case == "accepted":
        accepted[0] = 0
    if case == "width":
        indices = indices[:, :2]
    if case == "duplicate":
        indices[0, 1] = 0
    with pytest.raises(ValueError):
        module._sequence_plan(q, 2, "TND", cu, used, state, indices, accepted)
    assert torch.equal(state, before)


@pytest.mark.parametrize("layout", ("TND", "NTD", "BSND", "BNSD"))
def test_reject_legacy_state_orientation(layout):
    data, options = inputs(layout, dk=64, dv=96)
    legacy = torch.empty((2, 2, 96, 64), device="meta")
    with pytest.raises(ValueError, match="initial_state must have shape"):
        fused_recurrent_gdn(*data, legacy, **options)


def test_overlapping_pool_rejected_before_write():
    data, options = inputs(device="cpu")
    initial = torch.zeros(1, 2, 64, 96).expand(8, -1, -1, -1)
    before = initial.clone()
    options["ssm_state_indices"] = torch.tensor([0, 1], dtype=torch.int32)
    with pytest.raises(ValueError, match="overlapping"):
        module._infer_outputs(*data, initial, **options)
    assert torch.equal(initial, before)


def test_dsl_public_entry_matches_torch_contract():
    import inspect

    backend = importlib.import_module(
        "cann_ops_transformer_custom.ops.attention.fused_recurrent_gdn.op_dsl.fused_recurrent_gdn"
    )
    assert inspect.signature(backend.fused_recurrent_gdn) == inspect.signature(
        fused_recurrent_gdn
    )


@pytest.mark.parametrize("layout", ("TND", "NTD"))
@pytest.mark.parametrize(("tokens", "batch"), ((5, 0), (5, 1), (0, 0)))
def test_packed_zero_batch_contract(layout, tokens, batch):
    q_shape = (tokens, 1, 64) if layout == "TND" else (1, tokens, 64)
    v_shape = (tokens, 2, 96) if layout == "TND" else (2, tokens, 96)
    q = torch.empty(q_shape, device="meta", dtype=torch.bfloat16)
    v = torch.empty(v_shape, device="meta", dtype=torch.bfloat16)
    beta = torch.empty(v_shape[:-1], device="meta", dtype=torch.float32)
    lengths = torch.empty(batch + 1, device="meta", dtype=torch.int32)
    args = (q, q, v, beta, beta)
    options = dict(cu_seqlens=lengths, output_final_state=True, layout=layout)
    if tokens and not batch:
        with pytest.raises(ValueError, match="Nonempty packed input"):
            fused_recurrent_gdn(*args, **options)
    else:
        out, final = fused_recurrent_gdn(*args, **options)
        assert out.shape == v_shape
        assert final.shape == (batch, 2, 64, 96)
