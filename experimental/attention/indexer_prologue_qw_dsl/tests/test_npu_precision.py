# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""T=72 NPU precision: torch.ops ds41.indexer_prologue_qw vs the CPU golden."""

from __future__ import annotations

import pytest
import torch

from golden import indexer_prologue_qw_golden, make_inputs

DIM, Q_LORA, N, D, DR = 5120, 1280, 32, 128, 64
T = 72

torch_npu = pytest.importorskip("torch_npu", reason="torch_npu not installed")

if not torch.npu.is_available():  # pragma: no cover - depends on the host
    pytest.skip("no NPU visible", allow_module_level=True)


def _op():
    from cann_ops_transformer.ops.ds41 import indexer_prologue_qw

    return indexer_prologue_qw


def _to_nz():
    from ops.indexer_prologue_qw import to_nz

    return to_nz


def _to_npu(inputs):
    to_nz = _to_nz()
    tensors = {
        name: getattr(inputs, name).npu()
        for name in (
            "x",
            "qr",
            "wqb",
            "ww",
            "descale_qr",
            "descale_wqb",
            "rope_sin",
            "rope_cos",
        )
    }
    for name in ("wqb", "ww"):
        tensors[name] = to_nz(tensors[name])
    return tensors


def test_t72_matches_golden():
    inputs = make_inputs(T, DIM, Q_LORA, N, D, DR, softmax_scale=1.0, seed=0)
    want = indexer_prologue_qw_golden(inputs)
    got_q, got_dsc, got_w = _op()(**_to_npu(inputs), softmax_scale=1.0)
    got_q, got_dsc, got_w = got_q.cpu(), got_dsc.cpu(), got_w.cpu()

    assert got_dsc.shape == want.descale_q.shape
    n_bad = int((got_dsc != want.descale_q).sum())
    assert n_bad == 0, f"{n_bad} descale_q bytes differ"

    assert got_q.shape == want.q.shape
    bad = got_q != want.q
    n_bad = int(bad.sum())
    assert n_bad == 0, (
        f"{n_bad}/{want.q.numel()} q bytes differ, first at "
        f"{tuple(int(v) for v in torch.nonzero(bad)[0])}"
    )

    assert got_w.shape == want.w.shape
    torch.testing.assert_close(got_w, want.w, rtol=2e-2, atol=2e-2)
