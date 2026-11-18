# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Autograd smoke test: flash_kda forward plus the flash_kda_grad backward.

The forward runs through ``flash_kda`` and the backward dispatches to the
``attention/flash_kda_grad`` DSL operator, so a single ``torch.autograd.grad``
call exercises both directions end to end.
"""

import pytest

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.npu

BATCH = 1
TOKENS = 128
HEADS_QK = 1
HEADS_V = 2
HEAD_DIM = 128


def _require_npu():
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("NPU is not available")


@pytest.mark.parametrize("disable_recompute", [False, True])
def test_autograd_grad_reaches_every_input(disable_recompute):
    _require_npu()
    import attention.flash_kda_grad.torch_extension  # noqa: F401
    import attention.flash_kda.torch_extension.flash_kda  # noqa: F401

    flash_kda = torch.ops.cann_ops_transformer.flash_kda
    flash_kda_metadata = torch.ops.cann_ops_transformer.flash_kda_metadata

    device = "npu:0"
    torch.npu.set_device(0)
    cu_seqlens = torch.tensor([0, TOKENS], dtype=torch.int32, device=device)

    inputs = {
        "q": torch.randn(
            TOKENS, HEADS_QK, HEAD_DIM, device=device, dtype=torch.bfloat16
        ),
        "k": torch.randn(
            TOKENS, HEADS_QK, HEAD_DIM, device=device, dtype=torch.bfloat16
        ),
        "v": torch.randn(
            TOKENS, HEADS_V, HEAD_DIM, device=device, dtype=torch.bfloat16
        ),
        "g": torch.randn(
            TOKENS, HEADS_V, HEAD_DIM, device=device, dtype=torch.bfloat16
        ),
        "beta": torch.randn(TOKENS, HEADS_V, device=device, dtype=torch.bfloat16),
        "h0": torch.zeros(
            BATCH, HEADS_V, HEAD_DIM, HEAD_DIM, device=device, dtype=torch.float32
        ),
        "A_log": torch.zeros(HEADS_V, device=device, dtype=torch.float32),
        "dt_bias": torch.zeros(HEADS_V, HEAD_DIM, device=device, dtype=torch.float32),
    }
    for tensor in inputs.values():
        tensor.requires_grad_(True)

    with torch.no_grad():
        metadata = flash_kda_metadata(
            cu_seqlens,
            batch_size=BATCH,
            nheads_qk=HEADS_QK,
            nheads_v=HEADS_V,
            seqlens=TOKENS,
            enable_grad=True,
            layout="TND",
        )
    out, final_state, chunk_states = flash_kda(
        inputs["q"],
        inputs["k"],
        inputs["v"],
        inputs["g"],
        inputs["beta"],
        metadata,
        inputs["h0"],
        inputs["A_log"],
        inputs["dt_bias"],
        cu_seqlens=cu_seqlens,
        lower_bound=-0.06,
        layout="TND",
        output_final_state=True,
        disable_recompute=disable_recompute,
    )
    assert type(out.grad_fn).__name__ == "_FlashKDAEntryBackward"
    assert out.shape == (TOKENS, HEADS_V, HEAD_DIM)
    assert out.dtype == torch.bfloat16
    assert final_state.shape == (BATCH, HEADS_V, HEAD_DIM, HEAD_DIM)

    if disable_recompute:
        assert chunk_states.dtype == torch.bfloat16
        assert chunk_states.shape == (
            (TOKENS + 63) // 64 + BATCH,
            HEADS_V,
            HEAD_DIM,
            HEAD_DIM,
        )
        assert not chunk_states.requires_grad
    else:
        assert chunk_states is None

    grads = torch.autograd.grad(
        outputs=(out, final_state),
        inputs=tuple(inputs.values()),
        grad_outputs=(torch.ones_like(out), torch.ones_like(final_state)),
        retain_graph=False,
        allow_unused=False,
    )
    assert len(grads) == len(inputs)

    for (name, tensor), grad in zip(inputs.items(), grads):
        assert grad is not None, f"{name} received no gradient"
        assert grad.shape == tensor.shape, f"{name} gradient shape mismatch"
        assert torch.isfinite(grad.float()).all(), f"{name} gradient is not finite"
        assert (grad.float() != 0).any(), f"{name} gradient is all zero"
