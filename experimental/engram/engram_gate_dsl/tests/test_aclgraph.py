# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""ACLGraph capture/replay verification for engram_gate.

两层验证：
1. torch.npu.NPUGraph（torch_npu 原生 aclgraph 捕获）：DSL kernel launch 可被
   图捕获，重放时对新输入数据/新 mask 值结果正确（kernel 侧按运行时值读取）。
2. torchair npugraph_ex（依赖 torchair，缺失时跳过）：经 torch_extension 自定
   义算子边界（已安装 wheel 的 ``torch.ops.cann_ops_transformer.ds41``）被
   dynamo 全图捕获并重放。
"""

import pytest
import torch

from ops.engram_gate import engram_gate  # noqa: E402

torch_npu = pytest.importorskip("torch_npu", reason="torch_npu not installed")
if not torch.npu.is_available():  # pragma: no cover - host dependent
    pytest.skip("no NPU visible", allow_module_level=True)


def _golden(x, key, value, weight, eps, clamp_value, image_mask=None):
    x_f = x.float()
    rstd = torch.rsqrt(x_f.square().mean(-1) + eps) * torch.rsqrt(
        key.square().mean(-1) + eps
    )
    dot = (x_f * weight * key).sum(-1) * rstd * x.shape[-1] ** -0.5
    gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(clamp_value).sqrt(), dot))
    if image_mask is not None:
        gate = gate.masked_fill(image_mask.unsqueeze(-1), 0)
    return (x_f + gate.unsqueeze(-1) * value.float().unsqueeze(-2)).to(x.dtype)


def _make_inputs(t, hc, dim, seed):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(t, hc, dim, generator=generator).to(torch.bfloat16)
    key = torch.randn(t, hc, dim, generator=generator).to(torch.bfloat16)
    value = torch.randn(t, dim, generator=generator).to(torch.bfloat16)
    weight = torch.randn(hc, dim, generator=generator, dtype=torch.float32)
    return x, key, value, weight


@pytest.mark.npu
@pytest.mark.parametrize(
    "t,hc,dim,use_mask",
    [
        pytest.param(72, 4, 5120, False, id="model_shape_nomask"),
        pytest.param(72, 4, 5120, True, id="model_shape_partial_mask"),
        pytest.param(5, 1, 5120, False, id="two_dim_hc1"),
    ],
)
def test_engram_gate_aclgraph_capture_and_replay(t, hc, dim, use_mask):
    torch.npu.set_device(0)
    static_x, static_key, static_value, static_weight = (
        tensor.npu() for tensor in _make_inputs(t, hc, dim, 20260924)
    )
    static_mask = None
    if use_mask:
        static_mask = (torch.arange(t) % 2 == 1).npu()

    warmup_stream = torch.npu.Stream()
    warmup_stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(warmup_stream):
        for _ in range(3):
            engram_gate(static_x, static_key, static_value, static_weight, static_mask)
    torch.npu.current_stream().wait_stream(warmup_stream)
    torch.npu.synchronize()

    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        static_out = engram_gate(
            static_x, static_key, static_value, static_weight, static_mask
        )
    torch.npu.synchronize()

    for replay_idx in range(3):
        x, key, value, weight = _make_inputs(t, hc, dim, 20260925 + replay_idx)
        static_x.copy_(x.npu())
        static_key.copy_(key.npu())
        static_value.copy_(value.npu())
        static_weight.copy_(weight.npu())
        mask = None
        if use_mask:
            mask = (torch.arange(t) % 2 == (replay_idx % 2)).cpu()
            static_mask.copy_(mask.npu())
        torch.npu.synchronize()

        graph.replay()
        torch.npu.synchronize()

        expected = _golden(x, key, value, weight, 1e-6, 1e-6, mask)
        assert torch.allclose(
            static_out.cpu().float(), expected.float(), atol=2e-2, rtol=2e-2
        ), (
            f"replay {replay_idx}: max error {(static_out.cpu().float() - expected.float()).abs().max().item()}"
        )
        if mask is not None:
            assert torch.equal(
                static_out.cpu().view(t, hc, dim)[mask.view(-1)],
                x.view(t, hc, dim)[mask.view(-1)],
            )


@pytest.mark.npu
@pytest.mark.parametrize(
    "t,hc,dim,use_mask",
    [
        pytest.param(72, 4, 5120, False, id="model_shape_nomask"),
        pytest.param(72, 4, 5120, True, id="model_shape_partial_mask"),
    ],
)
def test_engram_gate_torchair_npugraph_ex(t, hc, dim, use_mask):
    torchair = pytest.importorskip("torchair", reason="torchair not installed")
    from torchair.configs.compiler_config import CompilerConfig

    # Explicit import surfaces DSL registration failures instead of testing
    # another implementation.
    from cann_ops_transformer.ops.engram.engram_gate_dsl import engram_gate_torch

    torch.npu.set_device(0)

    class _Model(torch.nn.Module):
        def forward(self, x, key, value, weight, mask):
            if mask is None:
                return engram_gate_torch(x, key, value, weight)
            return engram_gate_torch(x, key, value, weight, mask)

    static_x, static_key, static_value, static_weight = (
        tensor.npu() for tensor in _make_inputs(t, hc, dim, 20260926)
    )
    static_mask = (torch.arange(t) % 2 == 1).npu() if use_mask else None
    args = (
        (static_x, static_key, static_value, static_weight, static_mask)
        if use_mask
        else (static_x, static_key, static_value, static_weight, None)
    )

    eager_model = _Model().npu()
    eager_model(*args)
    torch.npu.synchronize()

    for tensor in args:
        if tensor is not None:
            torch._dynamo.mark_static(tensor)

    config = CompilerConfig()
    config.mode = "npugraph_ex"
    backend = torchair.get_npu_backend(compiler_config=config)
    torch._dynamo.reset()
    compiled_model = torch.compile(
        eager_model, fullgraph=True, backend=backend, dynamic=False
    )

    for replay_idx in range(2):
        x, key, value, weight = _make_inputs(t, hc, dim, 20260927 + replay_idx)
        static_x.copy_(x.npu())
        static_key.copy_(key.npu())
        static_value.copy_(value.npu())
        static_weight.copy_(weight.npu())
        mask = None
        if use_mask:
            mask = (torch.arange(t) % 2 == (replay_idx % 2)).cpu()
            static_mask.copy_(mask.npu())
        torch.npu.synchronize()

        result = compiled_model(*args)
        torch.npu.synchronize()

        expected = _golden(x, key, value, weight, 1e-6, 1e-6, mask)
        assert torch.allclose(
            result.cpu().float(), expected.float(), atol=2e-2, rtol=2e-2
        ), (
            f"replay {replay_idx}: max error {(result.cpu().float() - expected.float()).abs().max().item()}"
        )
        if mask is not None:
            assert torch.equal(
                result.cpu().view(t, hc, dim)[mask.view(-1)],
                x.view(t, hc, dim)[mask.view(-1)],
            )
