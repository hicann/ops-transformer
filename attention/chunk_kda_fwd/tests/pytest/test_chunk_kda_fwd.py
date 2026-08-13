# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import pytest
import torch

from common import (
    assert_binary_equal,
    assert_close,
    make_inputs,
    run,
    set_device,
    synchronize,
)


@pytest.fixture(scope="module", autouse=True)
def npu_device():
    set_device()


def test_layouts_and_bf16_output_determinism():
    inputs = make_inputs(tokens=128)
    reference = run(inputs, layout="BSND")
    repeated = run(inputs, layout="BSND")
    synchronize()
    for index in (0, 3, 4):
        assert torch.equal(reference[index], repeated[index])
    assert_close(reference[1], repeated[1])

    for layout in ("BNSD", "TND", "NTD"):
        actual = run(inputs, layout=layout)
        synchronize()
        for index in (0, 1, 3, 4):
            expected = reference[index]
            if index in (0, 3, 4) and layout in ("TND", "NTD"):
                expected = expected.squeeze(0)
            assert_close(actual[index], expected)


@pytest.mark.parametrize(
    "options,present",
    [
        ({}, {0, 1, 3, 4}),
        ({"return_intermediate_states": True}, {0, 1, 3, 4, 10}),
        ({"disable_recompute": True}, set(range(11))),
    ],
)
def test_optional_output_matrix(options, present):
    outputs = run(make_inputs(tokens=64), **options)
    synchronize()
    for index, output in enumerate(outputs):
        if index == 11:
            assert output is None
        elif index in present:
            assert output is not None, f"output {index} must be materialized"
            assert torch.isfinite(output).all().item()
        else:
            assert output is None, f"output {index} must not be materialized"


def test_bf16_gate_parameters_match_promoted_fp32():
    inputs = make_inputs(tokens=64)
    inputs.a_log = inputs.a_log.to(torch.bfloat16)
    inputs.dt_bias = inputs.dt_bias.to(torch.bfloat16)
    bf16_outputs = run(inputs)

    inputs.a_log = inputs.a_log.float()
    inputs.dt_bias = inputs.dt_bias.float()
    fp32_outputs = run(inputs)
    synchronize()
    for index in (0, 3, 4):
        assert torch.equal(bf16_outputs[index], fp32_outputs[index])
    assert_close(bf16_outputs[1], fp32_outputs[1])


def test_varlen_tail_with_host_cu_seqlens():
    inputs = make_inputs(tokens=65)
    outputs = run(
        inputs,
        layout="NTD",
        cu_seqlens=torch.tensor([0, 64, 65], dtype=torch.int64, device="npu"),
        cu_seqlens_cpu=[0, 64, 65],
    )
    repeated = run(
        inputs,
        layout="NTD",
        cu_seqlens=torch.tensor([0, 64, 65], dtype=torch.int64, device="npu"),
        cu_seqlens_cpu=[0, 64, 65],
    )
    synchronize()
    assert_binary_equal(outputs, repeated)
    assert outputs[0].shape == (65, 2, 128)
    assert outputs[1].shape == (2, 2, 128, 128)


def test_state_v_first_layout_contract():
    inputs = make_inputs(tokens=64)
    initial = torch.randn((1, 2, 128, 128), dtype=torch.float32).npu() * 0.01
    kv_outputs = run(inputs, initial_state=initial, state_v_first=False)
    vk_outputs = run(
        inputs,
        initial_state=initial.transpose(-1, -2).contiguous(),
        state_v_first=True,
    )
    synchronize()
    assert_close(vk_outputs[0], kv_outputs[0])
    assert_close(vk_outputs[1], kv_outputs[1].transpose(-1, -2).contiguous())
