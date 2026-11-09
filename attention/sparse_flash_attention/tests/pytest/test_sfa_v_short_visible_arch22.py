# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""V-template split-KV must preserve native short-visible-KV boundaries."""

import pytest
import torch

from test_sfa_ctemplate_split_kv_arch22 import LAYOUTS, make_case


def run_checked(spec, layouts, dtype, graph_mode):
    run, verify = make_case(spec, layouts, dtype)
    if graph_mode:
        run()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            out = run()
        for _ in range(3):
            graph.replay()
    else:
        out = run()
    torch.npu.synchronize()
    verify(out)


@pytest.mark.parametrize("short_kv", [1, 255, 256, 511, 512, 513, 1023, 1024, 1025])
@pytest.mark.parametrize("bs", [1, 2, 4])
@pytest.mark.parametrize("mode", [0, 3])
@pytest.mark.parametrize("layouts", LAYOUTS)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("graph_mode", [False, True])
def test_v_short_visible(short_kv, bs, mode, layouts, dtype, graph_mode):
    # Batch 0 keeps host-side storage large enough to enable split-KV. Batch 1
    # exercises a clipped last loop and whole partitions beyond visible KV.
    spec = dict(
        bs=bs,
        topk=2048,
        kind="full",
        mode=mode,
        b=2,
        q=4,
        g=64 if short_kv == 513 else 8,
        rope=short_kv % 2 != 0,
        kv_lengths=[2048 * bs + 128, short_kv],
        ordered_indices=True,
    )
    run_checked(spec, layouts, dtype, graph_mode)


@pytest.mark.parametrize("bs", [1, 2, 4])
@pytest.mark.parametrize("mode", [0, 3])
@pytest.mark.parametrize("layouts", LAYOUTS)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("graph_mode", [False, True])
def test_v_short_visible_mixed_padding(bs, mode, layouts, dtype, graph_mode):
    spec = dict(
        bs=bs,
        topk=2048,
        kind="mixed",
        mode=mode,
        b=2,
        q=4,
        g=128,
        rope=True,
        kv_lengths=[2048 * bs + 128, 513],
        ordered_indices=True,
    )
    run_checked(spec, layouts, dtype, graph_mode)


@pytest.mark.parametrize("bs", [1, 2, 4])
@pytest.mark.parametrize("layouts", LAYOUTS)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_v_short_visible_unordered(bs, layouts, dtype):
    spec = dict(
        bs=bs,
        topk=2048,
        kind="full",
        mode=0,
        b=2,
        q=4,
        g=8,
        rope=True,
        kv_lengths=[2048 * bs + 128, 1024],
    )
    run_checked(spec, layouts, dtype, True)
