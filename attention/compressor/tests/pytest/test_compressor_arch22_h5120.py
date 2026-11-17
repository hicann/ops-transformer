# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import pytest
import torch

from test_compressor_arch22 import _run_ring_sequence


@pytest.mark.ci
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("layout_th", [False, True])
@pytest.mark.parametrize("ratio", [32, 64, 128])
def test_arch22_h5120_d512_snapshot_chunks(dtype, layout_th, ratio):
    _run_ring_sequence(
        dtype,
        layout_th,
        [2 * ratio + 1],
        head_dim=512,
        hidden=5120,
        capacity=ratio + 3,
        ratio=ratio,
        starts=[2 * ratio - 1, 3 * ratio - 1],
        state_padding=1,
    )


@pytest.mark.ci
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("layout_th", [False, True])
def test_arch22_h5120_d512_snapshot_task_reuse(dtype, layout_th):
    _run_ring_sequence(
        dtype,
        layout_th,
        [257],
        head_dim=512,
        hidden=5120,
        capacity=131,
        ratio=128,
        starts=[255, 383, 511],
        state_padding=1,
    )


@pytest.mark.ci
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("layout_th", [False, True])
@pytest.mark.parametrize(
    "settings",
    [
        pytest.param({"starts": [255]}, id="single"),
        pytest.param({"starts": [255, 382, 510, 638]}, id="one-conflict"),
        pytest.param(
            {"starts": [254, 382, 511, 638, 766, 895, 1022, 1150]},
            id="sparse-conflicts",
        ),
        pytest.param({"starts": [255, 383, 511, 639]}, id="all-conflicts"),
        pytest.param({"starts": [254, 382, 510, 638]}, id="no-conflict"),
        pytest.param(
            {"starts": [255, 382, 384, 638], "used": [130, 130, 0, 1]},
            id="unused-and-no-output",
        ),
        pytest.param(
            {"starts": [128 * (batch + 2) - 1 for batch in range(8)]},
            id="many-conflicts",
        ),
    ],
)
def test_arch22_h5120_d512_selective_snapshot(dtype, layout_th, settings):
    _run_ring_sequence(
        dtype,
        layout_th,
        [130],
        head_dim=512,
        hidden=5120,
        capacity=256,
        ratio=128,
        state_padding=1,
        **settings,
    )


@pytest.mark.ci
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("head_dim", [128, 512])
@pytest.mark.parametrize(
    "layout_th,length",
    [
        (False, 1),
        (False, 1003),
        (False, 8192),
        (False, 10240),
        (True, 1),
        (True, 1003),
        (True, 8192),
        (True, 10240),
    ],
)
def test_arch22_h5120(dtype, layout_th, head_dim, length):
    capacity = 1024
    _run_ring_sequence(
        dtype,
        layout_th,
        [length, 1, 1, 1, 1],
        head_dim=head_dim,
        hidden=5120,
        capacity=capacity,
        starts=[capacity - 3, 2 * capacity - 2],
    )


@pytest.mark.ci
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "layout_th,length",
    [
        (False, 1),
        (False, 1003),
        (False, 8192),
        (False, 10240),
        (True, 1),
        (True, 1003),
        (True, 8192),
        (True, 10240),
    ],
)
def test_arch22_h5120_d512(dtype, layout_th, length):
    capacity = 1024
    _run_ring_sequence(
        dtype,
        layout_th,
        [length, 1, 1, 1, 1],
        head_dim=512,
        hidden=5120,
        capacity=capacity,
        starts=[capacity - 3, 2 * capacity - 2],
    )


@pytest.mark.ci
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("layout_th", [False, True])
@pytest.mark.parametrize(
    "lengths,ratio,capacity,starts,used,state_padding",
    [
        pytest.param([0], 4, 8, [0, 7], None, 0, id="empty-contiguous"),
        pytest.param([3], 4, 8, [3, 7], None, 1, id="before-compression-group"),
        pytest.param([4], 4, 8, [7, 15], None, 1, id="exact-compression-group"),
        pytest.param([5], 4, 8, [6, 14], None, 1, id="after-compression-group"),
        pytest.param([9], 4, 8, [7, 15], None, 1, id="wrap-after-capacity"),
        pytest.param([33], 4, 8, [7, 15], None, 1, id="multiple-ring-wraps"),
        pytest.param([3], 2, 4, [3, 7, 11], None, 1, id="ratio-2-three-batches"),
        pytest.param([8], 8, 16, [15, 31], None, 1, id="ratio-8-exact-boundary"),
        pytest.param([33], 8, 16, [15, 31], None, 1, id="ratio-8-long-wrap"),
        pytest.param([16], 16, 32, [31, 63], None, 1, id="ratio-16-exact-boundary"),
        pytest.param([65], 16, 32, [31, 63], None, 1, id="ratio-16-multiple-wraps"),
        pytest.param([3, 1, 1, 3], 4, 8, [7, 15], None, 1, id="group-spans-calls"),
        pytest.param([8], 4, 8, [7, 15], [0, 5], 1, id="partial-seqused-batch"),
        pytest.param([9], 4, 8, [0, 7, 15], None, 0, id="three-batches-contiguous"),
    ],
)
def test_arch22_h5120_d512_generalized(
    dtype, layout_th, lengths, ratio, capacity, starts, used, state_padding
):
    _run_ring_sequence(
        dtype,
        layout_th,
        lengths,
        head_dim=512,
        hidden=5120,
        capacity=capacity,
        ratio=ratio,
        starts=starts,
        used=used,
        state_padding=state_padding,
    )


@pytest.mark.ci
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "length,th_lengths,ratio,capacity,starts",
    [
        pytest.param(8, [0, 4, 8], 4, 8, [7, 15, 23], id="ragged-empty-and-full"),
        pytest.param(8, [1, 5, 7], 4, 8, [0, 3, 7], id="ragged-short-prefixes"),
        pytest.param(9, [2, 6, 9], 4, 8, [7, 15, 23], id="ragged-wrap-boundary"),
        pytest.param(16, [0, 8, 15], 8, 16, [15, 31, 47], id="ragged-ratio-8-wrap"),
    ],
)
def test_arch22_h5120_d512_ragged_th(
    dtype, length, th_lengths, ratio, capacity, starts
):
    _run_ring_sequence(
        dtype,
        True,
        [length],
        head_dim=512,
        hidden=5120,
        capacity=capacity,
        ratio=ratio,
        starts=starts,
        th_lengths=th_lengths,
    )


@pytest.mark.ci
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("layout_th", [False, True])
@pytest.mark.parametrize("mixed", [False, True])
def test_arch22_h5120_d512_parallel_batch_owners(dtype, layout_th, mixed):
    batch_size = 65
    starts = [
        0 if mixed and index % 3 == 0 else 7 + 8 * index for index in range(batch_size)
    ]
    used = [0 if mixed and index % 3 == 2 else 9 for index in range(batch_size)]
    _run_ring_sequence(
        dtype,
        layout_th,
        [9],
        head_dim=512,
        hidden=5120,
        capacity=8,
        ratio=4,
        starts=starts,
        used=used,
        state_padding=1,
    )
