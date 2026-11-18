# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Review regressions for state partitioning and unchanged packed record layouts."""

import importlib

import pytest
import torch


def _kernel_module():
    return importlib.import_module(
        "cann_ops_transformer.ops.attention.flash_gdn.op_dsl.flash_gdn_kernel"
    )


@pytest.mark.parametrize("nchains", [1, 2, 5])
def test_state_work_items_cover_each_value_row_once(nchains):
    kernel = _kernel_module()
    assert kernel.HEAD_DIM % kernel.STATE_ROWS_PER_WORK == 0
    actual = []
    for work in range(nchains * kernel.STATE_WORK_ITEMS):
        chain = work // kernel.STATE_WORK_ITEMS
        rowbase = (work % kernel.STATE_WORK_ITEMS) * kernel.STATE_ROWS_PER_WORK
        actual.extend(
            (chain, row) for row in range(rowbase, rowbase + kernel.STATE_ROWS_PER_WORK)
        )
    expected = [
        (chain, row) for chain in range(nchains) for row in range(kernel.HEAD_DIM)
    ]
    assert actual == expected


def test_named_strides_preserve_host_records_and_metadata_lengths(monkeypatch):
    kernel = _kernel_module()
    metadata = importlib.import_module(
        "cann_ops_transformer.ops.attention.flash_gdn.op_dsl.flash_gdn_metadata"
    )
    calls = []
    monkeypatch.setattr(kernel, "_compiled", lambda: lambda *args: calls.append(args))
    q = torch.zeros((2, 65, 2, kernel.HEAD_DIM), dtype=torch.bfloat16)
    g = torch.full((2, 65, 2), -0.5)
    beta = torch.full_like(g, 0.25)
    A_log = torch.tensor([1.0, 2.0])
    dt_bias = torch.tensor([3.0, 4.0])
    header = metadata.make_header(2, 2, 2, 65, kernel.HEAD_DIM, kernel.HEAD_DIM, "BSND")
    assert len(header) == kernel.HEADER_SIZE == metadata.HEADER_SIZE
    out, final_state, chunk_states = kernel.run_device(
        q,
        q,
        q,
        g,
        beta,
        None,
        1.0,
        "BSND",
        header + [65, 3],
        A_log,
        dt_bias,
        False,
        True,
        False,
        False,
        False,
        True,
    )
    assert len(calls) == 1
    args = calls[0]
    params = args[3].reshape(-1, kernel.TOKEN_PARAM_STRIDE)
    torch.testing.assert_close(params[:, 0], g.flatten())
    torch.testing.assert_close(params[:, 1], beta.flatten())
    torch.testing.assert_close(params[:, 2], A_log.repeat(2 * 65))
    torch.testing.assert_close(params[:, 3], dt_bias.repeat(2 * 65))
    assert torch.count_nonzero(params[:, kernel.TOKEN_PARAM_FIELDS :]).item() == 0
    tasks = args[4].reshape(-1, kernel.TASK_RECORD_STRIDE)
    chains = args[5].reshape(-1, kernel.CHAIN_RECORD_STRIDE)
    assert tasks[:, :3].tolist() == [
        [0, 64, 0],
        [64, 1, 0],
        [0, 64, 1],
        [64, 1, 1],
        [65, 3, 0],
        [65, 3, 1],
    ]
    assert chains[:, :3].tolist() == [[0, 2, 0], [2, 2, 1], [4, 1, 0], [5, 1, 1]]
    assert torch.count_nonzero(tasks[:, 3:]).item() == 0
    assert torch.count_nonzero(chains[:, 3:]).item() == 0
    assert (
        args[11].numel() == len(tasks) * kernel.CHUNK_SIZE * kernel.TOKEN_CACHE_STRIDE
    )
    assert args[-1] == min(kernel.MAX_LAUNCH_BLOCKS, len(tasks))
    assert out.shape == q.shape
    assert final_state.shape == (2, 2, kernel.HEAD_DIM, kernel.HEAD_DIM)
    assert chunk_states is None
