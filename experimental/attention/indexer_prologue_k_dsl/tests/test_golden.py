# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""CPU-only output layout tests for indexer_prologue_k."""

import pytest
import torch

from golden import indexer_prologue_k_golden, make_inputs


@pytest.mark.parametrize(
    "storage_mode,group,with_scale",
    [(0, -1, True), (0, -1, False), (1, 4, False), (1, 8, False)],
)
def test_golden_cache_contract(storage_mode, group, with_scale):
    inputs = make_inputs(
        17,
        64,
        96,
        70,
        storage_mode=storage_mode,
        combined_block_size=group,
        with_scale_cache=with_scale,
        seed=7,
    )
    before_cache = inputs["k_cache"].clone()
    before_scale = (
        None if inputs["k_scale_cache"] is None else inputs["k_scale_cache"].clone()
    )
    cache, scale = indexer_prologue_k_golden(
        inputs,
        storage_mode=storage_mode,
        norm_eps=1e-6,
        combined_block_size=group,
    )
    assert cache.shape == before_cache.shape
    assert cache.dtype == torch.uint8
    assert not torch.equal(cache, before_cache)
    assert torch.equal(inputs["k_cache"], before_cache)
    if before_scale is None:
        assert scale is None
    else:
        assert scale.shape == before_scale.shape
        assert not torch.equal(scale, before_scale)
        assert torch.equal(inputs["k_scale_cache"], before_scale)


def test_skipped_slot_does_not_write():
    inputs = make_inputs(2, 32, 64, 32, seed=11)
    skipped = inputs["cache_index"][-1].item()
    assert skipped == -1
    cache, _ = indexer_prologue_k_golden(inputs, storage_mode=0, norm_eps=1e-6)
    assert cache.shape == inputs["k_cache"].shape
