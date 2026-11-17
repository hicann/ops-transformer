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
import torch_npu

from cann_ops_transformer.ops import compressor, compressor_v2


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("coff", [1, 2])
@pytest.mark.parametrize("layout", ["BSH", "TH"])
@pytest.mark.parametrize("seq_len", [4, 16])
@pytest.mark.parametrize("ratio", [4, 128])
def test_optional_ape_matches_zero(dtype, coff, layout, seq_len, ratio):
    """Compare v2 without APE against the unchanged v1 with zero APE on arch22."""
    torch.manual_seed(20261102)
    batch, hidden, dim, block_size = 2, 1024, 128, 256
    x = (torch.randn(batch, seq_len, hidden) * 0.1).to(dtype).npu()
    if layout == "TH":
        x = x.reshape(batch * seq_len, hidden)
    wkv = (torch.randn(coff * dim, hidden) * 0.1).to(dtype).npu()
    wgate = (torch.randn(coff * dim, hidden) * 0.1).to(dtype).npu()
    # Reserve block zero (linear cache sentinel), and guard axis-0 padding.
    width = 2 * coff * dim
    storage = torch.full((batch + 1, block_size * width + 16), 17.0).npu()
    state = storage[:, : block_size * width].view(batch + 1, block_size, width)
    state.copy_(torch.randn(state.shape).npu() * 0.1)
    initial = storage.clone()
    table = torch.arange(1, batch + 1, dtype=torch.int32).reshape(batch, 1).npu()
    starts = torch.full((batch,), ratio - 1, dtype=torch.int32).npu()
    lengths = torch.full((batch,), seq_len, dtype=torch.int32).npu()
    cu = (
        torch.arange(0, (batch + 1) * seq_len, seq_len, dtype=torch.int32).npu()
        if layout == "TH"
        else None
    )
    kwargs = dict(
        state_block_table=table,
        cu_seqlens=cu,
        seqused=lengths,
        start_pos=starts,
        coff=coff,
        cache_mode=1,
    )
    zero_ape = torch.zeros((ratio, coff * dim), dtype=torch.float32).npu()
    with torch.no_grad():
        expected = compressor(x, wkv, wgate, state, zero_ape, ratio, **kwargs).cpu()
        expected_storage = storage.cpu().clone()
        for _ in range(3):
            storage.copy_(initial)
            actual = compressor_v2(x, wkv, wgate, state, ratio, **kwargs).cpu()
            valid = (ratio - 1 + seq_len) // ratio
            if layout == "BSH":
                actual, reference = actual[:, :valid], expected[:, :valid]
            else:
                actual, reference = actual[: batch * valid], expected[: batch * valid]
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)
            torch.testing.assert_close(storage.cpu(), expected_storage, rtol=0, atol=0)
            torch.testing.assert_close(
                storage[:, block_size * width :].cpu(),
                initial[:, block_size * width :].cpu(),
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(
                storage[0].cpu(), initial[0].cpu(), rtol=0, atol=0
            )
