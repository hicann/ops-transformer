# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Installed torch dispatcher vs CPU golden on Ascend 950."""

import pytest
import torch

from golden import indexer_prologue_k_golden, make_inputs

torch_npu = pytest.importorskip("torch_npu", reason="torch_npu not installed")
if not torch.npu.is_available():  # pragma: no cover - host dependent
    pytest.skip("no NPU visible", allow_module_level=True)


@pytest.mark.parametrize(
    "storage_mode,group,with_scale",
    [(0, -1, True), (0, -1, False), (1, 4, False), (1, 8, False)],
)
def test_indexer_prologue_k_matches_golden(storage_mode, group, with_scale):
    from cann_ops_transformer.ops.ds41 import indexer_prologue_k

    inputs = make_inputs(
        72,
        128,
        128,
        64,
        storage_mode=storage_mode,
        combined_block_size=group,
        with_scale_cache=with_scale,
        seed=storage_mode * 10 + max(group, 0),
    )
    expected_cache, expected_scale = indexer_prologue_k_golden(
        inputs,
        storage_mode=storage_mode,
        norm_eps=1e-6,
        combined_block_size=group,
    )
    wk_nz = torch_npu.npu_format_cast(inputs["wk"].npu(), torch_npu.Format.FRACTAL_NZ)
    cache_npu = inputs["k_cache"].npu()
    scale_npu = None
    if inputs["k_scale_cache"] is not None:
        scale_npu = inputs["k_scale_cache"].npu()
    result = indexer_prologue_k(
        inputs["latent"].npu(),
        wk_nz,
        inputs["norm_weight"].npu(),
        inputs["rope_sin"].npu(),
        inputs["rope_cos"].npu(),
        cache_npu,
        scale_npu,
        cache_index=inputs["cache_index"].npu(),
        storage_mode=storage_mode,
        norm_eps=1e-6,
        combined_block_size=group,
    )
    torch.npu.synchronize()
    assert result is cache_npu
    assert torch.equal(result.cpu(), expected_cache)
    if scale_npu is not None:
        assert torch.equal(scale_npu.cpu(), expected_scale)
