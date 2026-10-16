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

from compressor_golden import run_compressor_eager


@pytest.mark.parametrize("layout_th", [False, True], ids=["BSH", "TH"])
@pytest.mark.parametrize("coff", [1, 2], ids=["no-overlap", "overlap"])
@pytest.mark.parametrize("cmp_ratio", [2, 4, 8, 16])
@pytest.mark.parametrize("block_size", [2, 16, 128])
@pytest.mark.parametrize("cache_mode", [1], ids=["linear-cache"])
def test_compressor_grad_enabled_single_precision(
    layout_th, coff, cmp_ratio, block_size, cache_mode
):
    batch_size = 2
    seq_len = 32
    hidden_size = 4096
    head_dim = 128
    run_compressor_eager(
        batch_size,
        seq_len,
        head_dim,
        coff,
        cmp_ratio,
        layout_th,
        S=seq_len,
        block_size=block_size,
        data_type=torch.float16,
        hidden_size=hidden_size,
        cache_mode=cache_mode,
        grad_enabled=True,
    )


@pytest.mark.parametrize("coff", [1, 2], ids=["no-overlap", "overlap"])
@pytest.mark.parametrize("cmp_ratio", [2, 4])
def test_compressor_grad_enabled_full_load(coff, cmp_ratio):
    run_compressor_eager(
        2,
        4,
        128,
        coff,
        cmp_ratio,
        False,
        S=4,
        block_size=4,
        data_type=torch.float16,
        hidden_size=4096,
        cache_mode=1,
        grad_enabled=True,
    )
