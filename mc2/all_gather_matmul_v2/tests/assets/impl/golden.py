#!/usr/bin/python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Golden for aclnnAllGatherMatmulV2.

AllGatherMatmulV2 extends V1 with quant (fp8/mxfp/per_block) support.
Delegates to the shared all_gather golden logic which handles V1+V2 uniformly.
"""

import importlib.util
from pathlib import Path


def _load_shared_utils():
    """Load golden_utils from mc2/common/tests/assets."""
    utils_path = (
        Path(__file__).resolve().parents[4]
        / "common"
        / "tests"
        / "assets"
        / "golden_utils.py"
    )
    spec = importlib.util.spec_from_file_location(
        f"mc2_common_golden_utils_{abs(hash(utils_path))}", utils_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def all_gather_matmul_v2_multi_device_golden(
    thread_contexts, device_ids, all_precision
):
    """Use shared golden_all_gather_compare (handles V1+V2 uniformly)."""
    utils = _load_shared_utils()
    world_size = len(device_ids)
    utils.golden_all_gather_compare(
        thread_contexts, device_ids, all_precision, world_size
    )


def all_gather_matmul_v2_e2e_golden(
    cpu_inputs_per_rank, attrs, rank=None, ws=None, dist_avail=True
):
    import torch

    world_size = len(cpu_inputs_per_rank)
    all_x1 = [
        inp[0].float() if len(inp) > 0 and inp[0] is not None else None
        for inp in cpu_inputs_per_rank
    ]
    gathered = torch.cat([x for x in all_x1 if x is not None], dim=0)
    del all_x1
    goldens = []
    for r in range(world_size):
        inp = cpu_inputs_per_rank[r]
        x2 = inp[1].float() if len(inp) > 1 and inp[1] is not None else None
        if x2 is None:
            goldens.append(None)
            continue
        g = torch.matmul(gathered, x2)
        bias = inp[2] if len(inp) > 2 and inp[2] is not None else None
        if bias is not None and hasattr(bias, "numel") and bias.numel() > 0:
            g = g + bias.float()
        goldens.append(g.contiguous())
    del gathered
    return goldens
