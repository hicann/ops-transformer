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

"""TestSpec adapter for MatmulAllReduceAddRmsNorm TTK assets (multi-device ACLNN/E2E)."""

import importlib.util
from pathlib import Path

ASSET_IMPL_DIR = Path(__file__).with_name("impl")

_impl_cache = {}


def _load_impl_module(stem):
    if stem not in _impl_cache:
        path = ASSET_IMPL_DIR / f"{stem}.py"
        spec = importlib.util.spec_from_file_location(
            f"matmul_all_reduce_add_rms_norm_assets_impl_{stem}_{abs(hash(path))}", path
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _impl_cache[stem] = module
    return _impl_cache[stem]


class AclnnMatmulAllReduceAddRmsNormSpec:
    """TestSpec for aclnnMatmulAllReduceAddRmsNorm (ACLNN multi-device path)."""

    @staticmethod
    def golden(thread_contexts, device_ids, all_precision):
        return _load_impl_module(
            "golden"
        ).matmul_all_reduce_add_rms_norm_multi_device_golden(
            thread_contexts, device_ids, all_precision
        )


class MatmulAllReduceAddRmsNormE2ESpec:
    """TestSpec for torch_npu.npu_mm_all_reduce_add_rms_norm (E2E multi-device path)."""

    @staticmethod
    def golden(cpu_inputs_per_rank, attrs, rank=None, ws=None, dist_avail=True):
        return _load_impl_module("golden").matmul_all_reduce_add_rms_norm_e2e_golden(
            cpu_inputs_per_rank, attrs, rank, ws, dist_avail
        )


__spec__ = {
    "aclnnMatmulAllReduceAddRmsNorm": "AclnnMatmulAllReduceAddRmsNormSpec",
    "torch_npu.npu_mm_all_reduce_add_rms_norm": "MatmulAllReduceAddRmsNormE2ESpec",
}
