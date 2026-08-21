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

"""CPU Golden adapter for SparseFlashMla TTK cases."""

import importlib.util
import sys
from pathlib import Path


OPERATOR = "sparse_flash_mla"
PYTEST_MODULE_NAME = "smla_pytest_golden"
PYTEST_MODULE_FILE = "sparse_flash_mla_golden.py"


class CaseDataStore:
    """Share one input-to-Golden handoff without retaining completed cases."""

    def __init__(self):
        self.case_data = {}

    def put(self, testcase_name, data):
        if testcase_name is not None:
            self.case_data[str(testcase_name)] = data

    def get(self, testcase_name):
        if testcase_name is None:
            return None
        return self.case_data.get(str(testcase_name))

    def discard(self, data):
        """Drop every entry referring to a materialized case object."""
        for testcase_name, stored in tuple(self.case_data.items()):
            if stored is data:
                self.case_data.pop(testcase_name, None)


CASE_DATA = CaseDataStore()


def load_pytest_golden():
    """Load the pytest CPU reference only when the Golden stage needs it."""
    if PYTEST_MODULE_NAME in sys.modules:
        return sys.modules[PYTEST_MODULE_NAME]
    pytest_dir = Path(__file__).resolve().parents[2] / "pytest"
    path = pytest_dir / PYTEST_MODULE_FILE
    inserted = str(pytest_dir) not in sys.path
    if inserted:
        sys.path.insert(0, str(pytest_dir))
    try:
        spec = importlib.util.spec_from_file_location(PYTEST_MODULE_NAME, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot create import spec for {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[PYTEST_MODULE_NAME] = module
        spec.loader.exec_module(module)
        return module
    except Exception as exc:
        sys.modules.pop(PYTEST_MODULE_NAME, None)
        raise RuntimeError(
            "Failed to load SparseFlashMla pytest Golden module; "
            f"module={path.resolve()}; original error: {type(exc).__name__}: {exc}"
        ) from exc
    finally:
        if inserted:
            sys.path.remove(str(pytest_dir))


def get_case_data(testcase_name):
    return CASE_DATA.get(testcase_name)


def materialize_golden(data):
    if data.get("cpu_output") is None:
        load_pytest_golden().generate_cpu_golden(data)
    CASE_DATA.discard(data)
    return data


def activate_case_data(testcase_name):
    data = CASE_DATA.get(testcase_name)
    if data is None:
        raise RuntimeError(
            "SparseFlashMla Golden requires pytest data from the input stage"
        )
    return materialize_golden(data)


def cpu_sparse_flash_mla(
    q,
    *,
    return_softmax_lse=False,
    testcase_name=None,
    **kwargs,
):
    del q, kwargs
    data = activate_case_data(testcase_name)
    lse = data.get("softmax_lse") if bool(return_softmax_lse) else None
    return data["cpu_output"], lse


def cpu_aclnn_sparse_flash_mla(
    q,
    ori_kv,
    cmp_kv,
    ori_sparse_indices,
    cmp_sparse_indices,
    ori_block_table,
    cmp_block_table,
    cu_seqlens_q,
    cu_seqlens_ori_kv,
    cu_seqlens_cmp_kv,
    seqused_q,
    seqused_ori_kv,
    seqused_cmp_kv,
    cmp_residual_kv,
    ori_topk_length,
    cmp_topk_length,
    sinks,
    metadata,
    softmax_scale,
    cmp_ratio,
    ori_mask_mode,
    cmp_mask_mode,
    ori_win_left,
    ori_win_right,
    layout_q,
    layout_kv,
    topk_value_mode,
    return_softmax_lse,
    attn_out,
    softmax_lse_out,
    testcase_name=None,
    **kwargs,
):
    """Return the pytest Golden for the ACLNN C API parameter order."""
    del (
        ori_kv,
        cmp_kv,
        ori_sparse_indices,
        cmp_sparse_indices,
        ori_block_table,
        cmp_block_table,
        cu_seqlens_q,
        cu_seqlens_ori_kv,
        cu_seqlens_cmp_kv,
        seqused_q,
        seqused_ori_kv,
        seqused_cmp_kv,
        cmp_residual_kv,
        ori_topk_length,
        cmp_topk_length,
        sinks,
        metadata,
        softmax_scale,
        cmp_ratio,
        ori_mask_mode,
        cmp_mask_mode,
        ori_win_left,
        ori_win_right,
        layout_kv,
        topk_value_mode,
        attn_out,
        softmax_lse_out,
    )
    return cpu_sparse_flash_mla(
        q,
        return_softmax_lse=return_softmax_lse,
        testcase_name=testcase_name,
        layout_q=layout_q,
        **kwargs,
    )
