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

import torch


class CaseDataStore:
    """Share one pytest-generated case between input, golden, and compare callbacks."""

    def __init__(self):
        self.case_data = {}
        self.active_testcase_name = None

    def clear(self):
        self.case_data.clear()
        self.active_testcase_name = None

    def put(self, testcase_name, data):
        if testcase_name is not None:
            self.clear()
            self.case_data = {str(testcase_name): data}

    def activate(self, testcase_name):
        data = (
            self.case_data.get(str(testcase_name))
            if testcase_name is not None
            else None
        )
        if data is None:
            raise RuntimeError(
                "LightningIndexer TTK golden requires customize_inputs "
                "to generate pytest data first"
            )
        self.active_testcase_name = str(testcase_name)
        return data


CASE_DATA = CaseDataStore()


def get_compare_data(testcase_name):
    """Return pytest comparison context for the active or replayed case."""
    if testcase_name is None:
        return None
    return CASE_DATA.case_data.get(str(testcase_name))


def set_compare_data(testcase_name, data):
    CASE_DATA.active_testcase_name = str(testcase_name)
    CASE_DATA.case_data = {str(testcase_name): data}


__golden__ = {
    "e2e": {
        "torch_npu.npu_lightning_indexer": "cpu_lightning_indexer",
    },
    "aclnn": {
        "aclnnLightningIndexer": "cpu_lightning_indexer_aclnn",
    },
}


def cpu_lightning_indexer(
    query,
    key,
    weights,
    *,
    actual_seq_lengths_query=None,
    actual_seq_lengths_key=None,
    block_table=None,
    layout_query="BSND",
    layout_key="BSND",
    sparse_count=2048,
    sparse_mode=3,
    pre_tokens=(1 << 63) - 1,
    next_tokens=(1 << 63) - 1,
    return_value=False,
    testcase_name=None,
    **kwargs,
):
    """Return the CPU outputs produced while generating this exact pytest case."""
    del query, key, weights, actual_seq_lengths_query, actual_seq_lengths_key
    del block_table, layout_query, layout_key, sparse_count, sparse_mode
    del pre_tokens, next_tokens
    data = CASE_DATA.activate(testcase_name)
    return_value = bool(return_value or kwargs.get("return_values", False))
    if return_value:
        indices = data["cpu_result"]
        gather_idx = indices.clamp_min(0).to(torch.long)
        values = torch.gather(data["score_values"], -1, gather_idx)
        values = values.to(data["value_dtype"])
        values = torch.where(
            indices < 0, torch.full_like(values, -float("inf")), values
        )
        return data["cpu_result"], values
    return data["cpu_result"], torch.zeros(0, dtype=data["score_values"].dtype)


def cpu_lightning_indexer_aclnn(
    query,
    key,
    weights,
    actual_seq_lengths_query,
    actual_seq_lengths_key,
    block_table,
    layout_query,
    layout_key,
    sparse_count,
    sparse_mode,
    pre_tokens,
    next_tokens,
    return_value,
    sparse_indices_out,
    sparse_values_out,
    testcase_name=None,
    **kwargs,
):
    """Full aclnn param order. Return the CPU outputs produced while generating this exact pytest case."""
    del query, key, weights, actual_seq_lengths_query, actual_seq_lengths_key
    del block_table, layout_query, layout_key, sparse_count, sparse_mode
    del pre_tokens, next_tokens, sparse_indices_out, sparse_values_out
    data = CASE_DATA.activate(testcase_name)
    return_value = bool(return_value or kwargs.get("return_values", False))
    if return_value:
        indices = data["cpu_result"]
        gather_idx = indices.clamp_min(0).to(torch.long)
        values = torch.gather(data["score_values"], -1, gather_idx)
        values = values.to(data["value_dtype"])
        values = torch.where(
            indices < 0, torch.full_like(values, -float("inf")), values
        )
        return data["cpu_result"], values
    return data["cpu_result"], torch.zeros(0, dtype=data["score_values"].dtype)
