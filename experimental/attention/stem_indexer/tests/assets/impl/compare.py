#!/usr/bin/python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Compare helpers for stem_indexer TestSpec adapters."""

import numpy as np

MAX_FAILURES = 8
FAIL_RATIO_LIMIT = 0.001


def as_numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu()
        value = value.numpy()
    return np.asarray(value)


def compare(*outputs, **kwargs):
    """Compare NPU outputs against golden references.

    Expects [npu_sparse_indices, npu_sparse_seq_len, golden_sparse_indices, golden_sparse_seq_len].
    """
    if len(outputs) != 4:
        return {
            "pass": False,
            "precision": "invalid",
            "error_info": f"compare expects 4 outputs, got {len(outputs)}",
        }

    npu_indices = as_numpy(outputs[0])
    npu_seq_len = as_numpy(outputs[1])
    golden_indices = as_numpy(outputs[2])
    golden_seq_len = as_numpy(outputs[3])

    if npu_indices.shape != golden_indices.shape:
        return {
            "pass": False,
            "precision": "shape_mismatch",
            "error_info": f"sparse_indices shape mismatch: npu={npu_indices.shape}, golden={golden_indices.shape}",
        }

    if npu_seq_len.shape != golden_seq_len.shape:
        return {
            "pass": False,
            "precision": "shape_mismatch",
            "error_info": f"sparse_seq_len shape mismatch: npu={npu_seq_len.shape}, golden={golden_seq_len.shape}",
        }

    if not np.array_equal(npu_seq_len, golden_seq_len):
        diff_count = int(np.sum(npu_seq_len != golden_seq_len))
        diff_indices = np.argwhere(npu_seq_len != golden_seq_len)[:5].tolist()
        return {
            "pass": False,
            "precision": 0.0,
            "error_info": f"sparse_seq_len mismatch: {diff_count} elements differ, first={diff_indices}",
        }

    failures = []
    nonzero_mask = golden_seq_len > 0
    total_rows = 0
    for index in np.argwhere(nonzero_mask):
        b_idx, q_head_idx, q_block_idx = int(index[0]), int(index[1]), int(index[2])
        valid_len = int(golden_seq_len[b_idx, q_head_idx, q_block_idx])
        if valid_len == 0:
            continue
        total_rows += 1
        actual_set = set(
            npu_indices[b_idx, q_head_idx, q_block_idx, :valid_len].tolist()
        )
        expected_set = set(
            golden_indices[b_idx, q_head_idx, q_block_idx, :valid_len].tolist()
        )
        actual_valid = {x for x in actual_set if x >= 0}
        expected_valid = {x for x in expected_set if x >= 0}
        if actual_valid != expected_valid:
            only_actual = sorted(actual_valid - expected_valid)[:10]
            only_expected = sorted(expected_valid - actual_valid)[:10]
            failures.append(
                {
                    "b": b_idx,
                    "q_head": q_head_idx,
                    "q_block": q_block_idx,
                    "valid_len": valid_len,
                    "only_in_actual": only_actual,
                    "only_in_expected": only_expected,
                }
            )
            if len(failures) >= MAX_FAILURES:
                break

    if failures:
        fail_ratio = len(failures) / max(total_rows, 1)
        passed = fail_ratio <= FAIL_RATIO_LIMIT
        return {
            "pass": passed,
            "precision": (1 - fail_ratio) * 100,
            "error_info": f"sparse_indices set mismatch: {len(failures)}/{total_rows} rows differ (ratio={fail_ratio:.6g}), first={failures[0]}",
        }

    return {
        "pass": True,
        "precision": 100.0,
        "error_info": None,
    }
