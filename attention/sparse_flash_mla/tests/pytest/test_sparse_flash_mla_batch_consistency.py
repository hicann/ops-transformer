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

"""Opt-in SMLA deterministic-level-3 batch-consistency tests."""

from pathlib import Path

import pytest
import torch_npu

from batch_consistency.model import RunResult  # noqa: E402
from batch_consistency.pytest_support import (  # noqa: E402
    collect_case_matrix,
    collect_independent_relation_matrix,
    load_operator_module,
    run_independent_pytest_case,
    run_pytest_case,
)


PYTEST_PATH = Path(__file__).resolve().parent
result_compare_method = load_operator_module(
    "smla_consistency_result_compare",
    PYTEST_PATH / "result_compare_method.py",
)
sparse_flash_mla_process = load_operator_module(
    "smla_consistency_process",
    PYTEST_PATH / "batch/sparse_flash_mla_process.py",
)


class SparseFlashMlaExecutor:
    """Reuse the ordinary eager pytest call without changing its contract."""

    def __call__(self, data):
        output, softmax_lse = sparse_flash_mla_process.call_npu(data)
        return RunResult(output, softmax_lse)


TEST_MATRIX = collect_case_matrix("SMLA", "./data")
INDEPENDENT_RELATIONS = collect_independent_relation_matrix("SMLA", "smla")

pytestmark = pytest.mark.consistency


@pytest.mark.parametrize("case_path,mode", TEST_MATRIX)
def test_sparse_flash_mla_batch_consistency(case_path, mode):
    run_pytest_case(
        case_path,
        mode,
        "SMLA",
        "smla",
        SparseFlashMlaExecutor(),
        result_compare_method.check_result,
        lambda: torch_npu.npu.set_device(0),
    )


@pytest.mark.parametrize(
    "relation", INDEPENDENT_RELATIONS, ids=lambda value: value["id"]
)
def test_sparse_flash_mla_independent_batch_consistency(relation):
    run_independent_pytest_case(
        relation,
        "smla",
        SparseFlashMlaExecutor(),
        result_compare_method.check_result,
        lambda: torch_npu.npu.set_device(0),
    )
