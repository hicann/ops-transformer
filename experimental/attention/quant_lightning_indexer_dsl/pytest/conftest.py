# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


import gc
from pathlib import Path
import sys

import pytest
import torch

TEST_ROOT = Path(__file__).resolve().parent
for directory in (TEST_ROOT, TEST_ROOT.parents[1]):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from indexer_test_runtime import configure, report_sources

configure()


@pytest.fixture(scope="session", autouse=True)
def report_indexer_sources():
    report_sources(
        "qli" if TEST_ROOT.parent.name == "quant_lightning_indexer_dsl" else "qsli"
    )


@pytest.fixture(autouse=True)
def release_case_memory():
    yield
    gc.collect()
    if hasattr(torch, "npu") and torch.npu.is_initialized():
        torch.npu.empty_cache()
