# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""TTK E2E golden entry point for grouped_matmul_finalize_routing."""

import sys
from pathlib import Path
from runpy import run_path


_SHARED = Path(__file__).resolve().parents[3] / "common/tests/ttk/fused_spec.py"
_make_spec = run_path(str(_SHARED))["make_spec"]
sys.path.insert(0, str(_SHARED.parent))
try:
    from gmm_ttk_graph_models import RoutingGraph
finally:
    sys.path.pop(0)


E2ESpec = _make_spec("routing", "e2e")
E2ESpec.torch_graph = RoutingGraph


__spec__ = {
    "torch_npu.npu_grouped_matmul_finalize_routing": "E2ESpec",
}
