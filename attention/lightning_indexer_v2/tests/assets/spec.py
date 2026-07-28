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

"""TestSpec adapter for LightningIndexer V2 TTK assets."""

import importlib.util
import sys
from pathlib import Path

ASSET_IMPL_DIR = Path(__file__).with_name("impl")


def load_impl_module(stem):
    name = f"li_v2_ttk_{stem}"
    if name in sys.modules:
        return sys.modules[name]
    path = ASSET_IMPL_DIR / f"{stem}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


golden_module = load_impl_module("golden")
inputs_module = load_impl_module("inputs")
compare_module = load_impl_module("compare")
graph_module = load_impl_module("graph")


class LightningIndexerV2Spec:
    golden = golden_module.cpu_lightning_indexer_v2
    customize_inputs = inputs_module.generate_li_v2_inputs
    tolerance = {
        "float16": {"standard": "stat_rel_err"},
        "bfloat16": {"standard": "stat_rel_err"},
    }

    def compare(*outputs, **kwargs):
        score_layout, cu_seqlens_q = golden_module.get_score_context()
        return compare_module.compare(
            *outputs,
            scores=golden_module.get_topk_scores(),
            index_offsets=golden_module.get_index_offsets(),
            score_layout=score_layout,
            cu_seqlens_q=cu_seqlens_q,
        )

    torch_graph = graph_module.LightningIndexerV2AclGraph


__spec__ = {
    "li_v2_ttk_ops.lightning_indexer_v2": "LightningIndexerV2Spec",
}
