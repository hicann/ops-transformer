#!/usr/bin/python3
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


"""TTK TestSpec for quant_flash_mla_with_kvcache.

Maps two ops:
  - torch.ops.cann_ops_transformer.quant_flash_mla
  - torch.ops.cann_ops_transformer.quant_flash_mla_metadata
"""

import importlib.util
import sys
from pathlib import Path

ASSET_IMPL_DIR = Path(__file__).with_name("impl")

_impl_cache = {}


def load_impl_module(stem):
    """Lazily load impl/{stem}.py under a stable synthetic name.

    Deferred import avoids importing torch_npu in the main process
    before TTK forks workers (causes NPU re-init errors).
    """
    mod_name = f"qmla_assets_impl_{stem}"
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    path = ASSET_IMPL_DIR / f"{stem}.py"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    _impl_cache[stem] = module
    return _impl_cache[stem]


def _resolve_quant_mode(args, kwargs):
    """quant_mode is positional arg[6] after op-schema direct call."""
    qm = args[6] if len(args) > 6 else kwargs.get("quant_mode", 1)
    return int(qm) if qm is not None else 1


def _golden_dispatch(*args, **kwargs):
    return load_impl_module("golden").cpu_qmla_mxfp8(*args, **kwargs)


def _inputs_dispatch(*args, **kwargs):
    return load_impl_module("inputs").generate_qmla_inputs(*args, **kwargs)


class QuantMlaSpec:
    """quant_flash_mla_with_kvcache test spec (FP8_E4M3 full-quant).

    metadata 由 npu_preprocess 在主算子调用前生成并回填 metadata slot,
    主算子直接由 ttk 调用 torch.ops.cann_ops_transformer.quant_flash_mla。

    golden/customize_inputs/compare/npu_preprocess 均懒加载 impl 模块,
    避免主进程 fork 前 import torch_npu (与 flash_attn 资产一致)。
    """

    golden = staticmethod(_golden_dispatch)
    customize_inputs = staticmethod(_inputs_dispatch)

    @staticmethod
    def compare(*outputs, **kwargs):
        return load_impl_module("compare").compare(*outputs, **kwargs)

    @staticmethod
    def npu_preprocess(*args, **kwargs):
        return load_impl_module("npu_preprocess").run(*args, **kwargs)

    torch_graph = load_impl_module("graph").QuantMlaAclGraph

    tolerance = {
        "float16": {"rtol": 0.005, "ptol": 0.005, "atol": 0.000025},
        "bfloat16": {"rtol": 0.0078125, "ptol": 0.005, "atol": 0.0001},
    }


class QuantMlaMetadataSpec:
    """quant_flash_mla_with_kvcache_metadata spec (metadata-only op)."""

    customize_inputs = staticmethod(lambda *a, **kw: None)


__spec__ = {
    "torch.ops.cann_ops_transformer.quant_flash_mla": "QuantMlaSpec",
    "torch.ops.cann_ops_transformer.quant_flash_mla_metadata": "QuantMlaMetadataSpec",
}
