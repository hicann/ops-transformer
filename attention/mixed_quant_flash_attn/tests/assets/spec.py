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

"""TestSpec adapter for mixed_quant_flash_attn.

Registers the native api_name and wires the golden / customize_inputs / compare /
npu_preprocess functions from impl/*.py into the TestSpec class attributes that
TTK's get_spec_attr reads.

impl modules are loaded lazily (inside each callback) instead of at import time:
eager loading would cascade-import the pytest `core` package (impl/golden.py and
impl/inputs.py do `from core.* import ...`) in the main process, where the
`pytests` dir is not yet on sys.path, making the whole spec file fail to load
(``No module named 'core'``).  Lazy loading defers those imports to the forked
worker, where sitecustomize.py has already added ``../pytests`` to sys.path.
"""

import importlib.util
import sys
from pathlib import Path


_impl_dir = Path(__file__).with_name("impl")
_impl_cache = {}

# pytests 目录放在 sys.path 最前，让 assets/impl 模块里的
# ``from core.gen_data import generate_inputs`` / ``from core.cpu import ...`` 可解析。
# （sitecustomize.py 也会加这个路径，但它只在 assets 目录本就位于 sys.path 时才会被
# 解释器自动加载——ttk 的 --plugin 并不会把资产目录加进 sys.path，因此这里显式注入。）
_PYT_DIR = str((Path(__file__).parent.parent / "pytests").resolve())
if _PYT_DIR not in sys.path:
    sys.path.insert(0, _PYT_DIR)


def load_impl_module(stem):
    """Lazily load an impl/*.py module under a stable import name."""
    mod_name = f"mqfa_assets_impl_{stem}"
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    path = _impl_dir / f"{stem}.py"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    _impl_cache[stem] = module
    return _impl_cache[stem]


# 三方比对专用容差（仅 --compare cross_check 时生效）。与 tolerance 分开声明，
# 避免默认(stat_rel_err)模式下触发 TTK 的 golden Promote / 第三方抓取逻辑。
CROSS_CHECK_TOLERANCE = {
    "float16": {"standard": "cross_check", "level": "L2"},
    "bfloat16": {"standard": "cross_check", "level": "L2"},
}


class MixedQuantFlashAttnSpec:
    tolerance = {
        "float16": {"standard": "stat_rel_err"},
        "bfloat16": {"standard": "stat_rel_err"},
    }

    @staticmethod
    def golden(*args, **kwargs):
        return load_impl_module("golden").cpu_mixed_quant_flash_attn(*args, **kwargs)

    @staticmethod
    def customize_inputs(*args, **kwargs):
        return load_impl_module("inputs").generate_mixed_quant_inputs(*args, **kwargs)

    @staticmethod
    def npu_preprocess(*args, **kwargs):
        return load_impl_module("npu_preprocess").run(*args, **kwargs)

    @staticmethod
    def compare(*outputs, compare_context=None, **kwargs):
        compare_mod = load_impl_module("compare")
        compare_method = compare_mod._get_compare_method()
        if compare_method == "cross_check":
            # Rebuild 第三方(高精度 use_npu=True) benchmark 结果，保证 compare 进程里有 bench_outputs。
            golden_mod = load_impl_module("golden")
            golden_mod.rebuild_golden_context_from_compare_context(compare_context)
            ctx = golden_mod.get_golden_context()
            bench_outputs = ctx.get("bench_outputs")
            kwargs["data_type"] = ctx.get("data_type") or kwargs.get("data_type")
            if bench_outputs is not None and compare_mod.is_cross_check_available():
                return compare_mod._run_ttk_cross_check_e2e(
                    outputs, kwargs, bench_outputs, CROSS_CHECK_TOLERANCE
                )
        return compare_mod.compare(*outputs)


__spec__ = {
    "torch.ops.cann_ops_transformer.mixed_quant_flash_attn": "MixedQuantFlashAttnSpec",
}
