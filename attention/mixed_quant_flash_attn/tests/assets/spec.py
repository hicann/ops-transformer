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

Registers the 22-parameter signature and wires the golden / customize_inputs /
compare functions from impl/*.py into the TestSpec class attributes that TTK's
get_spec_attr reads.
"""

import importlib.util
from pathlib import Path


_impl_dir = Path(__file__).with_name("impl")


def _load_impl(stem):
    path = _impl_dir / f"{stem}.py"
    spec = importlib.util.spec_from_file_location(f"mqfa_impl_{stem}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_golden_mod = _load_impl("golden")
_inputs_mod = _load_impl("inputs")
_compare_mod = _load_impl("compare")


class MixedQuantFlashAttnSpec:
    golden = _golden_mod.cpu_mixed_quant_flash_attn
    customize_inputs = _inputs_mod.generate_mixed_quant_inputs
    tolerance = {
        "float16": {"standard": "stat_rel_err"},
        "bfloat16": {"standard": "stat_rel_err"},
    }

    def compare(*outputs, **kwargs):
        return _compare_mod.compare(*outputs)


__spec__ = {
    "mqfa_ttk_ops.call_npu": "MixedQuantFlashAttnSpec",
}
