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

"""注册 Quant MLA 的 TTK 插件及 CPU、设备和图执行钩子。

实现模块按需从 impl 目录加载，并将 assets 与 impl 目录加入模块搜索路径。
"""

import importlib
import importlib.util
import sys
from pathlib import Path

__spec__ = {
    "quant_mla_ttk_ops.quant_mla_prolog_ttk": "QuantMlaPrologSpec",
}

_ASSETS_DIR = str(Path(__file__).resolve().parent)
_ASSET_IMPL_DIR = str(Path(__file__).with_name("impl"))
for _dir in (_ASSETS_DIR, _ASSET_IMPL_DIR):
    if _dir not in sys.path:
        sys.path.insert(0, _dir)

_impl_cache = {}


def _load_impl_module(stem):
    """延迟加载实现模块，避免导入时依赖设备环境。"""
    if stem not in _impl_cache:
        path = Path(_ASSET_IMPL_DIR) / f"{stem}.py"
        spec = importlib.util.spec_from_file_location(
            f"quant_mla_prolog_ttk_assets_impl_{stem}", path
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _impl_cache[stem] = module
    return _impl_cache[stem]


class QuantMlaPrologSpec:
    """Quant MLA 的 TTK 端到端测试插件。"""

    @staticmethod
    def golden(*args, **kwargs):
        """调用独立的本地 CPU Golden，不依赖 pytest 目录。

        Golden 和双方、三方比较的数值定义统一保留在本地 pytest_reference 中。
        """
        return _load_impl_module("golden").quant_mla_prolog_golden(*args, **kwargs)

    @staticmethod
    def customize_inputs(*args, **kwargs):
        """按用例规则原地准备输入，TTK 不使用此钩子的返回值。"""
        return _load_impl_module("inputs").customize_inputs(*args, **kwargs)

    @staticmethod
    def npu_preprocess(*args, **kwargs):
        """在计时和图捕获前完成设备输入准备及适用的契约检查。"""
        return _load_impl_module("npu_preprocess").run(*args, **kwargs)

    # 比较钩子使用分段的双方比较和 MXFP8 三方验收规则，具体适配位于 compare.py。

    compare = _load_impl_module("compare").quant_mla_prolog_compare

    # 默认 stat_rel_err 避免提升 Golden 类型，双方比较保持相同精度的 CPU 参考。

    tolerance = _load_impl_module("compare").TOLERANCE

    # 图模式在构造阶段保留静态张量，forward 只执行公开设备调用。

    torch_graph = _load_impl_module("graph").QuantMlaPrologAclGraph
