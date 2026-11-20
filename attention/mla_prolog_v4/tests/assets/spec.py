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

"""注册 MLA v4 的 TTK 插件及 CPU、设备和图执行钩子。

实现模块按需从 impl 目录加载，并将 assets 目录加入模块搜索路径。
"""

import importlib.util
import sys
from pathlib import Path

__spec__ = {
    "mla_prolog_v4_ttk_ops.mla_prolog_v4_ttk": "MlaPrologV4Spec",
}

_ASSETS_DIR = str(Path(__file__).resolve().parent)
if _ASSETS_DIR not in sys.path:
    sys.path.insert(0, _ASSETS_DIR)

_ASSET_IMPL_DIR = Path(__file__).with_name("impl")

_impl_cache = {}


def _load_impl_module(stem):
    """延迟加载实现模块，避免导入时依赖设备环境。"""
    if stem not in _impl_cache:
        path = _ASSET_IMPL_DIR / f"{stem}.py"
        spec = importlib.util.spec_from_file_location(
            f"mla_prolog_v4_ttk_assets_impl_{stem}", path
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _impl_cache[stem] = module
    return _impl_cache[stem]


class MlaPrologV4Spec:
    """MLA v4 的 TTK 端到端测试插件。"""

    @staticmethod
    def golden(*args, **kwargs):
        """调用独立的本地 CPU Golden，不依赖 pytest 目录。

        双方比较遵循单 BF16 契约，三方比较使用 pytest_reference 中的独立 FP32 数学参考。
        """
        return _load_impl_module("golden").mla_prolog_v4_golden(*args, **kwargs)

    @staticmethod
    def customize_inputs(*args, **kwargs):
        """按用例规则原地准备输入，TTK 不使用此钩子的返回值。"""
        return _load_impl_module("inputs").customize_inputs(*args, **kwargs)

    @staticmethod
    def npu_preprocess(*args, **kwargs):
        """在计时和图捕获前完成设备输入准备及适用的契约检查。"""
        return _load_impl_module("npu_preprocess").run(*args, **kwargs)

    # 比较钩子处理量化载体；cross_check 请求三方比较，数值规则统一由本地参考实现提供。

    compare = _load_impl_module("compare").mla_prolog_v4_compare

    # 默认 stat_rel_err 避免提升 Golden 类型；cross_check 的参数由比较钩子解析。

    tolerance = _load_impl_module("compare").TOLERANCE

    # 图模式在构造阶段准备静态缓冲区，forward 只执行公开设备调用。

    torch_graph = _load_impl_module("graph").MlaPrologV4AclGraph
