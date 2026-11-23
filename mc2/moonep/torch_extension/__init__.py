# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""MoonEP operator adapter.

The implementation lives in ``op_dsl/`` and uses flat module imports
(``import moonep_plan``) so it can also be executed straight from the source
tree with ``PYTHONPATH=mc2/moonep/op_dsl``.  The Torch Extension wheel copies
``op_dsl/`` next to this file (``ops/mc2/moonep/op_dsl``); expose that
directory on ``sys.path`` before importing so both layouts behave the same.

Imports are lazy (PEP 562): ``import cann_ops_transformer.ops`` must stay cheap
and must not fail on hosts without torch_npu — only touching a MoonEP API
pulls in the accelerator runtime.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))

# Wheel layout: <package>/moonep/op_dsl; source layout: <op>/op_dsl.
_IMPL_DIRS = (
    os.path.join(_HERE, "op_dsl"),
    os.path.join(os.path.dirname(_HERE), "op_dsl"),
)

for _impl_dir in _IMPL_DIRS:
    if os.path.isdir(_impl_dir):
        if _impl_dir not in sys.path:
            sys.path.insert(0, _impl_dir)
        break

# Public attribute -> (module, attribute).  ``None`` exposes the module itself.
_EXPORTS = {
    "Buffer": ("moonep_buffer", "Buffer"),
    "MoonepPlan": ("moonep_buffer", "MoonepPlan"),
    "create_moonep_context": ("moonep_context", "create_moonep_context"),
    "moonep_plan": ("moonep_plan", "moonep_plan"),
    "prefetch_weight": ("prefetch_weight", "prefetch_weight"),
    "reduce_grad": ("reduce_grad", "reduce_grad"),
    "moonep_context": ("moonep_context", None),
    "moonep_layout": ("moonep_layout", None),
    "moonep_plan_module": ("moonep_plan", None),
    # moonep_topology 已删除：其 hook 的分布式建链接口在依赖升级后移除，
    # 模块失去运行意义（URMA 建链原生处理层选择），运行链路无任何引用。
    "moonep_comm": ("moonep_comm", None),
    "npu_blob": ("npu_blob", None),
    "prefetch_weight_module": ("prefetch_weight", None),
    "reduce_grad_module": ("reduce_grad", None),
}


def __getattr__(name):
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = target
    from importlib import import_module

    module = import_module(module_name)
    value = module if attribute is None else getattr(module, attribute)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_EXPORTS))


# Entry point target is ``...ops.mc2.moonep:moonep``.
moonep = sys.modules[__name__]
