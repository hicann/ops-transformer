# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
from cann_ops_transformer.op_builder import OpBuilder, get_as_library


class CommContextOpBuilder(OpBuilder):
    def __init__(self):
        super(CommContextOpBuilder, self).__init__("comm_context", category="mc2")

    def sources(self):
        return ["csrc/mc2/comm_context.cpp"]

    def schema(self):
        return None

    def register_meta(self):
        pass


comm_context_op_builder = CommContextOpBuilder()
comm_context_op_builder._ensure_initialized()


class _LazyClassProxy:
    def __init__(self, name, builder):
        self._name = name
        self._builder = builder
        self._real_cls = None

    def __call__(self, *args, **kwargs):
        return self._ensure_loaded()(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._ensure_loaded(), name)

    def _ensure_loaded(self):
        if self._real_cls is None:
            self._real_cls = getattr(self._builder.load(), self._name)
        return self._real_cls


def __getattr__(name):
    if name == "CommContextManager":
        return _LazyClassProxy("CommContextManager", comm_context_op_builder)
    raise AttributeError(f"module '{__name__}' has no attribute {name}")
