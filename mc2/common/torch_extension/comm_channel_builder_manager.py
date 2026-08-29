# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import os
from cann_ops_transformer.op_builder import OpBuilder


class CommChannelBuilderManagerOpBuilder(OpBuilder):
    def __init__(self):
        super(CommChannelBuilderManagerOpBuilder, self).__init__(
            "comm_channel_builder_manager", category="mc2"
        )

    def sources(self):
        return ["csrc/mc2/comm_channel_builder_manager.cpp"]

    def schema(self):
        return None

    def register_meta(self):
        pass

    def include_paths(self):
        paths = super().include_paths()
        candidate_paths = [
            os.path.join(
                self._cann_path,
                "opp/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common",
            ),
            os.path.join(
                self._cann_path,
                "vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common",
            ),
            os.path.join(
                self._cann_path,
                "opp/built-in/op_impl/ai_core/tbe/impl/ops_transformer/ascendc/common",
            ),
        ]
        for path in candidate_paths:
            if os.path.isdir(path):
                paths.append(path)
        return paths


comm_channel_builder_manager_op_builder = CommChannelBuilderManagerOpBuilder()


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
    if name == "CommChannelBuilderManager":
        return _LazyClassProxy(
            "CommChannelBuilderManager", comm_channel_builder_manager_op_builder
        )
    raise AttributeError(f"module '{__name__}' has no attribute {name}")
