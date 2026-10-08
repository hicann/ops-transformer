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
            os.path.join(
                self._cann_path,
                "opp/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/op_kernel",
            ),
            os.path.join(
                self._cann_path,
                "opp/built-in/op_impl/ai_core/tbe/impl/ops_transformer/ascendc/common/op_kernel",
            ),
        ]
        for path in candidate_paths:
            if os.path.isdir(path):
                paths.append(path)
        return paths


comm_channel_builder_manager_op_builder = CommChannelBuilderManagerOpBuilder()

# 低比特 MTE 算子族公共 context 的固定 tag，保证 eager/静态图/动态图共享同一份 context
QUANT_MTE_CONTEXT_TAG = "quant_lowbit_mte"

# context 的 int32 元素数 = sizeof(QuantMteContext) / sizeof(int32_t) = 8200 / 4，
# 与 csrc/comm_channel_builder_manager.cpp 中 QuantMteContext 布局一一对应，
# 修改结构体时需同步更新（含 static_assert 校验的 device 侧布局）
QUANT_MTE_CONTEXT_ELEM_NUM = 2050


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


class QuantMteContextManager:
    """低比特量化通信算子族（quant_all_reduce / quant_reduce_scatter）公共 context 管理类。"""

    def __init__(self, group: str):
        self._group = group

    def get_context(self):
        """获取幂等共享 context，返回 (NPU tensor, hccl buffer size)，eager 路径使用。"""
        module = comm_channel_builder_manager_op_builder.load()
        manager = module.CommChannelBuilderManager(self._group)
        return manager.create_quant_mte_context(QUANT_MTE_CONTEXT_TAG)

    def get_context_data(self):
        """获取幂等共享 context 的 host int32 内容，返回 (int32 列表, hccl buffer size)。

        FakeTensorMode 安全（不产生任何经过 dispatcher 的 tensor op），供 torchair
        GE converter 在 AOT 编译期（fake mode）构造 ge.Const 使用。
        """
        module = comm_channel_builder_manager_op_builder.load()
        manager = module.CommChannelBuilderManager(self._group)
        return manager.get_quant_mte_context_data(QUANT_MTE_CONTEXT_TAG)
