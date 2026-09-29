# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# GE Converter for Graph Mode（结构克隆自 lightning_indexer_v2；本算子当前无 GE 转换注册——
# graph 模式经 PrivateUse1/fallback 路径执行）

try:
    import torch
    import torch_npu  # noqa: F401
    import torchair  # noqa: F401

    _TORCHAIR_AVAILABLE = True
except ImportError:
    _TORCHAIR_AVAILABLE = False

# sparse_lightning_indexer 无 GE converter 注册（NPU-only 自定义算子；如需 graph 模式转换，
# 参照 graph_convert_lightning_indexer.py 模式在此注册 register_fx_node_ge_converter）
