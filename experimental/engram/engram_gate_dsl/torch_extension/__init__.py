# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""PyTorch custom-op boundary for the engram_gate CANNBotDSL operator.

本模块将算子注册进 ``cann_ops_transformer`` 库（PrivateUse1 + Meta），供
torch.compile / torchair npugraph_ex（ACL Graph）等图模式使用。
"""

from .engram_gate import engram_gate_torch

__all__ = ["engram_gate_torch"]
