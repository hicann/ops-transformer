# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""计算后端抽象基类。"""

from abc import ABC, abstractmethod
from typing import Any, Dict
import torch


class Backend(ABC):
    """算子计算后端统一接口。"""

    name: str = "base"

    @abstractmethod
    def is_available(self) -> bool: ...

    @abstractmethod
    def compute(
        self, inputs: Dict[str, torch.Tensor], params: Dict[str, Any]
    ) -> Dict[str, torch.Tensor]: ...

    @property
    @abstractmethod
    def device(self) -> torch.device: ...

    def clear_cache(self):
        pass
