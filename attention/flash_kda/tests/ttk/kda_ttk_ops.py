# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""CSV API dispatcher; test logic belongs to the selected architecture."""

from importlib import import_module

from kda_cases import ARCHITECTURES, architecture


def get_adapter(arch=None):
    arch = arch or architecture()
    if arch not in ARCHITECTURES:
        raise ValueError("FLASH_KDA_ARCH must be arch35 or arch92")
    return import_module(f"{arch}.kda_ttk_ops")


# Both adapters expose the same CSV signature. A running DUT process keeps its
# architecture fixed; host asset checks may explicitly load either adapter.
flash_kda = get_adapter().flash_kda
