# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""arch92 (dav-9201) stream resources for the Mega KDA implementation."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Architecture:
    name: str = "dav-9201"
    directory: str = "arch92"
    aiv_per_aic: int = 1
    mega_quotas: tuple[int, ...] = (8, 16, 24, 32, 64)


_TARGET = Architecture()


def architecture():
    return _TARGET


def available_aic(ref):
    import cannbotdsl
    import torch

    info = cannbotdsl.get_platform_info(stream=torch.npu.current_stream(ref.device))
    if info.npu_arch != _TARGET.name:
        raise ValueError("Mega KDA arch92 requires a dav-9201 device")
    cube, vector = int(info.cube_core_num), int(info.vector_core_num)
    if cube <= 0 or vector <= 0:
        raise ValueError("KDA requires positive AIC and AIV stream quotas")
    return min(cube, vector // _TARGET.aiv_per_aic)
