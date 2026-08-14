# ======================================================================================================================
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================


def pytest_addoption(parser):
    """选择验证通路：1=正反向全链路（通路5+通路4拼接，默认） 2=单反向PyPTO直跑
    3=小算子拼接golden 4=单反向直调（_compressor_backward） 5=单正向直调（_compressor_forward）
    compare-mode：two=两方（现有golden） three=三方（NPU小算子+float64高精度，默认）"""
    parser.addoption(
        "--pathway",
        action="store",
        type=int,
        default=1,
        help="验证通路: 1=全链路(默认) 2=单反向PyPTO直跑 3=小算子拼接golden "
        "4=单反向直调 5=单正向直调",
    )
    parser.addoption(
        "--compare-mode",
        action="store",
        type=int,
        default=3,
        choices=[2, 3],
        help="精度比对模式: 2=两方(现有golden, 回归用) "
        "3=三方(NPU小算子+float64高精度, 默认)",
    )
