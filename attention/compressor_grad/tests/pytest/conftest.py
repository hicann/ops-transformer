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


def pytest_runtest_call(item):
    """每个 case 执行前打印测试名和参数；skip 的 case（通路不匹配）不打印。"""
    # 测试函数名 → pathway 映射（与 test_compressor_grad.py 中 skip 逻辑一致）
    _PATHWAY_OF_TEST = {
        "test_compressor_grad": 1,
        "test_compressor_grad_backward": 2,
        "test_compressor_grad_small_ops": 3,
        "test_compressor_grad_backward_direct": 4,
        "test_compressor_grad_forward_direct": 5,
    }
    test_name = item.name.split("[")[0]
    pathway = _PATHWAY_OF_TEST.get(test_name)
    cur = item.config.getoption("--pathway")
    if pathway != cur:
        return  # 该 case 将被 pytest.skip，不打印
    case = item.callspec.params.get("case")
    if case:
        name = case.get("testcase_name", "unknown")
        params = {
            k: case.get(k)
            for k in (
                "B",
                "S1",
                "H",
                "D",
                "cmp_ratio",
                "coff",
                "dtype",
                "input_layout",
                "seqused_q",
            )
            if k in case
        }
        print(f"\n{'=' * 60}")
        print(f"[RUN] {name}")
        for k, v in params.items():
            if v is not None:
                print(f"      {k}={v}")
        print(f"{'=' * 60}")
