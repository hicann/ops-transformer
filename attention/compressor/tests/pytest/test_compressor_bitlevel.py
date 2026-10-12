# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Compressor 位级一致性测试入口.

用例来源: test_compressor_bitlevel_paramset.py (pair / position 两类)
执行方式: 在本目录下执行
    python3 -m pytest -rA -s test_compressor_bitlevel.py -v -m ci \
        -W ignore::UserWarning -W ignore::DeprecationWarning
环境变量:
    CMP_TEST_DEVICE_ID   指定设备号(默认0)
    CMP_BITLEVEL_STRICT  0 时位级失败降级为告警+阈值校验(默认1, 即严格位级断言)
"""

import itertools
import os

import pytest
import torch_npu

from test_compressor_bitlevel_paramset import BITLEVEL_TEST_PARAMS, ENABLED_PARAMS
import compressor_bitlevel_golden as golden


_CASE_TYPE_FIELDS = ("case_type", "pair_b", "batch_size")


def _expand_cases():
    """将参数组展开为具体用例: 普通字段做笛卡尔积, pair_b/batch_size 逐值展开, "dyn" 边界对自适应.

    仅展开 ENABLED_PARAMS 中选中的参数组(在 test_compressor_bitlevel_paramset.py 里
    注释掉 ENABLED_PARAMS 的对应行即可屏蔽该组用例).
    """
    aic = golden.get_aic_core_num()
    print(
        "[collect] 设备=%s, AIC核数=%d, PERF动态D分块悬崖边界B=%s"
        % (
            torch_npu.npu.get_device_name(golden.DEVICE_ID),
            aic,
            golden.calc_perf_dsplit_boundary_b(4, 512, 2, aic),
        )
    )
    cases = []
    for group_name, params in BITLEVEL_TEST_PARAMS.items():
        if params not in ENABLED_PARAMS:
            continue  # 未在 ENABLED_PARAMS 中启用的组, 跳过
        case_type = params.get("case_type", ["pair"])[0]
        common_fields = {
            key: value for key, value in params.items() if key not in _CASE_TYPE_FIELDS
        }
        # 普通字段笛卡尔积(保持与 test_compressor_single.py 一致的展开风格)
        keys = list(common_fields.keys())
        combos = [
            dict(zip(keys, combo))
            for combo in itertools.product(*[common_fields[k] for k in keys])
        ]
        for cfg in combos:
            if case_type == "pair":
                for pair in params["pair_b"]:
                    if pair == "dyn":
                        # 按当前设备 AIC 核数自适应推导悬崖边界对, 与固定 pair 重复时跳过
                        b_dyn = golden.calc_perf_dsplit_boundary_b(
                            cfg["seq_len"], cfg["head_dim"], cfg["coff"], aic
                        )
                        if b_dyn is None or b_dyn < 2:
                            print(
                                "[collect] AIC=%d, 不适用动态D分块边界推导, 跳过 dyn pair"
                                % aic
                            )
                            continue
                        if any(
                            c["kind"] == "pair" and c["b_small"] == b_dyn for c in cases
                        ):
                            print(
                                "[collect] 动态边界对(%d,%d)与固定用例重复, 跳过"
                                % (b_dyn, b_dyn + 1)
                            )
                            continue
                        cases.append(
                            {
                                "group": "%sDyn" % group_name,
                                "kind": "pair",
                                "b_small": b_dyn,
                                "b_large": b_dyn + 1,
                                "cfg": cfg,
                            }
                        )
                    else:
                        cases.append(
                            {
                                "group": group_name,
                                "kind": "pair",
                                "b_small": pair[0],
                                "b_large": pair[1],
                                "cfg": cfg,
                            }
                        )
            elif case_type == "single":
                # 单次运行探针: 只跑 golden校验+NPU直调, 不做位级对比(配合tiling调试探针使用)
                for batch_size in params["batch_size"]:
                    cases.append(
                        {
                            "group": group_name,
                            "kind": "single",
                            "B": batch_size,
                            "cfg": cfg,
                        }
                    )
            else:
                for batch_size in params["batch_size"]:
                    cases.append(
                        {
                            "group": group_name,
                            "kind": "position",
                            "B": batch_size,
                            "cfg": cfg,
                        }
                    )
    return cases


torch_npu.npu.set_deterministic_level(3)
_BITLEVEL_CASES = _expand_cases()


def _case_id(case):
    if case["kind"] == "pair":
        return "pair_%d_%d" % (case["b_small"], case["b_large"])
    if case["kind"] == "single":
        return "single_B%d_S%d" % (case["B"], case["cfg"]["seq_len"])
    return "pos_B%d" % case["B"]


_CASE_IDS = ["%s-%s" % (case["group"], _case_id(case)) for case in _BITLEVEL_CASES]


@pytest.mark.ci
@pytest.mark.parametrize("case", _BITLEVEL_CASES, ids=_CASE_IDS)
def test_compressor_bitlevel(case):
    torch_npu.npu.set_device(golden.DEVICE_ID)
    cfg = case["cfg"]
    if case["kind"] == "pair":
        # pair: B=(b_small, b_large) 两次运行共享batch位级对比(模板切换/动态D分块悬崖)
        golden.run_pair_case(cfg, case["b_small"], case["b_large"])
    elif case["kind"] == "single":
        # single: 单次运行探针(golden校验+NPU直调), 用于配合tiling侧调试打印验证模板选择
        golden.run_once(cfg, case["B"], tag="single_probe")
    else:
        # position: 同B, 目标样本换到首/中/尾位置位级对比(batch位置一致性)
        golden.run_position_case(cfg, case["B"])
