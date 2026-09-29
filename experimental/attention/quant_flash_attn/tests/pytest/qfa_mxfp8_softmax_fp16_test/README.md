<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# QFA MXFP8 Softmax FP16 测试用例执行指南

## 1. 概述

本目录承载 `quant_flash_attn` 的 MxFP8 **Softmax FP16**（`quant_mode=3`）pytest 测试框架。

当前为框架骨架阶段：pytest 框架与通用工具已就位，具体用例（golden 参考实现、参数集、测试入口）待补充。

## 2. 文件结构

```
qfa_mxfp8_softmax_fp16_test/
├── pytest.ini                                  # pytest 配置（自定义 marker）
├── conftest.py                                 # pytest 命令行选项（--golden-mode / --cache-dir / --msprof / --parse-prof / --perf-baseline）
├── quant_flash_attn_paramset_common.py         # 参数展开公共逻辑 + 默认值（quant_mode=3）
└── common/
    ├── __init__.py                             # 导入各工具模块
    ├── quant_flash_attn_golden.py              # 【待补】CPU golden 参考实现 + NPU 算子调用（当前 TODO 占位）
    ├── golden_cache.py                         # .pt 缓存工具模块
    ├── result_compare_method.py                # 精度对比工具
    ├── perf_parser.py                          # msprof op_summary.csv 解析 + baseline 比较（operator_name=QuantFlashAttn）
    └── test_runner.py                          # 共享测试执行逻辑（apply_params / execute_test / check_results）

# 【待补】用例文件：
#   quant_flash_attn_paramset_debug.py          # debug 参数集（少量用例）
#   quant_flash_attn_paramset_func_rdv.py       # 功能正确性参数集
#   quant_flash_attn_paramset_perf_rdv.py       # 性能/压力参数集
#   test_quant_flash_attn_debug.py              # debug 测试入口
#   test_quant_flash_attn_func_rdv.py           # 功能正确性测试入口
#   test_quant_flash_attn_perf_rdv.py           # 性能/压力测试入口
#   gen_csv_case_store.py                       # csv 用例生成脚本（可选）
```

## 3. 当前状态

| 类别 | 状态 |
| --- | --- |
| pytest 框架（pytest.ini / conftest.py） | ✅ 已就位 |
| common 工具（golden_cache / result_compare_method / perf_parser / test_runner） | ✅ 已就位 |
| paramset_common（参数展开 + 默认值） | ✅ 已就位（quant_mode=3） |
| golden 参考实现 | ⏳ 待补（TODO 占位） |
| 参数集（debug / func_rdv / perf_rdv） | ⏳ 待补 |
| 测试入口（test_*.py） | ⏳ 待补 |

## 4. 执行测试

用例补齐后可按 marker 运行：

```bash
pytest -v -m debug
pytest -v -m func_rdv
pytest -v -m perf_rdv
pytest -v -m ci
```

> 当前缺测试入口，pytest 会报 `no tests collected`，属空框架正常现象。

## 5. 环境准备

参考仓库根目录 README 安装 CANN + PyTorch + TorchNPU，每次运行前加载环境：

```bash
source <ascend_path>/set_env.sh
pip install pytest
cd experimental/attention/quant_flash_attn/tests/pytest/qfa_mxfp8_softmax_fp16_test
```
