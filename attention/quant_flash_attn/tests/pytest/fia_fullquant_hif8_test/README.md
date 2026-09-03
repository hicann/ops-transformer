<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# QFA HIF8 测试用例执行指南

quant_flash_attn HIF8 全量化（`quant_mode=0`，A8C8_QKV_HIF8_P_PER_TENSOR_SOFTMAX_FP32）功能测试。

## 1. 场景特性

| 特性 | 取值 | 说明 |
|------|------|------|
| quant_mode | 0 | HIF8 全量化 |
| q/k/v dtype | HIFLOAT8 | uint8 编码，csrc 侧转为 ACL_HIFLOAT8 |
| descale | FP32 标量 `(1,)` | q/k/v 各一个 per-tensor scale |
| p_scale | FP32 `(1,)` | P 量化 scale |
| head_dim | **仅 128** | tiling config 锁定 DAligned128 |
| layout | TND / BSND / BNSD | 不支持 PA（无 block_table） |
| layout_q_descale | BSND | HIF8 约束固定值 |
| attn_out | BF16 | — |
| softmax_lse | FP32 | TND 输出 `(N, T)`；非 TND 输出 `(B, N, S)` |

## 2. 环境准备

```bash
source /home/user/Ascend/cann/set_env.sh
conda activate your-env-name
pip install pytest
cd attention/quant_flash_attn/tests/pytest/fia_fullquant_hif8_test
```

## 3. 文件结构

```
fia_fullquant_hif8_test/
├── pytest.ini                                  # pytest 配置（自定义 marker）
├── conftest.py                                 # pytest 命令行选项（--golden-mode, --cache-dir, --msprof, --parse-prof, --perf-baseline）
├── common/
│   ├── __init__.py
│   ├── quant_flash_attn_golden.py              # CPU golden 参考实现 + NPU 算子调用
│   ├── golden_cache.py                         # .pt 缓存工具模块
│   ├── generate_hifloat8_data.py               # HIF8 编解码（hifloat8 <-> float）
│   ├── result_compare_method.py                # 精度对比工具
│   ├── perf_parser.py                          # msprof op_summary.csv 解析 + baseline 比较
│   └── test_runner.py                          # 共享测试执行逻辑（apply_params / execute_test / check_results）
├── quant_flash_attn_paramset_common.py         # 参数展开公共逻辑 + 默认值
├── quant_flash_attn_paramset_debug.py          # debug 参数集（少量用例，快速验证）
├── quant_flash_attn_paramset_func_rdv.py       # 功能正确性参数集（50 条）
├── quant_flash_attn_paramset_perf_rdv.py       # 性能/压力参数集（3 条）
├── test_quant_flash_attn_debug.py              # debug 测试入口
├── test_quant_flash_attn_func_rdv.py           # 功能正确性测试入口
└── test_quant_flash_attn_perf_rdv.py           # 性能/压力测试入口
```

## 4. 执行测试

### 4.1 三个测试入口

| 测试文件 | Marker | 参数集 | 用途 |
|----------|--------|--------|------|
| `test_quant_flash_attn_debug.py` | `@pytest.mark.debug` | debug（2 条） | 快速验证基本功能 |
| `test_quant_flash_attn_func_rdv.py` | `@pytest.mark.func_rdv` `@pytest.mark.ci` | func_rdv（50 条） | 功能正确性全覆盖 |
| `test_quant_flash_attn_perf_rdv.py` | `@pytest.mark.perf_rdv` | perf_rdv（3 条） | 性能/压力验证 |

### 4.2 基本执行

```bash
# 运行 debug 用例
pytest test_quant_flash_attn_debug.py -v

# 运行功能正确性用例
pytest test_quant_flash_attn_func_rdv.py -v

# 运行性能/压力用例
pytest test_quant_flash_attn_perf_rdv.py -v

# 按 marker 运行
pytest -m debug -v
pytest -m func_rdv -v
pytest -m ci -v
```

### 4.3 过滤用例（-k）

```bash
# 按 layout 类型
pytest test_quant_flash_attn_func_rdv.py -v -k "TND"
pytest test_quant_flash_attn_func_rdv.py -v -k "BSND"
pytest test_quant_flash_attn_func_rdv.py -v -k "BNSD"

# 精确指定单个用例
pytest test_quant_flash_attn_func_rdv.py -v -k "TND_B1_QS4_KVS1024_Nq1_Nkv1_D128_SP3"

# 排除某些用例
pytest -v -k "not BSND"
```

### 4.4 常用选项

| 选项 | 作用 |
|------|------|
| `-v` | 详细输出，显示每个用例名 |
| `-s` | 不截断 stdout/stderr |
| `-x` | 遇到失败立即停止 |
| `--tb=long` | 失败时显示完整堆栈 |
| `-m ci` | 只运行 CI 标记的用例 |

## 5. Golden 缓存模式（--golden-mode）

| 模式 | 作用 |
|------|------|
| `all` | 全流程：生成数据 → CPU → NPU → 精度对比（**默认值**） |
| `gen` | 仅生成并保存输入数据 |
| `cpu` | 加载输入缓存 → 跑 CPU 并保存输出 |
| `npu` | 加载输入缓存 → 跑 NPU 并保存输出 |
| `compare` | 加载 CPU/NPU 缓存输出 → 精度对比 |

模式支持逗号分隔组合，按 gen → cpu → npu → compare 顺序执行：

```bash
# 分步执行（适合跨机器调试）
pytest test_quant_flash_attn_func_rdv.py -v --golden-mode=gen
pytest test_quant_flash_attn_func_rdv.py -v --golden-mode=cpu
pytest test_quant_flash_attn_func_rdv.py -v --golden-mode=npu,compare

# 自定义缓存目录（默认 common/golden_cache/）
pytest test_quant_flash_attn_func_rdv.py -v --cache-dir=/tmp/my_cache
```

每个 case 生成 3 个 `.pt` 文件：

```
golden_cache/
├── {case_name}_input.pt        # 输入数据（Q/K/V hif8 + descale + p_scale）
├── {case_name}_cpu_output.pt   # CPU golden 输出（atten_out + lse）
└── {case_name}_npu_output.pt   # NPU 算子输出（atten_out + lse）
```

## 6. 独立脚本运行

`common/quant_flash_attn_golden.py` 也支持独立运行，同样支持 `--mode` 组合：

```bash
cd common/
python quant_flash_attn_golden.py --mode all --case-name my_case
python quant_flash_attn_golden.py --mode gen --case-name my_case
python quant_flash_attn_golden.py --mode npu,compare --case-name my_case
```

## 7. 用例命名规则

格式：`{LAYOUT}_B{batch}_QS{Q长度}_KVS{KV长度}_Nq{Q头数}_Nkv{KV头数}_D{维度}_SP{稀疏模式}`

| 字段 | 含义 | 示例 |
|------|------|------|
| `LAYOUT` | input_layout（TND / BSND / BNSD） | `BSND_B1_QS128_...` |
| `B` | batch size | `B1`, `B5` |
| `QS` | Q 序列长度（B>1 时为各 batch 求和） | `QS128` |
| `KVS` | KV 序列长度（B>1 时为各 batch 求和） | `KVS2048` |
| `Nq` | Q 头数 | `Nq80` |
| `Nkv` | KV 头数（N_q 需能被 N_kv 整除） | `Nkv8` |
| `D` | Head 维度（**固定 128**） | `D128` |
| `SP` | sparse_mode（0=无 mask, 3=causal+padding） | `SP3` |

## 8. 参数体系

### 8.1 HIF8 专属参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `input_layout` | str | `TND` | q/k/v/out 布局（`TND` / `BSND` / `BNSD`） |
| `q_scale_layout` | str | `BSND` | layout_q_descale（HIF8 固定为 BSND） |
| `p_scale` | float | 必填 | P 量化 scale（1.0 / 15.0 / 128.0 / 256.0） |

### 8.2 通用参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `B` | int | 必填 | batch size |
| `N_q` | int | 必填 | Q 头数 |
| `N_kv` | int | 必填 | KV 头数 |
| `D` | int | 必填 | Head 维度（仅 128） |
| `cu_seqlens_q/kv` | list | 必填 | 累计序列长度（TND 用） |
| `seqused_q/kv` | list | 必填 | 每 batch 实际序列长度 |
| `max_seqlen_q/kv` | int | -1 | 最大序列长度（-1 自动取 max(seqused)） |
| `mask_mode` | int | 必填 | 稀疏模式（0 / 3） |
| `enable_lse` | bool | 必填 | 是否返回 Log-Sum-Exp |
| `enable_pa` | bool | False | HIF8 不支持 PA，固定 False |
| `softmax_scale` | float | None | None 时取 1/sqrt(D) |
| `data_range_q/k/v` | float | 1.0 | 数据范围 |
| `device_id` | int | 0 | NPU 设备 ID |
| `is_contiguous` | bool | True | 输入是否 contiguous |

### 8.3 默认值机制

`TEST_PARAMS_DEFAULTS` 中的参数在 paramset 未显式指定时自动生效；`expand_paramset_to_cases()` 对每个配置的所有维度做笛卡尔积展开。

## 9. 新增用例

在对应 paramset 文件中添加条目：

```python
"BSND_B1_QS256_KVS1024_Nq8_Nkv2_D128_SP3": {
    "B": [1],
    "N_q": [8],
    "N_kv": [2],
    "D": [128],
    "cu_seqlens_q": [[0, 256]],
    "cu_seqlens_kv": [[0, 1024]],
    "seqused_q": [[256]],
    "seqused_kv": [[1024]],
    "max_seqlen_q": [256],
    "max_seqlen_kv": [1024],
    "mask_mode": [3],
    "q_scale_layout": ["BSND"],
    "p_scale": [1.0],
    "enable_lse": [True],
    "input_layout": ["BSND"],
},
```

注意事项：

- **D 只能取 128**（op 侧 HIF8 校验 `head_dim = 128`，UT 用例 `QFA_hif8_head_dim_64_unsupported` 验证非法值）
- **BSND/BNSD 用例建议各 batch 等长**（seqused == max_seqlen），避免非 TND 的 padding 行为差异
- **B>1 不等长序列**仅建议在 TND 用例中使用（cu_seqlens 累计求和作为 QS/KVS 命名值）
- 用例名 LAYOUT 前缀须与 `input_layout` 一致

## 10. 关联 UT（op_host）

host 侧 UT 位于 `tests/ut/op_host/`，CSV 中 `quant_compute_mode=0` 即 HIF8 场景：

| 文件 | HIF8 覆盖 |
|------|-----------|
| `arch35/test_quant_flash_attn_tiling.csv` | 14 条：TND/BSND/BNSD SUCCESS（含 LSE、GQA、QS=1）；FAILED（D=64、descale 2D/E8M0、q dtype E4M3、PA_NZ、layout 不一致、layout_q_descale=TND） |
| `test_quant_flash_attn_shape_infershape.csv` | 3 条：TND/BSND/BNSD infershape（含 LSE `(N,T)` / `(B,N,S)`） |
| `test_quant_flash_attn_dtype_infershape.csv` | 1 条：HIFLOAT8/FLOAT/BF16 dtype 推导 |
