# quant\_flash\_mla\_with\_kvcache 测试套件

MLA FP8 量化 Flash Attention（含 KV cache）算子的 TTK E2E 测试与 pytest 测试。

## 目录结构

```
tests/
├── assets/                              # TTK 插件目录（--plugin 指向此目录）
│   ├── _xlsx_minireader.py              # 无依赖 xlsx 读取器（zipfile + minidom）
│   ├── excel_to_csv.py                  # xlsx → TTK CSV 转换脚本
│   ├── qfmla_golden.py      # MLA FP8 golden 参考实现
│   ├── spec.py                          # TTK TestSpec 注册（含 torch_graph）
│   └── impl/                            # TTK 插件实现
│       ├── inputs.py                    # customize_inputs: 生成 BNSD FP8 数据 → 转目标 layout → 写入 tensor slot
│       ├── golden.py                    # golden: 设置全局变量 → 生成 CPU 数据 → 调 golden 函数 → 转目标 layout
│       ├── compare.py                   # compare: NPU vs golden 精度比对（rtol/atol/PctThd）
│       ├── npu_preprocess.py            # npu_preprocess: 调 metadata 算子生成 metadata → 回填 slot
│       └── graph.py                     # aclgraph 编译目标（QuantMlaAclGraph）
│
├── pytest/qfmla_mxfp8_test/             # pytest 测试目录（与 mxfp8 风格对齐）
│   ├── excel_to_pytest_paramset.py      # xlsx → pytest paramset（TEST_PARAMS 字典）
│   ├── pytest_paramset_to_csv.py        # pytest paramset → TTK e2e CSV
│   ├── quant_flash_mla_paramset_common.py  # 公共展开逻辑（expand_paramset_to_cases）
│   ├── quant_flash_mla_paramset_debug.py   # 自动生成的用例集
│   ├── test_quant_flash_mla_func_rdv.py # pytest 测试入口
│   ├── conftest.py                     # pytest 插件（--golden-mode, --cache-dir）
│   ├── pytest.ini                      # pytest marker 定义
│   └── common/
│       ├── __init__.py
│       └── test_runner.py              # 测试执行器（gen → cpu → npu → compare 四阶段）
│
├── qfmla_excel.csv                      # 当前 TTK 测试用 CSV（由 excel_to_csv.py 生成）
├── ttk_mla_run.sh                      # TTK 运行脚本（source CANN + conda + custom_transformer）
└── validate_red_tnd.py                 # TND 格式验证脚本
```

## 转换链

```
Excel (red_modified.xlsx)
  │
  ├─ excel_to_csv.py          ─→ TTK CSV (qfmla_excel.csv)
  │
  └─ excel_to_pytest_paramset.py ─→ pytest paramset (quant_flash_mla_paramset_debug.py)
                                     │
                                     └─ pytest_paramset_to_csv.py ─→ TTK CSV (qfmla_pytest.csv)
```

两种路径最终都生成 TTK e2e CSV，可互换使用。

## TTK E2E 测试

### 1. Excel → CSV

```bash
cd tests/assets
python excel_to_csv.py --xlsx /home/z00954345/red_modified.xlsx --out ../qfmla_excel.csv
```

可选参数：`--sheet 1` / `--list-sheets`

### 2. 运行 TTK

```bash
cd tests
source /root/miniconda3/etc/profile.d/conda.sh && conda activate zxx
source /home/wangke293/Ascend_0909/cann/set_env.sh
source /home/z00954345/Ascend_0909_custom/vendors/custom_transformer/bin/set_env.bash
export TORCH_EXTENSIONS_DIR=/home/z00954345/.cache/torch_extensions
export MULTIPROCESSING_START_METHOD=fork
```

直接运行：

```bash
bash ttk_mla_run.sh
```

或手动调用：

```bash
python -c "
import numpy as np
for name, repl in [('long', np.int64), ('ulong', np.uint64)]:
    if not hasattr(np, name): setattr(np, name, repl)
import scipy.sparse._sputils as _sputils
import multiprocessing; multiprocessing.set_start_method('fork', force=True)
import sys
sys.argv = ['ttk', 'e2e',
    '-i', 'qfmla_excel.csv',
    '--plugin', 'assets',
    '--single-log', '--proc-no-reuse', '--pc', '1']
from ttk.__main__ import main; main()
"
```

## pytest 测试

### 1. Excel → pytest paramset

```bash
cd tests/pytest/qfmla_mxfp8_test
python excel_to_pytest_paramset.py --xlsx /home/z00954345/red_modified.xlsx --sheet 1
```

### 2. 运行 pytest

```bash
cd tests/pytest/qfmla_mxfp8_test
pytest test_quant_flash_mla_func_rdv.py -v
```

支持 `--golden-mode` 分阶段执行：

| 模式 | 说明 |
|------|------|
| `all`（默认） | 全流程：生成数据 → CPU golden → NPU 执行 → 精度比对 |
| `gen` | 仅生成数据 |
| `cpu` | 仅跑 CPU golden |
| `npu` | 仅跑 NPU |
| `compare` | 仅精度比对 |

示例：`pytest test_quant_flash_mla_func_rdv.py -v --golden-mode=npu,compare`

### 3. pytest paramset → TTK CSV

```bash
python pytest_paramset_to_csv.py --paramset quant_flash_mla_paramset_debug --output qfmla_pytest.csv
```

## TTK 执行流程

```
TTK 读取 CSV
  │
  ├─ 1. customize_inputs (assets/impl/inputs.py)
  │     ├─ 设置 golden 全局变量（B, N_q, D, softmax_scale, SCALE_VALUE 等）
  │     ├─ generate_data() 生成 BNSD FP8 数据（CPU，与 golden 同种子）
  │     ├─ convert_q_bnsd_to_layout() → TND 布局
  │     ├─ 构建 PA K cache（BnNBsD / BnBsND / NZ）
  │     └─ 写入 10 个 tensor slot
  │
  ├─ 2. npu_preprocess (assets/impl/npu_preprocess.py)
  │     └─ 调 quant_flash_mla_metadata() → 回填 metadata slot
  │        （支持动态槽位 -1,-1 和静态槽位 resize_/copy_）
  │
  ├─ 3. NPU 执行
  │     └─ torch.ops.cann_ops_transformer.quant_flash_mla() → attn_out, lse_out
  │
  ├─ 4. golden (assets/impl/golden.py)
  │     ├─ 设置相同全局变量（含 SCALE_VALUE = softmax_scale）
  │     ├─ generate_data() 生成相同 BNSD FP8 数据
  │     ├─ fp8_fullquant_mla_golden() CPU 参考计算
  │     └─ 转目标 layout → bf16
  │
  └─ 5. compare (assets/impl/compare.py)
        └─ isclose(rtol=0.0078125, atol=0.0001) → PctThd=99.5% → Pass/Fail
```

## 10 个 tensor slot

| # | 名称 | dtype | 说明 |
|---|------|-------|------|
| 0 | q | float8_e4m3fn | Q 矩阵，TND layout `(T, N, D)` |
| 1 | k_cache | float8_e4m3fn | KV cache，PA layout |
| 2 | q_descale | float32 | Q 反量化 scale，shape `(T, N)` |
| 3 | k_descale | float32 | K 反量化 scale，shape `(1,)` |
| 4 | block_table | int32 | KV cache block 索引 |
| 5 | cache_seqlens | int32 | 每 batch KV cache 实际长度 |
| 6 | cu_seqlens_q | int32 | TND: 每 batch Q 的累积偏移 |
| 7 | seqused_q | int32 | 每 batch Q 的实际 token 数 |
| 8 | attn_mask | int8 | mask_mode=3 时提供 causal mask `(2048, 2048)` |
| 9 | metadata | int32 | metadata 算子生成的调度信息 |

## metadata slot 三种模式

| Excel metadata_shape | CSV 输出 | 行为 |
|----------------------|-----------|------|
| `-1,-1` | `(-1, -1)` | 动态槽位，npu_preprocess 生成并返回新 tensor |
| 具体值如 `2,4096` | `(2, 4096)` | 静态槽位，npu_preprocess resize_ + copy_ |
| 空 / `0,0` | `(2, 4096)` | 默认值（当前 NPU metadata op 实际输出） |

## 关键约定

| 项目 | 值 |
|------|-----|
| layout_q / layout_out | TND（强制） |
| quant_mode | 1（FP8_E4M3FN），原 HIFP8 自动转换 |
| TND padding | T = B × Q_S（padded），cu_seqlens_q[-1] = T |
| 精度阈值 | rtol=0.0078125, atol=0.0001, PctThd=99.5% |
| softmax_scale | Excel 指定时透传到 golden SCALE_VALUE；未指定时默认 1/sqrt(D+D_rope) |

## 环境依赖

- CANN 9.2.0（`/home/wangke293/Ascend_0909/cann`）
- custom_transformer vendor（`/home/z00954345/Ascend_0909_custom/vendors/custom_transformer`）
- conda env `zxx`（Python 3.x + torch + torch_npu）
- NPU: Ascend950DT_9582
