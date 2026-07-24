# FlashAttention精度测试框架

## 一、架构概览

```
test_flash_attn.py  (入口)
        │
        ├── case_loader.load_case_modules()    ← test_cases/*.py加载 + 笛卡尔积展开
        ├── case_loader.resolve_case_ids()     ← --case_id过滤
        │
        ▼  逐case循环
        ├── case_loader.normalize_params()     ← 补全默认值、TND/PA特殊处理
        │
        ▼
        ├── orchestrator.run_case()
        │       │
        │       ├── data.flash_attn_inputs.generate()   ← 声明式张量生成(Q/K/V)
        │       ├── data.build_flash_attn_params()      ← 构造metadata/kernel kwargs
        │       │
        │       ├── primary.compute(inputs, params)     ← NPU/GPU后端
        │       ├── golden.compute(cpu_inputs, params)  ← CPU参考(tfoward)
        │       │
        │       └── compare.check_result()              ← 精度对比
        │
        └── reporter.record() → 终端表格 + CSV
```

**核心模块** (`core/`):

| 模块 | 职责 |
|------|------|
| `backends/base.py` | Backend抽象基类(`is_available`, `compute`, `device`) |
| `backends/cpu.py` | CPU Golden —调用`backends/cpu_impl.tforward`做参考计算 |
| `backends/gpu.py` | GPU —封装`flash_attn`库 |
| `backends/npu.py` | NPU —调用`flash_attn_metadata` + `flash_attn`，含`print_metadata` |
| `data.py` | `InputSpec`声明式张量生成 + `build_flash_attn_params`参数组构造 |
| `case_loader.py` | 用例加载(`load_case_modules`)、参数规范化(`normalize_params`)、ID解析(`resolve_case_ids`) |
| `orchestrator.py` | 编排器—串联数据生成 → 后端计算 → 精度对比 |
| `reporter.py` | `Reporter` —终端表格 + 增量CSV输出 |

---

## 二、目录结构

```
pytests/
├── core/                           # 核心模块
│   ├── backends/
│   │   ├── base.py                 # Backend抽象基类
│   │   ├── cpu.py                  # CPU Golden (调用tforward)
│   │   ├── gpu.py                  # GPU (flash_attn库)
│   │   └── npu.py                  # NPU (flash_attn + print_metadata)
│   ├── data.py                     # InputSpec张量生成 + build_flash_attn_params
│   ├── case_loader.py              # 用例加载 + 参数规范化
│   ├── orchestrator.py             # 执行编排
│   └── reporter.py                 # 结果报告(终端 + CSV)
├── backends/                       # 底层实现(被core/backends调用)
│   ├── cpu_impl.py                 # tforward — CPU参考实现
│   ├── gpu_impl.py                 # flash_attn_gpu — GPU实现
│   └── npu_impl.py                 # NPU辅助函数
├── test_cases/                     # 测试用例定义
│   ├── functional_stc.py           # 功能STC用例
│   ├── performance_redline_train.py
│   ├── performance_redline_infer.py
│   ├── functional_redline_train.py
│   ├── functional_redline_train_tnd.py
│   ├── functional_redline_infer.py
│   ├── base.py                     # 基础用例
│   └── fia_stc.py                  # FIA STC用例
├── utils/                          # 工具函数
│   ├── compare.py                  # 精度对比 + 失败分布分析
│   ├── data.py                     # 数据生成、mask、layout转换
│   ├── io.py                       # Tensor文件读写
│   ├── perf_parser.py              # 性能数据解析
│   ├── perf_runner.py              # 性能测试运行器
│   └── precision_visual.py         # 精度热力图可视化
├── tools/
│   └── xlsx_to_testcase.py         # Excel → test case转换
├── test_flash_attn.py              # 主入口
├── test_npu_metadata.py            # metadata算子独立测试
└── readme.md
```

---

## 三、环境依赖

### 通用

| 依赖 | 说明 |
|------|------|
| Python 3.8+ | — |
| PyTorch 2.2+ | CPU/GPU计算基础 |
| `einops` | Layout转换(`pip install einops`) |
| `numpy` | 数值计算 |

### NPU模式

| 依赖 | 说明 |
|------|------|
| `TorchNPU` | PyTorch NPU扩展 |
| `cann_ops_transformer` | 提供`flash_attn`和`flash_attn_metadata` |

### GPU模式(可选)

| 依赖 | 说明 |
|------|------|
| `flash-attn` | FlashAttention库(`pip install flash-attn --no-build-isolation`) |
| CUDA 12.0+ | GPU环境 |

### 环境检查

```bash
# NPU
python -c "import torch_npu, cann_ops_transformer; print('NPU OK')"

# GPU
python -c "import torch, flash_attn, einops; print('GPU OK')"

# CPU-only
python -c "import torch, einops, numpy; print('CPU OK')"
```

---

## 四、运行方式

### 4.1 精度测试(默认)

```bash
# 运行所有case (NPU模式)
python test_flash_attn.py

# 指定case
python test_flash_attn.py --case_id TND_05,TND_06

# GPU模式
python test_flash_attn.py --case_id BASE_01 --use_gpu

# 指定case文件
python test_flash_attn.py --case_files functional_stc --case_id TND_05
```

### 4.2 Metadata调试

只调用AICPU metadata算子并打印分核结果，跳过kernel执行：

```bash
python test_flash_attn.py --case_id TND_05 --meta_only
```

输出包含：

- **Header**— `sectionNum`、`isFD`、`mBaseSize`、`s2BaseSize`
- **FA Metadata**—每个section的36个AIC core分核信息(BN2/M/S2的start/end)
- **FD Metadata**—活跃AIV core归约任务(仅M_NUM > 0)

```
==============================================================================
  [Metadata Header]
  sectionNum  = 1
  isFD        = 1
  mBaseSize   = 128
  s2BaseSize  = 64
------------------------------------------------------------------------------
  [Section 0] FA Metadata — AIC cores  (36 cores × 16 slots, 7 fields used)
------------------------------------------------------------------------------
  Core       BN2_START      M_START     S2_START      BN2_END        M_END       S2_END  FIRST_FD_WS_IDX
  AIC00              0            0            0              7          127           63                0
  ...
```

### 4.3 性能测试

```bash
# 批量性能采集
python test_flash_attn.py --case_files performance_redline_train --perf_mode --perf_output ./perf_out

# 逐个执行
python test_flash_attn.py --perf_mode --one_by_one
```

### 4.4 三方对比

```bash
# CPU vs GPU vs NPU实时对比
python test_flash_attn.py --case_id BASE_01 --compare_mode --use_gpu

# 离线对比(从dump文件)
python test_flash_attn.py --case_id BASE_01 --compare_mode \
    --load_gpu_dump ./dump/BASE_01/gpu_out.txt \
    --load_npu_dump ./dump/BASE_01/npu_out.txt
```

---

## 五、命令行参数

### 基础参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--case_files` | str | 自动扫描`test_cases/` | 用例文件，逗号分隔 |
| `--case_id` | str | `all` | 用例名，逗号分隔，支持前缀匹配 |
| `--device_id` | int | `0` | NPU设备ID |
| `--use_gpu` | flag | — | 使用GPU后端 |
| `--gpu_device` | int | `0` | GPU设备ID |

### Golden 持久化参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--save_golden` | str | — | 正常对比流程 + 将 inputs/golden 落盘到指定目录（.pt 格式） |
| `--load_golden` | str | — | 从指定目录加载 golden，只跑设备对比（跳过 CPU golden 计算） |
| `--case_timeout` | int | `300` | 子进程隔离模式下单个 case 超时时间（秒） |

### 调试参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--meta_only` | flag | — | 只调用 metadata 算子，打印分核结果 |
| `--verbose_diff` | flag | — | 精度不达标时输出全部超阈值元素（默认只打前10个） |
| `--visualize` | flag | — | 精度不达标时生成热力图 PNG |
| `--viz_dir` | str | `./viz_output` | 热力图/分析报告保存目录 |
| `--fail_analysis` | flag | — | 精度不达标时输出多维度失败分布分析报告 |
| `--compare_mode` | flag | — | 多后端对比 |
| `--graph_mode` | flag | — | 图模式执行 |

### 输出参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--result_csv` | str | `result.csv` | 精度结果CSV路径 |
| `--report_interval` | int | `20` | 中间统计打印间隔 |
| `--skip_accuracy` | flag | — | 跳过精度对比 |

### 性能参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--perf_mode` | flag | — | 性能测试模式 |
| `--perf_runs` | int | `5` | 性能采集轮数 |
| `--perf_cold_thr` | int | `16` | 冷/热case分界(S1) |
| `--perf_output` | str | `./perf_output` | 性能输出目录 |
| `--one_by_one` | flag | — | 逐个case执行 |

### 离线对比参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--load_gpu_dump` | str | None | GPU dump文件路径 |
| `--load_npu_dump` | str | None | NPU dump文件路径 |

---

## 六、测试用例格式

用例定义在`test_cases/*.py`的`TestCases`字典中。每个字段是列表，框架自动做笛卡尔积展开。

### 6.1 字段说明

```python
TestCases = {
    "MY_CASE": {
        # ── 必填 ──
        "B":        [2],          # batch size (TND下可省略)
        "N1":       [8],          # query head数
        "D":        [128],        # head dim
        "Dtype":    ["fp16"],     # fp16 / bf16

        # ── Layout ──
        "layout_q":   ["BNSD"],   # BNSD / BSND / TND
        "layout_kv":  ["BNSD"],   # BNSD / BSND / TND / PA_BBND / PA_BNBD
        "layout_out": ["BNSD"],   # BNSD / BSND / TND

        # ── 序列长度 ──
        "S1":       [64],         # query seq len (TND下为max_seqlen_q)
        "S2":       [1024],       # KV seq len (TND下为max_seqlen_kv)

        # ── 可选 ──
        "N2":       [1],          # KV head数，默认 = N1
        "DV":       [128],        # value head dim，默认 = D
        "scale":    [None],       # 缩放系数，默认1/sqrt(D)
        "mask_mode":[0],          # 0=无mask, 3=causal, 4=band+causal
        "win_left": [-1],
        "win_right":[-1],
        "return_softmax_lse": [False],

        # ── Q/K/V值域 ──
        "q_range": [(-5.0, 5.0)],
        "k_range": [(-5.0, 5.0)],
        "v_range": [(-5.0, 5.0)],

        # ── TND专用 ──
        "cu_seqlens_q":  [[None]],   # 累积序列长度(B+1个元素)
        "cu_seqlens_kv": [[None]],
        "seqused_q":     [[None]],   # 每batch实际序列长度
        "seqused_kv":    [[None]],
    },
}
```

### 6.2 TND用例

```python
"TND_05": {**BASE,
    "layout_q": ["TND"], "layout_kv": ["TND"], "layout_out": ["TND"],
    "cu_seqlens_q":  [[0, 64, 128, 192, 256, 320, 384, 448, 512]],
    "cu_seqlens_kv": [[0, 1024, 2048, 3072, 4096, 5120, 6144, 7168, 8192]]},
```

`normalize_params`会自动从`cu_seqlens`推导`seqused` (差分)和`B` (len-1)。

### 6.3 Paged Attention用例

```python
"PA_07": {**BASE,
    "layout_kv": ["PA_BBND", "PA_BNBD"],
    "block_size": [128]},
```

`normalize_params`自动从`seqused_kv` + `block_size`生成`block_table`。

---

## 七、执行流程详解

### 7.1 用例加载

```
test_cases/functional_stc.py::TestCases
        │
        ▼ load_case_modules()
  笛卡尔积展开: 每个参数组合 → 独立case
  命名: "test_cases.functional_stc/TND_05"
  多组合: "test_cases.functional_stc/GS1_14_0", "..._1", ...
        │
        ▼ resolve_case_ids()
  --case_id TND_05 → 匹配所有包含"/TND_05"的case
```

### 7.2 参数规范化

`normalize_params()`补全逻辑：

| 条件 | 自动补全 |
|------|----------|
| 缺少`N2` | `N2 = N1` |
| 缺少`S2` | `S2 = S1` |
| 缺少`DV` | `DV = D` |
| TND + 缺少`seqused_q` | 从`cu_seqlens_q`差分推导 |
| TND + 缺少`cu_seqlens_kv` | 复制`cu_seqlens_q` |
| PA + 缺少`block_table` | 从`seqused_kv` + `block_size`自动生成 |

### 7.3 张量生成

`InputSpec`声明式定义Q/K/V的shape、dtype、生成方法：

```python
flash_attn_inputs = (
    InputSpec("flash_attn")
    .tensor("q", layout={"BNSD": "B,N1,S1,D", "BSND": "B,S1,N1,D", "TND": "total_s1,N1,D"})
    .tensor("k", layout={...}, layout_override="layout_kv")
    .tensor("v", layout={...}, layout_override="layout_kv")
)
```

特殊token：`total_s1` = `cu_seqlens_q[-1]`，`total_s2` = `cu_seqlens_kv[-1]`。

### 7.4 参数组构造

`build_flash_attn_params()`返回`(meta_kwargs, kernel_kwargs, out_layout)`：

- **meta_kwargs** → 传给`flash_attn_metadata()`
  - `cu_seqlens_q/kv`, `seqused_q/kv`, `max_seqlen_q/kv`, `num_heads_q/kv`, `head_dim`, `batch_size`, `mask_mode`, `layout_*`
- **kernel_kwargs** → 传给`flash_attn()`
  - `softmax_scale`, `cu_seqlens_q/kv`, `seqused_q/kv`, `max_seqlen_q/kv` (TND时传实际值，否则 -1), `mask_mode`, `layout_*`

### 7.5 精度对比

`check_result()`判定标准：

```
单元素: diff ≤ max(|golden| × 0.5%, 0.000025)
整体:   超阈值元素占比 ≤ 0.5%
```

输出：`PASS` / `FAIL` + MaxAbsErr + MeanAbsErr + MaxRelErr + FailRatio。

---

## 八、Layout支持矩阵

| Layout | CPU | GPU | NPU | 说明 |
|--------|-----|-----|-----|------|
| **BNSD** | ✓ | ✓ | ✓ | (B, N, S, D)默认 |
| **BSND** | ✓ | ✓ | ✓ | (B, S, N, D) |
| **TND** | ✓ | ✓ | ✓ | (total_tokens, N, D)变长序列 |
| **PA_BBND** | ✗ | ✓ | ✓ | (num_blocks, block_size, N, D) Paged KV |
| **PA_BNBD** | ✗ | ✓ | ✓ | (num_blocks, N, block_size, D) Paged KV |

---

## 九、mask_mode

| 值 | 名称 | 说明 |
|----|------|------|
| `0` | NO_MASK | 全attention |
| `3` | RIGHT_DOWN_CAUSAL | 右下对齐因果mask |
| `4` | BAND_CAUSAL | BAND + CAUSAL混合，`win_left`/`win_right`生效 |

---

## 十、文件职责速查

| 文件 | 改动频率 | 典型改动 |
|------|----------|----------|
| `test_cases/*.py` | **经常** | 新增/修改测试case |
| `core/data.py` | 偶尔 | 参数组构造、张量生成规则 |
| `core/backends/npu.py` | 偶尔 | NPU调用、`print_metadata` |
| `test_flash_attn.py` | 偶尔 | CLI参数、功能入口 |
| `core/case_loader.py` | 很少 | 参数规范化逻辑 |
| `core/orchestrator.py` | 很少 | 执行流程编排 |
| `core/reporter.py` | 很少 | 报告格式 |
| `utils/compare.py` | 很少 | 精度对比、失败分析 |
| `utils/data.py` | 很少 | mask生成、layout转换 |
| `backends/cpu_impl.py` | 一般不改 | CPU golden (tfoward) |
| `backends/gpu_impl.py` | 很少 | GPU实现 |
| `utils/perf_*.py` | 很少 | 性能采集与解析 |

---

## 十一、常见问题

### Q1: `ModuleNotFoundError: No module named 'cann_ops_transformer'`

NPU模式需要安装`cann_ops_transformer`，或改用GPU模式：

```bash
python test_flash_attn.py --case_id BASE_01 --use_gpu
```

### Q2: TND case精度异常

检查`cu_seqlens_q`格式（必须包含前导0，长度 = batch + 1）：

```python
"cu_seqlens_q": [[0, 64, 128, 192]],  # 3个batch，各64 token
```

### Q3: TND case kernel挂死

确认`max_seqlen_q`/`max_seqlen_kv`正确传入。TND layout下kernel需要实际的max值：

```bash
python test_flash_attn.py --case_id TND_05 --meta_only  # 先检查metadata
```

### Q4: PA case `block_size`报错

不同后端对`block_size`有不同约束，请查阅对应后端的官方文档确认具体限制：

- GPU端参考`flash_attn`库文档
- NPU端参考`flash_attn`算子文档

### Q5: 如何查看某个case的实际参数

```bash
python -c "
from core.case_loader import load_case_modules, normalize_params
cases = load_case_modules(['test_cases.functional_stc'])
for name, raw in cases.items():
    if 'TND_05' in name:
        print(normalize_params(raw))
"
```
