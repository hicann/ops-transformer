# lightning_indexer_v2算子测试框架

## 功能说明

基于pytest测试框架，实现lightning_indexer_v2算子的功能验证：

- **CPU侧**：复现算子功能用以生成golden数据
- **NPU侧**：通过TorchNPU进行算子直调获取实际数据
- **精度对比**：进行CPU与NPU结果的精度对比验证算子功能
- **双模式执行隔离**：支持直接pytest多进程执行和shell层进程隔离两种批量模式
- **batch_exec 模式**：根据 Excel 表格筛选已有 pt 文件直接执行 NPU 测试（跳过 pt 生成步骤）
- **性能采集**：支持挂载msprof采集算子性能数据并汇总输出
- **运行模式切换**：支持eager直接调用和graph（torch.compile + torchair）两种算子调用模式

## 当前实现范围

### 参数限制

- **数据格式**:
  - **query_layout**：BSND、TND
  - **key_layout**：BSND、TND、PA_BBND

- **运行模式**:
   - **eager**：直接调用 `torch.ops.cann_ops_transformer.lightning_indexer`
   - **graph**：通过 `torch.compile` + `torchair` 后端编译执行（需torchair支持）

### 环境配置

#### 前置要求

1、 确认TorchNPU为最新版本
2、 参考Attention融合算子Experimental使用说明激活CANN包和自定义算子包

#### custom包调用

支持custom包调用

## 文件结构

#### pytest文件结构说明

- test_run_sh                                  # 执行脚本，支持single/candidate/candidate_reject/batch/batch_exec五种命令
- batch_isolated_run.sh                        # 批量隔离执行脚本（shell层进程隔离+msprof性能采集）
- lightning_indexer_v2_golden.py               # cpu侧算子golden实现
- lightning_indexer_v2_acl_graph.py            # graph模式torchair后端实现
- result_compare_method.py                     # cpu golden与npu输出精度对比
- collect_perf_data.py                         # msprof性能数据收集与汇总
- pytest.ini                                   # 创建测试标记

单用例测试：

- test_lightning_indexer_v2_single.py       # pytest测试单用例运行主程序
- test_lightning_indexer_v2_paramset.py     # 单用例入参配置（A2/arch22 老功能矩阵，仅 ascend910b/910_93 启用）
- test_lightning_indexer_v2_stc.py          # STC白盒用例配置（仅 Ascend950 启用）
- test_lightning_indexer_v2_candidate.py    # candidate（两级TopK候选源）单用例功能套件
- test_lightning_indexer_v2_candidate_reject.py  # candidate 属性 host 校验拒绝用例

批量测试：

- test_lightning_indexer_v2_batch.py        # 用例批量测试主程序并生成excel文件保存结果
- ./batch/lightning_indexer_v2_pt_loadprocess.py    # 读取pt文件并调用算子获取npu输出
- ./batch/lightning_indexer_v2_pt_save.py           # 读取excel表格批量生成用例pt文件
- ./batch/list_pt_from_excel.py                     # 从Excel提取Testcase_Name并按名匹配pt文件（batch_exec模式用）
- ./batch/replace_path.py                           # test_lightning_indexer_v2_batch.py占位符替换

## 架构说明

- **single 模式**：`liv2_output_acl_graph` 调用 `liv2_output_single(is_batch=True)` 即时生成数据 → `torch.compile` + `torchair` 执行
- **batch 模式**：从 Excel 生成 .pt 文件 → 读取 pt 执行 NPU 测试和精度对比 → 生成 result.xlsx
- **batch_exec 模式**：从 Excel 筛选已有 .pt 文件 → 跳过 pt 生成 → 直接执行 NPU 测试和精度对比 → 生成 result.xlsx
- 两路共用 `_liv2_prepare_tensors_and_metadata` 和 `_liv2_run_compiled_graph`，统一使用 `fullgraph=False`

## 使用方法

在pytest文件夹路径下执行：

### SuperKernel

```bash
bash test_run.sh single --paramset stc --super-kernel
bash test_run.sh batch -E ./excel/my.xlsx -S Sheet2 -P ./pt_path --super-kernel
```

批跑时在原命令末尾加 `--super-kernel`；默认沿用原执行模式。metadata 仍在图外生成，
启用时保留单算子满核 debug（`debug_per_op_max_core_num=1`，不是限制为一个核）。

### 运行测试用例

#### 单用例调测

`test_lightning_indexer_v2_paramset.py` 为 A2（ascend910b/910_93）老功能矩阵参数集
（29 组：dtype/layout 四组合、S1/G、S2 边界、topk 含 over-2K（1024 倍数）、mask/cmp_ratio、
变长、PA 块边界、return_value）；Ascend950 环境自动为空（走 STC 参数集）。默认参数集适合
临时调试时先用 `-C` 指定单条用例：

``` bash
bash test_run.sh single
bash test_run.sh single -C FP16_01,TND_04
```

STC 参数集位于 `test_lightning_indexer_v2_stc.py`，用于按代码路径执行白盒用例（仅
Ascend950 启用）：

``` bash
bash test_run.sh single --paramset stc
```

> 已知 A2 语义分歧（默认关闭）：TND + seqused_k 覆盖（seqused_k < cu_seqlens 差分）时，
> arch22 kernel 按 cu_seqlens 差分推导 k 有效长度、忽略 seqused_k 覆盖（BSND 路径正确
> 响应；950 kernel 两侧均响应，950 STC 的 ACTUAL_31 为同形态用例）。对应用例
> `ACTUAL_TND_SEQUSED_23` 默认不启用，kernel 修复后设 `LIV2_ENABLE_TND_SEQUSED_K=1`
> 启用回归验证。

可以通过 `-C` 或 `--cases` 指定一个或多个用例，多个名称使用逗号分隔：

``` bash
# 指定单组合 case
bash test_run.sh single --paramset stc -C META_TRAILING_K_ZERO_41

# 指定多组合 case，运行其展开的全部节点
bash test_run.sh single --paramset stc -C S2_16

# 指定其中一个展开节点；_003 对应 k_seq=128
bash test_run.sh single --paramset stc -C S2_16_003

# 一次运行多个指定用例
bash test_run.sh single --paramset stc -C S1_TOPK_14,META_TRAILING_Q_ZERO_42

# 指定 graph 模式；默认使用 eager 模式
bash test_run.sh single --paramset stc -C META_BSND_53_002 -M graph
```

#### 用例的批量生成与测试

##### 方式A：test_run.sh 批量执行

1、excel路径下存放用例excel表格

2、修改test_run.sh配置区中的DEFAULT_EXCEL（Excel表格路径）和DEFAULT_PT_PATH（pt文件存放路径），或通过命令行参数指定

3、执行指令：

``` bash
bash test_run.sh batch                                    # 使用配置区默认值（eager 模式）
bash test_run.sh batch -E ./excel/test_cases.xlsx        # 指定 Excel 路径
bash test_run.sh batch -S Sheet1 -P ./pt_path            # 指定 Sheet 和 pt 目录
bash test_run.sh batch -M graph                          # graph 模式
bash test_run.sh batch -E ./excel/my.xlsx -S Sheet2 -M graph  # 组合使用
bash test_run.sh batch graph                             # 兼容旧用法：直接跟 run_mode
```

4、配置区默认值：

| 变量 | 默认值 | 命令行参数 | 说明 |
|---|---|---|---|
| DEFAULT_EXCEL | `./excel/test_cases.xlsx` | `-E` | Excel 用例表格路径（**必须指定具体文件名**，不支持通配符如 `./excel/*`） |
| DEFAULT_PT_PATH | `./pt_path` | `-P` | pt 文件存放目录 |
| （无） | `Sheet1` | `-S` | Excel Sheet 页名 |
| （无） | `eager` | `-M` | 运行模式（eager/graph） |

#### 根据 Excel 表格筛选已有 pt 批量执行（batch_exec 模式）
>
> 仅重新执行 NPU 测试和精度对比，不重新生成 pt 文件。适用于已有 pt 文件、只需更新精度结果的场景。

1、确认 pt_path 下已有对应 .pt 文件（可通过 `batch` 模式或手工生成）

2、执行指令：

``` bash
bash test_run.sh batch_exec                                    # 默认: excel/test_cases.xlsx, Sheet1, pt_path
bash test_run.sh batch_exec -E ./excel/test_cases.xlsx         # 指定 Excel 路径
bash test_run.sh batch_exec -S Sheet1 -P ./pt_path             # 指定 Sheet 和 pt 目录
bash test_run.sh batch_exec -E ./excel/test_cases.xlsx -M graph  # graph 模式
```

3、执行流程：

- 从 Excel 表格读取 `Testcase_Name` 列
- 按 `<Testcase_Name>.pt` 在 pt_path 下匹配对应的 .pt 文件
- 仅对匹配到的 .pt 文件执行 NPU 测试和精度对比
- 生成 `result.xlsx` 测试结果表格
- 如果 Excel 中某条用例无对应的 .pt 文件，会输出警告并跳过该用例

4、与 `batch` 模式的区别：

|  | `batch` | `batch_exec` |
|---|---|---|
| pt 生成 | 每次重新生成 | 跳过 |
| 执行速度 | 较慢（含 pt 生成） | 较快 |
| 适用场景 | 首次运行 / 参数变更 | 精度复测 / 仅 NPU 结果更新 |

##### 方式B：手工分步执行

1、生成pt文件：

``` bash
python3 batch/lightning_indexer_v2_pt_save.py excel/test_cases.xlsx pt_path
python3 batch/lightning_indexer_v2_pt_save.py excel/test_cases.xlsx pt_path --sheet Sheet1  # 指定 Sheet 页
```

2、执行测试（pt目录默认读取 `./pt_path`，其他目录可通过 `LIV2_TESTCASE_DIR` 环境变量指定）：

``` bash
python3 -m pytest -rA -s test_lightning_indexer_v2_batch.py -v -m ci -W ignore::UserWarning -W ignore::DeprecationWarning
```

##### 方式C：批量隔离执行（推荐用于性能采集）

对每条用例单独拉起一个pytest进程，实现进程间完全隔离，避免单条用例崩溃影响其他用例。

``` bash
bash batch_isolated_run.sh ./pt_path 0         # 不采集性能
bash batch_isolated_run.sh ./pt_path 1         # 采集性能（挂载msprof）
bash batch_isolated_run.sh ./pt_path 0 graph   # graph模式 + 不采集性能
bash batch_isolated_run.sh ./pt_path 1 graph   # graph模式 + 性能采集
```

## Excel 用例表格式

`excel/test_cases.xlsx` 需包含以下列（Sheet1）：

| 列名 | 类型 | 示例 |
|---|---|---|
| Testcase_Name | str | `test_case_01` |
| batch_size | int | `8` |
| q_seq | int | `15` |
| k_seq | int | `111` |
| q_t_size | int | `8` |
| k_t_size | int | `15` |
| q_head_num | int | `64` |
| k_head_num | int | `1` |
| head_dim | int | `128` |
| block_size | int | `512` |
| block_num | int | `8` |
| qk_dtype | str | `FLOAT8_E4M3FN` / `INT8` / `HIFLOAT8` |
| cu_seqlens_q | None/str | `None` 或 `"[0, 1]"` |
| cu_seqlens_k | None/str | `None` 或 `"[0, 1]"` |
| seqused_q | None/str | `None` 或 `"[3,3,3,3,3,3,3,3]"` |
| seqused_k | str | `"[28,24,80,96,47,76,0,111]"` |
| cmp_residual_k | None/str | `None` 或 `"[0,0,0,0,0,0,0,0]"`（cmp_ratio>1时必填）|
| max_seqlen_q | int | `-1` |
| layout_q | str | `BSND` / `TND` |
| layout_k | str | `PA_BBND` |
| topk | int | `512` |
| mask_mode | int | `0` / `3` |
| query_datarange | str | `"[-448,448]"` |
| key_datarange | str | `"[-20,20]"` |
| weights_datarange | str | `"[-123,123]"` |
| cmp_ratio | int | `1` / `4` |
| return_value | int | `0` / `1` |
| output_idx_offset | None/str | `None` 或列表字符串 |

**注意事项**：

- `cmp_ratio > 1`且`sparse_mode != 0`时，`cmp_residual_k`必填（长度=batch_size的列表）

## 输出文件

| 文件 | 说明 |
|---|---|
| `result.xlsx` | 测试结果（精度、参数等） |
| `result_perf.xlsx` | 测试结果 + 性能数据（仅msprof模式） |
| `batch_summary.log` | 批量执行详细日志 |
| `batch_fail_list.log` | 失败用例清单 |
| `PROF_*/` | msprof性能原始数据目录 |

## candidate（两级 TopK candidate source）单用例功能与 host 拒绝套件

candidate 模式（`candidateTopkBlocks != -1`）在模型场景套之外另有轻量单用例矩阵与
host 校验拒绝套件，聚焦 candidate 属性面与 layout 泛化（含 TND）：

### 文件说明

- `test_lightning_indexer_v2_candidate.py`  # candidate 单用例功能套件（26 条用例）
- `test_lightning_indexer_v2_candidate_reject.py`  # candidate 属性 host 校验拒绝用例（11 条）

### 功能套件覆盖点（`test_lightning_indexer_v2_candidate.py`）

| 维度 | 用例 |
|---|---|
| candidate_block_size 全 6 档 | 2 / 4 / 8 / 16 / 32 / 64 |
| candidate_topk_blocks 边界 | 64（下界）/ 2048（上界）/ numBlocks<candBlocks 满覆盖退化 |
| layout 泛化 | BSND×BSND / TND×TND / BSND×PA_BBND / TND×PA_BBND（TND 输出 [T,N2,K]） |
| mask / cmp_ratio | mode3+cmp2（含残差）/ cmp128 上界 |
| dtype / 变长 | BF16 / B>1 变长 seqused_k（含 2049 非对齐） |
| topk 边界 | 1（下界）/ 2048（candidate 开启时上界） |
| return_value 维度（已放开） | rv=1：BSND 基线 / TND 变体 / mask3+cmp2（values shape==indices、float32、按同族口径 isclose）；rv=0 恒空 (0,) 回归 |
| rv 一致性 | 同输入 rv=0/rv=1 双跑：sparse_indices 与 candidate_block_indices 逐位一致 |
| 其他 | S2 尾块（513）/ S1 大行数 / off（-1）回归（输出 (0,) 且 sparse 不变） |

Per case 校验三件套与模型套一致：sparse 回归（source 不得改变位置级 topk）、
candidate_block_length 恒空、candidate_block_indices 集合比对。

### host 拒绝套件覆盖点（`test_lightning_indexer_v2_candidate_reject.py`）

| 规则 | 用例 |
|---|---|
| C3' candBlocks 非法 | 100（非 64 倍数）/ 2112（>2048）/ -2（非 -1 负值） |
| C4' blockSize 非法 | 12（非 2 幂）/ 1（<2）/ 128（>64）/ off 时同样拒绝（恒校验） |
| V1 topk 联动 | candidate 开启 + topk=3072（>2048） |
| V2 rv 值域越界 | candidate 开启 + return_value=2（TORCH_CHECK "Return_value only supports 0 or 1"） |
| 合法反向 | candBlocks=64 + bs=2 + topk=64 必须被接受；candidate 开启 + return_value=1（已放开）必须被接受 |

### 运行

```bash
# 全套（Ascend910B / 910_93）
bash test_run.sh candidate
bash test_run.sh candidate_reject

# 指定用例
bash test_run.sh candidate -C LIV2_CAND_BS_2,LIV2_CAND_TND_TND
# 或直接 pytest
LIV2_CAND_CASE_NAMES=LIV2_CAND_OFF_REGRESSION python3 -m pytest -s test_lightning_indexer_v2_candidate.py -m ci
```

## candidate（两级 TopK candidate source）模型场景套件

`candidateTopkBlocks != -1` 时开启的两级 TopK 候选源（source）模式有独立的模型场景测试套，与上述 single/paramset/batch 套件互不影响。

### 文件说明

- `liv2_candidate_golden.py`  # candidate golden 纯 CPU 实现（`select_candidate_blocks_ref` numpy/torch 双实现，仅依赖 numpy+torch，可在无 NPU 环境导入自检）
- `test_lightning_indexer_v2_model.py`  # candidate 模型场景测试套（10 条默认用例 + `LIV2_MODEL_LARGE=1` 追加重型边界用例）
- `lightning_indexer_v2_golden.py`（尾部 candidate 段）  # candidate golden + NPU 驱动（`liv2_output_candidate_single` / `liv2_candidate_host_reject` 950 拒绝探针 / `resolve_candidate_op`）
- `result_compare_method.py`（尾部 candidate 段）  # `check_result_candidate` 候选比较器

### 模型场景规格（BSND）

| 参数 | 取值 |
|---|---|
| layout | BSND×BSND（泛化：BSND×PA_BBND） |
| N1 (q_head_num) | 32 / 16 / 8 / 4 |
| D (head_dim) | 128 |
| topK | 512 |
| candidate_topk_blocks | 2048 |
| candidate_block_size | 8 |
| S1 / S2 | 4096 / 4096（S2 为压缩后 K 长度；`LIV2_MODEL_LARGE=1` 追加 16K 满覆盖边界与 24K 真实筛选） |
| cmp_ratio | 2 |
| sparse_mode (mask_mode) | 0 / 3 |
| 其余泛化 | dtype FP16/BF16、B>1 变长 seqused_k、PA_BBND key |

用例清单（默认 10 条，`LIV2_MODEL_LARGE=1` 追加 2 条重型边界用例）：

| 用例 | 覆盖点 |
|---|---|
| LIV2_MODEL_N1_32_MODE3 | 主规格：B=1，mask_mode=3 causal，cmp_residual_k=[1] |
| LIV2_MODEL_N1_16_MODE3 | N1=16，B=2 |
| LIV2_MODEL_N1_8_MODE3 | N1=8，B=2 |
| LIV2_MODEL_N1_4_MODE3 | N1=4，B=4 |
| LIV2_MODEL_N1_32_MODE0 | sparse_mode=0（no mask），cmp_residual_k=None |
| LIV2_MODEL_N1_4_MODE0 | sparse_mode=0，N1=4，B=2 |
| LIV2_MODEL_N1_32_MODE3_BF16 | BF16 dtype |
| LIV2_MODEL_N1_8_MODE0_BF16 | BF16 + sparse_mode=0 |
| LIV2_MODEL_VARLEN_MODE3 | B=2 seqused_k=[4096, 2049]（S2 非对齐） |
| LIV2_MODEL_PA_MODE3 | BSND q × PA_BBND k（block_table） |
| LIV2_MODEL_S2_16K_N1_4_MODE3 | k_seq=16384（numBlocks==candidate_topk_blocks 满覆盖边界），需 `LIV2_MODEL_LARGE=1` |
| LIV2_MODEL_S2_24K_N1_4_MODE3 | k_seq=24576（numBlocks>candidate_topk_blocks 块级真实筛选），需 `LIV2_MODEL_LARGE=1` |

Per case 校验三件套：

1. `sparse_indices` vs golden（`check_result`；source 不得改变位置级 topk）；
2. `candidate_block_length` 恒空 (0,)（预留接口）；
3. `candidate_block_indices` vs golden（`check_result_candidate`；行级块号集合 + -1 槽数硬校验）。

### candidate 比对契约（`check_result_candidate`）

candidate 输出是**无序块号集合**（归并排序 tie-break 不保证与 torch 一致），比对契约为五层：

1. multiset 门：有效（非 -1）块号集合相等 → PASS（顺序不敏感）；
2. 边界容差：集合不等但每个交换块对的 golden blkScore 在 `thres` 相对容差内 → PASS（bf16 tie 仲裁）；
3. -1 槽位硬校验：每行 -1 个数必须精确相等，否则 FAIL；
4. 重复块硬校验：行内出现重复块号 → FAIL；
5. pin 块硬校验：golden +inf（最新可达 token 所在块）必须出现在 NPU 输出行内，否则 FAIL。

### 运行

```bash
# 全套（Ascend910B / 910_93，需已编译 candidate 算子）
python3 -m pytest -rA -s test_lightning_indexer_v2_model.py -v -m ci

# 指定用例
LIV2_CASE_NAMES=LIV2_MODEL_N1_32_MODE3 python3 -m pytest -s test_lightning_indexer_v2_model.py -m ci

# 追加 16K 满覆盖边界 + 24K 真实筛选重型用例
LIV2_MODEL_LARGE=1 python3 -m pytest -s test_lightning_indexer_v2_model.py -m ci
```

> 说明：candidate source 仅支持 arch22（Ascend 910B / Ascend 910_93）；arch35（Ascend 950）上
> candidate-on 被 host 校验拒绝（tiling 入口守卫返回 GRAPH_FAILED），本套件在 950 环境直接
> skip。candidate torch 算子未随包构建/注册时自动 skip（`candidate_ops_unavailable_reason`）。
