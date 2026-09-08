# Chunk_gated_delta_rule算子TTK测试资产

## 功能说明

基于ops-test-kit（TTK）测试框架，实现Chunk_gated_delta_rule算子的E2E与ACLNN精度验证：

- **E2E模式**：通过TorchNPU Python API（`torch_npu.npu_chunk_gated_delta_rule`）直调或torch.compile图模式获取NPU实际数据，与CPU golden对比
- **ACLNN模式**：通过aclnn C API（`aclnnChunkGatedDeltaRule`）调用算子，与CPU golden对比
- **精度对比**：三方交叉校验（NPU vs golden vs benchmark），支持eager、aclgraph多种执行方式

## 当前实现范围

### 支持的测试方式

| 模式 | API | 说明 |
|------|-----|------|
| E2E eager | `torch_npu.npu_chunk_gated_delta_rule` | torch直调，含profiling |
| E2E aclgraph | 同上 | torch.compile reduce-overhead模式（npugraph_ex backend） |
| ACLNN | `aclnnChunkGatedDeltaRule` | aclnn C API调用 |

> **注意**：const graph（torchair GE backend）当前不支持，算子未在GE图引擎注册，执行会报 `Interface npu_chunk_gated_delta_rule is not supported in GE graph mode`。aclgraph模式走`npugraph_ex` backend，在aclnn层面捕获图，可正常执行。

### 精度标准

- tolerance声明社区标准`cross_check`（bfloat16），level `L1`。
- compare判定条件（`impl/compare.py`）：三方交叉校验，计算mare/mere/rmse/smra四项比率（NPU误差 / benchmark误差），阈值分别为 `CV_MAX_RE=5.0` / `CV_AVER_RE=1.5` / `CV_RMSE=1.5` / `CV_SMALL_VAL=2.0`，全部低于阈值方为Pass。比率分母floor为 `2^-8`（`_ERR_THRESHOLD`）。

### 环境配置

#### 前置要求

1. TorchNPU安装包下载路径（需及时更换为最新版本）：[TorchNPU安装教程](https://gitcode.com/Ascend/pytorch)
2. 完成环境安装和环境变量配置，具体操作请参考：[ops-transformer](../../../../README.md)
3. ops-test-kit测试框架（TTK）：[ops-test-kit](https://gitcode.com/cann/ops-test-kit)

## 文件结构

```
attention/chunk_gated_delta_rule/tests/assets/
├── convert_rdv_to_csv.py                        # RDV全量用例转TTK CSV脚本（E2E + ACLNN）
├── spec.py                                      # TestSpec：注册API + golden/inputs/tolerance/compare（e2e + aclnn）
└── impl/
    ├── golden.py                                # golden适配器（e2e + aclnn），加载CPU golden + 第三方benchmark
    ├── inputs.py                                # inputs适配器（e2e + aclnn），填充actual_seq_lengths + 约束裁剪
    └── compare.py                               # 三方交叉校验精度对比（output + state双输出）
```

### 文件职责

#### spec.py

- 定义`ChunkGatedDeltaRuleSpec`类（e2e），声明`golden`、`customize_inputs`、`tolerance`、`compare`。
- 定义`AclnnChunkGatedDeltaRuleSpec`类（aclnn），声明`golden`、`customize_inputs`、`tolerance`、`compare`。
- 通过`__spec__` dict注册：
    - `torch_npu.npu_chunk_gated_delta_rule` → `ChunkGatedDeltaRuleSpec`
    - `aclnnChunkGatedDeltaRule` → `AclnnChunkGatedDeltaRuleSpec`
- `compare`为自定义三方对比，golden计算时同时算benchmark并通过`_GOLDEN_CONTEXT`传递给compare。
- 动态加载`impl/`下三个模块，避免硬编码路径。

#### impl/golden.py

- 通过`importlib`加载CPU golden实现和第三方benchmark实现，不重复实现。
- **e2e适配器**`cpu_chunk_gated_delta_rule`：使用`*args, **kwargs`签名，按`_PARAM_ORDER`兼容位置参数与关键字参数，返回`[out, final_state]`。
- **aclnn适配器**`aclnn_chunk_gated_delta_rule_golden`：参数顺序对齐`aclnnChunkGatedDeltaRuleGetWorkspaceSize`函数签名，返回`[out_golden, finalState_golden]`，与`output_tensor_indexes`顺序一致。
- golden和benchmark在同一函数`_compute_and_store`中计算，benchmark结果存入`_GOLDEN_CONTEXT`供compare取用。
- **prepare持久化**`_save_bench`：`--dump in,golden`存golden时，benchmark同步存到manual-data目录的`<用例名>.bench/`兄弟目录（TTK的case目录schema固定，不允许放额外文件）；格式与TTK golden bin一致——裸字节、dtype/shape编码进文件名（`bench_<i>_<dtype>__shape_<DxN>.bin`，复用`dump_to_file`/`load_numpy_data`）。
- **replay加载**`resolve_bench`：全量存bin回放时golden函数被跳过、`_GOLDEN_CONTEXT`为空，此时从`.bench/`目录加载benchmark，与golden从bin加载的行为一致，均不重算；旧数据无`.bench`目录时回退为经`compare_context`（TTK注入的replay-safe输入上下文）从bin恢复的输入重算（日志有warning）。均按`testcase_name`缓存防止跨用例串数据。

#### impl/inputs.py

- 填充无法随机生成的int32张量与算子数值约束：
    - `actual_seq_lengths`：将T均分到B个batch。
    - q/k：L2归一化（算子要求预归一化输入）。
    - `g`：裁剪到 `[-1, 0]`。
    - `beta`：裁剪到 `(0, 1)`。

#### impl/compare.py

- 三方交叉校验：NPU输出 vs golden输出 vs benchmark输出，支持output + finalState双输出。
- 计算mare/mere/rmse/smra四项比率（NPU误差 / benchmark误差），阈值与floor值与算子侧`compare_cv`完全一致。
- 返回dict结构，含`pass`/`precision`/`error_info`/`metrics`（含四项比率及原始误差值）。

### convert_rdv_to_csv.py

- 将RDV全量用例（38条红线 + 34条STC，经FP32 state和非连续state扩展后翻倍）转换为TTK CSV。
- 覆盖连续/非连续state、bf16/fp32 state dtype组合，非连续state通过`tensor_storage_shapes`/`tensor_view_strides`/`tensor_view_offsets`描述。
- 生成两份CSV：E2E模式（`chunk_gated_delta_rule_rdv.csv`）与ACLNN模式（`aclnn_chunk_gated_delta_rule_rdv.csv`）。

```bash
cd attention/chunk_gated_delta_rule/tests/assets
python3 convert_rdv_to_csv.py
```

## 使用方法

使用时需准备E2E与ACLNN的CSV用例文件，并定义路径变量：

```bash
TTK_DIR=<ops-test-kit路径>
ASSETS=attention/chunk_gated_delta_rule/tests/assets
CSV_E2E=<E2E CSV用例路径>
CSV_ACLNN=<ACLNN CSV用例路径>
```

### E2E模式（eager + aclgraph）

```bash
cd $TTK_DIR
python3 -m ttk e2e \
  -i $CSV_E2E \
  --plugin $ASSETS \
  --aclgraph \
  -o <结果输出路径>
```

### ACLNN模式

```bash
cd $TTK_DIR
python3 -m ttk aclnn \
  -i $CSV_ACLNN \
  --plugin $ASSETS \
  -o <结果输出路径>
```

### 离线数据存bin与指定bin回放（两阶段执行）

将输入生成/CPU golden生成与设备执行拆为两步：prepare 阶段只生成并保存 input/golden 到 bin 文件（不跑设备），replay 阶段从 bin 恢复数据后跑设备 + compare。适用于准备机与执行机分离、或需复用固定输入复跑的场景。

```bash
MANUAL_DIR=/tmp/cgdr_manual   # bin 数据存放根目录
OUT=<结果输出CSV路径>
```

#### 1. prepare（存 bin）

完整数据（input + golden）：

```bash
# E2E
cd $TTK_DIR
python3 -m ttk e2e \
  -i $CSV_E2E --plugin $ASSETS \
  --no-prof --dump in,golden --dump-format bin \
  --manual-data-dirs $MANUAL_DIR --pc 1 \
  -o $OUT

# ACLNN
python3 -m ttk aclnn \
  -i $CSV_ACLNN --plugin $ASSETS \
  --no-prof --dump in,golden --dump-format bin \
  --manual-data-dirs $MANUAL_DIR --pc 1 \
  -o $OUT
```

仅准备 input（replay 时再现算 golden）：

```bash
python3 -m ttk e2e \
  -i $CSV_E2E --plugin $ASSETS \
  --no-prof --dump in --dump-format bin \
  --manual-data-dirs $MANUAL_DIR --pc 1 \
  -o $OUT
```

#### 2. replay（指定 bin 跑）

```bash
# E2E
python3 -m ttk e2e \
  -i $CSV_E2E --plugin $ASSETS \
  --manual-data-dirs $MANUAL_DIR --pc 1 \
  -o $OUT

# ACLNN
python3 -m ttk aclnn \
  -i $CSV_ACLNN --plugin $ASSETS \
  --manual-data-dirs $MANUAL_DIR --pc 1 \
  -o $OUT
```

- prepare 成功状态为 `MANUAL_DATA_PREPARED`，replay 命中后跳过随机输入生成，从 bin 恢复 input（及 golden），日志可见 `OnLoadManualData: loaded prepared input/scalar data from $MANUAL_DIR/...`。
- **三方benchmark随golden一起持久化、回放不重算**：`--dump in,golden` prepare 时 benchmark 存到 `$MANUAL_DIR/<用例名>.bench/bench_<i>_<dtype>__shape_<DxN>.bin`；replay 时 golden 从 bin 加载、benchmark 从该目录加载，行为一致。旧数据无 `.bench` 目录时回退为从 bin 输入重算（日志 warning）；仅 input 模式与直连模式的 benchmark 照常在 golden 阶段计算。
- `--pc 1` 单进程串行执行，本算子建议单进程。
- 两个阶段须使用同一份 CSV 与 assets；修改 input shape/view 或 attributes 后需重新 prepare。
- **`--plugin` 必传**：两阶段都依赖 spec.py 提供 golden/inputs/compare 适配器。
- **`--no-prof` 是双横杠**：写成 `-no-prof` 会报 `unrecognized arguments: -no-prof`。

### 关键参数

| 参数 | 作用 | 本算子取值 |
|------|------|-----------|
| `-o FILE` | 输出结果CSV（含耗时列） | profiling时建议带上 |
| `--aclgraph` | E2E测aclgraph模式（reduce-overhead） | 推荐，const graph不支持 |
| `-c` | E2E测const graph模式（GE backend） | 不支持，算子未在GE注册 |
| `--pc N` | 并发进程数 | 本算子建议 `--pc 1` |
| `--no-prof --dump in[,golden]` | prepare 阶段存 bin，不跑设备 | 详见"离线数据存bin"小节 |
| `--manual-data-dirs DIR` | replay 从 DIR 恢复 bin 数据 | 与 prepare 的 DIR 保持一致 |

### 仅校验CSV格式

```bash
# E2E
python3 -m ttk e2e -i $CSV_E2E --validate
# ACLNN
python3 -m ttk aclnn -i $CSV_ACLNN --plugin $ASSETS --validate
```
