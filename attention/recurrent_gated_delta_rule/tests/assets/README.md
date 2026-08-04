# Recurrent_gated_delta_rule算子TTK测试资产

## 功能说明

基于ops-test-kit（TTK）测试框架，实现Recurrent_gated_delta_rule算子的E2E与ACLNN精度验证：

- **E2E模式**：通过TorchNPU Python API（`torch_npu.npu_recurrent_gated_delta_rule`）直调或torch.compile图模式获取NPU实际数据，与CPU golden对比
- **ACLNN模式**：通过aclnn C API（`aclnnRecurrentGatedDeltaRule`）调用算子，与CPU golden对比
- **精度对比**：支持eager、静态图（const graph）多种执行方式，进行CPU与NPU结果的精度对比

## 当前实现范围

### 支持的测试方式

| 模式 | API | 说明 |
|------|-----|------|
| E2E eager | `torch_npu.npu_recurrent_gated_delta_rule` | torch直调，含profiling |
| E2E const graph | 同上 | torch.compile静态shape编译执行 |
| ACLNN | `aclnnRecurrentGatedDeltaRule` | aclnn C API调用 |

### 精度标准

- tolerance声明社区标准`stat_rel_err`（bfloat16）。
- compare判定条件（`impl/compare.py`）：`np.isclose(rtol=0.0078125, atol=0.0001)`，通过率 ≥ 99.5%且最大相对误差 < 10.0方为Pass。

### 环境配置

#### 前置要求

1. TorchNPU安装包下载路径（需及时更换为最新版本）：[TorchNPU安装教程](https://gitcode.com/Ascend/pytorch)
2. 完成环境安装和环境变量配置，具体操作请参考：[ops-transformer](../../../../README.md)
3. ops-test-kit测试框架（TTK）：[ops-test-kit](https://gitcode.com/cann/ops-test-kit)

## 文件结构

```
attention/recurrent_gated_delta_rule/tests/
├── pytest/
│   └── gen_ttk_csv.py                                # RDV全量用例转TTK CSV脚本（E2E + ACLNN）
└── assets/
    ├── spec.py                                       # TestSpec：注册API + golden/inputs/tolerance/compare/torch_graph（e2e + aclnn）
    └── impl/
        ├── golden.py                                 # golden适配器（e2e + aclnn），复用pytest的CPU golden
        ├── inputs.py                                 # inputs适配器（e2e + aclnn），填充index/length类张量
        ├── compare.py                                # 数值精度对比（output + state双输出）
        └── graph.py                                  # 自定义torch.nn.Module，torch.compile图模式专用
```

### 文件职责

#### spec.py

- 定义`RecurrentGatedDeltaRuleSpec`类（e2e），声明`golden`、`customize_inputs`、`torch_graph`、`tolerance`、`compare`。
- 定义`AclnnRgdrSpec`类（aclnn），声明`golden`、`customize_inputs`、`tolerance`、`compare`。
- 通过`__spec__` dict注册：
    - `torch_npu.npu_recurrent_gated_delta_rule` → `RecurrentGatedDeltaRuleSpec`
    - `aclnnRecurrentGatedDeltaRule` → `AclnnRgdrSpec`
- `torch_graph`指向`impl/graph.py`的`RgdrGraphModule`，供torch.compile图模式使用。
- 动态加载`impl/`下四个模块，避免硬编码路径。

#### impl/golden.py

- 通过`importlib`加载`tests/pytest/recurrent_gated_delta_rule_golden.py`中的CPU golden，不重复实现。
- **e2e适配器**`cpu_recurrent_gated_delta_rule`：使用`*args, **kwargs`签名，按`_PARAM_ORDER`兼容位置参数与关键字参数，返回`(output, state)`。
- **aclnn适配器**`aclnn_cpu_recurrent_gated_delta_rule`：参数顺序对齐`aclnnRecurrentGatedDeltaRuleGetWorkspaceSize`函数签名，返回`[output, state_out]`列表，与`output_tensor_indexes`顺序一致。

#### impl/inputs.py

- 填充无法随机生成的int32张量：
    - `actual_seq_lengths`：将T均分到B个batch。
    - `ssm_state_indices`：`arange(T)`。
    - `num_accepted_tokens`（可选）：每batch一个`[1, seq_len]`内的值。

#### impl/compare.py

- 对NPU输出与golden输出做数值精度对比，支持output + state双输出。

#### impl/graph.py

- `RgdrGraphModule`：torch.compile图模式自定义Module，显式返回`(output, state_out)`以便追踪in-place state修改。

### pytest/gen_ttk_csv.py

- 将`tests/pytest/test_recurrent_gated_delta_rule_paramset_rdv.py`中的全量RDV用例（148条）转换为TTK CSV。
- 覆盖连续/非连续state、bf16/fp32 state dtype组合，非连续state通过`tensor_storage_shapes`/`tensor_view_strides`/`tensor_view_offsets`描述。
- 生成两份CSV：E2E模式（`recurrent_gated_delta_rule_rdv.csv`）与ACLNN模式（`aclnn_recurrent_gated_delta_rule_rdv.csv`）。

```bash
cd attention/recurrent_gated_delta_rule/tests/pytest
python3 gen_ttk_csv.py
```

## 使用方法

使用时需准备E2E与ACLNN的CSV用例文件，并定义路径变量：

```bash
TTK_DIR=<ops-test-kit路径>
ASSETS=attention/recurrent_gated_delta_rule/tests/assets
CSV_E2E=<E2E CSV用例路径>
CSV_ACLNN=<ACLNN CSV用例路径>
```

### E2E torch直调（eager）

```bash
cd $TTK_DIR
python3 -m ttk e2e \
  -i $CSV_E2E \
  --plugin $ASSETS \
  --warmup false --run 1
```

### E2E 静态图（const graph）

```bash
cd $TTK_DIR
python3 -m ttk e2e \
  -i $CSV_E2E \
  --plugin $ASSETS \
  --warmup false --run 1 \
  -c \
  -o /tmp/rgdr_perf.csv
```

### ACLNN模式

```bash
cd $TTK_DIR
python3 -m ttk aclnn \
  -i $CSV_ACLNN \
  --plugin $ASSETS \
  -o /tmp/rgdr_aclnn_result.csv
```

### 关键参数

| 参数 | 作用 | 本算子取值 |
|------|------|-----------|
| `--run N` | 执行次数取平均 | E2E必须1（in-place约束）；ACLNN默认3 |
| `--warmup` | profiling前warmup | E2E必须false（in-place约束）；ACLNN默认true |
| `-o FILE` | 输出结果CSV（含耗时列） | profiling时建议带上 |
| `-c` | E2E额外测静态图模式耗时 | 可选 |

### 仅校验CSV格式

```bash
# E2E
python3 -m ttk e2e -i $CSV_E2E --validate
# ACLNN
python3 -m ttk aclnn -i $CSV_ACLNN --plugin $ASSETS --validate
```
