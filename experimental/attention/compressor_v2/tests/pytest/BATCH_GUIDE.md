# Compressor V2 批量测试操作指南

> 适用算子：`experimental/attention/compressor_v2`
> 测试方式：Excel 填用例 → 生成 .pt（含 CPU golden）→ 批跑 NPU 验证

---

## 前置依赖

```bash
pip install pandas openpyxl
```

---

## 一、填写测试用例表格

**模板文件**：`tests/pytest/excel/test_cases_template.csv`
（复制一份改名为 `test_cases.csv` 使用，或直接改模板）

### 列说明

| 列名 | 类型 | 说明 | 示例 |
|---|---|---|---|
| Testcase_Name | str | 用例名（必填，唯一，作为 .pt 文件名） | Prefill0 |
| batch_size | int | batch 大小 | 1 |
| hidden_size | int | 隐藏层维度 | 4096 |
| Seq_len | int | 序列长度 | 8192 |
| head_dim | int | 头维度（128 / 512） | 512 |
| block_size | int | 块大小（通常 128） | 128 |
| cmp_ratio | int | 压缩比（4 / 128） | 4 |
| start_p | int | 起始位置 | 0 |
| layout_x | str | TH / BSH | TH |
| data_type | str | BF16 / FP16 | BF16 |
| cu_seqlens | None 或 str | TH 布局必填：如 `0,8192`；None 自动生成 | None |
| seqused | None 或 str | 各 batch 实际使用长度，如 `8192` | None |
| start_pos | None 或 str | 各 batch 起始位置，如 `0` | None |
| x_datarange | str | 输入 x 数值范围 `低,高` | -10,10 |
| wkv_datarange | str | wkv 数值范围 | -10,10 |
| wgate_datarange | str | wgate 数值范围 | -10,10 |
| kv_state_datarange | str | kv_state 数值范围 | -10,10 |
| score_state_datarange | str | score_state 数值范围 | -10,10 |
| is_contiguous | bool | 可选。是否连续 state_cache：True=不加 pad（连续）；不填/False=随机 pad（非连续，默认） | False |

> 注：新版接口已移除 `coff`、`cache_mode`、`ape` 三个参数，测试中统一硬编码 coff=1、cache_mode=2（固定环形buffer）。
>
> 注：`is_contiguous` 为可选列，**不填则默认为非连续**（带 pad），与算子默认行为一致。

### 参数限制（README 说明）

- head_dim 支持 128 / 512
- hidden_size 支持 1K~10K，512 对齐
- cmp_ratio 支持 2、4、8、16、32、64、128
- cmp_ratio=4/128 时三种情况：
  - C4A:  D=512, cmp_ratio=4（原 coff=2）
  - C4Li: D=128, cmp_ratio=4（原 coff=2）
  - C128A: D=512, cmp_ratio=128（原 coff=1）
- layout_x=TH 时 cu_seqlens 长度 = batch_size+1

### 数据格式注意

- **datarange 不要带方括号**：写 `-10,10`，不要写 `[-10,10]`
- **cu_seqlens/seqused/start_pos 逗号分隔字符串**：`8192,8192`（batch>1 时每 batch 一个值）；单 batch 可写 `8192` 或留空 None
- 每行一个用例，Testcase_Name 不要重复

---

## 二、生成 .pt 文件（含 CPU golden）

```bash
cd experimental/attention/compressor_v2/tests/pytest
python -u batch/compressor_pt_save.py excel/test_cases.csv pt_path
```

- 输出：`pt_path/<Testcase_Name>.pt`
- 每个 .pt 内包含：输入张量 + CPU golden 结果 + 参数
- **可复用**：pt_path 目录保留，重跑测试无需再生成

---

## 三、批量执行测试

```bash
cd experimental/attention/compressor_v2/tests/pytest
python -u -m pytest -rA -s test_compressor_batch.py -v -m ci \
  -W ignore::UserWarning -W ignore::DeprecationWarning
```

- 自动读取 `pt_path/` 下所有 .pt 逐个执行
- 结果写入 `result.xlsx`（含每项精度对比 percent）
- **重要：全程串行执行，禁止两个 pytest 并行**（会 aicore timeout 抢核）

### 指定部分用例

方式一：用 Excel 过滤（环境变量 `TEST_CASE_EXCEL`，脚本只跑 Excel 里 Testcase_Name 存在的 .pt）

```bash
TEST_CASE_EXCEL=excel/test_cases.csv python -u -m pytest -rA -s test_compressor_batch.py -v -m ci ...
```

方式二：单条隔离执行（每条一个进程）

```bash
COMPRESSOR_TESTCASE_PATH=pt_path/Prefill0.pt python -u -m pytest -rA -s test_compressor_batch.py -v -m ci ...
```

### 单用例快速调测（不生成 .pt）

编辑 `test_compressor_paramset.py` 的 `ENABLED_PARAMS`，然后：

```bash
python -u -m pytest -rA -s test_compressor_single.py -v -m ci -k "param_combinations0"
```

---

## 四、结果查看

| 文件 | 说明 |
|---|---|
| result.xlsx | 每个用例的 5 项精度对比（result / kv_state_update / score_state_update / kv_state_origin / score_state_origin）+ percent |
| pytest 输出 | PASSED/FAILED 状态 |

---

## 五、常见问题

| 现象 | 处理 |
|---|---|
| 导入卡在 ninja/cc1plus | 首次 JIT 编译所有算子，等待 10 分钟左右；已有缓存则秒过 |
| `Failed to load op 'minimax_sparse_attention_split_kv'` | 正常警告，非 compressor 问题 |
| 两个 pytest 并行 aicore timeout | 串行执行 |
| 修改了 PTA csrc | `rm -rf /root/.cache/torch_extensions/py310_cpu/<op>` |
| .pt 重跑报精度全错 | 重新生成 .pt（golden 与 kernel 参数需一致） |
