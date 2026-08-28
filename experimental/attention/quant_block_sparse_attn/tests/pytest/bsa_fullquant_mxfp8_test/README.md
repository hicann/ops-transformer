# QBSA MXFP8 Pytest 用例说明

本目录用于运行 QuantBlockSparseAttn 的 MXFP8 full-quant golden/NPU 对比用例。

Golden 模拟 master kernel 的 softmax 状态，不再使用额外的 `has_valid_position` 判断：在线 max 从
`-FLT_MAX` 开始；OUT 以 `max == -FLT_MAX` 判断空行，LSE 还会将 `sum == 0` 映射为
`EMPTY_LSE`。若某行的所有合法 QK score 均为 `-Inf`，则与 kernel 一致保留
`attention_out=0`、`lse=EMPTY_LSE`；由 `NaN` 或 `+Inf`
触发的有效 softmax 行仍传播 `NaN`。为了兼容旧版算子和已生成的 golden，MXFP8 的
`EMPTY_LSE` 使用 FP32 `-FLT_MAX`（`-3.4028235e+38`）；该语义同时适用于单 tile
和多 tile 的最终归一化。

非有限输入按 CUDA SDPA 语义保持：Q/K 产生 `NaN` 或 `+Inf` score 时，
`attention_out` 和 LSE 传播 `NaN`；Q/K 使所有 score 为 `-Inf` 时输出
`0 / -FLT_MAX`；V 中的 `NaN/±Inf` 只传播到 `attention_out`，LSE 仍由 Q/K 决定。

STC 的 Q/K/V 数据范围支持标量 `inf`、`-inf`、`nan` 以及区间 `[-inf, inf]`。
无界区间会在相邻 D 向量的不同 MX group 中分别放入两个无穷端点；`±Inf` 使用 E4M3FN 的 `±448`
与最大有限 E8M0 descale `2^127` 配对，使 FP32 反量化得到同符号 Inf。`NaN` 保留在
E4M3FN payload 中，并使用该组的合法有限 descale（全 NaN 组使用 1）。

## 文件结构

- `qbsa_mxfp8_golden.py`：执行入口，负责生成输入、CPU golden、NPU 调用和精度对比。
- `qbsa_mxfp8_test_cases.py`：默认 testcase 文件，保留少量典型 case，便于日常快速回归。
- `qbsa_mxfp8_test_cases.csv`：批量 testcase 文件，每行对应一个 case。
- `qbsa_mxfp8_test_cases_full.csv`：单 batch 全量因子组合表。
- `qbsa_mxfp8_test_cases_batch_large.csv`：多 batch、ragged 和规模压力典型组合表。
- `golden_cache.py`：输入、CPU 输出、NPU 输出缓存工具。
- `result_compare_method.py`：精度比较工具。

## 前置条件

运行 NPU 路径前需要先完成 PTA 扩展编译/安装，确保 `custom_ops` 可以导入，并注册：

```bash
cd experimental/attention/quant_block_sparse_attn/torch_ops_extension
bash build_and_install.sh
```

## 查看用例

从仓库根目录执行：

```bash
python experimental/attention/quant_block_sparse_attn/tests/pytest/bsa_fullquant_mxfp8_test/qbsa_mxfp8_golden.py \
  --list_cases
```

输出会列出 case 名称和 `enable` 状态。

## 默认执行

不指定数据源和 case 名称时，脚本默认执行 `qbsa_mxfp8_test_cases.py` 中所有 `enable=True` 的 case。

```bash
python experimental/attention/quant_block_sparse_attn/tests/pytest/bsa_fullquant_mxfp8_test/qbsa_mxfp8_golden.py
```

默认 `--mode all`，等价于依次执行：

- `gen`：生成输入并保存缓存
- `cpu`：生成 CPU golden
- `npu`：调用 NPU 算子
- `compare`：比较 CPU/NPU 输出

每个 case 开始时会打印 `CASE START [idx/total]` 标志；一次运行多个 case 时，脚本末尾会输出
`QBSA MXFP8 CASE SUMMARY` 汇总表。汇总表各列含义：

| 列名 | 含义 |
|------|------|
| `No.` | 用例序号 |
| `Status` | `Pass` / `Failed` / `Generated` / `CpuDone` / `NpuDone` / `Error` |
| `Time(s)` | 该 case 总耗时（秒） |
| `mfu*time` | MFU×时间(us)，即 FLOPS / 算力；`-` 表示未计算（如未走数据生成阶段） |
| `OutPct` / `OutMaxErr` | Attention 输出的通过率与最大误差 |
| `LsePct` / `LseMaxErr` | softmax LSE 的通过率与最大误差 |
| `Case` | case 名称 |

其中 `mfu*time` 在每个 case 的数据生成阶段会同时打印 FLOPS 与算力的完整计算过程，包括基本块数量、
基本块 shape、单基本块计算量、FLOPS 公式、最小分型 shape、算力公式，以及最终
`FLOPS / 算力 = MFU * 时间(us)`。

## 指定单个 Case

通过 `--case_name` 指定 case 名称：

```bash
python experimental/attention/quant_block_sparse_attn/tests/pytest/bsa_fullquant_mxfp8_test/qbsa_mxfp8_golden.py \
  --case_name STC_B1_BS128_MIN_PAGE_EQ_MASK3
```

`--case_id` 保留为兼容参数，效果等同于 `--case_name`。

也支持用逗号一次指定多个 case：

```bash
python experimental/attention/quant_block_sparse_attn/tests/pytest/bsa_fullquant_mxfp8_test/qbsa_mxfp8_golden.py \
  --case_name STC_B1_BS128_MIN_PAGE_EQ_MASK3,STC_B1_NOMASK_BS128_EQ_TAIL
```

## 使用 Case CSV

使用 CSV 数据源时必须增加 `--csv`。此时未指定 `--case_files` 会读取同目录下的
`qbsa_mxfp8_test_cases.csv`。CSV 使用 Python 标准库读取，不依赖 `pandas` 或 `openpyxl`。

```bash
python experimental/attention/quant_block_sparse_attn/tests/pytest/bsa_fullquant_mxfp8_test/qbsa_mxfp8_golden.py \
  --csv
```

此时不指定 `--case_name`，会运行 CSV 中所有 `enable=true` 的 case；指定单个 case 的方式为：

```bash
python experimental/attention/quant_block_sparse_attn/tests/pytest/bsa_fullquant_mxfp8_test/qbsa_mxfp8_golden.py \
  --csv --case_name F_BS64_R5_N3_GT_AA
```

指定其他 CSV 或一次加载多个 CSV：

```bash
python experimental/attention/quant_block_sparse_attn/tests/pytest/bsa_fullquant_mxfp8_test/qbsa_mxfp8_golden.py \
  --csv --case_files /path/to/cases_a.csv,/path/to/cases_b.csv
```

执行 300 条多 batch 与规模压力用例：

```bash
python experimental/attention/quant_block_sparse_attn/tests/pytest/bsa_fullquant_mxfp8_test/qbsa_mxfp8_golden.py \
  --csv --case_files qbsa_mxfp8_test_cases_batch_large.csv
```

## 复用 CPU Golden

增加 `--rdv` 后，如果缓存目录中已存在当前 case 对应的 `<case_name>_cpu_output.pt`，脚本会直接加载
该 CPU golden，跳过耗时的 CPU 计算；缓存不存在时仍会按 `--mode` 正常生成并保存。输入生成、NPU
调用和精度对比不受影响。

```bash
python experimental/attention/quant_block_sparse_attn/tests/pytest/bsa_fullquant_mxfp8_test/qbsa_mxfp8_golden.py \
  --csv --rdv
```

`--rdv` 默认从 `golden_cache/` 查找，也可以配合 `--cache-dir` 指定缓存目录。缓存按 case 名称匹配；
case 参数发生变化后，应删除旧缓存或使用新的 case 名称，避免复用不匹配的 golden。
Debug 模式包含 `gen` 阶段时，新生成的 PT 仍写入 `debug/<case_name>/pt/`，RDV 查找仍使用正式缓存目录。

不带 `--csv` 时，`--case_files` 按 Python testcase 文件解析：

```bash
python experimental/attention/quant_block_sparse_attn/tests/pytest/bsa_fullquant_mxfp8_test/qbsa_mxfp8_golden.py \
  --case_files /path/to/custom_test_cases.py
```

## 批量用例设计

`qbsa_mxfp8_test_cases.csv` 包含 212 条用例，其中 188 条默认启用，覆盖 `block_size=64/128`、重点和非重点
head 组合、11 组可实现的 S1/S2 大小关系与对齐状态、`mask_mode=0/3`、不同稀疏模式、block-table
顺序以及空 Tensor 场景。

`blocknum` 表示可供 `block_table` 映射的物理页池大小。case 显式传入正数时直接使用该值；未传入或传入
非正数时，才根据每个 batch 的实际 KV 长度推导默认值：

```text
blocknum = sum(ceil(seqused_kv[b] / block_size))
```

`block_table` 的宽度为
`max(max_block_per_batch, max(ceil(seqused_kv[b] / block_size)))`，确保每个 batch 的有效逻辑页都可索引。
表中每个位置从 `[0, blocknum)` 物理页池中有放回随机采样：一次选中的物理页不会从候选池移除，
因此同一 batch 内或不同 batch 间的多个逻辑页允许映射到同一个 physical ID，用于覆盖物理页共享场景。
`blocknum` 不要求等于全部逻辑页数量之和：小于逻辑页总数时可以通过复用完成映射，大于逻辑页总数时
也允许存在未被引用的物理页。超过各 batch 有效 KV 长度的 `block_table` 填充列不会被 kernel 读取。

以下字段仅用于记录用例设计，统一放在 case 属性末尾，不参与 pytest 的类型转换、校验或计算：

```text
case_group, s1_base_size
```

CSV 按列名而非列位置解析，允许保留其他扩展参考列。因此已有外部泛化 CSV 即使列顺序不同或包含
pytest 未使用的字段，也可以继续加载；实际运行字段仍需提供合法值。

S1/S2 相等时，“S1 不对齐 128、S2 对齐 512”在数学上不可实现：一个数只要对齐 512，就必然对齐
128。因此三个关系与四种对齐状态共有 11 个有效组合，而不是 12 个。

`qbsa_mxfp8_test_cases_full.csv` 包含 2948 条单 batch 用例，补齐 `mask_mode=0/3` 与
`return_softmax_lse=true/false` 的排列组合；物理页数统一由实际 KV 序列自动推导。

`qbsa_mxfp8_test_cases_batch_large.csv` 包含 300 条全部启用的多 batch 典型与大规模用例：

- 48 条等长主路径，交叉 `B=2/4/8`、两种 block size、mask、LSE 和重点 head；
- 24 条 ragged batch，覆盖递减、交替、一长多短和 Q/KV 长度反向变化；
- 16 条规模压力用例，最大 `S1=16384`、`S2=32768`，并以低头数限制内存；
- 8 条 sparse/block-table 顺序、反序、随机、尾块和部分稀疏用例；
- 4 条 `B=16`、缩放参数、部分零长度 batch 和空 `actualSeqLengths` 边界用例；
- 80 条扩展典型用例，补齐多 batch 下的 `N2=2/5/6/7`；
- 80 条 `B=3/12` 中大型等长和一长多短用例；
- 40 条扩展 ragged/部分空 batch/部分 sparse 用例，将 batch 覆盖扩展到 `B=32`。

默认 Python testcase 文件当前保留 6 条 BASE/MIX case，并包含 37 条默认启用的 STC，覆盖单 batch、
多 batch、ragged、稀疏拼接、scale、mask、LSE、空输入和不同 block-table 顺序。

## Debug 模式

`--debug` 必须与一个 `--case_name` 配合使用，只允许执行单个 case。默认（不指定 `--mode` 或
`--mode all`）执行完整的 `gen/cpu/npu/compare` 流程：

```bash
python experimental/attention/quant_block_sparse_attn/tests/pytest/bsa_fullquant_mxfp8_test/qbsa_mxfp8_golden.py \
  --debug --case_name BASE_01
```

CSV case 同样支持 debug：

```bash
python experimental/attention/quant_block_sparse_attn/tests/pytest/bsa_fullquant_mxfp8_test/qbsa_mxfp8_golden.py \
  --csv --debug --case_name F_BS64_R5_N3_GT_AA
```

也支持指定 `--mode` 为非 `all` 的组合，例如只复用已有缓存跑 NPU 对比，不再重新生成数据和
CPU golden：

```bash
python experimental/attention/quant_block_sparse_attn/tests/pytest/bsa_fullquant_mxfp8_test/qbsa_mxfp8_golden.py \
  --debug --case_name F_BS64_R5_N3_GT_AA --csv --mode="npu,compare"
```

输出目录固定为 `debug/<case_name>/`：

```text
debug/<case_name>/
├── run.log
├── pt/
│   ├── <case_name>_input.pt
│   ├── <case_name>_cpu_output.pt  # RDV 命中正式缓存时不重复写入
│   └── <case_name>_npu_output.pt
└── precision/
    ├── attention_out.png
    └── softmax_lse.png        # return_softmax_lse=true 时生成
```

`cpu_output.pt` 保存 CPU golden，`npu_output.pt` 保存算子输出。精度图逐元素使用与
`check_result` 相同的 `rtol/atol`：白色表示通过，蓝色表示失败。图片标题会标注 golden/算子 shape、
逻辑矩阵大小、显示大小、元素数和失败数。超大 Tensor 会按区域聚合；某个显示像素对应的原始区域中
只要存在一个失败元素，该像素就是蓝色。因此整体比较允许少量失败元素而最终为 Pass 时，图片中仍可能
出现少量蓝点。生成 PNG 需要安装 `matplotlib`。

Debug 模式的数据读写位置取决于 `--mode` 是否包含 `gen`：

- **包含 `gen`**（如 `all`、`gen`、`gen,cpu`）：数据生成后写入 `debug/<case_name>/pt/`，即上述目录
  中的 `pt/`，此模式下不允许指定 `--cache-dir`（debug 自行管理 pt 目录）。
- **不含 `gen`**（如 `npu,compare`、`compare`、`npu`）：从已有缓存读取 `input.pt`、`cpu_output.pt`
  等数据。默认读取本目录下的 `golden_cache/`；可通过 `--cache-dir` 指定其他缓存来源。precision PNG
  始终输出到 `debug/<case_name>/precision/`。

无论哪种 `--mode`，Debug 模式都不允许使用 `--cache_case_name`（命名由 debug 内部管理），也不能与
`--list_cases` 组合使用。

## 指定执行阶段

`--mode` 支持 `all/gen/cpu/npu/compare`，也支持逗号组合：

```bash
# 只生成输入
python experimental/attention/quant_block_sparse_attn/tests/pytest/bsa_fullquant_mxfp8_test/qbsa_mxfp8_golden.py \
  --mode gen

# 使用已有缓存，只跑 NPU 并比较
python experimental/attention/quant_block_sparse_attn/tests/pytest/bsa_fullquant_mxfp8_test/qbsa_mxfp8_golden.py \
  --mode npu,compare
```

## 缓存目录

默认缓存目录为本目录下的 `golden_cache/`。可以通过 `--cache-dir` 指定：

```bash
python experimental/attention/quant_block_sparse_attn/tests/pytest/bsa_fullquant_mxfp8_test/qbsa_mxfp8_golden.py \
  --cache-dir /tmp/qbsa_mxfp8_cache
```

缓存文件名前缀默认使用 case 名称。需要自定义时使用：

```bash
python experimental/attention/quant_block_sparse_attn/tests/pytest/bsa_fullquant_mxfp8_test/qbsa_mxfp8_golden.py \
  --cache_case_name my_case
```

## 新增 Case

默认方式是在 `qbsa_mxfp8_test_cases.py` 的 `TestCases` 中新增 dict。也可以在
`qbsa_mxfp8_test_cases.csv` 中新增一行，并通过 `--csv` 执行。两种方式都要求
`name` 唯一；`enable=True` 表示默认运行，关闭的 case 仍可通过 `--case_name` 单独执行。

字段格式：

- `actual_seq_q`、`actual_seq_kv` 使用 JSON 数组文本，例如 `[256]`。
- 等长多 batch 可以使用单元素总 T，例如 `B=8, S1=65536, actual_seq_q=[65536]` 会在运行时展开为
  8 个 8192。
- `enable`、`is_contiguous`、`return_softmax_lse`、`empty_actual_seq` 使用 `true/false`。
- `softmax_scale` 留空时，脚本按 `1 / sqrt(D)` 自动补齐。
- `N1` 必须能被 `N2` 整除。
- `s2_base_size` 是 golden 划分 C1 子块的必需运行字段，当前 MXFP8 模板固定为 `512`。
- `sparse_q_block_size` 必须等于 `sparse_kv_block_size`；`block_size`（PA 物理块大小）必须是 `sparse_kv_block_size` 的正整数倍（支持 `block_size = m * sparse_kv_block_size`）。
- `s1_base_size` 仅作为覆盖场景说明保留，不参与 pytest 计算或校验。
- `p_scale_type` 仅支持 `float8_e8m0fnu` 或 `float32` 的输入，传入其他值或不传会默认为 `float8_e8m0fnu`。
- CSV 可直接扩展到多行；case 名称在一个或多个输入文件之间都不能重复。
