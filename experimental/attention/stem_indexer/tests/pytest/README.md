# StemIndexer算子pytest测试框架

## 功能说明

基于pytest测试框架，实现StemIndexer算子的功能验证：

- CPU侧：按StemIndexer设计方案实现golden。
- NPU侧：先通过`torch.ops.custom.npu_stem_indexer_metadata`生成分核信息，再通过
  `torch.ops.custom.npu_stem_indexer`获取实际结果（接口位于
  `experimental/attention/stem_indexer/torch_ops_extension`，由 `custom_ops.py` 加载注册）。
  支持两种调用模式：eager（直接调用算子）和 graph（通过 `torch.compile` + `torchair` 编译为 aclgraph 图）。
  通过环境变量 `STEM_INDEXER_MODE` 切换，默认 `eager`。
- 结果比对：比较`sparse_seq_len`，并只比较`sparse_indices`中`sparse_seq_len`范围内的有效前缀；尾部未定义区域不校验。

## 用例来源

白盒用例方案：

```text
test_stem_indexer_paramset.py
csv/stem_indexer_generalized_cases.csv
```

single与batch模式分别维护各自的用例：single模式的用例直接维护在`ENABLED_PARAMS`中，每条case前通过注释记录覆盖点和设计原因；batch模式的用例维护在`csv/stem_indexer_generalized_cases.csv`中，作为批量模式生成`.pt`文件的输入。

两种用例源均包含`topk_score_precision`：1表示uint32，2表示uint16，未显式配置时默认1。
其中`SI_WB_001`～`SI_WB_100`固定使用uint32 TopK score路径；`SI_WB_101`～`SI_WB_150`
镜像前50条功能场景，固定使用uint16 TopK score路径。
`SI_WB_001_1`、`SI_WB_001_2`及其uint16镜像`SI_WB_101_1`、`SI_WB_101_2`额外覆盖batch=8的64K/128K等长prefill场景。

## 当前覆盖点

- q/kv尾块，以及batch内`q_seq_lens=0`、`kv_seq_lens=0`、二者同时为0的空序列边界。
- `initial_blocks=4`与`window_size=4`在不同`s2Valid`下的完全重叠、部分交集、无交集、短序列裁剪。
- causal与non-causal路径。
- TPD `alpha`无衰减、普通衰减、强衰减。
- 动态TopK预算的small/medium/large prompt block分支。
- S2方向`baseN=256`整块和尾块。
- M方向`baseM=64`整块和尾块。
- GQA组合：`q_heads`为32/64，`kv_heads`为2/4/8，覆盖6种合法组合。
- 多batch变长、prefill/decode混合、单token decode，batch覆盖1到19、23到31、37、39等非2次幂和较大batch场景。
- 长序列量级覆盖：`kv_seq_lens`和`num_prompt_tokens`覆盖32K/64K/128K/256K/1M token基准线；`q_seq_lens`以保留query chunk、decode和尾块覆盖目的为主，仅在张量规模可控的case中放大，同时保留小于32K的短序列、尾块和短路场景。
- OAM `vbias`影响选块、`scoreScale=1/64`路径。
- `num_prompt_tokens`不能整除`stem_block_size`。
- 动态TopK small/medium/large分段边界：55/56/159/160 blocks。
- `curS2Len == initial + dynamicTopK + window`和刚越过该边界的短路2路径。

## Metadata说明

`metadata`是StemIndexer主算子的前置输入，pytest正例和普通单case脚本都会先调用
`stem_indexer_metadata`按`16 + B * kv_heads * (36 + 72) * 16`生成动态容量的metadata，
并向上对齐到4096个INT32元素，再传给`stem_indexer`。
当前case表只保留可运行并可与golden比对的正例。

当前StemIndexer主算子使用BNSD布局，`q_seq_lens`和`kv_seq_lens`按batch实际长度传入，
`num_prompt_tokens`按batch传入动态TopK预算基准长度，正例中保持`num_prompt_tokens >= kv_seq_lens`；
该输入缺省时由OpHost通过TilingData通知Kernel复用`kv_seq_lens`。`metadata`虽然在接口层声明为可选输入，
但当前主算子计算必须传入有效Metadata，缺省时会在Tiling阶段返回参数错误。
测试用例不再单独维护额外token长度辅助字段；
`qflat`、`kflat`的shape由`q_seq_lens`、`kv_seq_lens`的最大值推导。

## 文件结构

```text
test_run.sh                             # 执行脚本
test_stem_indexer_paramset.py           # single用例参数表
stem_indexer_golden.py                  # CPU侧golden实现
result_compare_method.py                # sparse输出比较
test_stem_indexer_single.py             # single主执行入口
test_stem_indexer_batch.py              # batch主执行入口
stem_indexer_aclgraph.py                # single/batch共用的aclgraph(graph)调用实现
test_npu_stem_indexer.py                # 参考LI写法的普通单case脚本
pytest.ini                              # pytest标记
csv/stem_indexer_generalized_cases.csv  # batch用例表
batch/stem_indexer_pt_save.py           # 读取CSV并生成pt
batch/stem_indexer_pt_loadprocess.py    # 读取pt并调用算子
batch/replace_path.py                   # 替换batch pytest中的pt路径
```

## 使用方法

在当前pytest目录执行：

```bash
bash test_run.sh single        # single (eager)
bash test_run.sh single_graph  # single (graph)
```

single和batch模式都支持通过`STEM_INDEXER_CASE_ID`只运行指定用例，多个case_id使用逗号分隔：

```bash
STEM_INDEXER_CASE_ID=SI_WB_001,SI_WB_002 python3 -m pytest test_stem_indexer_single.py
STEM_INDEXER_CASE_ID=SI_WB_001,SI_WB_002 python3 -m pytest test_stem_indexer_batch.py
```

批量测试：

```bash
bash test_run.sh batch        # eager 模式
bash test_run.sh batch_graph  # graph 模式
```

复跑已生成的`.pt`文件：

```bash
python3 -m pytest test_stem_indexer_batch.py
```

batch模式流程与QLI保持一致：

```text
1. 读取csv/stem_indexer_generalized_cases.csv。
2. 生成每条case的.pt文件，保存输入和CPU golden。
3. pytest逐个读取.pt文件，运行时动态构造metadata并调用NPU算子。
4. 与.pt中保存的golden比对。
5. 生成result.csv记录批量执行结果。
```

生成`.pt`时也使用`STEM_INDEXER_CASE_ID`只选择指定用例，多个case_id使用逗号分隔：

```bash
STEM_INDEXER_CASE_ID=SI_WB_001_1,SI_WB_101_1 \
    python3 batch/stem_indexer_pt_save.py csv/stem_indexer_generalized_cases.csv pt_path
```

生成脚本默认使用全部可用CPU核按case并行。大用例并行时内存占用较高，可通过`--workers`限制进程数：

```bash
python3 batch/stem_indexer_pt_save.py csv/stem_indexer_generalized_cases.csv pt_path --workers 8
```

`--workers 0`表示使用全部可用CPU核，`--workers 1`表示按原方式串行生成。

`.pt`文件和`result.csv`是本地生成产物，不需要提交。

普通单case脚本可直接执行：

```bash
python3 test_npu_stem_indexer.py
```
