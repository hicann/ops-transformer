# Chunk_gated_delta_rule算子测试框架

## 功能说明

基于pytest测试框架，实现Chunk_gated_delta_rule算子的功能验证：

- **CPU侧**：复现算子功能用以生成golden数据
- **NPU侧**：通过TorchNPU进行算子直调获取实际数据
- **精度对比**：进行CPU与NPU结果的精度对比验证算子功能

## 当前实现范围

### 参数限制

- 支持batch_size大于0。
- 支持seqlen序列长度。
- 支持NK、NV head数，NV需要为NK倍数。
- 支持DK、DV不超过128。
- 支持data_type为BF16。

### 环境配置

#### 前置要求

1. TorchNPU安装包下载路径（需及时更换为最新版本）：[TorchNPU安装教程](https://gitcode.com/Ascend/pytorch)
2. 完成环境安装和环境变量配置，具体操作请参考：[ops-transformer](../../../../README.md)

#### custom包调用

支持custom包调用

## 文件结构

#### pytest文件结构说明

- test_run.sh                               # 执行脚本
- conftest.py                               # pytest钩子：逐用例记录参数/结果/种子，会话结束落CSV
- chunk_gated_delta_rule_golden.py          # cpu侧算子golden实现
- chunk_gated_delta_rule_main.py            # cpu golden与npu结果精度对比及主调测逻辑
- pytest.ini                                # 创建ci单算子和graph图模式的测试标记

单用例测试:

- test_chunk_gated_delta_rule_single.py     # 测试单用例运行主程序
- chunk_gated_delta_rule_operator_single.py # CPU侧算子逻辑实现获取golden与npu算子直调
- test_chunk_gated_delta_rule_paramset.py   # 单用例入参配置
- test_chunk_gated_delta_rule_paramset_rdv.py   # RDV测试入参配置

## 使用方法

在pytest文件夹路径下执行：

### 运行测试用例

#### 单用例调测

1、手动配置test_chunk_gated_delta_rule_paramset.py的参数

2、执行指令：

``` bash
bash test_run.sh single
```

#### RDV测试

1、手动配置test_chunk_gated_delta_rule_paramset_rdv.py的参数

2、执行指令：

``` bash
bash test_run.sh rdv
```

#### 随机用例测试

随机生成N条用例并执行（含CPU golden精度对比），可用`RANDOM_SEED`环境变量固定随机种子复现（不指定则自动生成并记录到CSV）：

``` bash
bash test_run.sh random 100
```

#### 随机用例测试（仅NPU）

随机生成N条用例，设置`SKIP_GOLDEN=1`跳过CPU golden计算与精度对比，仅执行NPU算子，加快执行速度：

``` bash
bash test_run.sh random_npu 100
```

#### 随机用例生成规则

random/random_npu 模式均使用同一套随机参数生成器（`_generate_random_param_dict`），在算子约束内从0随机生成，不依赖 single/rdv 参数池。每条用例的入参生成规则如下：

| 接口入参 | 随机规则 | 约束/说明 |
|----------|---------|-----------|
| B | choice([1,2,4,8,16,32,64,128]) | batch size |
| seqlen | choice([1,3,7,32,64,100,128,200,256,300,512,1000,1024,2048,4096,5000,8192,10000,16384,32768,65535]) | 序列长度；含非 chunk 对齐值（如 100, 300, 5000, 65535）测试 partial chunk 路径 |
| seqlen（变长） | B>1 时 30% 概率生成 list，每个 batch 独立随机 seqlen | 覆盖 actual_seq_lengths 变长路径 |
| nk | randint(1,64) | key头数 |
| nv | nk × randint(1,64//nk) | Nv>=Nk 且 Nv%Nk==0 |
| dk | randint(1,min(128,budget)) | key维度，受 state 元素上限约束 |
| dv | randint(1,min(128,budget//dk)) | value维度，受 state 元素上限约束 |
| chunk_size | 固定 64 | 算子 tiling 硬编码，golden 须与 NPU 一致 |
| data_type | 固定 bfloat16 | |
| state_data_type | choice([bfloat16,float32]) | 状态数据类型 |
| has_g | choice([True,False]) | 50%概率启用门控 |
| is_contiguous | choice([True,False]) | 50%概率非连续 |
| query_datarange | 固定 [-1,1] | q 经 L2 归一化 |
| key_datarange | 固定 [-1,1] | k 经 L2 归一化 |
| value_datarange | choice([-10,10], [-1,1]) | 随机数据范围 |
| gamma_datarange | choice([-1,0], [-0.5,0], [-0.1,0], [-1,-0.5]) | 文档约束 [-1,0]，g 经 exp 衰减 |
| beta_datarange | 固定 [0,1] | |
| state_datarange | 固定 [-10,10] | |

**shape 约束**：B>0、seqlen>0、0<Nk<=64、0<Nv<=64 且 Nv>=Nk 且 Nv%Nk==0、0<Dk<=128、0<Dv<=128。

**内存约束**：
- Dk×Dv 受 state 元素数上限 `_STATE_ELEM_CAP=2.0B` 约束（`budget = STATE_ELEM_CAP // (B × Nv)`）
- T×max(Nk×Dk, Nv×Dv) 受 QKV 张量元素上限 `_QKV_ELEM_CAP=1.0B` 约束（`max_seqlen = QKV_ELEM_CAP // (B × qkv_per_token)`）
- 两项约束共同防止单进程 OOM

**随机种子机制**：
- `RANDOM_SEED` 控制 shape/参数序列（一个 seed 对应一组确定的 N 条用例参数）
- `TORCH_SEED` 控制张量数值（每条用例独立，conftest 自动生成并记 CSV）
- 不设 `RANDOM_SEED` 时自动生成并回写 `os.environ`，conftest 落 CSV
- 复现：`RANDOM_SEED=<seed> bash test_run.sh random N`

### 结果输出与复现

所有模式执行后均输出到`output/`目录（已gitignore）：

- `run_<时间戳>.log`：完整执行日志（tee屏显）
- `result_<时间戳>.csv`：逐用例结果表，每行一条用例

| 列 | 说明 |
|----|------|
| random_seed | 随机shape序列种子（random模式；single/rdv为固定参数集无此值） |
| seed | 本条用例张量数值种子（每条独立记录） |
| test_name | 用例名称 |
| test_mode | single/rdv/random |
| check_type | precision=带golden精度对比 / execution_only=仅NPU执行 |
| model | 执行模式（torch直调/aclgraph） |
| status | pytest执行结果（PASSED/FAILED/SKIPPED） |
| B...is_continue | 本条用例全部入参 |
| errmsg | 失败详情（截断2000字符） |
| durations | 算子耗时（仅prof模式） |

失败用例复现：

``` bash
# 整批复现（同shape序列）：CSV取random_seed
RANDOM_SEED=<random_seed> bash test_run.sh random N

# 单条数值级复现（同shape+同张量数值）：CSV取入参与seed
TORCH_SEED=<tensor_seed> bash test_run.sh random 1
```

### 环境变量汇总

| 变量 | 作用 | 适用模式 |
|------|------|---------|
| RANDOM_SEED | 固定随机shape序列种子（不设则自动生成并记CSV） | random/random_npu |
| TORCH_SEED | 固定张量数值种子（不设则每条自动生成并记CSV） | 全部 |
| RANDOM_CASE_COUNT | 随机用例条数（test_run.sh已透传） | random系 |
| SKIP_GOLDEN | =1跳过CPU golden与精度对比，仅NPU执行 | random_npu |
| CSV_FILE | 指定CSV输出路径（test_run.sh已自动设置） | 全部 |
| CSV_APPEND | =1时CSV追加写入 | 全部 |
| USE_GRAPH | =true启用aclgraph模式 | 全部 |
