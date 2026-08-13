# ChunkKdaFwd算子测试框架

## 功能说明

基于pytest和msopprof实现ChunkKdaFwd算子的功能、确定性和性能验证：

- **功能验证**：通过`cann_ops_transformer.ops.chunk_kda_fwd`调用算子，验证四种输入布局、可选输出矩阵、BF16 gate参数提升、变长序列尾块和`state_v_first`。
- **一致性验证**：将BNSD、TND、NTD输入的结果与BSND输入结果进行交叉布局对比。
- **确定性验证**：对同一输入重复执行，验证指定输出二进制一致。
- **性能验证**：使用msopprof采集关键场景的device侧耗时，不使用Python wall time作为性能结论。

## 当前实现范围

### 用例覆盖

| 测试项 | 覆盖内容 |
| --- | --- |
| 输入布局 | BSND、BNSD、TND、NTD |
| 数据类型 | BF16主输入，FP32 gate，BF16/FP32 `A_log`和`dt_bias` |
| 可选输出 | 默认返回、`return_intermediate_states=True`、`disable_recompute=True` |
| 变长序列 | NTD，`cu_seqlens=[0,64,65]`，序列尾块长度为1 |
| 状态布局 | `state_v_first=False`和`state_v_first=True` |
| 性能场景 | BF16、BSND、B=1、S=16384、H=HV=96、K=V=128、`chunk_size=64`、`safe_gate=True`、`use_gate_in_kernel=True` |

### 环境配置

#### 前置要求

1. 完成CANN、Ascend for PyTorch和ops-transformer安装及环境变量配置，具体操作请参考[ops-transformer](../../../../README.md)。
2. 安装包中已包含`ChunkKdaFwd`算子及`cann_ops_transformer.ops.chunk_kda_fwd` Python接口。
3. 性能验证前确认`msopprof`命令可用。

## 文件结构

```text
pytest/
├── README.md                    # 测试框架说明
├── common.py                    # 输入构造、layout转换和通用断言
├── test_chunk_kda_fwd.py        # 功能、一致性和确定性用例
├── profile_a5_h96_t16k.py       # A5关键场景性能与50次确定性入口
└── check_msopprof.py             # msopprof结果解析和性能门禁
```

## 使用方法

在ops-transformer仓根目录下执行。

### 运行功能用例

```bash
python3 -m pytest -rA -s \
  attention/chunk_kda_fwd/tests/pytest/test_chunk_kda_fwd.py \
  -v
```

### 运行50次二进制确定性验证

```bash
python3 attention/chunk_kda_fwd/tests/pytest/profile_a5_h96_t16k.py \
  --check-determinism \
  --determinism-repeats 50
```

确定性验证对`attn_out`、`final_state`、`Aqk`和`Akk`执行二进制一致性比较。

### 运行A5关键场景性能验证

创建性能数据目录：

```bash
mkdir -m 700 -p output/chunk_kda_fwd_profiles
```

采集`ChunkKdaFwd`主kernel：

```bash
msopprof \
  --application="python3 attention/chunk_kda_fwd/tests/pytest/profile_a5_h96_t16k.py --repeats 8" \
  --output=output/chunk_kda_fwd_profiles/a5_h96_t16k \
  --aic-metrics=BasicInfo \
  --kernel-name=ChunkKdaFwd \
  --launch-count=5 \
  --warm-up=3 \
  --kill=off
```

采集BSND `beta`布局转换kernel：

```bash
msopprof \
  --application="python3 attention/chunk_kda_fwd/tests/pytest/profile_a5_h96_t16k.py --repeats 8" \
  --output=output/chunk_kda_fwd_profiles/a5_h96_t16k_beta \
  --aic-metrics=BasicInfo \
  --kernel-name=Transpose_float16_int64_high_performance_10001 \
  --launch-count=5 \
  --warm-up=3 \
  --kill=off
```

校验端到端device侧耗时：

```bash
python3 attention/chunk_kda_fwd/tests/pytest/check_msopprof.py \
  output/chunk_kda_fwd_profiles/a5_h96_t16k \
  --component Transpose_float16_int64_high_performance_10001 \
    output/chunk_kda_fwd_profiles/a5_h96_t16k_beta \
  --max-ms 11
```

`check_msopprof.py`分别取主kernel和`beta`布局转换kernel的有效样本最大值后求和。合计耗时超过11 ms，或任一组件未采集到有效样本时，性能门禁失败。

## 输出说明

- pytest功能用例直接输出每条用例的PASS/FAIL结果。
- `profile_a5_h96_t16k.py`在确定性通过时输出`binary_determinism=PASS`。
- `check_msopprof.py`输出每个组件的样本数、最小/平均/最大耗时及端到端最大耗时。
- `output/chunk_kda_fwd_profiles/`为本地性能分析产物，不应提交到代码仓。
