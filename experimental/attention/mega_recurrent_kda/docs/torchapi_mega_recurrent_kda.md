# mega_recurrent_kda

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT系列产品</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3系列产品</term>：不支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2系列产品</term>：不支持
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2推理产品</term>：不支持
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas推理系列产品</term>：不支持
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas训练系列产品</term>：不支持
<!-- end id6 -->

## 功能说明

对BSH或TH输入执行融合的KDA递推，读取分页checkpoint并更新Conv和replay池。

## 函数原型

调用`torch.ops.cann_ops_transformer.mega_recurrent_kda`。

```python
torch.ops.cann_ops_transformer.mega_recurrent_kda(
    hidden_states: torch.Tensor,
    qkv_projection_weight: torch.Tensor,
    decay_projection_a_weight: torch.Tensor,
    decay_projection_b_weight: torch.Tensor,
    beta_projection_weight: torch.Tensor,
    output_gate_projection_weight: torch.Tensor,
    output_norm_weight: torch.Tensor,
    output_projection_weight: torch.Tensor,
    conv1d_weight: torch.Tensor,
    conv_state: torch.Tensor,
    checkpoint_state: torch.Tensor,
    replay_u: torch.Tensor,
    replay_k: torch.Tensor,
    replay_decay: torch.Tensor,
    state_indices: torch.Tensor,
    conv_num_accepted_tokens: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    scale: float,
    lower_bound: float,
    rms_norm_eps: float = 1e-5,
    cu_seqlens: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
```

## 参数说明

本文按从1开始的方式描述张量的第几维。代码索引如`shape[2]`和`stride(0)`从0开始。

### 维度符号说明

- `B`本次调用的逻辑请求数。BSH取输入第1维，TH取`cu_seqlens.numel() - 1`。
- `b`优化路径条件中的模型批量，满足`B = b * TP`。
- `seq_len`BSH中每个请求的token数。
- `total_tokens`TH中各请求token数之和。
- `H`隐藏维度，固定为7168。
- `N`当前TP实例的head数。TP4、TP8、TP16分别对应24、12、6。
- `D`每个head的维度，固定为128。
- `P`投影通道数，`P = N * D`。
- `block_num`五个状态池共用的物理块数，对应各池第1维。
- `L`每个Conv块的行数，对应`conv_state`第2维，至少为最长序列长度加2。
- `S`每个replay块中每个head的token容量，对应replay池第3维，至少覆盖最长序列。

### 参数表

| 参数名 | 参数类型 | 输入/输出/属性 | 必选/可选 | 数据类型/格式 | 形状或取值 | 描述 | 非连续Tensor |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `hidden_states` | Tensor | 输入 | 必选 | BF16 / ND | `[B,seq_len,H]`或`[total_tokens,H]` | BSH或TH输入。 | × |
| `qkv_projection_weight` | Tensor | 输入 | 必选 | BF16 / NZ | `[3*P,H]` | Q、K、V投影权重。 | × |
| `decay_projection_a_weight` | Tensor | 输入 | 必选 | BF16 / NZ | `[D,H]` | 第一级decay投影。 | × |
| `decay_projection_b_weight` | Tensor | 输入 | 必选 | BF16 / NZ | `[P,D]` | 第二级decay投影。 | × |
| `beta_projection_weight` | Tensor | 输入 | 必选 | BF16 / NZ | `[N,H]` | beta投影。 | × |
| `output_gate_projection_weight` | Tensor | 输入 | 必选 | BF16 / NZ | `[P,H]` | 输出门控投影。 | × |
| `output_norm_weight` | Tensor | 输入 | 必选 | FP32 / ND | `[D]` | 输出RMSNorm权重。 | × |
| `output_projection_weight` | Tensor | 输入 | 必选 | BF16 / NZ | `[H,P]` | 输出投影。 | × |
| `conv1d_weight` | Tensor | 输入 | 必选 | BF16 / ND | `[4,3*P]`、`[3,N,4,D]`或`[3,4,D]` | Conv权重，卷积核长度为4。 | × |
| `conv_state` | Tensor | 输入/输出 | 必选 | BF16 / ND | `[block_num,L,3*P]` | PA Conv池，原地更新。 | 支持第0轴非连续 |
| `checkpoint_state` | Tensor | 输入 | 必选 | FP32 / ND | `[block_num,N,D,D]` | PA checkpoint池，只读。 | 支持第0轴非连续 |
| `replay_u` | Tensor | 输入/输出 | 必选 | FP32 / ND | `[block_num,N,S,D]` | PA更新向量U记录池，原地更新。 | 支持第0轴非连续 |
| `replay_k` | Tensor | 输入/输出 | 必选 | FP16 / ND | `[block_num,N,S,D]` | PA归一化K记录池，原地更新。 | 支持第0轴非连续 |
| `replay_decay` | Tensor | 输入/输出 | 必选 | FP32 / ND | `[block_num,N,S,D]` | PA指数化衰减系数记录池，原地更新。 | 支持第0轴非连续 |
| `state_indices` | Tensor | 输入 | 必选 | int32 / ND | `[B]` | 每个逻辑batch在五个池中共用的块号。 | × |
| `conv_num_accepted_tokens` | Tensor | 输入 | 必选 | int32 / ND | `[B]` | 选择Conv历史窗口的计数。 | × |
| `A_log` | Tensor | 输入 | 必选 | FP32 / ND | `[N]` | log decay参数。 | × |
| `dt_bias` | Tensor | 输入 | 必选 | FP32 / ND | `[N,D]` | decay bias。 | × |
| `scale` | float | 属性 | 必选 | FP32 scalar | — | 有限query scale，常用`D**-0.5`。 | × |
| `lower_bound` | float | 属性 | 必选 | FP32 scalar | — | decay gate下界，在`[-5,0]`内。 | × |
| `rms_norm_eps` | float | 属性 | 可选（默认`1e-5`） | FP32 scalar | — | 输出RMSNorm epsilon。 | × |
| `cu_seqlens` | Tensor | 输入 | 可选（TH必传） | int32/int64 / ND | `[B+1]`或None | TH的序列前缀和；BSH必须为None。 | × |
| `out` | Tensor | 输出 | — | BF16 / ND | `[B,seq_len,H]` / `[total_tokens,H]` | 主输出，形状与输入一致。 | × |

回放元素偏移按各池自身数据类型的元素数计算。

```text
slot * replay.stride(0) + (head * S + token) * D + channel
```

## 返回值说明

返回`(out, conv_state, replay_u, replay_k, replay_decay)`。`out`是与`hidden_states`同形状的BF16张量，后四项依次返回对应的输入池并保留其存储布局。`checkpoint_state`为只读输入。

## 约束说明

- 该接口支持推理场景下使用。
- 调用方式：支持单算子模式和aclgraph图模式（ACL Graph）调用；其中BSH输入支持aclgraph，TH输入需在主机读取序列边界，不支持aclgraph。
- 所有输入张量的维度均为正，不支持空Tensor。

## 确定性/Batch一致性

- 确定性说明：默认支持确定性计算。
- Batch一致性说明：默认不支持Batch一致性；不同批量可能命中不同执行路径（等长4或8 token的优化路径与通用路径），数值结果可能不一致。

## 调用示例

下面的BSH示例使用形状为`[2,4,7168]`的BF16/ND`hidden_states`。六个NZ投影权重、`conv1d_weight`、`output_norm_weight`、`A_log`和`dt_bias`按参数表准备。安装仓库构建的算子包和PyTorch扩展后，导入扩展以注册接口。示例使用默认扩展包名`cann_ops_transformer`，实际包名随构建配置而定。

```python
import torch
import cann_ops_transformer

block_num, N, D = 6, 12, 128
S, L = 8, 10
P = N * D
device = hidden_states.device

conv_state = torch.zeros(2*block_num, L, 3*P, dtype=torch.bfloat16, device=device)[::2]
checkpoint_state = torch.zeros(3*block_num, N, D, D, dtype=torch.float32, device=device)[::3]
replay_u = torch.zeros(2*block_num, N, S, D, dtype=torch.float32, device=device)[::2]
replay_k = torch.zeros(3*block_num, N, S, D, dtype=torch.float16, device=device)[::3]
replay_decay = torch.zeros(4*block_num, N, S, D, dtype=torch.float32, device=device)[::4]
state_indices = torch.tensor([5, 0], dtype=torch.int32, device=device)
conv_num_accepted_tokens = torch.zeros(2, dtype=torch.int32, device=device)

out, conv_state, replay_u, replay_k, replay_decay = torch.ops.cann_ops_transformer.mega_recurrent_kda(
    hidden_states=hidden_states,
    qkv_projection_weight=qkv_projection_weight,
    decay_projection_a_weight=decay_projection_a_weight,
    decay_projection_b_weight=decay_projection_b_weight,
    beta_projection_weight=beta_projection_weight,
    output_gate_projection_weight=output_gate_projection_weight,
    output_norm_weight=output_norm_weight,
    output_projection_weight=output_projection_weight,
    conv1d_weight=conv1d_weight,
    conv_state=conv_state,
    checkpoint_state=checkpoint_state,
    replay_u=replay_u,
    replay_k=replay_k,
    replay_decay=replay_decay,
    state_indices=state_indices,
    conv_num_accepted_tokens=conv_num_accepted_tokens,
    A_log=A_log,
    dt_bias=dt_bias,
    scale=D**-0.5,
    lower_bound=-5.0,
)
# 第一个请求的记录写入块5，第二个请求的记录写入块0。
```

示例返回的后四项分别引用传入的四个可写池。

TH调用使用相同接口。例如，两个请求各有1和3个token时，将输入按请求顺序拼成`[4,7168]`，传入同设备的`cu_seqlens=[0,1,4]`（int32或int64张量），仍为每个请求提供一个块号和一个Conv计数。
