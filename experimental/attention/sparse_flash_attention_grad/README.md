# SparseFlashAttentionGrad

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                        |    ×     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>        |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>        |    √     |
| <term>Atlas 200I/500 A2 推理系列产品</term>                    |    ×     |
| <term>Atlas 推理系列产品</term>                                |    ×     |
| <term>Atlas 训练系列产品</term>                                |    ×     |

## 功能说明

- API功能：`SparseFlashAttentionGrad`算子实现了`SparseFlashAttention`算子的反向传播计算。该算子根据前向输出的`out`、softmax统计量（`softmax_max`、`softmax_sum`）和上游梯度`d_out`，重计算注意力权重P并计算`query`、`key`、`value`（及可选的`query_rope`、`key_rope`）的梯度，常用于MLA（Multi-head Latent Attention）架构的训练场景。

- 计算公式：

    $$
    \text{dV}_{sparse} = P^T \cdot \text{dO}
    $$

    $$
    \text{dP} = \text{dO} \cdot V_{sparse}^T
    $$

    $$
    \text{dS} = P \odot (\text{dP} - \text{rowsum}(\text{dP} \odot P))
    $$

    $$
    \text{dQ} = \text{scale\_value} \cdot \text{dS} \cdot K_{sparse}
    $$

    $$
    \text{dK}_{sparse} = \text{scale\_value} \cdot \text{dS}^T \cdot Q
    $$

    其中：
    - `P = softmax(Q · K_sparse^T · scale_value)`为前向注意力权重，通过`out`、`softmax_max`、`softmax_sum`重计算得到；
    - `dO`为上游梯度`d_out`；
    - 输出`d_query`、`d_key`、`d_value`分别为Q、K、V的梯度；
    - 若提供了`query_rope`、`key_rope`，还输出`d_query_rope`、`d_key_rope`。

## 参数说明

| 参数名                 | 输入/输出/属性 | 描述                                                         | 数据类型                        | 数据格式 |
| ---------------------- | -------------- | ------------------------------------------------------------ | ------------------------------- | :------: |
| query                  | 输入           | 前向查询张量，layout为"BSND"时shape为[B, S1, N1, D]；layout为"TND"时shape为[T1, N1, D]，N1为query head数。 | FLOAT16、BFLOAT16               | ND       |
| key                    | 输入           | 前向键张量，layout为"BSND"时shape为[B, S2, N2, D]；layout为"TND"时shape为[T2, N2, D]，N2为KV head数。 | FLOAT16、BFLOAT16               | ND       |
| value                  | 输入           | 前向值张量，形状与key相同。                                   | FLOAT16、BFLOAT16               | ND       |
| sparse_indices         | 输入           | 稀疏块索引，layout为"BSND"时shape为[B, S1, N2, K]；layout为"TND"时shape为[T1, N2, K]，K为每个query token选取的稀疏块数（sparse_block_count）。 | INT32                           | ND       |
| d_out                  | 输入           | 上游梯度（loss对前向输出的梯度），shape与query一致。           | FLOAT16、BFLOAT16               | ND       |
| out                    | 输入           | 前向输出（用于重计算softmax），shape与query一致。              | FLOAT16、BFLOAT16               | ND       |
| softmax_max            | 输入           | 前向softmax行最大值，layout为"BSND"时shape为[B, N2, S1, G]；layout为"TND"时shape为[N2, T1, G]。 | FLOAT                           | ND       |
| softmax_sum            | 输入           | 前向softmax行求和值，shape规则与softmax_max一致。              | FLOAT                           | ND       |
| cur_seq_lengths_query  | 可选输入       | query累积序列长度（前缀和，首元素为0），shape为[B+1]。        | INT64                           | ND       |
| cur_seq_lengths_kv     | 可选输入       | KV累积序列长度（前缀和，首元素为0），shape为[B+1]。           | INT64                           | ND       |
| query_rope             | 可选输入       | 前向query的RoPE编码部分，layout为"BSND"时shape为[B, S1, N1, D_rope]；layout为"TND"时shape为[T1, N1, D_rope]。 | FLOAT16、BFLOAT16               | ND       |
| key_rope               | 可选输入       | 前向key的RoPE编码部分，shape规则与key一致。                    | FLOAT16、BFLOAT16               | ND       |
| scale_value            | 属性           | 缩放因子，与前向一致，通常取head_dim^(-0.5)，必选。            | FLOAT                           | -        |
| sparse_block_size      | 属性           | 稀疏块大小，与前向一致，必选。                                 | INT64                           | -        |
| layout                 | 可选属性       | 数据布局，支持"BSND"和"TND"，默认值为"BSND"。                  | STRING                          | -        |
| sparse_mode            | 可选属性       | 稀疏模式：0表示全量稀疏，3表示下三角因果稀疏，默认值为3。     | INT64                           | -        |
| pre_tokens             | 可选属性       | 注意力窗口前方token数，默认值为INT64_MAX。                    | INT64                           | -        |
| next_tokens            | 可选属性       | 注意力窗口后方token数，默认值为INT64_MAX。                    | INT64                           | -        |
| deterministic          | 可选属性       | 是否使用确定性算法（可能牺牲性能换取结果一致性），默认值为False。 | BOOL                            | -        |
| d_query                | 输出           | query的梯度，shape与query一致。                               | FLOAT16、BFLOAT16               | ND       |
| d_key                  | 输出           | key的梯度，shape与key一致。                                   | FLOAT16、BFLOAT16               | ND       |
| d_value                | 输出           | value的梯度，shape与value一致。                               | FLOAT16、BFLOAT16               | ND       |
| d_query_rope           | 可选输出       | query_rope的梯度（仅提供query_rope时有效），shape与query_rope一致。 | FLOAT16、BFLOAT16               | ND       |
| d_key_rope             | 可选输出       | key_rope的梯度（仅提供key_rope时有效），shape与key_rope一致。 | FLOAT16、BFLOAT16               | ND       |

## 约束说明

- 该接口支持训练场景下使用。
- 该接口支持aclgraph模式。
- 所有输入张量的dtype和形状必须与前向调用时一致。
- `softmax_max`和`softmax_sum`必须由前向算子（`return_softmax_lse=true`）产出。
- `query`的head维度D固定为512，RoPE head维度固定为64（MLA架构约束）。
- GQA分组约束：`N1`（query head数）必须是`N2`（KV head数）的整数倍，即`G = N1 / N2`。
- TND布局下，`cur_seq_lengths_query`和`cur_seq_lengths_kv`为必选参数，存储累积序列长度（前缀和形式，首元素为0）。
- 若提供了`query_rope`，则`key_rope`也必须提供，且`d_query_rope`、`d_key_rope`输出同时有效。
- 目前所有输入不支持传入空tensor。

## 调用说明

### 单算子模式调用

```python
import torch
import torch_npu
import numpy as np

device = torch.device('npu:0')
torch.npu.set_device(device)

# 参数设置
B = 2
seqlen_q = [4, 4]
seqlen_kv = [500, 600]
n1 = 32          # query head 数
n2 = 1           # KV head 数
head_dim = 512
rope_head_dim = 64
sparse_block_size = 1
sparse_block_count = 2048
scale_value = 192 ** -0.5

# 累积序列长度（前缀和）
acc_seqlen_q = torch.tensor(np.cumsum([0] + seqlen_q), dtype=torch.int64, device=device)
acc_seqlen_kv = torch.tensor(np.cumsum([0] + seqlen_kv), dtype=torch.int64, device=device)
T1, T2 = sum(seqlen_q), sum(seqlen_kv)

# TND 布局输入
query = torch.randn((T1, n1, head_dim), dtype=torch.float16, device=device)
key = torch.randn((T2, n2, head_dim), dtype=torch.float16, device=device)
value = key.clone()
query_rope = torch.randn((T1, n1, rope_head_dim), dtype=torch.float16, device=device)
key_rope = torch.randn((T2, n2, rope_head_dim), dtype=torch.float16, device=device)

# 生成稀疏索引 (T1, N2, K)
sparse_indices = torch.zeros((T1, n2, sparse_block_count), dtype=torch.int32, device=device) - 1
# ... 填充有效的稀疏块索引 ...

# 前向调用（需要 softmax_max/softmax_sum）
attention_out, softmax_max, softmax_sum = torch_npu.sparse_flash_attention(
    query, key, value, sparse_indices, scale_value,
    cur_seq_lengths_query=acc_seqlen_q,
    cur_seq_lengths_kv=acc_seqlen_kv,
    query_rope=query_rope,
    key_rope=key_rope,
    sparse_block_size=sparse_block_size,
    layout_query="TND",
    layout_kv="TND",
    sparse_mode=3,
    return_softmax_lse=True,
    return_float_output=False,
)

# 模拟上游梯度
d_out = torch.randn_like(attention_out)

# 反向调用
d_query, d_key, d_value, d_query_rope, d_key_rope = torch_npu.sparse_flash_attention_grad(
    query, key, value, sparse_indices, d_out, attention_out,
    softmax_max, softmax_sum,
    cur_seq_lengths_query=acc_seqlen_q,
    cur_seq_lengths_kv=acc_seqlen_kv,
    query_rope=query_rope,
    key_rope=key_rope,
    scale_value=scale_value,
    sparse_block_size=sparse_block_size,
    layout="TND",
    sparse_mode=3,
)
```

更多使用示例见[pytest示例](./tests/pytest/README.md)。
