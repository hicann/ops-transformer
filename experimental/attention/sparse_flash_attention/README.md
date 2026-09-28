# SparseFlashAttention

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

- API功能：`SparseFlashAttention`算子实现了基于稀疏索引的FlashAttention计算。该算子根据`sparse_indices`指定的稀疏块索引，仅对选定的KV块执行注意力计算（QK^T -> Softmax -> V），避免对全量KV序列进行密集注意力运算，常用于MLA（Multi-head Latent Attention）架构的长序列推理加速场景。

- 计算公式：

    $$
    \text{Attention}(Q, K, V) = \text{softmax}(Q \cdot K_{sparse}^T \cdot \text{scale\_value}) \cdot V_{sparse}
    $$

    其中：
    - `Q`为查询张量，可选拼接`query_rope`（RoPE位置编码部分）参与QK^T计算；
    - `K_sparse`、`V_sparse`为根据`sparse_indices`从完整KV中选取的稀疏子集，K可选拼接`key_rope`；
    - `d`为head维度大小，`scale_value`通常取`d^(-0.5)`；
    - 输出`attention_out`的head维度与`value`一致（不含RoPE部分）。

## 参数说明

| 参数名                 | 输入/输出/属性 | 描述                                                         | 数据类型                        | 数据格式 |
| ---------------------- | -------------- | ------------------------------------------------------------ | ------------------------------- | :------: |
| query                  | 输入           | 查询张量，layout_query为"BSND"时shape为[B, S1, N1, D]；layout_query为"TND"时shape为[T1, N1, D]，N1为query head数。 | FLOAT16、BFLOAT16               | ND       |
| key                    | 输入           | 键张量，layout_kv为"BSND"时shape为[B, S2, N2, D]；layout_kv为"TND"时shape为[T2, N2, D]；layout_kv为"PA_BSND"时shape为[TotalBlocks, BlockSize, N2, D]，N2为KV head数。 | FLOAT16、BFLOAT16               | ND       |
| value                  | 输入           | 值张量，形状与key相同。                                       | FLOAT16、BFLOAT16               | ND       |
| sparse_indices         | 输入           | 稀疏块索引，layout_query为"BSND"时shape为[B, S1, N2, K]；layout_query为"TND"时shape为[T1, N2, K]，K为每个query token选取的稀疏块数（sparse_block_count）。 | INT32                           | ND       |
| block_table            | 可选输入       | PageAttention场景下的块映射表，shape为[B, MaxBlockNum]。      | INT32                           | ND       |
| cur_seq_lengths_query  | 可选输入       | query累积序列长度（前缀和，首元素为0），shape为[B+1]。        | INT64                           | ND       |
| cur_seq_lengths_kv     | 可选输入       | KV累积序列长度（前缀和，首元素为0），shape为[B+1]。           | INT64                           | ND       |
| query_rope             | 必选输入       | query的RoPE编码部分，拼接到query参与QK^T计算，layout_query为"BSND"时shape为[B, S1, N1, D_rope]；layout_query为"TND"时shape为[T1, N1, D_rope]。 | FLOAT16、BFLOAT16               | ND       |
| key_rope               | 必选输入       | key的RoPE编码部分，拼接到key参与QK^T计算，shape规则与key一致。 | FLOAT16、BFLOAT16               | ND       |
| scale_value            | 可选属性       | 缩放因子，通常取head_dim^(-0.5)，默认值为1.0。                | FLOAT                           | -        |
| sparse_block_size      | 可选属性       | 稀疏块大小，每个sparse_index对应连续的KV token数，取值范围[1,128]且为2的幂次方，默认值为1。 | INT64                           | -        |
| layout_query           | 可选属性       | query和输出的数据布局，支持"BSND"和"TND"，默认值为"BSND"。    | STRING                          | -        |
| layout_kv              | 可选属性       | KV的数据布局，支持"BSND"、"TND"和"PA_BSND"（PageAttention），默认值为"BSND"。 | STRING                          | -        |
| sparse_mode            | 可选属性       | 稀疏模式：0表示全量稀疏，3表示下三角因果稀疏，默认值为3。     | INT64                           | -        |
| pre_tokens             | 可选属性       | 注意力窗口前方token数，默认值为INT64_MAX。                    | INT64                           | -        |
| next_tokens            | 可选属性       | 注意力窗口后方token数，默认值为INT64_MAX。                    | INT64                           | -        |
| attention_mode         | 可选属性       | 注意力模式，默认值为0。                                       | INT64                           | -        |
| return_softmax_lse     | 可选属性       | 是否输出softmax的log-sum-exp统计量（max和sum），默认值为False。 | BOOL                            | -        |
| return_float_output    | 可选属性       | 是否以float32类型输出attention_out，默认值为False。           | BOOL                            | -        |
| attention_out          | 输出           | 注意力计算结果，shape与query一致。                            | FLOAT16、BFLOAT16、FLOAT        | ND       |
| softmax_max            | 输出           | softmax的行最大值（仅return_softmax_lse为True时有效），layout_query为"BSND"时shape为[B, N2, S1, G]；layout_query为"TND"时shape为[N2, T1, G]，其中G = N1 / N2；return_softmax_lse为False时shape为[0]。 | FLOAT                           | ND       |
| softmax_sum            | 输出           | softmax的行求和值（仅return_softmax_lse为True时有效），shape规则与softmax_max一致；return_softmax_lse为False时shape为[0]。 | FLOAT                           | ND       |

## 约束说明

- 该接口支持推理场景下使用。
- 该接口支持aclgraph模式。
- `query`、`key`、`value`的dtype必须一致。
- `query_rope`、`key_rope`为必选输入，dtype需与`query`、`key`一致。
- `query`的head维度D固定为512，RoPE head维度固定为64（MLA架构约束）。
- `key`和`value`的形状必须完全一致。
- GQA分组约束：`N1`（query head数）必须是`N2`（KV head数）的整数倍，即`G = N1 / N2`，其中N2仅支持1。
- TND布局下，`cur_seq_lengths_query`和`cur_seq_lengths_kv`为必选参数，存储累积序列长度（前缀和形式，首元素为0）。
- PageAttention场景（`layout_kv="PA_BSND"`）下，`block_table`为必选参数。
- `sparse_indices`中的值表示KV序列中的块索引，值为-1表示无效块（提前终止）。
- `layout_query`和`layout_kv`必须匹配：`layout_query`为"BSND"时，`layout_kv`可为"BSND"或"PA_BSND"；`layout_query`为"TND"时，`layout_kv`可为"TND"或"PA_BSND"。
- PageAttention场景下，block_size为一个block的token数，block_size取值为16的倍数，且最大支持1024，且要求block_size能被sparse_block_size整除。
- 目前所有输入不支持传入空tensor。
- Q_S和S1表示query shape中的S，S2表示key/value shape中的S；Q_N和N1表示num_q_heads，KV_N和N2表示num_kv_heads；Q_T和T1表示query shape中的输入样本序列长度的累加和，KV_T和T2表示key/value shape中的输入样本序列长度的累加和。

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

# 调用算子（TND 布局，不带 PageAttention）
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
    return_softmax_lse=False,
    return_float_output=True,
)
```

更多使用示例见[pytest示例](./tests/pytest/README.md)。
