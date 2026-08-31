# kv_quant_sparse_flash_attention

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR/Ascend 950DT</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：不支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：不支持
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2 推理产品</term>：不支持
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas 推理系列产品</term>：不支持
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas 训练系列产品</term>：不支持
<!-- end id6 -->

## 功能说明

- 接口功能：
  `kv_quant_sparse_flash_attention`是`cann_ops_transformer`的扩展`torch`接口，用于调用`KvQuantSparseFlashAttentionV2`算子完成量化和稀疏场景下的注意力计算。该算子仅支持Ascend 950系列产品（A5架构）。

  随着大模型上下文长度的增加，Sparse Attention的重要性与日俱增。该技术通过“只计算关键部分”大幅减少计算量，然而会引入大量的离散访存，造成数据搬运时间增加，进而影响整体性能。`kv_quant_sparse_flash_attention`在`sparse_flash_attention`的基础上支持了Per-Token-Head-Tile-128量化输入，并针对离散访存进行了指令缩减及搬运聚合的细致优化。

- 计算公式：

  $$
  Attention=\text{softmax}(\frac{Q @ \text{Dequant}({\tilde{K}^{INT8}},{Scale_K})^T}{\sqrt{d_k}})@\text{Dequant}(\tilde{V}^{INT8},{Scale_V}),
  $$

  其中$\tilde{K},\tilde{V}$为基于某种选择算法（如`LightningIndexer`）得到的重要性较高的Key和Value，一般具有稀疏或分块稀疏的特征，$d_k$为$Q,\tilde{K}$每一个头的维度，$\text{Dequant}(\cdot,\cdot)$为反量化函数。

## 函数原型

```python
cann_ops_transformer.kv_quant_sparse_flash_attention(
    query,
    key,
    value,
    sparse_indices,
    scale_value,
    key_quant_mode,
    value_quant_mode,
    *,
    key_dequant_scale=None,
    value_dequant_scale=None,
    block_table=None,
    actual_seq_lengths_query=None,
    actual_seq_lengths_kv=None,
    sparse_block_size=1,
    layout_query="BSND",
    layout_kv="BSND",
    sparse_mode=3,
    pre_tokens=9223372036854775807,
    next_tokens=9223372036854775807,
    attention_mode=0,
    quant_scale_repo_mode=1,
    tile_size=128,
    rope_head_dim=64,
    key_dtype=None,
    value_dtype=None,
    sinks=None,
    return_softmax_lse=False
) -> (Tensor, Tensor, Tensor)
```

## 参数说明

### 常见字段释义

|    命名    |                            含义                            |
| :---------: | :---------------------------------------------------------: |
|      B      |      输入样本batch大小                                        |
|     Q_S     |      输入query的序列长度                                       |
|    KV_S     |      输入key/value的序列长度                                   |
|     Q_N     |      输入query的头数                                          |
|    KV_N     |      输入key/value的头数                                       |
|     Q_D     |      输入query的注意力头维度                                   |
|    KV_D     |      输入key/value的注意力头维度                                |
|     Q_T     |      输入query所有batch序列长度的累加和（Total Tokens）           |
|    KV_T     |      输入key/value所有batch序列长度的累加和（Total Tokens）       |
|  sparse_size |      一次离散选取的block数                                      |
|   block_num |      PageAttention场景下的block总数                            |
|  block_size |      PageAttention场景下每个block的token数                      |

### kv_quant_sparse_flash_attention

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 数据格式 | 维度 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| query | Tensor | 必选 | attention结构的Q输入，不支持非连续。由相同数据类型的`q_nope`和`q_rope`按D维度拼接得到。`layout_query`为`BSND`时shape为`(B, Q_S, Q_N, Q_D)`；`layout_query`为`TND`时shape为`(Q_T, Q_N, Q_D)`。其中Q_D值仅支持576，即q_nope+q_rope=512+64；Q_N值支持1/2/4/8/16/32/48/64/128。 | float16、bfloat16 | ND | `BSND`：(B, Q_S, Q_N, Q_D)<br>`TND`：(Q_T, Q_N, Q_D) |
| key | Tensor | 必选 | attention结构的K输入。`k_nope`、与query相同数据类型的`k_rope`和float32的量化参数按D维度拼接得到。`layout_kv`为`BSND`时shape为`(B, KV_S, KV_N, KV_D)`；`layout_kv`为`TND`时shape为`(KV_T, KV_N, KV_D)`；`layout_kv`为`PA_BSND`（PageAttention）时shape为`(block_num, block_size, KV_N, KV_D)`，其中block_size取值为16的整数倍，最大支持到1024。KV_N仅支持1；KV_D值仅支持656，即nope+rope*2+dequant_scale*4=512+64*2+4*4。 | float8_e4m3、int8、hifloat8 | ND | `BSND`：(B, KV_S, KV_N, KV_D)<br>`TND`：(KV_T, KV_N, KV_D)<br>`PA_BSND`：(block_num, block_size, KV_N, KV_D) |
| value | Tensor | 必选 | attention结构的V输入。数据类型与key相同。 | float8_e4m3、int8、hifloat8 | ND | 与key保持一致 |
| sparse_indices | Tensor | 必选 | 代表离散取kvCache的索引，不支持非连续。`layout_query`为`BSND`时shape为`(B, Q_S, KV_N, sparse_size)`；`layout_query`为`TND`时shape为`(Q_T, KV_N, sparse_size)`。每行有效值均在前半部分，无效值（-1）均在后半部分，且sparse_size需大于0。当key和value的数据类型为hifloat8时，sparse_size仅支持2048。 | int32 | ND | `BSND`：(B, Q_S, KV_N, sparse_size)<br>`TND`：(Q_T, KV_N, sparse_size) |
| scale_value | float | 必选 | 公式中$d_k$开根号的倒数，代表缩放系数，作为query和key矩阵乘后Muls的scalar值。 | float32 | - | - |
| key_quant_mode | int64 | 必选 | 代表key的量化模式，仅支持传入2，即per_tile量化模式。 | int64 | - | - |
| value_quant_mode | int64 | 必选 | 代表value的量化模式，仅支持传入2，即per_tile量化模式。 | int64 | - | - |
| key_dequant_scale | Tensor | 可选 | 预留参数。 | - | - | - |
| value_dequant_scale | Tensor | 可选 | 预留参数。 | - | - | - |
| block_table | Tensor | 可选 | 表示PageAttention中kvCache存储使用的block映射表。shape为`(B, ceil(KV_S_max/block_size))`，其中第一维长度为B，第二维长度不小于所有batch中最大的KV_S对应的block数量，即KV_S_max/block_size向上取整。 | int32 | ND | (B, ceil(KV_S_max/block_size)) |
| actual_seq_lengths_query | Tensor | 可选 | 表示不同Batch中query的有效token数。不传（None）时表示与query的Q_S长度相同。shape为`(B,)`，每个Batch的有效token数不超过query中的Q_S大小且不小于0。当`layout_query`为`TND`时，该入参必须传入，并以该入参元素的数量作为B值；该入参每个元素的值表示当前batch与之前所有batch的token数总和（前缀和），后一个元素的值必须大于等于前一个元素的值。 | int32 | ND | (B,) |
| actual_seq_lengths_kv | Tensor | 可选 | 表示不同Batch中key和value的有效token数。不传（None）时表示与KV_S长度相同。shape为`(B,)`，每个Batch的有效token数不超过key/value中的KV_S大小且不小于0。当`layout_kv`为`TND`或`PA_BSND`时，该入参必须传入；`layout_kv`为`TND`时，该参数每个元素的值表示当前batch与之前所有batch的token数总和（前缀和），后一个元素的值必须大于等于前一个元素的值。 | int32 | ND | (B,) |
| sinks | Tensor | 可选 | 表示attention结构中可学习的sinks信息，用于维持长文本推理时的稳定性。不支持非连续，shape为`(Q_N,)`。 | float32 | ND | (Q_N,) |
| sparse_block_size | int64 | 可选 | 代表sparse阶段的block大小。sparse_block_size为1时，为Token-wise稀疏化场景；sparse_block_size大于1且小于等于128时，为Block-wise稀疏化场景，块内token共享相同的稀疏化决策。默认值为1。 | int64 | - | - |
| layout_query | string | 可选 | 用于标识输入query的数据排布格式，支持传入`BSND`和`TND`。默认值为`BSND`。 | string | - | - |
| layout_kv | string | 可选 | 用于标识输入key的数据排布格式，支持传入`BSND`、`TND`和`PA_BSND`，`PA_BSND`在开启PageAttention时使用。默认值为`BSND`。 | string | - | - |
| sparse_mode | int64 | 可选 | 表示sparse的模式。sparse_mode为0时，代表全部计算；sparse_mode为3时，代表rightDownCausal模式的mask，对应以右下顶点往左上为划分线的下三角场景。默认值为3。 | int64 | - | - |
| pre_tokens | int64 | 可选 | 用于稀疏计算，表示attention需要和前几个Token计算关联，仅支持$2^{63}-1$。默认值为$2^{63}-1$。 | int64 | - | - |
| next_tokens | int64 | 可选 | 用于稀疏计算，表示attention需要和后几个Token计算关联，仅支持$2^{63}-1$。默认值为$2^{63}-1$。 | int64 | - | - |
| attention_mode | int64 | 可选 | 表示attention的模式，仅支持传入2，表示MLA-absorb模式，即QK的D包含rope和nope两部分，且KV是同一份。 | int64 | - | - |
| quant_scale_repo_mode | int64 | 可选 | 表示量化参数的存放模式，仅支持传入1，表示combine模式，即量化参数和数据混合存放。默认值为1。 | int64 | - | - |
| tile_size | int64 | 可选 | 表示per_tile时每个参数对应的数据块大小，仅在per_tile时有效，仅支持128。默认值为128。 | int64 | - | - |
| rope_head_dim | int64 | 可选 | 表示MLA架构下的rope_head_dim大小，仅在attention_mode为2时有效，仅支持64。默认值为64。 | int64 | - | - |
| key_dtype | int | 可选 | 表示key的数据类型标识。当key为hifloat8类型时可传入`torch_npu.hifloat8`，其他数据类型下保持默认值None。 | int64 | - | - |
| value_dtype | int | 可选 | 表示value的数据类型标识。当value为hifloat8类型时可传入`torch_npu.hifloat8`，其他数据类型下保持默认值None。 | int64 | - | - |
| return_softmax_lse | bool | 可选 | 表示是否返回softmax的lse结果。为True时随attention_out返回softmax_max与softmax_sum两个输出；为False时返回空的占位Tensor。默认值为False。 | bool | - | - |

## 返回值说明

### kv_quant_sparse_flash_attention

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 数据格式 | 维度 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| attention_out | Tensor | 必选 | 代表公式中的输出Attention。输出shape与入参query的shape保持一致，`layout_query`为`BSND`时shape为`(B, Q_S, Q_N, Q_out_D)`，`layout_query`为`TND`时shape为`(Q_T, Q_N, Q_out_D)`，其中Q_out_D=Q_D-rope_head_dim。 | float16、bfloat16 | ND | `BSND`：(B, Q_S, Q_N, Q_D-rope_head_dim)<br>`TND`：(Q_T, Q_N, Q_D-rope_head_dim) |
| softmax_max | Tensor | 可选 | 表示softmax计算过程中每个Query位置的最大值（softmax分子$x-\max(x)$中的$\max(x)$）。`return_softmax_lse=True`时返回有效结果。 | float32 | ND | `BSND`：(B, KV_N, Q_S, Q_N/KV_N)<br>`TND`：(KV_N, Q_T, Q_N/KV_N) |
| softmax_sum | Tensor | 可选 | 表示softmax计算过程中每个Query位置的求和值（softmax分母$\sum\exp(x-\max(x))$）。`return_softmax_lse=True`时返回有效结果。 | float32 | ND | `BSND`：(B, KV_N, Q_S, Q_N/KV_N)<br>`TND`：(KV_N, Q_T, Q_N/KV_N) |

## 约束说明

- 该接口支持单算子模式和TorchAir图模式调用。
- 非PageAttention场景`layout_query`和`layout_kv`取值需要保持一致。
- `return_softmax_lse=False`时返回空shape为`[0]`的float32占位Tensor；`return_softmax_lse=True`时返回float32的softmax_max与softmax_sum结果。
- <term>Ascend 950PR/Ascend 950DT</term>：
  - 参数key、value数据类型仅支持float8_e4m3、int8、hifloat8数据类型。
  - 参数`sparse_block_size`仅支持1。
  - 仅在`layout_kv`为`PA_BSND`时，key支持0轴非连续。
  - 支持可选入参sinks。
  - 当key/value为hifloat8时，`sparse_size`（sparse_indices最后一维）仅支持2048。

## 调用示例

### PageAttention场景（layout_query=BSND，layout_kv=PA_BSND）

```python
import torch
import torch_npu
import math
import random
import cann_ops_transformer

torch_npu.npu.set_device(0)

query_dtype = torch.bfloat16
kv_dtype = torch.int8
B = 1
Q_S = 1
KV_S = 8192
KV_S_act = 4096            # 实际参与计算的KV长度
Q_N = 128
KV_N = 1
Q_D = 512                  # q_nope长度
D_rope = 64
tile_size = 128
block_size = 256
sparse_size = 2048
scale_value = 1.0 / math.sqrt(Q_D)

# query：q_nope(512, bf16) + q_rope(64, bf16) 按D维拼接
query = torch.randn(B, Q_S, Q_N, Q_D + D_rope, dtype=query_dtype).npu()

# key：k_nope(512, int8) + k_rope(64, bf16) + dequant_scale(512/128, fp32) 按D维按字节拼接
block_num = B * (KV_S // block_size)
k_nope = torch.randint(-128, 127, (block_num, block_size, KV_N, Q_D), dtype=kv_dtype).npu()
k_rope = torch.randn(block_num, block_size, KV_N, D_rope, dtype=query_dtype).npu()
dequant_scale = torch.rand(block_num, block_size, KV_N, Q_D // tile_size, dtype=torch.float32).npu()
key = torch.cat((k_nope, k_rope.view(torch.int8), dequant_scale.view(torch.int8)), dim=-1)
value = key.clone()

# sparse_indices：每个query位置离散选取的block索引，无效位置填-1
sparse_indices = torch.full((B, Q_S, KV_N, sparse_size), -1, dtype=torch.int32).npu()
idxs = random.sample(range(KV_S_act - Q_S + 1), sparse_size)
sparse_indices[0, 0, 0, :] = torch.tensor(idxs, dtype=torch.int32).npu()

# block_table：PageAttention的block映射表
block_table = torch.arange(block_num, dtype=torch.int32).reshape(B, -1).npu()

actual_seq_q = torch.tensor([Q_S] * B, dtype=torch.int32).npu()
actual_seq_kv = torch.tensor([KV_S_act] * B, dtype=torch.int32).npu()
sinks = torch.zeros(Q_N, dtype=torch.float32).npu()

attention_out, softmax_max, softmax_sum = torch.ops.cann_ops_transformer.kv_quant_sparse_flash_attention(
    query,
    key,
    value,
    sparse_indices,
    scale_value,
    key_quant_mode=2,
    value_quant_mode=2,
    block_table=block_table,
    actual_seq_lengths_query=actual_seq_q,
    actual_seq_lengths_kv=actual_seq_kv,
    sparse_block_size=1,
    layout_query="BSND",
    layout_kv="PA_BSND",
    sparse_mode=3,
    attention_mode=2,
    quant_scale_repo_mode=1,
    tile_size=tile_size,
    rope_head_dim=64,
    sinks=sinks,
    return_softmax_lse=False)

torch.npu.synchronize()
```

### 全计算场景（layout_query=BSND，layout_kv=BSND，开启return_softmax_lse）

```python
import torch
import torch_npu
import math
import cann_ops_transformer

torch_npu.npu.set_device(0)

query_dtype = torch.float16
kv_dtype = torch.int8
B = 1
Q_S = 16
KV_S = 128
Q_N = 8
KV_N = 1
Q_D = 512
D_rope = 64
tile_size = 128
sparse_size = 128
scale_value = 1.0 / math.sqrt(Q_D)

query = torch.randn(B, Q_S, Q_N, Q_D + D_rope, dtype=query_dtype).npu()

k_nope = torch.randint(-128, 127, (B, KV_S, KV_N, Q_D), dtype=kv_dtype).npu()
k_rope = torch.randn(B, KV_S, KV_N, D_rope, dtype=query_dtype).npu()
dequant_scale = torch.rand(B, KV_S, KV_N, Q_D // tile_size, dtype=torch.float32).npu()
key = torch.cat((k_nope, k_rope.view(torch.int8), dequant_scale.view(torch.int8)), dim=-1)
value = key.clone()

sparse_indices = torch.full((B, Q_S, KV_N, sparse_size), -1, dtype=torch.int32).npu()
for s in range(Q_S):
    sparse_indices[0, s, 0, :sparse_size] = torch.arange(sparse_size, dtype=torch.int32)

attention_out, softmax_max, softmax_sum = torch.ops.cann_ops_transformer.kv_quant_sparse_flash_attention(
    query,
    key,
    value,
    sparse_indices,
    scale_value,
    key_quant_mode=2,
    value_quant_mode=2,
    sparse_block_size=1,
    layout_query="BSND",
    layout_kv="BSND",
    sparse_mode=0,
    attention_mode=2,
    quant_scale_repo_mode=1,
    tile_size=tile_size,
    rope_head_dim=64,
    return_softmax_lse=True)

torch.npu.synchronize()
```
