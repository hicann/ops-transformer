# pool\_key\_indexer

## 产品支持情况

<!-- npu="950" id1 -->

- <term>Ascend 950PR/Ascend 950DT</term>：支持

<!-- end id1 -->

<!-- npu="A3" id2 -->

- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：支持

<!-- end id2 -->

<!-- npu="910b" id3 -->

- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：支持

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

- **接口功能**：

  `pool_key_indexer`接口基于一系列操作得到每一个token对应的top-k个位置。与`lightning_indexer`不同，`pool_key_indexer`在Key侧引入了 **池化（Pooling）** 机制，将原始Key序列按`pool_size`个token为一组进行聚合（每个pool为一个token），先在pool级别做TopK选取，再将选中的pool展开为原始token索引。

  主要计算过程为：

  1. 将某个token对应的输入参数`query`（$Q_{index}\in\R^{g\times d}$）乘以池化后的上下文`pool_key`（$K_{index}^{Pool}\in\R^{S_{k}^{pool}\times d}$），其中$S_{k}^{pool} = \lfloor S_{k} / pool\_size \rfloor$，再乘以缩放系数$\frac{1}{\sqrt{d}}$（防止点积过大），得到相关性分数。
  2. 通过激活函数$ReLU$过滤无效负相关信号后，得到当前Token与所有前序Token的pool级相关性分数向量。
  3. 将其与权重系数`weights`（$W$）相乘后，沿g的方向聚合。
  4. 在pool空间选取前$Top\text{-}k / pool\_size$个pool索引，展开为$Top\text{-}k$个token索引。
  5. 追加尾部pool中未被完整pool_size覆盖的有效token（由`pool_tail_k`指定），无效位置填-1。
- **计算公式**：

  $$
  Top\text{-}k \leftarrow Expand\left(Top\text{-}\frac{k}{p}\left\{[1]_{1\times g}@\left[\left(W@[1]_{1\times S_{k}^{pool}}\right)\odot ReLU\left(\left(Q_{index}@{K_{index}^{Pool}}^T\right)\odot\frac{1}{\sqrt{d}}\right)\right]\right\}\right) + AppendTail
  $$

  其中$p = pool\_size$，$d$为注意力头维度（headDim），缩放系数$\frac{1}{\sqrt{d}}$防止点积过大；$Expand$将每个pool索引展开为$p$个连续token索引，$AppendTail$追加尾部pool中`pool_tail_k`个有效token。

## 函数原型

```python
cann_ops_transformer.pool_key_indexer(
  query,
  pool_key,
  weights,
  pool_tail_k,
  *,
  actual_seq_q=None,
  actual_seq_k=None,
  block_table=None,
  q_descale=None,
  k_descale=None,
  layout_q="BSND",
  layout_k="BSND",
  topk=128,
  pool_size=1,
  mask_mode=0,
  quant_mode=-1,
  return_value=False
) -> (Tensor, Tensor)
```

## 参数说明

> **说明：**
>
> - query、pool_key、weights参数维度含义：B（Batch Size）表示输入样本批量大小、S1表示query的输入样本序列长度、S2表示pool_key的输入样本序列长度（pool级别）、N1表示query的多头数、N2表示pool_key的多头数、D（Head Dim）表示注意力头的维度、T1表示query的输入样本序列长度的累加和、T2表示pool_key的输入样本序列长度的累加和。参数query中的D和参数pool_key中的D值相等，当前仅支持128。N2当前仅支持1。

### pool_key_indexer

| 参数名       | 参数类型 | 可选/必选 | 描述                                                                                                                         | 数据类型          | 维度(shape)                                                                                                                |
| ------------ | -------- | --------- | ---------------------------------------------------------------------------------------------------------------------------- | ----------------- | -------------------------------------------------------------------------------------------------------------------------- |
| query        | Tensor   | 必选      | 公式中的输入Q。不支持空tensor。数据格式为ND。                                                                                | bfloat16、float16 | layout_q为BSND时shape为(B,S1,N1,D)；layout_q为TND时shape为(T1,N1,D)                                                        |
| pool_key     | Tensor   | 必选      | 公式中的池化后输入K。不支持空tensor。数据格式为ND，支持非连续的Tensor（仅PA_BBND场景下0轴支持非连续）。                      | bfloat16、float16 | layout_k为BSND时shape为(B,S2,N2,D)；layout_k为TND时shape为(T2,N2,D)；layout_k为PA_BBND时shape为(block_num,block_size,N2,D) |
| weights      | Tensor   | 必选      | 公式中的输入W。不支持空tensor。数据格式为ND。                                                                                | bfloat16、float16 | layout_q为BSND时shape为(B,S1,N1)；layout_q为TND时shape为(T1,N1)                                                            |
| pool_tail_k  | Tensor   | 必选      | 表示每个Batch中最后一个pool未满pool_size时的有效token数。取值范围[0, pool_size-1]。数据格式为ND。                            | int32             | (B,)                                                                                                                       |
| actual_seq_q | Tensor   | 可选      | 表示不同Batch中Query的有效Sequence Length。仅layout_q为TND场景下必传，第一个值固定为0。数据格式为ND，支持非连续的Tensor。    | int32             | (B,) 或 (B+1,)                                                                                                             |
| actual_seq_k | Tensor   | 可选      | 表示不同Batch中pool_key的有效Sequence Length。仅layout_k为TND场景下必传，第一个值固定为0。数据格式为ND，支持非连续的Tensor。 | int32             | (B,) 或 (B+1,)                                                                                                             |
| block_table  | Tensor   | 可选      | 表示PageAttention中KV存储使用的block映射表。仅layout_k为PA_BBND场景下必传。数据格式为ND。                                    | int32             | (B, max_block_num_per_seq)                                                                                                 |
| q_descale    | Tensor   | 可选      | 表示Query的反量化系数。仅quant_mode>=0时有效。数据格式为ND。                                                                 | float             | layout_q为BSND时shape为(B,S1,N1)；layout_q为TND时shape为(T1,N1)                                                            |
| k_descale    | Tensor   | 可选      | 表示pool_key的反量化系数。仅quant_mode>=0时有效。数据格式为ND。                                                              | float             | layout_k为BSND时shape为(B,S2,N2)；layout_k为TND时shape为(T2,N2)；layout_k为PA_BBND时shape为(block_num,block_size,N2)       |
| layout_q     | str      | 可选      | 表示Query的排列格式，支持BSND、TND，默认值为BSND。                                                                           | string            | -                                                                                                                          |
| layout_k     | str      | 可选      | 表示pool_key的排列格式，支持BSND、TND、PA_BBND，默认值为BSND。                                                               | string            | -                                                                                                                          |
| topk         | int      | 可选      | 表示从Query中筛选出的关键稀疏token的个数。取值范围[1, 8192]，需能被pool_size整除。默认值为128。                              | int32             | -                                                                                                                          |
| pool_size    | int      | 可选      | 表示Key侧池化窗口大小。取值范围[1, 128]，默认值为1，表示无池化（退化为lightning_indexer行为）。                              | int32             | -                                                                                                                          |
| mask_mode    | int      | 可选      | 表示mask的模式，0表示No mask，3表示rightDownCausal模式，默认值为0。                                                          | int32             | -                                                                                                                          |
| quant_mode   | int      | 可选      | 表示量化模式。-1表示不量化，0表示FP8 per-token-head量化，1表示mxFP8量化。默认值为-1。                                        | int32             | -                                                                                                                          |
| return_value | bool     | 可选      | 表示是否需要返回Indices对应的Values值。False表示不返回，True表示返回值。默认值为False。                                      | bool              | -                                                                                                                          |

## 返回值说明

| 参数名         | 参数类型 | 可选/必选 | 描述                                                                                                                                               | 数据类型 | 维度(shape)                                                                                                                 |
| -------------- | -------- | --------- | -------------------------------------------------------------------------------------------------------------------------------------------------- | -------- | --------------------------------------------------------------------------------------------------------------------------- |
| sparse_indices | Tensor   | 必选      | 公式中的Indices输出。不支持空tensor。无效部分填-1。数据格式为ND。                                                                                  | int32    | layout_q为BSND时shape为(B,S1,topk+pool_size-1)；layout_q为TND时shape为(T1,topk+pool_size-1)                                 |
| sparse_values  | Tensor   | 条件输出  | 公式中的Indices对应的Values输出。当return_value为True时输出对应值；当return_value为False时输出shape为[0]的空tensor。无效部分填-inf。数据格式为ND。 | float    | layout_q为BSND时shape为(B,S1,topk//pool_size)；layout_q为TND时shape为(T1,topk//pool_size)；return_value为False时shape为(0,) |

## 约束说明

- 该接口支持推理场景下使用。
- 该接口支持单算子模式、TorchAir（GE图）模式和TorchAir（aclgraph）图模式调用。
- B（Batch）表示输入样本批量大小。
- 参数query、pool_key的数据类型应保持一致。
- topk需能被pool_size整除。
- pool_tail_k取值范围为[0, pool_size-1]。取值校验仅对CPU输入在host侧执行；NPU输入经aclnn Tensor变体直传device数据，host侧不做值校验（与TorchAir图模式一致），调用方须自行保证取值合法，否则kernel行为未定义。
- pool_tail_k、actual_seq_q、actual_seq_k支持CPU或NPU输入：全部为CPU时走host取值链路（aclgraph捕获后值为图常量，修改需重新捕获）；任一为NPU时自动切换为device输入链路（无阻塞D2H，aclgraph捕获后修改buffer内容replay自动生效）。
- 当pool_size=1且pool_tail_k=0时，退化为lightning_indexer行为，输出维度为topk。
- sparse_indices无效部分填-1；sparse_values无效部分填-inf。
- 参数actual_seq_q、actual_seq_k要求其值为当前Batch与前序Batch有效token数的累加值（TND场景），后一个元素的值必须大于等于前一个元素的值。
- pool_key支持PA_BBND场景下0轴非连续，其余轴必须连续。
- mask_mode所表示的mask模式的详细介绍见[sparse_mode参数说明](../../../docs/zh/context/sparse_mode_introduction.md)。
- 当layout_k为PA_BBND时，必须传入block_table和actual_seq_k；当layout_k不为PA_BBND时，不支持传入block_table。
- 当quant_mode=-1时，不支持传入q_descale和k_descale。

<!-- npu="A3,910b" id7 -->

- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>:
  - topk取值范围当前仅支持[1, 2048]，以及3072、4096、5120、6144、7168、8192。
  - 当layout_k为PA_BBND时，必须传入actual_seq_k。
  - 当layout_q为BSND时，不支持传入actual_seq_q；当layout_k为BSND时，不支持传入actual_seq_k。
  - quant_mode当前仅支持-1（不量化），quant_mode=0/1暂不支持。

<!-- end id7 -->

<!-- npu="950" id8 -->

- <term>Ascend 950PR/Ascend 950DT</term>:
  - 当layout_q为BSND时，不支持传入actual_seq_q；当layout_k为BSND或PA_BBND时，不支持传入actual_seq_k。
  - 当layout_q为TND时，必须传入actual_seq_q。
  - quant_mode支持-1（不量化）、0（FP8 per-token-head量化）、1（mxFP8量化）。

<!-- end id8 -->

## 确定性计算

默认支持确定性计算。

## 调用示例

- 单算子模式调用（BSND + 因果掩码 + 池化）：

  ```python
  import torch
  import torch_npu
  from cann_ops_transformer import pool_key_indexer

  B = 2
  S1 = 64
  S2 = 256  # pool级别Key序列长度
  N1 = 8
  N2 = 1
  D = 128
  topk = 128
  pool_size = 16
  mask_mode = 3  # 因果掩码
  return_value = True

  # 构造输入
  query = torch.randn(B, S1, N1, D, dtype=torch.float16).npu()
  pool_key = torch.randn(B, S2, N2, D, dtype=torch.float16).npu()
  weights = torch.randn(B, S1, N1, dtype=torch.float16).npu()
  pool_tail_k = torch.tensor([0, 8], dtype=torch.int64).npu()  # batch0尾块满, batch1尾块8个token

  # 执行pool_key_indexer
  sparse_indices, sparse_values = pool_key_indexer(
      query, pool_key, weights, pool_tail_k,
      layout_q="BSND",
      layout_k="BSND",
      topk=topk,
      pool_size=pool_size,
      mask_mode=mask_mode,
      quant_mode=-1,
      return_value=return_value
  )
  print(f"sparse_indices shape: {sparse_indices.shape}")  # (2, 64, 143)
  print(f"sparse_values shape: {sparse_values.shape}")    # (2, 64, 8)
  ```
- 单算子模式调用（TND + 无掩码 + 退化场景pool_size=1）：

  ```python
  import torch
  import torch_npu
  from cann_ops_transformer import pool_key_indexer

  q_lens = [32, 48]
  k_lens = [128, 192]
  N1 = 8
  N2 = 1
  D = 128
  topk = 64
  pool_size = 1
  mask_mode = 0

  total_q = sum(q_lens)
  total_k = sum(k_lens)

  query = torch.randn(total_q, N1, D, dtype=torch.float16).npu()
  pool_key = torch.randn(total_k, N2, D, dtype=torch.float16).npu()
  weights = torch.randn(total_q, N1, dtype=torch.float16).npu()
  pool_tail_k = torch.tensor([0, 0], dtype=torch.int64).npu()
  # TND前缀和
  actual_seq_q = torch.tensor([q_lens[0], q_lens[0] + q_lens[1]], dtype=torch.int64).npu()
  actual_seq_k = torch.tensor([k_lens[0], k_lens[0] + k_lens[1]], dtype=torch.int64).npu()

  sparse_indices, sparse_values = pool_key_indexer(
      query, pool_key, weights, pool_tail_k,
      actual_seq_q=actual_seq_q,
      actual_seq_k=actual_seq_k,
      layout_q="TND",
      layout_k="TND",
      topk=topk,
      pool_size=pool_size,
      mask_mode=mask_mode,
      quant_mode=-1,
      return_value=False
  )
  print(f"sparse_indices shape: {sparse_indices.shape}")  # (80, 64)
  # return_value=False时sparse_values为空tensor
  ```
- TorchAir（GE图）模式调用：

  ```python
  import torch
  import torch_npu
  import torchair
  from cann_ops_transformer import pool_key_indexer

  B = 1
  S1 = 64
  S2 = 256
  N1 = 8
  N2 = 1
  D = 128

  query = torch.randn(B, S1, N1, D, dtype=torch.float16).npu()
  pool_key = torch.randn(B, S2, N2, D, dtype=torch.float16).npu()
  weights = torch.randn(B, S1, N1, dtype=torch.float16).npu()
  pool_tail_k = torch.tensor([0], dtype=torch.int64).npu()

  class PoolKeyIndexerNetwork(torch.nn.Module):
      def forward(self, query, pool_key, weights, pool_tail_k):
          return torch.ops.cann_ops_transformer.pool_key_indexer(
              query, pool_key, weights, pool_tail_k,
              layout_q="BSND",
              layout_k="BSND",
              topk=128,
              pool_size=16,
              mask_mode=3,
              quant_mode=-1,
              return_value=True
          )

  from torchair.configs.compiler_config import CompilerConfig
  config = CompilerConfig()
  config.mode = "max-autotune"
  npu_backend = torchair.get_npu_backend(compiler_config=config)
  torch._dynamo.reset()
  npu_mode = torch.compile(PoolKeyIndexerNetwork(), fullgraph=True, backend=npu_backend, dynamic=False)
  sparse_indices, sparse_values = npu_mode(query, pool_key, weights, pool_tail_k)
  print(f"sparse_indices shape: {sparse_indices.shape}")
  ```
- TorchAir（aclgraph）图模式调用：

  ```python
  import torch
  import torch_npu
  import torchair
  from cann_ops_transformer import pool_key_indexer

  B = 1
  S1 = 64
  S2 = 256
  N1 = 8
  N2 = 1
  D = 128

  query = torch.randn(B, S1, N1, D, dtype=torch.float16).npu()
  pool_key = torch.randn(B, S2, N2, D, dtype=torch.float16).npu()
  weights = torch.randn(B, S1, N1, dtype=torch.float16).npu()
  pool_tail_k = torch.tensor([0], dtype=torch.int64).npu()

  class PoolKeyIndexerNetwork(torch.nn.Module):
      def forward(self, query, pool_key, weights, pool_tail_k):
          return torch.ops.cann_ops_transformer.pool_key_indexer(
              query, pool_key, weights, pool_tail_k,
              layout_q="BSND",
              layout_k="BSND",
              topk=128,
              pool_size=16,
              mask_mode=3,
              quant_mode=-1,
              return_value=True
          )

  from torchair.configs.compiler_config import CompilerConfig
  config = CompilerConfig()
  # npugraph_ex 为 aclgraph 捕获/回放路径（等价于已废弃的 reduce-overhead）
  config.mode = "npugraph_ex"
  npu_backend = torchair.get_npu_backend(compiler_config=config)
  torch._dynamo.reset()
  npu_mode = torch.compile(PoolKeyIndexerNetwork(), fullgraph=True, backend=npu_backend, dynamic=False)
  # 首次调用触发捕获，其后调用走回放；值输入为NPU tensor时为图输入，
  # 回放期读取buffer当前内容，修改后无需重新捕获
  npu_mode(query, pool_key, weights, pool_tail_k)
  sparse_indices, sparse_values = npu_mode(query, pool_key, weights, pool_tail_k)
  print(f"sparse_indices shape: {sparse_indices.shape}")
  ```
