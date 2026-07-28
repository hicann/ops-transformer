# quant\_lightning\_indexer

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
<!-- npu="310p" id8 -->
- <term>Atlas 推理系列产品</term>：不支持
<!-- end id8 -->
<!-- npu="910" id9 -->
- <term>Atlas 训练系列产品</term>：不支持
<!-- end id9 -->

## 功能说明

- **接口功能**：

  `quant_lightning_indexer_metadata`接口用于生成一个任务列表，包含每个AIcore的Attention计算任务的起止点的Batch、Head、以及Q和K的分块的索引，供后续`quant_lightning_indexer`算子使用。

  `quant_lightning_indexer`接口基于一系列操作得到每一个token对应的top-k个位置。主要计算过程为：

  1. 将某个token对应的输入参数`query`（$Q_{index}^{Quant}\in\R^{g\times d}$）乘以给定上下文`key`（$K_{index}^{Quant}\in\R^{S_{k}\times d}$），得到相关性。
  2. 相关性结果与`query`和`key`对应的反量化系数`query_dequant_scale`（$Scale_Q$）和`key_dequant_scale`（$Scale_K^T$）相乘，通过激活函数$ReLU$过滤无效负相关信号后，得到当前Token与所有前序Token的相关性分数向量。
  3. 将其与权重系数`weights`（$W$）相乘后，沿g的方向，选取前$Top-k$个索引值得到输出$sparseIndices$，并输出对应的$sparseValues$，作为Attention的输入。

- **计算公式**：

  $$
  out = Top\text{-}k\left\{[1]_{1\times g}@\left[\left(W@[1]_{1\times S_{k}}\right)\odot ReLU\left(\left(Scale_Q@Scale_K^T\right)\odot\left(Q_{index}^{Quant}@{\left(K_{index}^{Quant}\right)}^T\right)\right)\right]\right\}
  $$

## 函数原型

调用quant_lightning_indexer接口之前，先调用前置接口quant_lightning_indexer_metadata，完成quant_lightning_indexer负载均衡的计算。

```python
cann_ops_transformer.quant_lightning_indexer_metadata(
  num_heads_q,
  num_heads_k,
  head_dim,
  topk,
  quant_mode,
  *,
  cu_seqlens_q=None,
  cu_seqlens_k=None,
  seqused_q=None,
  seqused_k=None,
  cmp_residual_k=None,
  batch_size=0,
  max_seqlen_q=-1,
  max_seqlen_k=-1,
  layout_q="BSND",
  layout_k="BSND",
  mask_mode=0,
  cmp_ratio=1
) -> Tensor
```

```python
cann_ops_transformer.quant_lightning_indexer(
  query,
  key,
  weights,
  query_dequant_scale,
  key_dequant_scale,
  topk,
  quant_mode,
  *,
  cu_seqlens_q=None,
  cu_seqlens_k=None,
  seqused_q=None,
  seqused_k=None,
  cmp_residual_k=None,
  block_table=None,
  output_idx_offset=None,
  metadata=None,
  max_seqlen_q=-1,
  layout_q="BSND",
  layout_k="BSND",
  mask_mode=0,
  cmp_ratio=1,
  return_value=0
) -> (Tensor, Tensor)
```

## 参数说明

>**说明：**
>
> query、key、weights、query_dequant_scale、key_dequant_scale参数维度含义：B（Batch Size）表示输入样本批量大小、S1表示query的输入样本序列长度、S2表示key的输入样本序列长度、N1表示query的多头数、N2表示key的多头数、D（Head Dim）表示注意力头的维度、T1表示query的输入样本序列长度的累加和、T2表示key的输入样本序列长度的累加和、block_num表示PageAttention场景下的block总数、block_size表示PageAttention场景下每个block的token数、g表示GQA的group size（g = N1 / N2）。参数query中的D和参数key中的D值相等，当前仅支持128。

### quant_lightning_indexer_metadata

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
|--------|----------|-----------|------|----------|-------------|
| num_heads_q | int | 必选 | 表示Query的head个数，当前仅支持32/64。 | int32 | - |
| num_heads_k | int | 必选 | 表示Key的head个数，当前仅支持1。 | int32 | - |
| head_dim | int | 必选 | 表示注意力头的维度，当前仅支持128。 | int32 | - |
| topk | int | 必选 | 表示从Query中筛选出的关键稀疏token的个数，当前仅支持[1, 2048]。 | int32 | - |
| quant_mode | int | 必选 | 表示量化模式，当前支持1/2/3/4/5。1表示qk: fp8(e4m3fn) per-token-head, scale: fp32；2表示qk: int8 per-token-head, scale: fp16, w: fp16；3表示qk: mxfp8(e4m3fn), scale: fp8(e8m0)；4表示qk: hifloat8 per-tensor, scale: fp32；5表示qk: mxfp4(e2m1), scale: fp8(e8m0)。 | int32 | - |
| cu_seqlens_q | Tensor | 可选 | 表示不同Batch中Query的有效Sequence Length，仅layout_q为TND场景下必传，第一个值固定为0。数据格式为ND，支持非连续的Tensor。 | int32 | (B+1, ) |
| cu_seqlens_k | Tensor | 可选 | 表示不同Batch中Key的有效Sequence Length，仅layout_k为TND场景下必传，第一个值固定为0。数据格式为ND，支持非连续的Tensor。 | int32 | (B+1, ) |
| seqused_q | Tensor | 可选 | 表示不同Batch中Query实际参与运算的Sequence Length。数据格式为ND，支持非连续的Tensor。 | int32 | (B, ) |
| seqused_k | Tensor | 可选 | 表示不同Batch中Key实际参与运算的Sequence Length。数据格式为ND，支持非连续的Tensor。 | int32 | (B, ) |
| cmp_residual_k | Tensor | 可选 | 表示不同Batch中cmp_kv压缩后Sequence Length的余数，配合cmp_ratio实现cmp_kv部分的mask和负载计算。cmp_ratio不为1且mask_mode为3场景下必传。需满足cmp_residual_k\[i\] \< cmp_ratio。数据格式为ND，支持非连续的Tensor。 | int32 | (B, ) |
| batch_size | int | 可选 | 表示Batch数量，默认值为0。 | int32 | - |
| max_seqlen_q | int | 可选 | 表示Query的最长Sequence Length，-1表示任意可能长度，默认值为-1。 | int32 | - |
| max_seqlen_k | int | 可选 | 表示Key的最长Sequence Length，-1表示任意可能长度，默认值为-1。 | int32 | - |
| layout_q | str | 可选 | 表示Query的排列格式，支持BSND、TND，默认值为BSND。 | string | - |
| layout_k | str | 可选 | 表示Key的排列格式，支持BSND、TND、PA_BBND，默认值为BSND。 | string | - |
| mask_mode | int | 可选 | 表示sparse模式，0表示No mask，3表示rightDownCausal模式，默认值为0。 | int32 | - |
| cmp_ratio | int | 可选 | 表示Key的压缩率，取值范围[1, 128]，默认值为1，表示无压缩。 | int32 | - |

<!-- npu="950" id5 -->
- <term>Ascend 950PR/Ascend 950DT</term>：不支持quant_mode = 2。
<!-- end id5 -->
<!-- npu="A3" id6 -->
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：不支持num_heads_q = 32，不支持quant_mode = 1/3/4/5，不支持layout_k = BSND/TND，不支持cmp_ratio在[1, 128]任意取值，仅支持cmp_ratio = 1/2/4/8/16/32/64/128。
<!-- end id6 -->
<!-- npu="910b" id7 -->
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：不支持num_heads_q = 32，不支持quant_mode = 1/3/4/5，不支持layout_k = BSND/TND，不支持cmp_ratio在[1, 128]任意取值，仅支持cmp_ratio = 1/2/4/8/16/32/64/128。
<!-- end id7 -->

### quant_lightning_indexer

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
|--------|----------|-----------|------|----------|-------------|
| query | Tensor | 必选 | 公式中的量化输入$Q_{index}^{Quant}$。不支持空tensor。数据格式为ND。 | int8、float8_e4m3fn、HIfloat8、float4_e2m1 | layout_q为BSND时shape为(B,S1,N1,D)；layout_q为TND时shape为(T1,N1,D) |
| key | Tensor | 必选 | 公式中的量化输入$K_{index}^{Quant}$。不支持空tensor。数据格式为ND，仅PA_BBND场景下0轴支持非连续。PA_BBND场景下block_size取值为16的倍数，最大支持1024。 | int8、float8_e4m3fn、HIfloat8、float4_e2m1 | layout_k为BSND时shape为(B,S2,N2,D)；layout_k为TND时shape为(T2,N2,D)；layout_k为PA_BBND时shape为(block_num,block_size,N2,D) |
| weights | Tensor | 必选 | 公式中的权重系数$W$。不支持空tensor。数据格式为ND。 | float16、float32 | layout_q为BSND时shape为(B,S1,N1)；layout_q为TND时shape为(T1,N1) |
| query_dequant_scale | Tensor | 必选 | 公式中的$Scale_Q$，表示Index Query的反量化系数。不支持空tensor。数据格式为ND。 | float16、float32、float8_e8m0 | quant_mode为3/5时，layout_q为BSND时shape为(B,S1,N1,D/64,2)，layout_q为TND时shape为(T1,N1,D/64,2)；quant_mode为4时shape必须为(1,)；其他场景shape与weights保持一致 |
| key_dequant_scale | Tensor | 必选 | 公式中的$Scale_K$，表示Index Key的反量化系数。不支持空tensor。数据格式为ND，仅PA_BBND场景下0轴支持非连续。 | float16、float32、float8_e8m0 | quant_mode为3/5时，layout_k为PA_BBND时shape为(block_num,block_size,N2,D/64,2)，layout_k为BSND时shape为(B,S2,N2,D/64,2)，layout_k为TND时shape为(T2,N2,D/64,2)；quant_mode为4时shape必须为(1,)；其他场景分别为(block_num,block_size,N2)、(B,S2,N2)或(T2,N2) |
| topk | int | 必选 | topK阶段需要保留的block数量，当前支持[1, 2048]。 | int32 | - |
| quant_mode | int | 必选 | 表示量化模式，当前支持1/2/3/4/5。1表示qk: fp8(e4m3fn) per-token-head, scale: fp32；2表示qk: int8 per-token-head, scale: fp16, w: fp16；3表示qk: mxfp8(e4m3fn), scale: fp8(e8m0)；4表示qk: hifloat8 per-tensor, scale: fp32；5表示qk: mxfp4(e2m1), scale: fp8(e8m0)。 | int32 | - |
| cu_seqlens_q | Tensor | 可选 | 当前Batch及前序Batch中query的有效token数的累加和，后一个元素的值必须大于等于前一个元素的值。仅layout_q为TND场景下必传，第一个值固定为0。数据格式为ND。 | int32 | (B+1,) |
| cu_seqlens_k | Tensor | 可选 | 当前Batch及前序Batch中key的有效token数的累加和，后一个元素的值必须大于等于前一个元素的值。仅layout_k为TND场景下必传，第一个值固定为0。数据格式为ND。 | int32 | (B+1,) |
| seqused_q | Tensor | 可选 | 不同Batch中query的真实使用长度，每个Batch的有效token数不超过query中的维度S大小且不小于0。数据格式为ND。 | int32 | (B,) |
| seqused_k | Tensor | 可选 | 不同Batch中key的真实使用长度，每个Batch的有效token数不超过key中的维度S大小且不小于0。数据格式为ND。layout_k为PA_BBND时必须传入。 | int32 | (B,) |
| cmp_residual_k | Tensor | 可选 | 表示k压缩前token数量除以cmp_ratio的余数，需满足cmp_residual_k\[i\] \< cmp_ratio。需要在mask_mode等于3、cmp_ratio不等于1的场景下使用。数据格式为ND。 | int32 | (B,) |
| block_table | Tensor | 可选 | 表示PageAttention中KV存储使用的block映射表。不支持空tensor。layout_k为PA_BBND时必须传入。数据格式为ND。 | int32 | (B, S2_max/block_size) |
| output_idx_offset | Tensor | 可选 | 表示topK结果输出索引所需要加上的偏移。值必须大于0，加上偏移后topK index不能超过int32最大值。数据格式为ND。 | int32 | (B,) |
| metadata | Tensor | 可选 | quant_lightning_indexer_metadata算子传入的分核信息，包含使用核数、分块大小以及每个核处理数据的起始点等内容。不支持空tensor。数据格式为ND。 | int32 | (1024,) |
| max_seqlen_q | int | 可选 | query的最大序列长度。-1表示任意可能长度，默认值为-1。 | int32 | - |
| layout_q | str | 可选 | 用于标识输入query的数据排布格式，支持BSND、TND，默认值为BSND。 | string | - |
| layout_k | str | 可选 | 用于标识输入key的数据排布格式，默认值为BSND。 | string | - |
| mask_mode | int | 可选 | 表示mask的模式，0代表defaultMask模式，3代表rightDownCausal模式，默认值为0。 | int32 | - |
| cmp_ratio | int | 可选 | 用于稀疏计算，表示key的压缩倍数。默认值为1，表示无压缩。 | int32 | - |
| return_value | int | 可选 | 代表是否需要返回sparseIndices对应的sparseValues值。0代表不返回，1代表返回值，默认值为0。 | int32 | - |

<!-- npu="950" id10 -->
- <term>Ascend 950PR/Ascend 950DT</term>:
  - query、key在quant_mode为1/3时支持float8_e4m3fn，quant_mode为4时支持HIfloat8，quant_mode为5时支持float4_e2m1，不支持int8。
  - query_dequant_scale和key_dequant_scale在quant_mode为3/5时支持float8_e8m0，quant_mode为1/4时支持float32。
  - weights支持float32，不支持float16。
  - query的N支持32或64。
<!-- end id10 -->
<!-- npu="A3,910b" id11 -->
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>:
  - query、key仅支持int8，不支持float8_e4m3fn和HIfloat8。
  - weights、query_dequant_scale和key_dequant_scale支持float16，不支持float32。
  - query的N仅支持64。
  - layout_k仅支持PA_BBND。
  - cmp_ratio仅支持2的幂次方值：1/2/4/8/16/32/64/128。
  - 不支持return_value功能，不建议传入该参数。
<!-- end id11 -->

## 返回值说明

### quant_lightning_indexer_metadata

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
|--------|----------|-----------|------|----------|-------------|
| metadata | Tensor | 必选 | 每个AIcore的Attention计算任务的Batch、Head、以及Q和K的分块的索引。数据格式为ND，不支持非连续的Tensor。 | int32 | (1024, )  |

### quant_lightning_indexer

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
|--------|----------|-----------|------|----------|-------------|
| sparse_indices | Tensor | 必选 | 公式中的Indices输出。不支持空tensor。无效部分填-1。数据格式为ND。 | int32 | layout_q为BSND时shape为(B,S1,N2,topk)；layout_q为TND时shape为(T1,N2,topk) |
| sparse_values | Tensor | 条件输出 | 公式中的Indices对应的Values输出。当return_value为1时输出对应值；当return_value为0时输出shape为[0]的空tensor。无效部分填-inf。数据格式为ND。 | bfloat16 | layout_q为BSND时shape为(B,S1,N2,topk)；layout_q为TND时shape为(T1,N2,topk)；return_value为0时shape为(0,) |

## 约束说明

- 该接口支持推理场景下使用，支持单算子模式和aclgraph图模式调用。
- quant_lightning_indexer_metadata接口需与quant_lightning_indexer算子配套使用。
- mask_mode所表示的mask模式的详细介绍见[sparse_mode参数说明](../../../../docs/zh/context/sparse_mode_introduction.md)。
<!-- npu="A3,910b" id15 -->
- 该接口要求$W \odot Scale_Q$的结果在float16的表示范围内（Atlas A3/A2）。
<!-- end id15 -->
- 该接口的TopK过程对NaN排序是未定义行为。
- sparse_indices无效部分填-1；sparse_values无效部分填-inf。

## 确定性计算

- 默认支持确定性计算。

## 调用示例

- 单算子模式调用：

  ```python
  import torch
  import torch_npu
  from cann_ops_transformer.ops import quant_lightning_indexer, quant_lightning_indexer_metadata

  B = 2
  S1 = 64
  N1 = 64
  N2 = 1
  D = 128
  block_num = 128
  block_size = 128
  topk = 32

  # 构造输入
  query = torch.randn(B, S1, N1, D).to(torch.float8_e4m3fn).npu()
  key = torch.randn(block_num, block_size, N2, D).to(torch.float8_e4m3fn).npu()
  weights = torch.randn(B, S1, N1).npu()
  query_dequant_scale = torch.randn(B, S1, N1).npu()
  key_dequant_scale = torch.randn(block_num, block_size, N2).npu()
  block_table = torch.arange(block_num, dtype=torch.int32).expand(B, -1).npu()
  seqused_k = torch.full((B,), block_num * block_size, dtype=torch.int32).npu()

  # 生成metadata
  metadata = quant_lightning_indexer_metadata(
      num_heads_q=N1,
      num_heads_k=N2,
      head_dim=D,
      topk=topk,
      quant_mode=1,
      seqused_k=seqused_k,
      batch_size=B,
      max_seqlen_q=S1,
      max_seqlen_k=block_num * block_size,
      layout_q="BSND",
      layout_k="PA_BBND",
      mask_mode=0,
      cmp_ratio=1
  )

  # 执行quant_lightning_indexer
  sparse_indices, sparse_values = quant_lightning_indexer(
      query, key, weights, query_dequant_scale, key_dequant_scale,
      topk=topk,
      quant_mode=1,
      block_table=block_table,
      metadata=metadata,
      seqused_k=seqused_k,
      max_seqlen_q=-1,
      layout_q="BSND",
      layout_k="PA_BBND",
      mask_mode=0,
      cmp_ratio=1,
      return_value=1
  )
  print(f"sparse_indices shape: {sparse_indices.shape}")
  print(f"sparse_values shape: {sparse_values.shape}")
  ```

- aclgraph图模式调用：

  ```python
  import torch
  import torch_npu
  import torchair
  from cann_ops_transformer.ops import quant_lightning_indexer, quant_lightning_indexer_metadata

  B = 2
  S1 = 64
  S2 = 128
  N1 = 64
  N2 = 1
  D = 128
  topk = 32

  query = torch.randn(B, S1, N1, D).to(torch.float8_e4m3fn).npu()
  key = torch.randn(B, S2, N2, D).to(torch.float8_e4m3fn).npu()
  weights = torch.randn(B, S1, N1).npu()
  query_dequant_scale = torch.randn(B, S1, N1).npu()
  key_dequant_scale = torch.randn(B, S2, N2).npu()

  class QuantLightningIndexerNetwork(torch.nn.Module):
      def __init__(self):
          super(QuantLightningIndexerNetwork, self).__init__()

      def forward(self, query, key, weights, query_dequant_scale, key_dequant_scale):
          metadata = torch.ops.cann_ops_transformer.quant_lightning_indexer_metadata(
              num_heads_q=N1,
              num_heads_k=N2,
              head_dim=D,
              topk=topk,
              quant_mode=1,
              batch_size=B,
              max_seqlen_q=S1,
              max_seqlen_k=S2,
              layout_q="BSND",
              layout_k="BSND",
              mask_mode=0,
              cmp_ratio=1
          )

          return torch.ops.cann_ops_transformer.quant_lightning_indexer(
              query, key, weights, query_dequant_scale, key_dequant_scale,
              topk=topk,
              quant_mode=1,
              metadata=metadata,
              max_seqlen_q=-1,
              layout_q="BSND",
              layout_k="BSND",
              mask_mode=0,
              cmp_ratio=1,
              return_value=1
          )

  from torchair.configs.compiler_config import CompilerConfig
  config = CompilerConfig()
  config.mode = "reduce-overhead"
  npu_backend = torchair.get_npu_backend(compiler_config=config)
  torch._dynamo.reset()
  npu_mode = torch.compile(QuantLightningIndexerNetwork(), fullgraph=True, backend=npu_backend, dynamic=False)
  sparse_indices, sparse_values = npu_mode(query, key, weights, query_dequant_scale, key_dequant_scale)
  print(f"sparse_indices shape: {sparse_indices.shape}")
  print(f"sparse_values shape: {sparse_values.shape}")
  ```
