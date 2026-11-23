# quant_sparse_lightning_indexer

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT系列产品</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3系列产品</term>：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2系列产品</term>：支持
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

- 接口功能：本接口用于两级TopK（先按块筛选，再在块内筛选）方案的第二级，在第一级给出的候选块内计算量化Query与Key的加权相关性，选出token级索引供稀疏attention使用。
- 调用流程：取得候选块索引及其有效长度后，调用[quant_sparse_lightning_indexer_metadata](../../quant_sparse_lightning_indexer_metadata/docs/torchapi_quant_sparse_lightning_indexer_metadata.md)生成本级分核信息，再将候选输入和metadata传入本接口。
- 计算公式（每 batch b、query 行 i）：

  $$
  \begin{aligned}
  score(b,i,:) &= \mathrm{DoReduce}(W \odot \mathrm{ReLU}(Scale_Q \odot Scale_K^T \odot (Q_{index}^{Quant} @ (K_{index}^{Quant})^{T}))) \\
  score'(b,i,p) &=
  \begin{cases}
      score(b,i,p) & \lfloor p / candidateBlockSize \rfloor \in candSet(b,i) \\
      -\infty      & \text{其它（不参与选择）}
  \end{cases} \\
  nvalid(b,i) &= \left|\{\, p : score'(b,i,p) > -\infty \,\}\right| \\
  sparseIndices(b,i,k) &=
  \begin{cases}
      \mathrm{topK}(score'(b,i,:))_k & k < \min(topk,\ nvalid(b,i)) \\
      -1 & \min(topk,\ nvalid(b,i)) \le k < topk
  \end{cases}
  \end{aligned}
  $$

  其中 `candidateBlockSize` = 属性 `candidate_block_size`，`candSet` 为候选块号集合。

  `DoReduce`表示沿Query的head维度求和，`topK`返回Key位置索引；不满足有效序列长度和掩码约束的位置不参与选择。候选输入规则及输出顺序见[约束说明](#约束说明)。

## 函数原型

```python
cann_ops_transformer.quant_sparse_lightning_indexer(
    query,
    key,
    weights,
    query_dequant_scale,
    key_dequant_scale,
    topk,
    quant_mode,
    *,
    candidate_block_indices=None,
    candidate_block_length=None,
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
    candidate_block_size=8,
    return_value=0
) -> Tuple[Tensor, Tensor]
```

## 参数说明

b表示batch数量，q_s、k_s分别表示Query和Key的序列长度，q_t、k_t表示TND布局下的token总数，q_n、k_n表示head数。d表示注意力头的逻辑维度，当前仅支持128；d_q、d_k表示存储后的物理末维大小，block_num表示物理页总数，block_size表示每页的token数。

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
|--------|----------|-----------|------|----------|-------------|
| query | Tensor | 必选 | 公式中的量化输入$Q_{index}^{Quant}$。不支持空tensor。数据格式为ND。 | int8、float4_e2m1 | layout_q为BSND时shape为(b,q_s,q_n,d_q)；layout_q为TND时shape为(q_t,q_n,d_q) |
| key | Tensor | 必选 | 公式中的量化输入$K_{index}^{Quant}$。不支持空tensor。数据格式为ND，非连续及分页存储规格见约束说明。 | int8、float4_e2m1 | layout_k为BSND时shape为(b,k_s,k_n,d_k)；layout_k为TND时shape为(k_t,k_n,d_k)；PA_BBND布局的存储shape见约束说明 |
| weights | Tensor | 必选 | 公式中的权重系数$W$。不支持空tensor。数据格式为ND。 | float16、float32 | layout_q为BSND时shape为(b,q_s,q_n)；layout_q为TND时shape为(q_t,q_n) |
| query_dequant_scale | Tensor | 必选 | 公式中的$Scale_Q$，表示Index query的反量化系数。不支持空tensor。数据格式为ND。 | float16、float8_e8m0 | quant_mode为5时，layout_q为BSND时shape为(b,q_s,q_n,d/64,2)，layout_q为TND时shape为(q_t,q_n,d/64,2)；quant_mode为2时shape与weights保持一致 |
| key_dequant_scale | Tensor | 必选 | 公式中的$Scale_K$，表示Index key的反量化系数。quant_mode为5且layout_k为PA_BBND时传入空Tensor，其他场景不支持空tensor。数据格式为ND，非连续规格见约束说明。 | float16、float8_e8m0 | quant_mode为5时，layout_k为BSND时shape为(b,k_s,k_n,d/64,2)，layout_k为TND时shape为(k_t,k_n,d/64,2)，layout_k为PA_BBND时为(0,)；quant_mode为2时shape为(block_num,block_size,k_n) |
| topk | int | 必选 | 表示topK阶段需要保留的key token索引数量，当前支持[1, 8192]。 | int64 | - |
| quant_mode | int | 必选 | 表示量化模式，当前支持2/5。2表示qk: int8 per-token-head, scale: fp16, weights: fp16；5表示qk: mxfp4(e2m1), scale: fp8(e8m0)。 | int64 | - |
| candidate_block_indices | Tensor | 可选 | 第一级算子输出的候选逻辑块索引。默认值为None，实际调用时必须传入，须与candidate_block_size对应。数据格式为ND，有效性要求见约束说明。 | int32 | BSND为(b,q_s,k_n,2048)，TND为(q_t,k_n,2048) |
| candidate_block_length | Tensor | 可选 | 每行实际有效候选块数，取值范围为[0,2048]。默认值为None。数据格式为ND，传入规则见约束说明。 | int32 | BSND为(b,q_s,k_n)，TND为(q_t,k_n)，或空Tensor |
| cu_seqlens_q | Tensor | 可选 | 当前batch及前序batch中query的有效token数的累加和，后一个元素的值必须大于等于前一个元素的值。仅layout_q为TND场景下必传，第一个值固定为0。数据格式为ND。layout_q为BSND时不传入。 | int32 | (b+1,) |
| cu_seqlens_k | Tensor | 可选 | 当前batch及前序batch中key的有效token数的累加和，后一个元素的值必须大于等于前一个元素的值。仅layout_k为TND场景下必传，第一个值固定为0。数据格式为ND。其余布局不传入。 | int32 | (b+1,) |
| seqused_q | Tensor | 可选 | 不同batch中query的真实使用长度，每个batch的有效token数不超过query中的维度S大小且不小于0。数据格式为ND。 | int32 | (b,) |
| seqused_k | Tensor | 可选 | 不同batch中key的真实使用长度，每个batch的有效token数不超过key中的维度S大小且不小于0。数据格式为ND，layout_k为PA_BBND时的传入规则见约束说明。 | int32 | (b,) |
| cmp_residual_k | Tensor | 可选 | 表示key压缩前token数量除以cmp_ratio的余数，需满足0 \<= cmp_residual_k\[i\] \< cmp_ratio。需要在mask_mode等于3、cmp_ratio不等于1的场景下使用。数据格式为ND。 | int32 | (b,) |
| block_table | Tensor | 可选 | 表示PageAttention中KV存储使用的block映射表。默认值为None。数据格式为ND，传入规则见约束说明。 | int32 | (b,max_block_num_per_seq) |
| output_idx_offset | Tensor | 可选 | 表示topK结果输出索引所需要加上的偏移，仅叠加到有效索引，无效索引保持-1。默认值为None，表示不叠加偏移。数据格式为ND。 | int32 | BSND为(b,q_s,k_n)，TND为(q_t,k_n) |
| metadata | Tensor | 可选 | 配套quant_sparse_lightning_indexer_metadata生成的分核信息，包含使用核数、分块大小以及每个核的任务起止位置。不支持空tensor。数据格式为ND。 | int32 | (1024,) |
| max_seqlen_q | int | 可选 | query的最大序列长度。-1表示任意可能长度，默认值为-1。 | int64 | - |
| layout_q | str | 可选 | 用于标识输入query的数据排布格式，支持BSND、TND，默认值为BSND。 | string | - |
| layout_k | str | 可选 | 用于标识输入key的数据排布格式，支持BSND、TND、PA_BBND，默认值为BSND。非分页场景须与layout_q相同。 | string | - |
| mask_mode | int | 可选 | 表示mask的模式，0代表defaultMask模式，3代表rightDownCausal模式，默认值为0。 | int64 | - |
| cmp_ratio | int | 可选 | 用于稀疏计算，表示key的压缩倍数。默认值为1，表示无压缩。取值范围为[1, 128]。 | int64 | - |
| candidate_block_size | int | 可选 | 每个候选块包含的逻辑Key位置数，必须与第一级算子使用相同配置。默认值为8，当前仅支持8。 | int64 | - |
| return_value | int | 可选 | 代表是否需要返回sparse_indices对应的sparse_values值。0代表不返回，1代表返回值，默认值为0。 | int64 | - |

## 返回值说明

返回(sparse_indices, sparse_values)。

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
|--------|----------|-----------|------|----------|-------------|
| sparse_indices | Tensor | 必选 | 候选范围内的token索引输出，无效部分填-1。数据格式为ND。 | int32 | layout_q为BSND时为(b,q_s,k_n,topk)；layout_q为TND时为(q_t,k_n,topk) |
| sparse_values | Tensor | 可选 | sparse_indices对应的分数。return_value为1时输出对应分数，无效部分填-inf；为0时返回空Tensor。数据格式为ND。 | bfloat16 | return_value为1时与sparse_indices相同；为0时为(0,) |

## 约束说明

- 该接口支持推理场景下的单算子模式调用。
- 所有输入Tensor必须位于同一NPU设备。
- `query`、`weights`、`query_dequant_scale`和`candidate_block_indices`必须连续。
- `candidate_block_indices`必须传入，`metadata`必须由[quant_sparse_lightning_indexer_metadata](../../quant_sparse_lightning_indexer_metadata/docs/torchapi_quant_sparse_lightning_indexer_metadata.md)生成，不能与第一级算子的metadata混用。
- `key`的head数为1，Query和Key的逻辑head维度均为128。
- `mask_mode`为3且`cmp_ratio`不为1时，必须传入`cmp_residual_k`。

<!-- npu="950" id7 -->
- <term>Ascend 950PR&950DT系列产品</term>：
  - `quant_mode`仅支持5；`query`和非分页`key`支持float4_e2m1，物理末维大小为64。
  - `weights`仅支持float32；`query_dequant_scale`和非分页`key_dequant_scale`支持float8_e8m0。
  - Query的head数仅支持[1,32]。
  - `layout_k`为PA_BBND时，`key`使用uint8存储，shape为(block_num,block_size/8,544)。`block_size`为[16,1024]内16的倍数；每8个token的512字节Key数据和32字节反量化系数合并存储，`key_dequant_scale`必须传入空uint8 Tensor。
  - 当`layout_k`为PA_BBND时，必须传入`seqused_k`和非空的`block_table`，不支持传入`cu_seqlens_k`。
  - 非分页布局不支持传入`block_table`。
  - 当`layout_k`为TND或BSND时，`key`和`key_dequant_scale`必须连续。当`layout_k`为PA_BBND时，`key`仅支持0轴非连续，0轴stride不得小于单页容量。
  - `candidate_block_length`必须传入且连续，每个元素表示对应行的实际有效候选数。对应行前`candidate_block_length`个候选块索引必须合法且互不重复，不能以候选容量代替实际有效候选数。
  - `return_value`支持0、1；输出索引不保证按分数排序，返回分数与索引逐项对应。
  - 仅在`mask_mode`为3且`cmp_ratio`不为1时传入`cmp_residual_k`，其余情况不传入。
  - 取得第一级的候选索引和实际有效候选数后，需生成本级metadata，再调用计算接口。候选有效长度或序列信息变化后，必须重新生成metadata。
  - 支持aclgraph图模式调用。
<!-- end id7 -->

<!-- npu="A3,910b" id8 -->
- <term>Atlas A3系列产品</term>、<term>Atlas A2系列产品</term>：
  - `quant_mode`仅支持2；`query`、`key`仅支持int8，物理末维大小为128；`weights`、`query_dequant_scale`和`key_dequant_scale`仅支持float16。
  - Query的head数仅支持32、64；`topk`仅支持[1,2048]。
  - `layout_k`仅支持PA_BBND，`key`的shape为(block_num,block_size,1,128)，`block_size`为16的整数倍。`block_table`和`seqused_k`必须传入。
  - `key`和`key_dequant_scale`支持第0轴带padding的非连续Tensor，其余轴必须连续。
  - `candidate_block_length`为预留参数，应保持默认值None或传入空Tensor，不支持非空Tensor。
  - `cmp_ratio`仅支持1、2、4、8、16、32、64、128。
  - 候选块索引使用-1标识无效项，无效或越界候选不参与计算。
  - `return_value`仅支持0；输出的有效索引按对应分数降序排列。
  - 图模式暂不支持。
<!-- end id8 -->

## 确定性计算

- 确定性说明：默认支持确定性计算。

## 调用示例

  ```python
  import torch
  import torch_npu
  import cann_ops_transformer

  b, s1, n1, d = 2, 8, 64, 128
  pages_per_batch, block_size, n2 = 4, 64, 1
  block_num = b * pages_per_batch
  key_length = pages_per_batch * block_size
  topk = 2048

  q = torch.randint(-128, 127, (b, s1, n1, d), dtype=torch.int8).npu()
  k = torch.randint(-128, 127, (block_num, block_size, n2, d), dtype=torch.int8).npu()
  w = torch.rand(b, s1, n1, dtype=torch.float16).npu()
  q_scale = torch.rand(b, s1, n1, dtype=torch.float16).npu()
  k_scale = torch.rand(block_num, block_size, n2, dtype=torch.float16).npu()
  seqused_k = torch.full((b,), key_length, dtype=torch.int32).npu()
  block_table = torch.arange(block_num, dtype=torch.int32).npu().reshape(b, pages_per_batch)

  # 构造覆盖全部有效Key位置的候选块；实际业务中使用第一级的候选输出。
  candidate_count = key_length // 8
  cand = torch.full((b, s1, n2, 2048), -1, dtype=torch.int32).npu()
  cand[..., :candidate_count] = torch.arange(candidate_count, dtype=torch.int32).npu()

  # 先生成第二级分核信息，再在候选范围内执行TopK计算。
  metadata = cann_ops_transformer.quant_sparse_lightning_indexer_metadata(
      num_heads_q=n1, num_heads_k=n2, head_dim=d, topk=topk, quant_mode=2,
      seqused_k=seqused_k, batch_size=b, max_seqlen_q=s1,
      max_seqlen_k=key_length, layout_q="BSND", layout_k="PA_BBND",
      mask_mode=0, cmp_ratio=1, candidate_block_size=8)
  out_idx, out_val = cann_ops_transformer.quant_sparse_lightning_indexer(
      q, k, w, q_scale, k_scale, topk, 2,
      candidate_block_indices=cand,
      seqused_k=seqused_k, block_table=block_table, metadata=metadata,
      layout_q="BSND", layout_k="PA_BBND", candidate_block_size=8)
  torch.npu.synchronize()
  print(out_idx.shape)  # torch.Size([2, 8, 1, 2048])
  print(out_val.shape)  # torch.Size([0])
  ```
