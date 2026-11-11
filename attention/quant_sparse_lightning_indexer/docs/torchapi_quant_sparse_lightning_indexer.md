# quant_sparse_lightning_indexer

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT系列产品</term>：不支持
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

- **接口功能**：

  `quant_sparse_lightning_indexer` 是两级 TopK（two-level TopK，先按块粗筛再在块内精筛）方案的
  **第二级**（consumer，候选消费级）接口。它在 `quant_lightning_indexer`（第一级，source）
  的量化加权 ReLU 打分框架上，消费第一级输出的块级候选索引 `candidate_block_indices`，
  只在候选块覆盖的位置上做 token 级 TopK：非候选块位置的分数被置为负无穷（不参与选择），
  有效位置不足 `topk` 时剩余槽位输出 `-1`。

- **计算公式**（每 batch b、query 行 i）：

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

  其中 `candidateBlockSize` = 属性 `candidate_block_size`，`candSet` 为候选块号集合
  （重复/越界/-1 按"非候选"处理）。

- **调用流程**：`quant_sparse_lightning_indexer_metadata`（本算子的前置接口，生成分核信息 `metadata`）→
  `quant_lightning_indexer`（`candidate_topk_blocks=2048` 开启 source，输出块级候选索引）→
  本接口（输入 `candidate_block_indices`）→ 稀疏 attention。

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
    candidate_block_indices,
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
    candidate_block_size=8
) -> (Tensor, Tensor)
```

返回 `(sparse_indices, sparse_values)`：

- `sparse_indices`：INT32，BSND `[B,S1,N2,topk]` / TND `[T,N2,topk]`（候选内 TopK 索引，无效槽一律 -1）；
- `sparse_values`：arch22（Atlas A2/A3）上 `return_value` 固定为 0，恒为空 tensor `(0,)`，BFLOAT16。

## 参数说明

>**说明：**<br>
>
>- query、key、weights、query_dequant_scale、key_dequant_scale 的参数维度含义与 `quant_lightning_indexer` 一致：
>  b（batch size）、q_s/q_t（q 的序列长度/累加和）、k_s/k_t（k 的序列长度/累加和）、q_n/k_n（q/k 的多头数）、
>  d（head dim，当前仅支持 128）。
>- PageAttention 场景下 key 的 shape 为 (block_num, block_size, k_n, d)，block_num 为 block 总数，
>  block_size 为一个 block 的 token 数（为 16 的整数倍）。

|  参数名   | 参数类型|  可选/必选 |    描述    |     数据类型    |   维度(shape)   |
|---------|------------|----------|---------------|---------------|------------|
|  query  |  Tensor |  必选  |  公式中的 $Q_{index}^{Quant}$，不支持非连续。layout_q 为 BSND 时 shape 为 (b,q_s,q_n,d)；layout_q 为 TND 时 shape 为 (q_t,q_n,d)。  |  int8   |  3-4  |
|  key  |  Tensor |  必选  |  公式中的 $K_{index}^{Quant}$，`key`/`key_dequant_scale` 支持 0 轴 padding 型非连续（仅 PA_BBND 布局，其余轴必须连续）。layout_k 为 PA_BBND 时 shape 为 (block_num,block_size,k_n,d)。  |  int8   |  3-4  |
|  weights  |  Tensor |  必选  |  公式中的 $W$，不支持非连续。layout_q 为 BSND 时 shape 为 (b,q_s,q_n)；layout_q 为 TND 时 shape 为 (q_t,q_n)。  |  float16   |  2-3  |
|  query_dequant_scale  |  Tensor |  必选  |  公式中的 $Scale_Q$，不支持非连续，shape 与 `weights` 一致。  |  float16   |  2-3  |
|  key_dequant_scale  |  Tensor |  必选  |  公式中的 $Scale_K$，支持 0 轴非连续，shape 为移除 `key` 的 D 轴。  |  float16   |  2-3  |
|  topk  |  int  |  必选  |  Top-k 的 k 值，取值 [1, 2048]。  |  int64   |  -  |
|  quant_mode  |  int  |  必选  |  量化模式，本接口**仅支持 2**（INT8 量化），其它取值被 host 侧校验拒绝。  |  int64   |  -  |
|  candidate_block_indices  |  Tensor  |  必选  |  第一级算子 `quant_lightning_indexer` 开启 source（`candidate_topk_blocks=2048`）时输出的块级候选索引。**末维即候选块个数（当前仅支持 2048）**；元素个数必须等于 BSND：B\*S1\*N2\*末维，TND：T\*N2\*末维。值域 [0,numBlocks) 或 -1（无效槽）。  |  int32   |  3-4  |
|  candidate_block_length  |  Tensor  |  可选  |  预留接口（本期不消费）。约束见下文“约束说明”。  |  int32   |  -  |
|  cu_seqlens_q  |  Tensor  |  可选  |  q 有效长度前缀和，第一个元素固定为 0。layout_q 为 TND 时必传；BSND 时不能传入。shape 为 (b+1,)。  |  int32   |  1  |
|  cu_seqlens_k  |  Tensor  |  可选  |  k 有效长度前缀和。layout_k 为 PA_BBND 时不能传入。  |  int32   |  1  |
|  seqused_q  |  Tensor  |  可选  |  q 每 batch 的有效 token 数，layout_q 为 BSND 时可选传入。shape 为 (b,)。  |  int32   |  1  |
|  seqused_k  |  Tensor  |  可选  |  k 每 batch 的有效长度；layout_k 为 PA_BBND 时**必传**。shape 为 (b,)。  |  int32   |  1  |
|  cmp_residual_k  |  Tensor  |  可选  |  压缩场景下 key 的残余长度，需满足 0 <= cmp_residual_k[i] < cmp_ratio。shape 为 (b,)。  |  int32   |  1  |
|  block_table  |  Tensor  |  可选  |  PageAttention 中 KV 存储使用的 block 映射表；layout_k 为 PA_BBND 时**必传**。shape 为 (b, max_block_num_per_seq)。  |  int32   |  2  |
|  output_idx_offset  |  Tensor  |  可选  |  输出索引的偏移量，仅加到 `sparse_indices` 的有效槽上（-1 填充槽保持 -1）；维数比 `sparse_indices` 少 1：BSND 为 (b,q_s,k_n)，TND 为 (q_t,k_n)。  |  int32   |  2-3  |
|  metadata  |  Tensor  |  必选  |  `quant_sparse_lightning_indexer_metadata` 生成的分核信息，**不能为空**。shape 固定为 (1024,)。  |  int32   |  1  |
|  max_seqlen_q  |  int  |  可选  |  query 的最大序列长度，默认 -1 表示任意可能长度。  |  int64   |  -  |
|  layout_q  |  str  |  可选  |  query 的数据排布格式，支持 "BSND"、"TND"，默认 "BSND"。  |  string   |  -  |
|  layout_k  |  str  |  可选  |  key 的数据排布格式，默认 "BSND"；arch22 上**仅支持 "PA_BBND"**（按 block_table 分页寻址）。  |  string   |  -  |
|  mask_mode  |  int  |  可选  |  mask 模式，0 表示无 mask，3 表示 rightDownCausal（因果）mask，默认 0。  |  int64   |  -  |
|  cmp_ratio  |  int  |  可选  |  key 的压缩倍数，取值 (0, 128] 且必须为 2 的幂，默认 1。  |  int64   |  -  |
|  candidate_block_size  |  int  |  可选  |  候选块粒度（每个块包含的位置数），须 > 0，当前仅支持 8（BlockReduceMax 以 32B 块即 8 个 float32 为归约粒度）。与第一级算子必须一致。  |  int64   |  -  |

## 返回值说明

|  参数名   | 参数类型|  可选/必选 |    描述    |     数据类型    |   维度(shape)   |
|---------|------------|----------|---------------|---------------|------------|
|  sparse_indices  |  Tensor  |  必选  |  候选块内的 Token 级 TopK 索引。有效槽为候选内可达位置按分数降序排列，不足 topk 的槽位与无效位置一律填 -1。layout_q 为 BSND 时 shape 为 (b,q_s,k_n,topk)；layout_q 为 TND 时 shape 为 (q_t,k_n,topk)。  |  int32   |  3-4  |
|  sparse_values  |  Tensor  |  必选  |  arch22 上 `return_value` 固定为 0，恒为空 tensor (0,)。  |  bfloat16   |  1  |

## 约束说明

- 该接口支持推理场景下使用。
- 该接口支持单算子模式调用，图模式暂不支持。
- `quant_mode` 仅支持 2；`query`、`key` 支持 INT8，不支持 FLOAT8_e4m3fn、HIFLOAT8、FLOAT4_e2m1。
- `query_dequant_scale`、`key_dequant_scale` 支持 FLOAT16；`weights` 支持 FLOAT16。
- `topk` 支持 [1, 2048]；不支持 `return_value=1`（`sparse_values` 恒为 shape (0,)）。
- `key` 的 N 仅支持 1；`query` 的 N 与 `key` 的 N 之比仅支持 64 或 32。
- `layout_k` 仅支持 `PA_BBND`，因此 `block_table` 与 `seqused_k` 必传、`cu_seqlens_k` 不能传入；
  `layout_q` 支持 `BSND`/`TND`，为 `TND` 时必须传入 `cu_seqlens_q` 且 `layout_k` 必须为 `PA_BBND`。
- `candidate_block_indices` 必传且 INT32；末维（候选块个数）当前仅支持 2048；元素个数必须等于
  BSND：`B*S1*N2*末维`，TND：`T*N2*末维`，否则 tiling 报错。
- `candidate_block_size` 当前仅支持 8；`candidate_block_length` 为预留接口（本期不消费），仅接受空 tensor，传入非空 tensor 会在 tiling 报错。
- `metadata` 必传且不能为空；`output_idx_offset` 的维数必须比 `sparse_indices` 少 1。
- 其余约束（head_dim=128、block_size 为 16 的整数倍等）同第一级算子，详见
  aclnn 文档 [aclnnQuantSparseLightningIndexer.md](aclnnQuantSparseLightningIndexer.md)。

## 确定性计算

- 确定性说明：默认支持确定性计算。

## 调用示例

- 单算子模式调用（第一级 source + 第二级 consumer 配对，仅 Atlas A2/A3 系列产品支持开启候选）：

  ```python
  import torch
  import torch_npu
  import cann_ops_transformer

  b, s1, n1, d = 2, 8, 64, 128
  block_num, block_size, n2 = 4, 64, 1
  cand_blocks = 2048

  q = torch.randint(-128, 127, (b, s1, n1, d), dtype=torch.int8).npu()
  k = torch.randint(-128, 127, (block_num, block_size, n2, d), dtype=torch.int8).npu()
  w = torch.rand(b, s1, n1, dtype=torch.float16).npu()
  q_scale = torch.rand(b, s1, n1, dtype=torch.float16).npu()
  k_scale = torch.rand(block_num, block_size, n2, dtype=torch.float16).npu()
  seqused_k = torch.full((b,), block_num * block_size, dtype=torch.int32).npu()
  block_table = torch.arange(b * block_num, dtype=torch.int32).npu().reshape(b, block_num)

  # 第一级：source（候选块索引），其分核信息由第一级 metadata 接口生成
  meta_v2 = torch.ops.cann_ops_transformer.quant_lightning_indexer_metadata(
      q, k, seqused_k, block_table, 2048, 2)
  _, _, cand = torch.ops.cann_ops_transformer.quant_lightning_indexer(
      q, k, w, q_scale, k_scale, 2048, 2,
      seqused_k=seqused_k, block_table=block_table, metadata=meta_v2,
      layout_q="BSND", layout_k="PA_BBND", candidate_topk_blocks=2048,
      candidate_block_size=8)

  # 第二级：分核信息由本算子的前置接口生成（输出固定 1024），再消费候选做 token 级 TopK
  metadata = torch.ops.cann_ops_transformer.quant_sparse_lightning_indexer_metadata(
      num_heads_q=n1, num_heads_k=1, head_dim=d, topk=2048, quant_mode=2,
      seqused_k=seqused_k, batch_size=b, max_seqlen_q=s1,
      max_seqlen_k=block_num * block_size,
      layout_q="BSND", layout_k="PA_BBND", mask_mode=0, cmp_ratio=1)
  out_idx, out_val = cann_ops_transformer.quant_sparse_lightning_indexer(
      q, k, w, q_scale, k_scale, 2048, 2,
      candidate_block_indices=cand,
      seqused_k=seqused_k, block_table=block_table, metadata=metadata,
      layout_q="BSND", layout_k="PA_BBND", candidate_block_size=8)

  # out_idx: [b, s1, n2, topk]，无效槽为 -1；out_val: 空 tensor (0,)
  ```
