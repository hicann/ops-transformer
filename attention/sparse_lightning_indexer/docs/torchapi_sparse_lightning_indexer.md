# sparse_lightning_indexer

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

  `sparse_lightning_indexer`（candidate consumer）在 LightningIndexerV2 加权 ReLU 打分框架上消费
  source 算子（`lightning_indexer`，即 aclnnLightningIndexerV3 candidate 模式）
  输出的候选块索引，做无 leak 候选内 TopK（可达且命中候选块的位置以真实分数参与排序，
  **非候选块索引一律不进入 topk 输出**，候选内有效数不足 topk 时剩余槽位填 -1），
  输出每个 query 行的 top-k 索引与（`return_value=1` 时的）TopK 分数。

- **计算公式**（每 batch b、query 行 i，N2=1）：

  $$
  \begin{aligned}
  score(b,i,:) &= \mathrm{DoReduce}(W \odot ReLU(Q_{index} @ K_{index}^{T})) \in \R^{S_2} \\
  score'(b,i,p) &= \begin{cases}
      score(b,i,p) & p < vl(b,i) \text{ 且 } \lfloor p / blockSize \rfloor \in candSet(b,i) \\
      -\infty      & \text{其它（} p \geq vl(b,i) \text{，或可达但不在候选内）}
  \end{cases} \\
  nvalid(b,i) &= \left|\{\, p : score'(b,i,p) > -\infty \,\}\right| \\
  sparseIndices(b,i,k) &= \begin{cases}
      \mathrm{topK}(score'(b,i,:))_k & k < \min(topk,\ nvalid(b,i)) \\
      -1 & \min(topk,\ nvalid(b,i)) \le k < topk
  \end{cases} \\
  sparseValues(b,i,k) &= \begin{cases}
      score(b,i,\ sparseIndices(b,i,k)) & k < \min(topk,\ nvalid(b,i)),\ return\_value=1 \\
      \text{恒空 tensor } (0,) & return\_value=0
  \end{cases}
  \end{aligned}
  $$

  其中 `vl(b,i)` 为行可达前缀长度（由 causal mask、seqused_k/cu_seqlens_k 与 cmp_ratio 推导），
  `blockSize` = `candidate_block_size`，`candSet` 为候选块号集合（重复/越界/-1 按"非候选"处理）。

- **调用流程**：先调用 `lightning_indexer`（source，candidate_topk_blocks 非 -1）生成候选块索引，
  再将第 3 个输出 `candidate_block_indices` 作为本接口输入完成 consumer 计算，两侧需在同一 forward 内配对使用。

## 函数原型

```python
cann_ops_transformer.sparse_lightning_indexer(
    q,
    k,
    w,
    topk,
    candidate_block_indices,
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
    return_value=0,
    candidate_block_length=None,
    candidate_block_size=8
) -> (Tensor, Tensor)
```

返回 `(sparse_indices, sparse_values)`：

- `sparse_indices`：INT32，BSND `[B,S1,N2,topk]` / TND `[T,N2,topk]`（无 leak TopK 索引，所有无效槽一律 -1）；
- `sparse_values`：`return_value=0`（或缺省）时恒空 tensor `(0,)`；`return_value=1` 时与
  `sparse_indices` 同 shape，FLOAT32，有效槽为入选位置真实 TopK 分数、-1 槽对应位填 0。

## 参数说明

>**说明：**<br>
>
>- q、k、w 参数维度含义与 `lightning_indexer` 一致：b（batch size）、q_s/q_t（q 的序列长度/累加和）、k_s/k_t（k 的序列长度/累加和）、q_n/k_n（q/k 的多头数）、d（head dim，当前仅支持 128）。
>- PageAttention场景下k的shape为(block_num, block_size, k_n, d)，block_num为block总数，block_size为一个block的token数（取值[16, 1024]且为16的倍数）。

|  参数名   | 参数类型|  可选/必选 |    描述    |     数据类型    |   维度(shape)   |
|---------|------------|----------|---------------|---------------|------------|
|  q  |  Tensor |  必选  |  公式中的输入Q（与`lightning_indexer`相同）。不支持空tensor。layout_q为BSND时shape为(b,q_s,q_n,d)；layout_q为TND时shape为(q_t,q_n,d)。  |   bfloat16、float16（与k一致）   |  3-4  |
|  k  |  Tensor |  必选  |  公式中的输入K（与`lightning_indexer`相同）。不支持空tensor。layout_k为BSND时shape为(b,k_s,k_n,d)；layout_k为TND时shape为(k_t,k_n,d)；layout_k为PA_BBND时shape为(block_num,block_size,k_n,d)，仅PA_BBND下0轴支持非连续。  |   bfloat16、float16   |  3-4  |
|  w  |  Tensor |  必选  |  公式中的输入W（与`lightning_indexer`相同）。不支持空tensor。layout_q为BSND时shape为(b,q_s,q_n)；layout_q为TND时shape为(q_t,q_n)。  |   float   |  2-3  |
|  topk  |  int  |  必选  |  TopK的k值，取值(0, 2048]。  |   int64   |  -  |
|  candidate_block_indices  |  Tensor  |  必选  |  候选块索引，source算子`lightning_indexer`第3个输出直连。不支持空tensor。layout_q为BSND时shape为(b,q_s,k_n,candBlocks)；layout_q为TND时shape为(q_t,k_n,candBlocks)。  |   int32   |  3-4  |
|  cu_seqlens_q  |  Tensor  |  可选  |  q有效长度前缀和，第一个元素固定为0。layout_q为TND时必传。shape为(b+1,)。  |   int32   |  1  |
|  cu_seqlens_k  |  Tensor  |  可选  |  k有效长度前缀和，第一个元素固定为0。layout_k为TND时必传、BSND时传入报错、PA_BBND时不传。shape为(b+1,)。  |   int32   |  1  |
|  seqused_q  |  Tensor  |  可选  |  接口对齐保留位，本算子（arch22）不支持传入，非None即被host侧校验拒绝，保持None即可。  |   int32   |  -  |
|  seqused_k  |  Tensor  |  可选  |  k每batch实际（压缩后）长度。layout_k为PA_BBND时必传；BSND/TND可选传入，传入则参与k有效长度计算（actS2Orig = seqused_k×cmp_ratio + cmp_residual_k）。shape为(b,)。  |   int32   |  1  |
|  cmp_residual_k  |  Tensor  |  可选  |  k压缩前token数量除以cmp_ratio的余数。host侧不校验shape/dtype（跨算子约束）。shape为(b,)。  |   int32   |  1  |
|  block_table  |  Tensor  |  可选  |  PageAttention中KV存储使用的block映射表。layout_k为PA_BBND时必传（dim0必须等于b），非PA时传入报错。shape为(b, max_block_num_per_seq)。  |   int32   |  2  |
|  output_idx_offset  |  Tensor  |  可选  |  接口对齐保留位，本算子（arch22）不支持传入，非None即被host侧校验拒绝（输出索引不加偏移），保持None即可。  |   int32   |  -  |
|  metadata  |  Tensor  |  可选  |  接口对齐保留位，本算子（arch22）不消费metadata（arch22使用运行时分核），传入非空tensor会被host侧校验拒绝，保持None即可。  |   int32   |  -  |
|  max_seqlen_q  |  int  |  可选  |  q的最大序列长度。仅支持-1（缺省），非-1即被host侧校验拒绝（arch22 kernel不消费）。  |   int64   |  -  |
|  layout_q  |  str  |  可选  |  q的数据排布格式，支持BSND、TND，默认值为BSND。  |   string   |  -  |
|  layout_k  |  str  |  可选  |  k的数据排布格式，支持BSND、TND、PA_BBND，默认值为BSND；非PA_BBND时需与layout_q一致。  |   string   |  -  |
|  mask_mode  |  int  |  可选  |  mask模式，0表示无mask，3表示rightDownCausal（因果）mask，默认值为0。  |   int64   |  -  |
|  cmp_ratio  |  int  |  可选  |  k的压缩倍数，取值(0, 128]且必须为2的幂，默认值为1；需与source侧配置保持一致。  |   int64   |  -  |
|  return_value  |  int  |  可选  |  是否输出TopK分数，取值∈{0,1}，默认值为0；1=sparse_values输出入选位置真实TopK分数（与sparse_indices同shape，-1槽对应位为0），其它值报错拒绝。  |   int64   |  -  |
|  candidate_block_length  |  Tensor  |  可选  |  预留接口，仅None或numel==0的空tensor（非空报错）。shape为(0,)。  |   int32   |  1  |
|  candidate_block_size  |  int  |  可选  |  候选块粒度，即压缩后K空间中每个候选块包含的位置数。[2,64]内的2的幂（2/4/8/16/32/64），默认值为8；与source侧必须一致。  |   int64   |  -  |

## 返回值说明

|  参数名   | 参数类型|  可选/必选 |    描述    |     数据类型    |   维度(shape)   |
|---------|------------|----------|---------------|---------------|------------|
|  sparse_indices  |  Tensor  |  必选  |  topk索引输出（无 leak TopK）。有效槽为候选内可达位置按分数降序排列；所有无效槽（不可达/可达但非候选）一律填-1，候选内有效数不足topk时剩余槽位全-1（不输出任何候选外索引），全无效行输出全-1。layout_q为BSND时shape为(b,q_s,k_n,topk)；layout_q为TND时shape为(q_t,k_n,topk)。  |   int32   |  3-4  |
|  sparse_values  |  Tensor  |  必选  |  <li>return_value=0（或缺省）时恒为空tensor (0,)；return_value=1时与sparse_indices同shape，有效槽为入选位置真实TopK分数（fp32打分域直出），-1槽对应位填0。</li><li>shape为1（return_value=0）或3-4（return_value=1）</li>  |   float   |  1或3-4  |

## 约束说明

- 该接口支持推理场景下使用。
- 该接口支持单算子模式调用。
- `candidate_block_indices` 必传且 INT32；candBlocks（shape 末维）∈ (0, 2048] 且为 64 的倍数
  （由输入 shape 推导，无属性）；值域 [0,numBlocks) 或 -1、槽位无序、允许重复/越界/-1 脏数据为跨算子约束
  （host/device 不校验，越界候选自然按非候选降级）。
- `candidate_block_size` ∈ {2,4,8,16,32,64}，默认 8；与 source 侧必须一致。
- `topk` ∈ (0, 2048]；`return_value` ∈ {0,1}（缺省 0），其它值报错拒绝
  （"Return_value only supports 0 or 1"）。
- `candidate_block_length` 预留，仅 None 或 numel==0 的空 tensor（非空报错）。
- `cmp_ratio` 由本算子 host 强校验 (0, 128] 且 2 的幂；上游 source 算子 host 仅校验 (0, 128]，两侧配置需保持一致。
- seqused_q / output_idx_offset / metadata / max_seqlen_q≠-1 由 arch22 host 侧校验拒绝（保持缺省即可）。
- 其余约束（layout 组合、PA block_table、N2=1、head_dim=128 等）同上游算子 `lightning_indexer`，
  详见 aclnn 文档 [aclnnSparseLightningIndexer.md](aclnnSparseLightningIndexer.md)。

## 确定性计算

- 确定性说明：默认支持确定性计算。

## 调用示例

- 单算子模式调用（source + consumer 配对，仅 Atlas A2/A3 系列产品支持开启 candidate）：

  ```python
  import torch
  import torch_npu
  import cann_ops_transformer

  q = torch.randn(2, 8, 8, 128, dtype=torch.float16).npu()
  k = torch.randn(2, 2048, 1, 128, dtype=torch.float16).npu()
  w = torch.rand(2, 8, 8, dtype=torch.float32).npu()

  # source（lightning_indexer candidate 模式，仅Atlas A2/A3支持开启）：生成候选块索引（同一 forward 内与
  # consumer 配对；返回4元组，第3个输出 candidate_block_indices 即 consumer 的候选输入）
  sparse_idx, _, cand, _ = torch.ops.cann_ops_transformer.lightning_indexer(
      q, k, w, 128, candidate_topk_blocks=2048, candidate_block_size=8,
      mask_mode=3, cmp_ratio=1)

  # consumer：消费候选，输出无 leak topk（metadata 缺省 None 纯透传，arch22 kernel 不消费）
  out_idx, out_val = cann_ops_transformer.sparse_lightning_indexer(
      q, k, w, 128, cand,
      candidate_block_size=8, mask_mode=3, cmp_ratio=1)

  # return_value=1：sparse_values 输出真实 TopK 分数（与 sparse_indices 同 shape，fp32；
  # 有效槽为入选位置真实分数，-1 槽对应位为 0）
  out_idx, out_val = cann_ops_transformer.sparse_lightning_indexer(
      q, k, w, 128, cand,
      candidate_block_size=8, mask_mode=3, cmp_ratio=1, return_value=1)
  ```
