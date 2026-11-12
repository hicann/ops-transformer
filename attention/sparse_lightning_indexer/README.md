# SparseLightningIndexer

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT系列产品</term>                      |    ×     |
| <term>Atlas A3系列产品</term>                                |    √     |
| <term>Atlas A2系列产品</term>                                |    √     |
| <term>Atlas 200I/500 A2推理产品</term>                       |    ×     |
| <term>Atlas推理系列产品</term>                               |    ×     |
| <term>Atlas训练系列产品</term>                               |    ×     |

## 功能说明

- 算子功能：SparseLightningIndexer（candidate consumer）在
  LightningIndexerV2 的加权 ReLU 打分 + TopK 框架上，**消费上游 source 算子输出的候选块索引**
  （`candidate_block_indices`），对每个 query 行做**无 leak 候选内 TopK**：
  可达且命中候选块的位置以真实分数参与排序，其余位置（不可达 / 可达但不在候选内）一律无效不入选；
  **非候选块索引一律不进入 topk 输出**，候选内有效数不足 topk 时剩余槽位填 -1。
  与 `lightning_indexer`（source 模式，`candidate_topk_blocks != -1`）
  配对使用，构成候选块两级 TopK 的 consumer 侧。

- 计算公式（每 batch b、query 行 i，N2=1）：

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
      score(b,i,\ sparseIndices(b,i,k)) & k < \min(topk,\ nvalid(b,i)),\ \text{return\_value}=1 \\
      \text{恒空 tensor } (0,) & \text{return\_value}=0
  \end{cases}
  \end{aligned}
  $$

  其中 `vl(b,i)` 为行可达前缀长度（由 causal mask、seqused_k/cu_seqlens_k 与 cmp_ratio 推导），
  `blockSize` = `candidate_block_size`，`candSet` 为候选块号集合（重复/越界/-1 按"非候选"处理）。

- 主要计算过程为：

  1. 与 LIV2 相同的打分流程：QK 相关性 → ReLU → W 加权 g 维求和得到 score 行。
  2. 依据候选块索引构建位置级候选掩码：候选块覆盖的位置为候选，其余为非候选。
  3. 候选外可达位置与不可达位置并轨为同一档无效位（score 置 -inf、索引置 -1），不入选。
  4. 沿用 LIV2 的位置级 TopK 管线输出索引（nvalid 不足 topk 时剩余槽位为 -1，全无效行输出全 -1）。

## 参数说明

> **说明：**<br>
> 参数维度含义：B表示Batch Size、S1和S2分别表示q和k的Sequence Length、N1和N2分别表示q和k的Head Num（N2仅支持1）、D表示Head Dim（仅支持128）、T表示q/k的Total Tokens；block_num和block_size分别表示PageAttention场景下的block总数和每个block包含的token数（block_size取值[16, 1024]且为16的倍数）。

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
|--------|---------------|------|----------|----------|
| q | 输入 | 公式中的输入Q（与source算子相同）。不支持空tensor。layout_q为BSND时，shape为[B, S1, N1, D]；layout_q为TND时，shape为[T, N1, D]。 | FLOAT16、BFLOAT16 | ND |
| k | 输入 | 公式中的输入K（与source算子相同）。不支持空tensor。layout_k为PA_BBND时，shape为[block_num, block_size, N2, D]；layout_k为BSND时，shape为[B, S2, N2, D]；layout_k为TND时，shape为[T, N2, D]。 | FLOAT16、BFLOAT16 | ND |
| w | 输入 | 公式中的输入W（与source算子相同）。不支持空tensor。layout_q为BSND时，shape为[B, S1, N1]；layout_q为TND时，shape为[T, N1]。 | FLOAT | ND |
| cu_seqlens_q | 输入 | 当前Batch及前序Batch中q的有效token数的累加和（前缀和）。layout_q为TND时必传；BSND时不建议传入。shape为[B+1, ]。 | INT32 | ND |
| cu_seqlens_k | 输入 | 当前Batch及前序Batch中k的有效token数的累加和（前缀和）。layout_k为TND时必传；BSND时传入报错；PA_BBND时不传。shape为[B+1, ]。 | INT32 | ND |
| seqused_q | 输入 | 接口对齐保留位，本算子（arch22）不支持传入，非空即被host侧校验拒绝，传空即可。 | INT32 | ND |
| seqused_k | 输入 | 不同Batch中k的真实使用长度（压缩后）。layout_k为PA_BBND时必传；BSND/TND时可选传入，传入则参与k有效长度计算（actS2Orig = seqused_k×cmp_ratio + cmp_residual_k）。shape为[B, ]。 | INT32 | ND |
| cmp_residual_k | 输入 | 表示k压缩前token数量除以cmp_ratio的余数。host侧不校验shape/dtype（跨算子约束）。shape为[B, ]。 | INT32 | ND |
| block_table | 输入 | 表示PageAttention中KV存储使用的block映射表。layout_k为PA_BBND时必传（dim0必须等于B），非PA时传入报错。shape为[B, maxBlockNumPerSeq]。 | INT32 | ND |
| output_idx_offset | 输入 | 接口对齐保留位，本算子（arch22）不支持传入，非空即被host侧校验拒绝（输出索引不加偏移），传空即可。 | INT32 | ND |
| metadata | 输入 | 元数据（接口对齐保留位），arch22 kernel不消费，非空即被host侧校验拒绝，传空即可。 | INT32 | ND |
| candidate_block_indices | 输入 | 候选块索引（source算子lightning_indexer输出直连）。必传，不支持空tensor。值为batch内压缩后K位置空间的相对块号（块号×candidate_block_size为起始位置）或-1（无效槽），槽位无序，允许重复/越界/-1（按"非候选"处理），不叠加output_idx_offset。layout_q为BSND时，shape为[B, S1, N2, candBlocks]；layout_q为TND时，shape为[T, N2, candBlocks]；candBlocks为shape末维，取值(0, 2048]且为64的倍数。 | INT32 | ND |
| candidate_block_length | 输入 | 预留接口，仅接受空tensor（None/numel==0，非空报错）。shape为(0,)。 | INT32 | ND |
| sparse_indices | 输出 | 公式中的输出Top-k索引（无leak TopK索引）。有效槽为候选内可达位置按分数降序排列（同分次序不承诺）；候选内有效数nvalid < topk时，其后槽位全部填-1（含候选外可达位置在内的所有无效槽一律-1，不输出任何候选外索引）；全无效行输出全-1。layout_q为BSND时，shape为[B, S1, N2, topk]；layout_q为TND时，shape为[T, N2, topk]。 | INT32 | ND |
| sparse_values | 输出 | 公式中的输出Top-k值。return_value为1时与sparse_indices同shape（dtype FLOAT32），有效槽为入选位置真实TopK分数（fp32打分域直出），-1槽对应位填0；return_value为0（或缺省）时恒为空tensor (0,)，kernel不写入。 | FLOAT | ND |
| topk | 属性 | Top-k的k值，取值范围(0, 2048]（超限直接报错）。 | INT64 | - |
| max_seqlen_q | 属性 | q的最大序列长度，仅支持-1（arch22 kernel不消费），非-1报错。 | INT64 | - |
| layout_q | 属性 | q的数据排布格式，支持BSND/TND，默认值为BSND。 | STRING | - |
| layout_k | 属性 | k的数据排布格式，支持BSND/TND/PA_BBND，默认值为BSND；非PA_BBND时需与layout_q一致。 | STRING | - |
| mask_mode | 属性 | mask模式，0表示不使用mask，3表示因果mask（rightDownCausal）。默认值为0。 | INT64 | - |
| cmp_ratio | 属性 | 压缩倍数，取值(0, 128]且为2的幂，默认值为1；需与source侧配置保持一致。 | INT64 | - |
| return_value | 属性 | 是否输出TopK分数，取值{0,1}，默认值为0；1=输出真实TopK分数（sparse_values与sparse_indices同shape，无效槽value=0）；其它值拒绝。 | INT64 | - |
| candidate_block_size | 属性 | 候选块粒度，[2,64]内的2的幂（2/4/8/16/32/64），默认值为8；与source侧必须一致。 | INT64 | - |

## 约束说明

1. 该接口支持推理场景下使用，支持单算子模式调用。
2. `candidate_block_indices` 必传（REQUIRED，INT32）；candBlocks（shape 末维）∈ (0, 2048]
   且为 64 的倍数；本算子无 `candidate_topk_blocks` 属性（宽度由输入推导）。
3. `candidate_block_size` ∈ [2, 64] 内的2的幂，默认值为8；传其它值拒绝。
4. `topk` ∈ (0, 2048] 恒校验（>2048 报错）；`return_value` ∈ {0,1}，其它值报错拒绝。
5. `candidate_block_length` 预留：仅接受 None/空 tensor，非空报错
   "candidate_block_length is reserved and only empty tensor is supported yet"。
6. layout 组合、PA block_table、N2=1、head_dim=128 等其余约束同上游算子 `lightning_indexer`。
7. 跨算子约束（source → consumer）：
   - 与 `lightning_indexer` source 模式（`candidate_topk_blocks != -1`）配对使用；
   - 两侧 `candidate_block_size` 必须一致——**调用方约束，框架层无法跨算子校验**（两侧 host 各自仅校验
     值域；错配行为未定义）；
   - 候选仅**同一 forward 内**跨层共享有效；source 与 consumer 应共享同一 KV 域——
     consumer 侧 KV 更短时，越界候选块号自然降级为"非候选"（鲁棒，无需额外校验）；
   - 候选为 batch 内**压缩后位置域**的相对块号（块号 × blockSize = 起始位置），
     -1 为无效槽，槽位无序，**不加 output_idx_offset**；
   - 无 leak 语义：**非候选块索引一律不进入 topk 输出**——候选外可达位置与不可达位置同为无效位
     （score 置 -inf、索引置 -1，不参与有效排序、不入选）；候选内有效数 nvalid < topk
     （含 regime 2：候选内有效数 < topk < 可达数）时剩余槽位全部填 -1，不输出任何候选外索引，
     全无效行输出全 -1；
   - 输出槽位顺序（含 -1 槽填充次序与真实分同分次序）不作对外承诺，仅保证有效槽集合一致与 -1 槽数正确。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
|---------|--------|------|
| aclnn API | [test_aclnn_sparse_lightning_indexer](examples/test_aclnn_sparse_lightning_indexer.cpp) | 通过[aclnnSparseLightningIndexer](docs/aclnnSparseLightningIndexer.md)接口方式调用SparseLightningIndexer算子。 |
| PyTorch API | [sparse_lightning_indexer](docs/torchapi_sparse_lightning_indexer.md) | 通过[cann_ops_transformer.sparse_lightning_indexer](docs/torchapi_sparse_lightning_indexer.md)接口调用SparseLightningIndexer算子（与lightning_indexer source模式配对使用）。 |
