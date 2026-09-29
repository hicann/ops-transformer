# SparseLightningIndexer

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>    |    √     |
| <term>Ascend 950PR/Ascend 950DT</term>                     |     ×    |
| <term>Atlas 200I/500 A2 推理产品</term>                      |     ×    |
| <term>Atlas 推理系列产品</term>                              |     ×    |
| <term>Atlas 训练系列产品</term>                              |     ×    |

## 功能说明

- 算子功能：SparseLightningIndexer（candidate consumer）在
  LightningIndexerV2 的加权 ReLU 打分 + TopK 框架上，**消费上游 source 算子输出的候选块索引**
  （`candidate_topk_indices`），对每个 query 行做"候选内 leak 降级 TopK"：
  候选外**可达**位置的分数降级为 NEG_HUGE(-1e30) 参与排序（不取消入选资格），
  不可达位置输出 -1。与 `lightning_indexer_v2`（source 模式，`candidate_topk_blocks != -1`）
  配对使用，构成候选块两级 TopK 的 consumer 侧。

- 计算公式（每 batch b、query 行 i，N2=1）：

  $$
  \begin{aligned}
  score(b,i,:) &= \mathrm{DoReduce}(W \odot ReLU(Q_{index} @ K_{index}^{T})) \in \R^{S_2} \\
  score'(b,i,p) &= \begin{cases}
      score(b,i,p) & p < vl(b,i) \text{ 且 } \lfloor p / blockSize \rfloor \in candSet(b,i) \\
      -10^{30}     & p < vl(b,i) \text{ 且可达但不在候选内（leak 降级，仍可作填充入选）} \\
      -\infty      & p \geq vl(b,i) \text{（不可达）}
  \end{cases} \\
  sparseIndices(b,i,:) &= \mathrm{topK}(score')
  \end{aligned}
  $$

  其中 `vl(b,i)` 为行可达前缀长度（由 causal mask、seqused_k/cu_seqlens_k 与 cmp_ratio 推导），
  `blockSize` = `candidate_block_size`，`candSet` 为候选块号集合（重复/越界/-1 按"非候选"处理）。

- 主要计算过程为：

  1. 与 LIV2 相同的打分流程：QK 相关性 → ReLU → W 加权 g 维求和得到 score 行。
  2. 依据候选块索引构建位置级候选掩码：候选块覆盖的位置为候选，其余为非候选。
  3. 候选外可达位置的分数降级为 NEG_HUGE 参与排序（leak 语义，不取消入选资格）。
  4. 沿用 LIV2 的位置级 TopK 管线输出索引（-1 槽仅出现在不可达位置）。

## 参数说明

  <table style="undefined;table-layout: fixed; width: 1080px"><colgroup>
  <col style="width: 200px">
  <col style="width: 150px">
  <col style="width: 280px">
  <col style="width: 330px">
  <col style="width: 120px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出/属性</th>
      <th>描述</th>
      <th>数据类型</th>
      <th>数据格式</th>
    </tr></thead>
  <tbody>
    <tr>
    <td>q</td>
    <td>输入</td>
    <td>公式中的输入Q（与 LIV2 相同）。</td>
    <td>BFLOAT16、FLOAT16</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>k</td>
    <td>输入</td>
    <td>公式中的输入K（与 LIV2 相同）。</td>
    <td>BFLOAT16、FLOAT16</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>w</td>
    <td>输入</td>
    <td>公式中的输入W（与 LIV2 相同）。</td>
    <td>FLOAT</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>cuSeqlensQOptional</td>
    <td>输入</td>
    <td>当前Batch及前序Batch中q的有效token数的累加和。</td>
    <td>INT32</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>cuSeqlensKOptional</td>
    <td>输入</td>
    <td>当前Batch及前序Batch中k的有效token数的累加和。</td>
    <td>INT32</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>sequsedQOptional</td>
    <td>输入</td>
    <td>不同Batch中q的真实使用长度。</td>
    <td>INT32</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>sequsedKOptional</td>
    <td>输入</td>
    <td>不同Batch中k的真实使用长度。</td>
    <td>INT32</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>cmpResidualKOptional</td>
    <td>输入</td>
    <td>表示k压缩前token数量除以cmpRatio的余数。</td>
    <td>INT32</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>blockTableOptional</td>
    <td>输入</td>
    <td>表示PageAttention中KV存储使用的block映射表。</td>
    <td>INT32</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>outputIdxOffsetOptional</td>
    <td>输入</td>
    <td>表示topK结果输出索引所需要加上的偏移。</td>
    <td>INT32</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>metadataOptional</td>
    <td>输入</td>
    <td>元数据（arch22 kernel 不消费，接口一致性保留）。</td>
    <td>INT32</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>candidate_topk_indices</td>
    <td>输入</td>
    <td>候选块索引（source 算子输出直连）。BSND [B,S1,N2,candBlocks] / TND [T,N2,candBlocks]（TND 为 query.dim0 专式）；值域 [0,numBlocks) 或 -1；槽位无序；允许重复/越界/-1（按"非候选"处理）；candBlocks = shape 末维，∈ (0,2048] 且 64 的倍数（由输入 shape 推导）。</td>
    <td>INT32</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>candidate_block_length</td>
    <td>输入</td>
    <td>预留接口，仅接受空 tensor（None/numel==0，非空报错）。</td>
    <td>INT32</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>sparseIndices</td>
    <td>输出</td>
    <td>公式中的输出Top-k索引（leak 降级 topk；候选外可达位置以 -1e30 降级参与排序；仅不可达位置输出 -1）。</td>
    <td>INT32</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>sparseValues</td>
    <td>输出</td>
    <td>公式中的输出Top-k值。return_value 恒 0，本输出恒为空 tensor。</td>
    <td>FLOAT</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>topk</td>
    <td>属性</td>
    <td>Top-k的k值，取值范围 (0, 2048]（超限直接报错）。</td>
    <td>INT</td>
    <td>-</td>
    </tr>
    <tr>
    <td>max_seqlen_q</td>
    <td>属性</td>
    <td>q的最大序列长度，缺省-1。</td>
    <td>INT</td>
    <td>-</td>
    </tr>
    <tr>
    <td>layout_q</td>
    <td>属性</td>
    <td>q的数据排布，取值BSND/TND。</td>
    <td>STRING</td>
    <td>-</td>
    </tr>
    <tr>
    <td>layout_k</td>
    <td>属性</td>
    <td>k的数据排布，取值BSND/TND/PA_BBND。</td>
    <td>STRING</td>
    <td>-</td>
    </tr>
    <tr>
    <td>mask_mode</td>
    <td>属性</td>
    <td>mask类型，0表示不使用mask，3表示因果mask。</td>
    <td>INT</td>
    <td>-</td>
    </tr>
    <tr>
    <td>cmp_ratio</td>
    <td>属性</td>
    <td>压缩比，1~128（2 的幂）。</td>
    <td>INT</td>
    <td>-</td>
    </tr>
    <tr>
    <td>return_value</td>
    <td>属性</td>
    <td>必须 0（≠0 拒绝；leak 降级语义下 Values 输出无意义，恒不开放；torch 层不暴露）。</td>
    <td>INT</td>
    <td>-</td>
    </tr>
    <tr>
    <td>candidate_block_size</td>
    <td>属性</td>
    <td>候选块粒度，[2,64] 且为 2 的幂（2/4/8/16/32/64），默认 8；与 source 侧必须一致。</td>
    <td>INT</td>
    <td>-</td>
    </tr>
  </tbody>
  </table>

## 约束说明

1. 仅支持 ascend910b / ascend910_93（arch22），不支持 ascend950。
2. `candidate_topk_indices` 必传（REQUIRED，INT32）；candBlocks（shape 末维）∈ (0, 2048]
   且为 64 的倍数；本算子无 `candidate_topk_blocks` 属性（宽度由输入推导）。
3. `candidate_block_size` ∈ {2,4,8,16,32,64}，默认 8；传其它值拒绝。
4. `topk` ≤ 2048 恒校验（>2048 报错）；`return_value` 必须为 0。
5. `candidate_block_length` 预留：仅接受 None/空 tensor，非空报错
   "candidate_block_length is reserved and only empty tensor is supported yet"。
6. layout 组合、PA block_table、N2=1、head_dim=128 等其余约束同 LIV2。
7. 跨算子契约（source → consumer）：
   - 与 `lightning_indexer_v2` source 模式（`candidate_topk_blocks != -1`）配对使用；
   - 两侧 `candidate_block_size` 必须一致——**调用方契约，框架层无法跨算子校验**（两侧 host 各自仅校验
     值域；错配行为未定义）；
   - 候选仅**同一 forward 内**跨层共享有效；source 与 consumer 应共享同一 KV 域——
     consumer 侧 KV 更短时，越界候选块号自然降级为"非候选"（鲁棒，无需额外校验）；
   - 候选为 batch 内**压缩后位置域**的相对块号（块号 × blockSize = 起始位置），
     -1 为无效槽，槽位无序，**不加 output_idx_offset**；
   - leak 语义：候选只降级排序（NEG_HUGE > -inf 两档），不取消入选资格；regime 2
     （候选内有效数 < topk < 可达数）时候选外可达位置以降级分数填充入选；
   - 输出槽位顺序（含候选外填充次序）不作对外承诺，仅保证有效槽集合一致与 -1 槽数正确。

## 调用示例

- aclnn 调用示例：[examples/test_aclnn_sparse_lightning_indexer.cpp](examples/test_aclnn_sparse_lightning_indexer.cpp)
- aclnn 接口说明：[docs/aclnnSparseLightningIndexer.md](docs/aclnnSparseLightningIndexer.md)
- torch 接口说明：[docs/torchapi_sparse_lightning_indexer.md](docs/torchapi_sparse_lightning_indexer.md)
