# aclnnPoolKeyIndexer

[📄 查看源码](https://gitcode.com/cann/ops-transformer/tree/master/attention/pool_key_indexer)

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

- 接口功能：`pool_key_indexer`将多个连续token打包成一个pool（池），以pool为单位计算注意力相关性分数并选取top-k位置，从而在保持稀疏注意力优势的同时减少索引开销。该算子适用于基于key pool的稀疏注意力场景下的稀疏索引选择。支持FLOAT16、BFLOAT16浮点输入，并支持配合反量化系数的FLOAT8_E4M3FN量化输入（FP8 per-token-head或mxFP8量化模式）。

- 计算公式：

  完整计算流程分为输入反量化、分数计算与选择、索引展开与追加三个部分。

  **输入反量化（仅量化场景，quantMode为0或1）：**

  当quantMode为-1（不量化）时，直接使用原始输入：

  $$
  Q = query,\quad K_{pool} = poolKey
  $$

  当quantMode为0或1（量化）时，先对FLOAT8_E4M3FN输入结合反量化系数进行反量化：

  $$
  Q = \text{Dequant}(query, q\_descale)
  $$

  $$
  K_{pool} = \text{Dequant}(poolKey, k\_descale)
  $$

  其中quantMode为0表示FP8 per-token-head量化模式，反量化系数为FLOAT类型；quantMode为1表示mxFP8量化模式，反量化系数为FLOAT8_E8M0类型。

  **分数计算与选择：**

  1. 缩放点积：

  $$
  S = Q \otimes K_{pool}^T \odot \frac{1}{\sqrt{headDim}}
  $$

  2. ReLU激活，过滤负相关信号：

  $$
  S' = \text{ReLU}(S)
  $$

  3. 多头权重聚合：

  $$
  PoolScores = W \otimes S'
  $$

  4. 可见性过滤：根据因果性和padding状态标记有效候选pool，将无效pool分数置为$-\infty$。

  5. Top-k选择：

  $$
  TopK\_pool = \text{Top-}k(PoolScores)
  $$

  其中$k = \lfloor topk / pool\_size \rfloor$，即选取前$k$个pool。

  **索引展开与追加：**

  $$
  SparseIndices = \text{Append}(\text{Expand}(TopK\_pool, PoolIndices))
  $$

  其中$\text{Expand}$将选中的pool索引通过$PoolIndices$展开为原始token索引，$\text{Append}$追加不完整pool的尾部有效token，确保短序列尾部不遗漏（先展开后追加）。尾部有效token数由poolTailK指定。

  **输出：**

  $$
  SparseValues = PoolScores[TopK\_pool]
  $$

  当returnValue为true时输出$SparseValues$，即选中pool的分数值；returnValue为false时不输出。

  符号说明如下：

  - $SparseIndices$：输出的稀疏token索引。
  - $SparseValues$：输出的选中pool分数（returnValue为true时输出）。
  - $PoolScores$：每个pool的综合分数。
  - $TopK\_pool$：被选中的top-k pool索引。
  - $Top$-$k$：取前$k$个最大分数对应的pool索引，$k = \lfloor topk / pool\_size \rfloor$。
  - $W$：多头权重投影矩阵。
  - $\otimes$：矩阵乘法（matmul）。
  - $\text{ReLU}$：激活函数$ReLU(x) = \max(0, x)$。
  - $Q$：Query向量（量化场景下为反量化后的Query）。
  - $K_{pool}$：Pool的加权聚合key，$K_{pool}^T$为其最后两维转置（量化场景下为反量化后的Key）。
  - $\odot$：逐元素乘法（Hadamard积）。
  - $headDim$：头维度，缩放因子$\frac{1}{\sqrt{headDim}}$防止点积过大。
  - $PoolIndices$：每个pool包含的原始token索引。
  - $query$、$poolKey$：原始输入tensor（量化场景下为FLOAT8_E4M3FN）。
  - $q\_descale$、$k\_descale$：Query与Key的反量化系数。
  - $poolTailK$：每个batch尾部不完整pool的有效token数。

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/two_phase_api.md)，必须先调用“aclnnPoolKeyIndexerGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnPoolKeyIndexer”接口执行计算。

```Cpp
aclnnStatus aclnnPoolKeyIndexerGetWorkspaceSize(
    const aclTensor   *query,
    const aclTensor   *poolKey,
    const aclTensor   *weights,
    const aclIntArray *poolTailK,
    const aclIntArray *actualSeqLengthsQueryOptional,
    const aclIntArray *actualSeqLengthsKeyOptional,
    const aclTensor   *blockTableOptional,
    const aclTensor   *qDescaleOptional,
    const aclTensor   *kDescaleOptional,
    int64_t            topk,
    int64_t            poolSize,
    char              *layoutQueryOptional,
    char              *layoutKeyOptional,
    int64_t            maskMode,
    int64_t            quantMode,
    bool               returnValue,
    int64_t            keyStride0,
    int64_t            kDescaleStride0,
    const aclTensor   *sparseIndicesOut,
    const aclTensor   *sparseValuesOut,
    uint64_t          *workspaceSize,
    aclOpExecutor    **executor)
```

```Cpp
aclnnStatus aclnnPoolKeyIndexer(
    void             *workspace,
    uint64_t          workspaceSize,
    aclOpExecutor    *executor,
    const aclrtStream stream)
```

## aclnnPoolKeyIndexerGetWorkspaceSize

- **参数说明：**

  > [!NOTE]
  >
  > - query、poolKey、weights参数维度含义：B（Batch Size）表示输入样本批量大小、S（Sequence Length）表示输入样本序列长度、N（Head Num）表示多头数、D（Head Dim）表示每个头的维度、T表示所有Batch输入样本序列长度的累加和。
  > - S1表示query shape中的S，S2表示poolKey shape中的S，T1表示query shape中的T，T2表示poolKey shape中的T，N1表示query shape中的N，N2表示poolKey shape中的N。
  > - block_num为PageAttention时block总数，block_size为一个block的token数。
  > - poolTailK、actualSeqLengthsQueryOptional、actualSeqLengthsKeyOptional为Host侧取值输入（ValueDepend），接口层以aclIntArray类型传入（元素类型INT64），非aclTensor。

  <table style="undefined;table-layout: fixed; width: 1601px"><colgroup>
  <col style="width: 264px">
  <col style="width: 132px">
  <col style="width: 232px">
  <col style="width: 330px">
  <col style="width: 164px">
  <col style="width: 119px">
  <col style="width: 215px">
  <col style="width: 145px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出</th>
      <th>描述</th>
      <th>使用说明</th>
      <th>数据类型</th>
      <th>数据格式</th>
      <th>维度(shape)</th>
      <th>非连续Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>query（aclTensor*）</td>
      <td>输入</td>
      <td>公式中的输入Q。</td>
      <td>不支持空tensor。</td>
      <td>FLOAT16、BFLOAT16、FLOAT8_E4M3FN</td>
      <td>ND</td>
      <td>
          <ul>
                <li>layoutQueryOptional为BSND时，shape为(B,S1,N1,D)。</li>
                <li>layoutQueryOptional为TND时，shape为(T1,N1,D)。</li>
          </ul>
      </td>
      <td>√</td>
    </tr>
    <tr>
      <td>poolKey（aclTensor*）</td>
      <td>输入</td>
      <td>公式中的输入K<sub>pool</sub>。</td>
      <td>
          <ul>
                <li>不支持空tensor。</li>
                <li>block_num为PageAttention时block总数，block_size为一个block的pool数（块内每行为一个pool的key，行距为headDim，不做token展开）。</li>
                <li>数据类型需与query保持一致。</li>
                <li>数据布局需与query保持一致，PageAttention场景（layoutKeyOptional为PA_BBND）除外。</li>
                <li>layoutKeyOptional为PA_BBND时，shape为(block_num, block_size, N2, D)。</li>
                <li>layoutKeyOptional为BSND时，shape为(B, S2, N2, D)。</li>
                <li>layoutKeyOptional为TND时，shape为(T2, N2, D)。</li>
          </ul>
      </td>
      <td>FLOAT16、BFLOAT16、FLOAT8_E4M3FN</td>
      <td>ND</td>
      <td>2-5</td>
      <td>支持0轴非连续（仅PA_BBND布局，需配合keyStride0属性；BSND/TND布局要求连续输入）</td>
    </tr>
    <tr>
      <td>weights（aclTensor*）</td>
      <td>输入</td>
      <td>公式中的输入W。</td>
      <td>
          <ul>
                <li>不支持空tensor。</li>
                <li>layoutQueryOptional为BSND时，shape为(B,S1,N1)。</li>
                <li>layoutQueryOptional为TND时，shape为(T1,N1)。</li>
          </ul>
      </td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>2-3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>poolTailK（aclIntArray*）</td>
      <td>输入</td>
      <td>每个Batch中尾部不完整pool的有效token数。</td>
      <td>
          <ul>
                <li>不支持空tensor。</li>
                <li>取值范围为[0, pool_size-1]，0表示无尾部有效token。</li>
                <li>shape为(B,)。</li>
          </ul>
      </td>
      <td>INT64</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>actualSeqLengthsQueryOptional（aclIntArray*）</td>
      <td>输入</td>
      <td>每个Batch中，Query的有效token数。</td>
      <td>
          <ul>
                <li>可选输入，不指定seqlen时可传入空指针，表示与query的shape的S长度相同。</li>
                <li>不支持空Tensor。</li>
                <li>该入参中每个Batch的有效token数不超过query中的维度S大小且不小于0，支持长度为B的一维tensor。</li>
                <li>当layoutQueryOptional为TND时，该入参必须传入，且以该入参元素的数量作为B值，该入参中每个元素的值表示当前batch与之前所有batch的token数总和，即前缀和，因此后一个元素的值必须大于等于前一个元素的值。</li>
                <li>shape为(B,)。</li>
          </ul>
      </td>
      <td>INT64</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>actualSeqLengthsKeyOptional（aclIntArray*）</td>
      <td>输入</td>
      <td>每个Batch中，Key的有效token数。</td>
      <td>
          <ul>
                <li>可选输入，不指定seqlen时可传入空指针，表示与poolKey的shape的S长度相同。</li>
                <li>不支持空Tensor。</li>
                <li>该参数中每个Batch的有效pool数不超过poolKey的维度S（BSND/TND）或blockSize总池数（PA_BBND）大小且不小于0，支持长度为B的一维tensor。</li>
                <li>当layoutKeyOptional为TND或PA_BBND时，该入参必须传入。layoutKeyOptional为TND时，该参数中每个元素的值表示当前batch与之前所有batch的token数总和，即前缀和，因此后一个元素的值必须大于等于前一个元素的值；layoutKeyOptional为PA_BBND时，该参数中每个元素的值表示当前batch的pool数（即该batch有效token数除以pool_size后向下取整的池数，非前缀和形式）。</li>
                <li>shape为(B,)。</li>
          </ul>
      </td>
      <td>INT64</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>blockTableOptional（aclTensor*）</td>
      <td>输入</td>
      <td>表示PageAttention中KV存储使用的block映射表。</td>
      <td>
          <ul>
                <li>可选输入，非PageAttention场景可传入空指针。</li>
                <li>不支持空Tensor。</li>
                <li>PageAttention场景下，block_table必须为二维，第一维长度需要等于B，第二维长度不能小于maxBlockNumPerSeq（maxBlockNumPerSeq为每个batch中最大actual_seq_lengths_key对应的block数量）。</li>
                <li>shape为(B, maxBlockNumPerSeq)。</li>
          </ul>
      </td>
      <td>INT32</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>qDescaleOptional（aclTensor*）</td>
      <td>输入</td>
      <td>Query的反量化系数。</td>
      <td>
          <ul>
                <li>可选输入，非量化场景（quantMode为-1）可传入空指针。</li>
                <li>不支持空Tensor。</li>
                <li>量化场景下必传，数据类型由quantMode决定：quantMode为0时为FLOAT（FP8 per-token-head量化），quantMode为1时为FLOAT8_E8M0（mxFP8量化）。</li>
                <li>quantMode为0时：layoutQueryOptional为BSND时shape为(B,S1,N1)；layoutQueryOptional为TND时shape为(T1,N1)。</li>
                <li>quantMode为1时（mxFP8，每32个D维度元素共享一个E8M0 scale）：layoutQueryOptional为BSND时shape为(B,S1,N1,D/64,2)；layoutQueryOptional为TND时shape为(T1,N1,D/64,2)。</li>
          </ul>
      </td>
      <td>FLOAT、FLOAT8_E8M0</td>
      <td>ND</td>
      <td>2-5</td>
      <td>√</td>
    </tr>
    <tr>
      <td>kDescaleOptional（aclTensor*）</td>
      <td>输入</td>
      <td>Key的反量化系数。</td>
      <td>
          <ul>
                <li>可选输入，非量化场景（quantMode为-1）可传入空指针。</li>
                <li>不支持空Tensor。</li>
                <li>量化场景下必传，数据类型由quantMode决定：quantMode为0时为FLOAT（FP8 per-token-head量化），quantMode为1时为FLOAT8_E8M0（mxFP8量化）。</li>
                <li>quantMode为0时：layoutKeyOptional为BSND时shape为(B,S2,N2)；layoutKeyOptional为TND时shape为(T2,N2)；layoutKeyOptional为PA_BBND时shape为(block_num, block_size, N2)。</li>
                <li>quantMode为1时（mxFP8）：layoutKeyOptional为BSND时shape为(B,S2,N2,D/64,2)；layoutKeyOptional为TND时shape为(T2,N2,D/64,2)；layoutKeyOptional为PA_BBND时shape为(block_num, block_size, N2, D/64,2)。</li>
          </ul>
      </td>
      <td>FLOAT、FLOAT8_E8M0</td>
      <td>ND</td>
      <td>2-5</td>
      <td>支持0轴非连续（仅PA_BBND布局且量化场景，需配合kDescaleStride0属性；BSND/TND布局要求连续输入）</td>
    </tr>
    <tr>
      <td>topk（int64_t）</td>
      <td>输入</td>
      <td>展开后需要保留的token数量，对应公式中的topk。可选属性，建议值为2048。</td>
      <td>
          <ul>
                <li>支持[1, 2048]以及3072、4096、5120、6144、7168、8192。</li>
                <li>需满足topk % pool_size == 0。</li>
          </ul>
      </td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>poolSize（int64_t）</td>
      <td>输入</td>
      <td>每个pool包含的token数量，对应公式中的pool_size。可选属性，建议值为16。</td>
      <td>支持[1, 128]。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layoutQueryOptional（char*）</td>
      <td>输入</td>
      <td>用于标识输入Query的数据排布格式。</td>
      <td>
          <ul>
                <li>用户不特意指定时可传入建议值"BSND"。</li>
                <li>当前支持BSND、TND。</li>
          </ul>
      </td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layoutKeyOptional（char*）</td>
      <td>输入</td>
      <td>用于标识输入Key的数据排布格式。</td>
      <td>
          <ul>
                <li>用户不特意指定时可传入建议值"BSND"。</li>
                <li>当前支持PA_BBND、BSND、TND。</li>
          </ul>
      </td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>maskMode（int64_t）</td>
      <td>输入</td>
      <td>表示mask的模式。</td>
      <td>
          <ul>
                <li>maskMode为0时，代表defaultMask模式。</li>
                <li>maskMode为3时，代表rightDownCausal模式的mask，对应以右顶点为划分的下三角场景。</li>
          </ul>
      </td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>quantMode（int64_t）</td>
      <td>输入</td>
      <td>表示Query/Key的量化模式。</td>
      <td>
          <ul>
                <li>quantMode为-1时，表示不量化。</li>
                <li>quantMode为0时，表示FP8 per-token-head量化模式，反量化系数为FLOAT。</li>
                <li>quantMode为1时，表示mxFP8量化模式，反量化系数为FLOAT8_E8M0。</li>
          </ul>
      </td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>returnValue（bool）</td>
      <td>输入</td>
      <td>表示是否输出sparseValuesOut。可选属性，建议值为false。</td>
      <td>
          <ul>
                <li>returnValue为false时，表示不输出sparseValuesOut。</li>
                <li>returnValue为true时，表示输出sparseValuesOut。</li>
          </ul>
      </td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>keyStride0（int64_t）</td>
      <td>输入</td>
      <td>poolKey第0轴的stride（元素单位），用于layoutKeyOptional为PA_BBND场景下0轴非连续poolKey的寻址。可选属性，建议值为-1。</td>
      <td>
          <ul>
                <li>-1表示未指定，按连续输入由shape推导stride。</li>
                <li>指定时取值必须不小于block_size * N2 * D（连续stride）。</li>
                <li>指定值与poolKey运行时实际stride冲突时报错。</li>
          </ul>
      </td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>kDescaleStride0（int64_t）</td>
      <td>输入</td>
      <td>kDescaleOptional第0轴的stride（元素单位），用于layoutKeyOptional为PA_BBND场景下0轴非连续kDescaleOptional的寻址。可选属性，建议值为-1。</td>
      <td>
          <ul>
                <li>仅量化场景（quantMode为0或1）有效。</li>
                <li>-1表示未指定，按连续输入由shape推导stride。</li>
                <li>指定时取值必须不小于连续stride：quantMode为0时为block_size * N2，quantMode为1时为block_size * N2 * (D / 32)。</li>
                <li>指定值与kDescaleOptional运行时实际stride冲突时报错。</li>
          </ul>
      </td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>sparseIndicesOut（aclTensor*）</td>
      <td>输出</td>
      <td>公式中的SparseIndices输出，展开后的稀疏token索引。</td>
      <td>
          <ul>
                <li>不支持空tensor。</li>
                <li>layoutQueryOptional为BSND时，输出shape为(B, S1, topk + pool_size - 1)。</li>
                <li>layoutQueryOptional为TND时，输出shape为(T1, topk + pool_size - 1)。</li>
          </ul>
      </td>
      <td>INT32</td>
      <td>ND</td>
      <td>2-3</td>
      <td>x</td>
    </tr>
    <tr>
      <td>sparseValuesOut（aclTensor*）</td>
      <td>输出</td>
      <td>公式中的SparseValues输出，即选中pool对应的分数值。</td>
      <td>
          <ul>
                <li>当returnValue为true时，输出有效结果，不支持空tensor。</li>
                <li>当returnValue为false时，输出shape为(0,)的空tensor。</li>
                <li>layoutQueryOptional为BSND时，输出shape为(B, S1, topk // pool_size)。</li>
                <li>layoutQueryOptional为TND时，输出shape为(T1, topk // pool_size)。</li>
          </ul>
      </td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>2-3</td>
      <td>x</td>
    </tr>
    <tr>
      <td>workspaceSize（uint64_t*）</td>
      <td>输出</td>
      <td>返回需要在Device侧申请的workspace大小。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor（aclOpExecutor**）</td>
      <td>输出</td>
      <td>返回op执行器，包含了算子计算流程。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

  第一段接口会完成入参校验，出现以下场景时报错：

    <table style="undefined;table-layout: fixed;width: 1155px"><colgroup>
    <col style="width: 319px">
    <col style="width: 144px">
    <col style="width: 671px">
    </colgroup>
        <thead>
            <tr>
                <th>返回值</th>
                <th>错误码</th>
                <th>描述</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>ACLNN_ERR_PARAM_NULLPTR</td>
                <td>161001</td>
                <td>query、poolKey、weights、poolTailK、sparseIndicesOut、sparseValuesOut为空指针（可选输入为空指针时除外）。</td>
            </tr>
            <tr>
                <td rowspan="4">ACLNN_ERR_PARAM_INVALID</td>
                <td rowspan="4">161002</td>
                <td>query、poolKey、weights、poolTailK、sparseIndicesOut、sparseValuesOut的数据类型和数据格式不在支持的范围内。</td>
            </tr>
            <tr>
                <td>query、poolKey、weights的shape维度不在支持的范围内。</td>
            </tr>
            <tr>
                <td>topk与poolSize不满足topk % pool_size == 0的约束。</td>
            </tr>
            <tr>
                <td>topk或poolSize的取值不在支持的范围内。</td>
            </tr>
        </tbody>
    </table>

## aclnnPoolKeyIndexer

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1151px"><colgroup>
  <col style="width: 184px">
  <col style="width: 134px">
  <col style="width: 833px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出</th>
      <th>描述</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>workspace</td>
      <td>输入</td>
      <td>在Device侧申请的workspace内存地址。</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>输入</td>
      <td>在Device侧申请的workspace大小，由第一段接口aclnnPoolKeyIndexerGetWorkspaceSize获取。</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>输入</td>
      <td>op执行器，包含了算子计算流程。</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>输入</td>
      <td>指定执行任务的Stream。</td>
    </tr>
  </tbody>
  </table>

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

- 确定性实现：
  - aclnnPoolKeyIndexer默认确定性实现。
- 输入shape限制：
  - 参数query的N1支持小于等于64，poolKey的N2必须为1，headDim支持128，且poolKey的D必须与query的headDim保持一致。
  - 不支持query、poolKey中对S1、S2 pad无效数据。
  - 输出sparseIndicesOut最后一维为topk + pool_size - 1，输出sparseValuesOut最后一维为topk // pool_size。
  - 维度一致性约束：weights、qDescaleOptional的各维度（BSND时为B、S1、N1，TND时为T1、N1）必须与query对应维度一致；kDescaleOptional的各维度必须与poolKey对应维度一致（PA_BBND时blockNum、blockSize、N2需与poolKey一致）；poolTailK的元素数必须等于B；actualSeqLengthsKeyOptional的元素数必须等于B。
  - 前缀和约束：当layoutQueryOptional为TND时，actualSeqLengthsQueryOptional的前缀和最后一个元素的值必须等于query的T1；当layoutKeyOptional为TND时，actualSeqLengthsKeyOptional的前缀和最后一个元素的值必须等于poolKey的T2。
  - 输出维度约束：sparseIndicesOut和sparseValuesOut的B、S1（或T1）维度必须与query对应维度一致。
- 输入属性限制：
  - topk需满足topk % pool_size == 0，pool_size支持[1, 128]，topk支持[1, 2048]以及3072、4096、5120、6144、7168、8192。
  - block_size取值为16的倍数，最大支持2048。
  - 当layoutKeyOptional为PA_BBND且poolKey为0轴非连续输入时，keyStride0应传入poolKey的实际stride(0)（元素单位，且不小于block_size * N2 * D）；keyStride0为-1时表示未指定，按连续输入推导；keyStride0指定值与poolKey运行时实际stride冲突时报错。
  - 当layoutKeyOptional为PA_BBND且kDescaleOptional为0轴非连续输入时（量化场景），kDescaleStride0应传入kDescaleOptional的实际stride(0)（元素单位，且不小于连续stride：quantMode为0时为block_size * N2，quantMode为1时为block_size * N2 * (D / 32)）；kDescaleStride0为-1时表示未指定，按连续输入推导；kDescaleStride0指定值与kDescaleOptional运行时实际stride冲突时报错。
- 输入数据类型限制：
  - 参数query、poolKey的数据类型应保持一致。
  - 非量化场景（quantMode为-1）下，参数weights的数据类型应与query、poolKey保持一致；量化场景（quantMode为0或1）下，参数weights为FLOAT16或BFLOAT16。
  <!-- npu="950" id8 -->
  - <term>Ascend 950PR/Ascend 950DT</term>：支持FLOAT8_E4M3FN（query、poolKey）数据类型，并支持量化场景（quantMode为0时反量化系数为FLOAT，quantMode为1时反量化系数为FLOAT8_E8M0）。
  <!-- end id8 -->
  <!-- npu="A3,910b" id7 -->
  - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：不支持FLOAT8_E4M3FN与FLOAT8_E8M0数据类型，不支持量化场景。
  <!-- end id7 -->
- 输入数据布局限制：
  - 参数query、poolKey的数据布局（layoutQueryOptional与layoutKeyOptional）需保持一致，PageAttention场景（layoutKeyOptional为PA_BBND时）除外。
- 其他限制：
  - poolKey支持0轴非连续（仅PA_BBND布局，需配合keyStride0属性；BSND/TND布局要求连续输入）。
  - kDescaleOptional支持0轴非连续（仅PA_BBND布局且量化场景，需配合kDescaleStride0属性；BSND/TND布局要求连续输入）。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。

```Cpp
#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include "securec.h"
#include "acl/acl.h"
#include "aclnnop/aclnn_pool_key_indexer.h"

using namespace std;

namespace {

#define CHECK_RET(cond) ((cond) ? true :(false))

#define LOG_PRINT(message, ...)     \
  do {                              \
    (void)printf(message, ##__VA_ARGS__); \
  } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
  int64_t shapeSize = 1;
  for (auto i : shape) {
    shapeSize *= i;
  }
  return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream) {
  auto ret = aclInit(nullptr);
  if (!CHECK_RET(ret == ACL_SUCCESS)) {
    LOG_PRINT("aclInit failed. ERROR: %d\n", ret);
    return ret;
  }
  ret = aclrtSetDevice(deviceId);
  if (!CHECK_RET(ret == ACL_SUCCESS)) {
    LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret);
    return ret;
  }
  ret = aclrtCreateStream(stream);
  if (!CHECK_RET(ret == ACL_SUCCESS)) {
    LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret);
    return ret;
  }
  return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor) {
  auto size = GetShapeSize(shape) * sizeof(T);
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  if (!CHECK_RET(ret == ACL_SUCCESS)) {
    LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret);
    return ret;
  }

  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  if (!CHECK_RET(ret == ACL_SUCCESS)) {
    LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret);
    return ret;
  }

  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

struct TensorResources {
    void* queryDeviceAddr = nullptr;
    void* poolKeyDeviceAddr = nullptr;
    void* weightsDeviceAddr = nullptr;
    void* sparseIndicesDeviceAddr = nullptr;
    void* sparseValuesDeviceAddr = nullptr;

    aclTensor* queryTensor = nullptr;
    aclTensor* poolKeyTensor = nullptr;
    aclTensor* weightsTensor = nullptr;
    aclTensor* sparseIndicesTensor = nullptr;
    aclTensor* sparseValuesTensor = nullptr;
};

int InitializeTensors(TensorResources& resources) {
    // topk=2048, pool_size=16
    // sparseIndicesOut最后一维 = topk + pool_size - 1 = 2063
    // sparseValuesOut最后一维 = topk // pool_size = 128
    std::vector<int64_t> queryShape = {1, 2, 1, 128};
    std::vector<int64_t> poolKeyShape = {1, 2, 1, 128};
    std::vector<int64_t> weightsShape = {1, 2, 1};
    std::vector<int64_t> sparseIndicesShape = {1, 2, 2063};
    std::vector<int64_t> sparseValuesShape = {1, 2, 128};

    int64_t queryShapeSize = GetShapeSize(queryShape);
    int64_t poolKeyShapeSize = GetShapeSize(poolKeyShape);
    int64_t weightsShapeSize = GetShapeSize(weightsShape);
    int64_t sparseIndicesShapeSize = GetShapeSize(sparseIndicesShape);
    int64_t sparseValuesShapeSize = GetShapeSize(sparseValuesShape);

    std::vector<float> queryHostData(queryShapeSize, 1);
    std::vector<float> poolKeyHostData(poolKeyShapeSize, 1);
    std::vector<float> weightsHostData(weightsShapeSize, 1);
    std::vector<int32_t> sparseIndicesHostData(sparseIndicesShapeSize, 1);
    std::vector<float> sparseValuesHostData(sparseValuesShapeSize, 1);

    int ret = CreateAclTensor(queryHostData, queryShape, &resources.queryDeviceAddr,
                              aclDataType::ACL_FLOAT16, &resources.queryTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
      return ret;
    }

    ret = CreateAclTensor(poolKeyHostData, poolKeyShape, &resources.poolKeyDeviceAddr,
                          aclDataType::ACL_FLOAT16, &resources.poolKeyTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
      return ret;
    }

    ret = CreateAclTensor(weightsHostData, weightsShape, &resources.weightsDeviceAddr,
                          aclDataType::ACL_FLOAT16, &resources.weightsTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
      return ret;
    }

    ret = CreateAclTensor(sparseIndicesHostData, sparseIndicesShape, &resources.sparseIndicesDeviceAddr,
                          aclDataType::ACL_INT32, &resources.sparseIndicesTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
      return ret;
    }

    ret = CreateAclTensor(sparseValuesHostData, sparseValuesShape, &resources.sparseValuesDeviceAddr,
                         aclDataType::ACL_FLOAT, &resources.sparseValuesTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
      return ret;
    }
    return ACL_SUCCESS;
}

int ExecutePoolKeyIndexer(TensorResources& resources, aclrtStream stream,
                        void** workspaceAddr, uint64_t* workspaceSize) {
    int64_t topk = 2048;
    int64_t poolSize = 16;
    int64_t maskMode = 3;
    int64_t quantMode = -1;
    bool returnValue = true;
    int64_t keyStride0 = -1;  // -1表示未指定，按连续输入推导stride
    int64_t kDescaleStride0 = -1;  // -1表示未指定，按连续输入推导stride(量化场景k_descale 0轴非连续寻址)
    constexpr const char layerOutStr[] = "BSND";
    constexpr size_t layerOutLen = sizeof(layerOutStr);
    char layoutQuery[layerOutLen];
    char layoutKey[layerOutLen];
    errno_t memcpyRet = memcpy_s(layoutQuery, sizeof(layoutQuery), layerOutStr, layerOutLen);
    if (!CHECK_RET(memcpyRet == 0)) {
        LOG_PRINT("memcpy_s layoutQuery failed. ERROR: %d\n", memcpyRet);
        return -1;
    }
    memcpyRet = memcpy_s(layoutKey, sizeof(layoutKey), layerOutStr, layerOutLen);
    if (!CHECK_RET(memcpyRet == 0)) {
        LOG_PRINT("memcpy_s layoutKey failed. ERROR: %d\n", memcpyRet);
        return -1;
    }
    // poolTailK为Host侧取值输入（ValueDepend），以aclIntArray类型传入
    std::vector<int64_t> poolTailKHostData = {0};
    aclIntArray* poolTailK = aclCreateIntArray(poolTailKHostData.data(), poolTailKHostData.size());
    aclOpExecutor* executor;

    int ret = aclnnPoolKeyIndexerGetWorkspaceSize(resources.queryTensor, resources.poolKeyTensor,
                                                 resources.weightsTensor, poolTailK,
                                                 nullptr, nullptr, nullptr, nullptr, nullptr,
                                                 topk, poolSize, layoutQuery, layoutKey,
                                                 maskMode, quantMode, returnValue, keyStride0, kDescaleStride0,
                                                 resources.sparseIndicesTensor, resources.sparseValuesTensor,
                                                 workspaceSize, &executor);

    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclnnPoolKeyIndexerGetWorkspaceSize failed. ERROR: %d\n", ret);
        return ret;
    }

    if (*workspaceSize > 0ULL) {
        ret = aclrtMalloc(workspaceAddr, *workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (!CHECK_RET(ret == ACL_SUCCESS)) {
            LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
            return ret;
        }
    }

    ret = aclnnPoolKeyIndexer(*workspaceAddr, *workspaceSize, executor, stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclnnPoolKeyIndexer failed. ERROR: %d\n", ret);
        return ret;
    }
    aclDestroyIntArray(poolTailK);

    return ACL_SUCCESS;
}

int PrintValuesOutResult(std::vector<int64_t> &shape, void** deviceAddr) {
  auto size = GetShapeSize(shape);
  std::vector<float> resultData(size, 0);
  auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]),
                         *deviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret);
        return ret;
  }
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("values result[%ld] is: %f\n", i, resultData[i]);
  }
  return ACL_SUCCESS;
}

int PrintIndicesOutResult(std::vector<int64_t> &shape, void** deviceAddr) {
  auto size = GetShapeSize(shape);
  std::vector<int32_t> resultData(size, 0);
  auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]),
                         *deviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret);
        return ret;
  }
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("indices result[%ld] is: %d\n", i, resultData[i]);
  }
  return ACL_SUCCESS;
}

void CleanupResources(TensorResources& resources, void* workspaceAddr,
                     aclrtStream stream, int32_t deviceId) {
    if (resources.queryTensor) {
      aclDestroyTensor(resources.queryTensor);
    }
    if (resources.poolKeyTensor) {
      aclDestroyTensor(resources.poolKeyTensor);
    }
    if (resources.weightsTensor) {
      aclDestroyTensor(resources.weightsTensor);
    }
    if (resources.sparseIndicesTensor) {
      aclDestroyTensor(resources.sparseIndicesTensor);
    }
    if (resources.sparseValuesTensor) {
      aclDestroyTensor(resources.sparseValuesTensor);
    }

    if (resources.queryDeviceAddr) {
      aclrtFree(resources.queryDeviceAddr);
    }
    if (resources.poolKeyDeviceAddr) {
      aclrtFree(resources.poolKeyDeviceAddr);
    }
    if (resources.weightsDeviceAddr) {
      aclrtFree(resources.weightsDeviceAddr);
    }
    if (resources.sparseIndicesDeviceAddr) {
      aclrtFree(resources.sparseIndicesDeviceAddr);
    }
    if (resources.sparseValuesDeviceAddr) {
      aclrtFree(resources.sparseValuesDeviceAddr);
    }

    if (workspaceAddr) {
      aclrtFree(workspaceAddr);
    }
    if (stream) {
      aclrtDestroyStream(stream);
    }
    aclrtResetDevice(deviceId);
    aclFinalize();
}

} // namespace

int main() {
    int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    TensorResources resources = {};
    void* workspaceAddr = nullptr;
    uint64_t workspaceSize = 0;
    std::vector<int64_t> sparseIndicesShape = {1, 2, 2063};
    std::vector<int64_t> sparseValuesShape = {1, 2, 128};
    int ret = ACL_SUCCESS;

    // 1. Initialize device and stream
    ret = Init(deviceId, &stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("Init acl failed. ERROR: %d\n", ret);
        return ret;
    }

    // 2. Initialize tensors
    ret = InitializeTensors(resources);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        CleanupResources(resources, workspaceAddr, stream, deviceId);
        return ret;
    }

    // 3. Execute the operation
    ret = ExecutePoolKeyIndexer(resources, stream, &workspaceAddr, &workspaceSize);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        CleanupResources(resources, workspaceAddr, stream, deviceId);
        return ret;
    }

    // 4. Synchronize stream
    ret = aclrtSynchronizeStream(stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
        CleanupResources(resources, workspaceAddr, stream, deviceId);
        return ret;
    }

    // 5. Process results
    PrintIndicesOutResult(sparseIndicesShape, &resources.sparseIndicesDeviceAddr);
    PrintValuesOutResult(sparseValuesShape, &resources.sparseValuesDeviceAddr);

    // 6. Cleanup resources
    CleanupResources(resources, workspaceAddr, stream, deviceId);
    return 0;
}
```
