# aclnnQuantSparseLightningIndexer

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR/Ascend 950DT</term>：不支持
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

- 算子功能：`QuantSparseLightningIndexer`是两级TopK（two-level TopK，先按块粗筛再在块内精筛）方案的**第二级**算子，负责在候选块范围内选出关键的稀疏token，并对输入query和key进行量化实现存8算8。

- 算子定位：本算子必须与第一级算子`aclnnQuantLightningIndexerV2`配合使用。第一级算子以`candidate_topk_blocks=2048`开启 source 后，按`candidate_block_size`个位置为一个块对每个token的行做块级打分，选出`candidate_topk_blocks`个候选块并输出块级索引`candidate_topk_index_out`；本算子以该索引作为必传输入`candidateTopkIndex`，只在候选块覆盖的位置上做token级TopK。**该算子不建议单独使用。**

- 计算公式：

$$
out = \text{Top-}k\left\{[1]_{1\times g}@\left[(W@[1]_{1\times S_{k}})\odot\text{ReLU}\left(\left(Scale_Q@Scale_K^T\right)\odot\left(Q_{index}^{Quant}@{\left(K_{index}^{Quant}\right)}^T\right)\right)\right]\right\},\quad p \in \text{候选块}
$$

主要计算过程为：

1. 将某个token对应的输入参数`query`（$Q_{index}^{Quant}\in\R^{g\times d}$）乘以给定上下文`key`（$K_{index}^{Quant}\in\R^{S_{k}\times d}$），得到相关性。
2. 相关性结果与`query`和`key`对应的反量化系数`queryDequantScale`（$Scale_Q$）和`keyDequantScale`（$Scale_K^T$）相乘，通过激活函数$ReLU$过滤无效负相关信号后，得到当前Token与所有前序Token的相关性分数向量。
3. 分数向量按候选块掩码降级（位置$p$属于块$\lfloor p / \text{candidate\_block\_size} \rfloor$，块号不在`candidateTopkIndex`中的位置分数被降级，不进入TopK）后，与权重系数`weights`（$W$）相乘，沿g的方向选取前$Top-k$个索引值得到输出$out$。

- 典型调用链：`aclnnQuantLightningIndexerV2Metadata`（生成分核信息`metadata`）→ `aclnnQuantLightningIndexerV2`（`candidate_topk_blocks=2048` 开启 source，输出`candidate_topk_index_out`）→ `aclnnQuantSparseLightningIndexer`（本算子）→ 稀疏attention。

## 函数原型

每个算子分为[两段式接口](../../../../docs/zh/context/two_phase_api.md)，必须先调用"aclnnQuantSparseLightningIndexerGetWorkspaceSize"接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用"aclnnQuantSparseLightningIndexer"接口执行计算。

```Cpp
aclnnStatus aclnnQuantSparseLightningIndexerGetWorkspaceSize(
    const aclTensor *q,
    const aclTensor *k,
    const aclTensor *w,
    const aclTensor *qDescale,
    const aclTensor *kDescale,
    const aclTensor *cuSeqlensQOptional,
    const aclTensor *cuSeqlensKOptional,
    const aclTensor *sequsedQOptional,
    const aclTensor *sequsedKOptional,
    const aclTensor *cmpResidualKOptional,
    const aclTensor *blockTableOptional,
    const aclTensor *outputIdxOffsetOptional,
    const aclTensor *metadataOptional,
    const aclTensor *candidateTopkIndex,
    const aclTensor *candidateBlockLengthOptional,
    int64_t          topk,
    int64_t          quantMode,
    int64_t          maxSeqlenQ,
    char            *layoutQOptional,
    char            *layoutKOptional,
    int64_t          maskMode,
    int64_t          cmpRatio,
    int64_t          returnValue,
    int64_t          candidateBlockSize,
    const aclTensor *sparseIndicesOut,
    const aclTensor *sparseValuesOut,
    uint64_t        *workspaceSize,
    aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnQuantSparseLightningIndexer(
    void             *workspace,
    uint64_t          workspaceSize,
    aclOpExecutor    *executor,
    const aclrtStream stream)
```

## aclnnQuantSparseLightningIndexerGetWorkspaceSize

- **参数说明：**

> [!NOTE]
>
> - query、key、weights参数维度含义：B（Batch Size）表示输入样本批量大小、S（Sequence Length）表示输入样本序列长度、H（Head Size）表示hidden层的大小、N（Head Num）表示多头数、D（Head Dim）表示hidden层最小的单元尺寸，且满足D=H/N、T表示所有Batch输入样本序列长度的累加和。
> - S1表示query shape中的S，S2表示key shape中的S，T1表示query shape中的T，N1表示query shape中的N，N2表示key shape中的N。
> - block_num为PageAttention时block总数，block_size为一个block的token数，candidateBlockSize为候选块大小（每个块包含的位置数）。

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
      <td>q</td>
      <td>输入</td>
      <td>公式中量化后的 Query。</td>
      <td>不支持空tensor。</td>
      <td>INT8</td>
      <td>ND</td>
      <td>
          <ul>
                <li>layout_q为BSND时，shape为(B,S1,N1,D)。</li>
                <li>layout_q为TND时，shape为(T1,N1,D)。</li>
          </ul>
      </td>
      <td>x</td>
    </tr>
    <tr>
      <td>k</td>
      <td>输入</td>
      <td>公式中量化后的 Key。</td>
      <td>
          <ul>
                <li>不支持空tensor。</li>
                <li>layout_k仅支持PA_BBND（按blockTable分页寻址）；支持0轴padding型非连续，stride由输入描述符携带。</li>
          </ul>
      </td>
      <td>INT8</td>
      <td>ND</td>
      <td>
          <ul>
                <li>layout_k为PA_BBND时，shape为(block_num, block_size, N2, D)。</li>
          </ul>
      </td>
      <td>支持0轴非连续</td>
    </tr>
    <tr>
      <td>w</td>
      <td>输入</td>
      <td>公式中的权重系数 W。</td>
      <td>不支持空tensor。</td>
      <td>FLOAT16</td>
      <td>ND</td>
      <td>
          <ul>
                <li>layout_q为BSND时，shape为(B,S1,N1)。</li>
                <li>layout_q为TND时，shape为(T1,N1)。</li>
          </ul>
      </td>
      <td>x</td>
    </tr>
    <tr>
      <td>qDescale</td>
      <td>输入</td>
      <td>公式中 Query 的反量化系数。</td>
      <td>不支持空tensor。</td>
      <td>FLOAT16</td>
      <td>ND</td>
      <td>
          <ul>
                <li>shape与weights一致。</li>
          </ul>
      </td>
      <td>x</td>
    </tr>
    <tr>
      <td>kDescale</td>
      <td>输入</td>
      <td>公式中 Key 的反量化系数。</td>
      <td>不支持空tensor。</td>
      <td>FLOAT16</td>
      <td>ND</td>
      <td>
          <ul>
                <li>shape为移除key的D轴。</li>
          </ul>
      </td>
      <td>支持0轴非连续</td>
    </tr>
    <tr>
      <td>cuSeqlensQOptional</td>
      <td>可选输入</td>
      <td>每个Batch中query的有效token数前缀和。</td>
      <td>layout_q为TND时必须传入；layout_q为BSND时不能传入。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>
          <ul>
                <li>shape为(B+1,)。</li>
          </ul>
      </td>
      <td>x</td>
    </tr>
    <tr>
      <td>cuSeqlensKOptional</td>
      <td>可选输入</td>
      <td>每个Batch中key的有效token数前缀和。</td>
      <td>layout_k为PA_BBND时不能传入。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>
          <ul>
                <li>shape为(B+1,)。</li>
          </ul>
      </td>
      <td>x</td>
    </tr>
    <tr>
      <td>sequsedQOptional</td>
      <td>可选输入</td>
      <td>每个Batch中query的有效token数。</td>
      <td>layout_q为BSND时可选传入。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>
          <ul>
                <li>shape为(B,)。</li>
          </ul>
      </td>
      <td>x</td>
    </tr>
    <tr>
      <td>sequsedKOptional</td>
      <td>可选输入</td>
      <td>每个Batch中key的有效token数。</td>
      <td>layout_k为PA_BBND时必传。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>
          <ul>
                <li>shape为(B,)。</li>
          </ul>
      </td>
      <td>x</td>
    </tr>
    <tr>
      <td>cmpResidualKOptional</td>
      <td>可选输入</td>
      <td>压缩场景下Key的残余长度。</td>
      <td>需满足0 &lt;= cmpResidualK[i] &lt; cmpRatio。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>
          <ul>
                <li>shape为(B,)。</li>
          </ul>
      </td>
      <td>x</td>
    </tr>
    <tr>
      <td>blockTableOptional</td>
      <td>可选输入</td>
      <td>PageAttention中KV存储使用的block映射表。</td>
      <td>layout_k为PA_BBND时必传。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>
          <ul>
                <li>shape为(B, maxBlockNumPerSeq)。</li>
          </ul>
      </td>
      <td>x</td>
    </tr>
    <tr>
      <td>outputIdxOffsetOptional</td>
      <td>可选输入</td>
      <td>输出索引的偏移量，仅加到sparseIndicesOut上。</td>
      <td>维数必须比sparseIndicesOut少1。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>
          <ul>
                <li>layout_q为BSND时，shape为(B,S1,N2)。</li>
                <li>layout_q为TND时，shape为(T1,N2)。</li>
          </ul>
      </td>
      <td>x</td>
    </tr>
    <tr>
      <td>metadataOptional</td>
      <td>可选输入</td>
      <td>aclnnQuantLightningIndexerV2Metadata算子传入的分核信息。</td>
      <td>必传，不能为空。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>
          <ul>
                <li>shape由metadata算子决定。</li>
          </ul>
      </td>
      <td>x</td>
    </tr>
    <tr>
      <td>candidateTopkIndex</td>
      <td>输入</td>
      <td>第一级算子aclnnQuantLightningIndexerV2在开启 source（`candidate_topk_blocks=2048`）时输出的块级候选索引。</td>
      <td>
          <ul>
                <li>必传，不支持空tensor。</li>
                <li>值域为[0, numBlocks)或-1（无效槽）。</li>
                <li>末维即候选块个数（当前仅支持 2048），元素个数必须等于B*S1*N2*末维（BSND）或T1*N2*末维（TND）。</li>
          </ul>
      </td>
      <td>INT32</td>
      <td>ND</td>
      <td>
          <ul>
                <li>layout_q为BSND时，shape为(B,S1,N2,候选块个数)。</li>
                <li>layout_q为TND时，shape为(T1,N2,候选块个数)。</li>
          </ul>
      </td>
      <td>x</td>
    </tr>
    <tr>
      <td>candidateBlockLengthOptional</td>
      <td>可选输入</td>
      <td>预留接口（N4/O13，本期不消费）。</td>
      <td>
          <ul>
                <li>仅接受空 tensor，传入非空 tensor 会在 tiling 报错。</li>
          </ul>
      </td>
      <td>INT32</td>
      <td>ND</td>
      <td>
          <ul>
                <li>shape 为 (0,)。</li>
          </ul>
      </td>
      <td>x</td>
    </tr>
    <tr>
      <td>topk</td>
      <td>属性</td>
      <td>topK阶段需要保留的索引数量。</td>
      <td>取值[1, 2048]，默认值2048。</td>
      <td>INT32</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>quantMode</td>
      <td>属性</td>
      <td>用于标识输入的量化模式。</td>
      <td>本算子仅支持2（INT8量化）。</td>
      <td>INT32</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>maxSeqlenQ</td>
      <td>可选属性</td>
      <td>Query的最大序列长度。</td>
      <td>默认值-1表示任意可能长度。</td>
      <td>INT32</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layoutQOptional</td>
      <td>可选属性</td>
      <td>用于标识输入query的数据排布格式。</td>
      <td>支持"BSND"、"TND"，默认值"BSND"。</td>
      <td>STRING</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layoutKOptional</td>
      <td>可选属性</td>
      <td>用于标识输入key的数据排布格式。</td>
      <td>仅支持"PA_BBND"。</td>
      <td>STRING</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>maskMode</td>
      <td>可选属性</td>
      <td>表示mask的模式。</td>
      <td>仅支持0（无mask）或3（下三角），默认值0。</td>
      <td>INT32</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cmpRatio</td>
      <td>可选属性</td>
      <td>用于稀疏计算，表示key的压缩倍数。</td>
      <td>默认值1。</td>
      <td>INT32</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>returnValue</td>
      <td>可选属性</td>
      <td>表示是否输出sparseValuesOut。</td>
      <td>仅支持0（不支持输出value），默认值0。</td>
      <td>INT32</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>candidateBlockSize</td>
      <td>可选属性</td>
      <td>候选块大小（每个块包含的位置数）。</td>
      <td>须 > 0，当前仅支持8，默认值8。</td>
      <td>INT32</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>sparseIndicesOut</td>
      <td>输出</td>
      <td>参与稀疏attention计算的token索引值。</td>
      <td>不支持空tensor。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>
          <ul>
                <li>layout_q为BSND时，shape为(B,S1,N2,topk)。</li>
                <li>layout_q为TND时，shape为(T1,N2,topk)。</li>
          </ul>
      </td>
      <td>x</td>
    </tr>
    <tr>
      <td>sparseValuesOut</td>
      <td>输出</td>
      <td>与sparseIndicesOut对应的value值。</td>
      <td>returnValueOptional为0时shape为(0,)。</td>
      <td>BFLOAT16</td>
      <td>ND</td>
      <td>
          <ul>
                <li>returnValueOptional为0时shape为(0,)。</li>
          </ul>
      </td>
      <td>x</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>输出</td>
      <td>返回需要在Device侧申请的workspace大小。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor</td>
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

  <table style="undefined;table-layout: fixed; width: 688px"><colgroup>
    <col style="width: 160px">
    <col style="width: 100px">
    <col style="width: 428px">
    </colgroup>
        <thead>
            <th>返回值</th>
            <th>错误码</th>
            <th>描述</th>
        </thead>
        <tbody>
            <tr>
                <td>ACLNN_ERR_PARAM_NULLPTR</td>
                <td>161001</td>
                <td>如果传入参数是必选输入，输出或者必选属性，且是空指针，则返回161001。</td>
            </tr>
            <tr>
                <td>ACLNN_ERR_PARAM_INVALID</td>
                <td>161002</td>
                <td>q、k、w、qDescale、kDescale、cuSeqlensQOptional、cuSeqlensKOptional、sequsedQOptional、sequsedKOptional、cmpResidualKOptional、blockTableOptional、outputIdxOffsetOptional、metadataOptional、candidateTopkIndex、layoutQOptional、layoutKOptional、topk、quantMode、maskMode、cmpRatio、returnValue、candidateBlockSize、sparseIndicesOut、sparseValuesOut的数据类型、数据格式或取值不在支持的范围内。</td>
            </tr>
        </tbody>
    </table>

## aclnnQuantSparseLightningIndexer

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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnQuantSparseLightningIndexerGetWorkspaceSize获取。</td>
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

  <table style="undefined;table-layout: fixed; width: 688px"><colgroup>
    <col style="width: 160px">
    <col style="width: 100px">
    <col style="width: 428px">
    </colgroup>
        <thead>
            <th>返回值</th>
            <th>错误码</th>
            <th>描述</th>
        </thead>
        <tbody>
            <tr>
                <td>ACLNN_ERR_PARAM_NULLPTR</td>
                <td>161001</td>
                <td>如果传入参数是必选输入，输出或者必选属性，且是空指针，则返回161001。</td>
            </tr>
            <tr>
                <td>ACLNN_ERR_PARAM_INVALID</td>
                <td>161002</td>
                <td>传入参数的数据类型、数据格式或取值不在支持的范围内。</td>
            </tr>
        </tbody>
    </table>

## 约束说明

- 本算子仅在<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>（arch22）上支持；<term>Ascend 950PR/Ascend 950DT</term>不支持。
- `quantMode`仅支持2；`q`、`k`支持INT8，`qDescale`、`kDescale`、`w`支持FLOAT16。
- `k`的N仅支持1；`q`的N与`k`的N之比仅支持64或32（即`q`的N支持64或32）。
- `topk`支持[1, 2048]。
- `layout_k`仅支持`PA_BBND`，因此`blockTableOptional`与`sequsedKOptional`必传，`cuSeqlensKOptional`不能传入。
- `layout_q`支持`BSND`与`TND`；为`TND`时必须传入`cuSeqlensQOptional`，且`layout_k`必须为`PA_BBND`。
- `candidateTopkIndex`为必传输入，数据类型为INT32；末维为候选块个数（当前仅支持 2048），元素个数必须为B*S1*N2*末维（BSND）或T1*N2*末维（TND），否则报错。
- `candidateBlockSize`当前仅支持8。
- `candidateBlockLengthOptional`为预留接口（N4/O13，本期不消费），仅接受空 tensor，传入非空值会在 tiling 报错。
- `metadataOptional`必传，不能为空，需由`aclnnQuantLightningIndexerV2Metadata`生成且与本次调用的query/key形状匹配。
- 不支持`returnValue`为1，`sparseValuesOut`恒为shape (0,)。
- 当传入的`layoutQOptional`为TND时，若同时传入`sequsedQOptional`，应保证由`sequsedQOptional`传入的各个batch的query长度不超过根据`cuSeqlensQOptional`计算出的各个batch的q序列长度；此时会启用TND Padding功能，将该batch的无效部分的`sparseIndicesOut`和`sparseValuesOut`置为无效值。部分长序列场景下，如果需要填充的无效数据过多，由于硬件限制可能会导致aicore执行超时，可以通过(seqlen2 - seqlen1) * topk来计算需要填充的数据量，建议将这个数据量控制在4亿以内。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../../docs/zh/context/compile_and_run_sample.md)。

- [test_aclnn_quant_sparse_lightning_indexer.cpp](../examples/test_aclnn_quant_sparse_lightning_indexer.cpp)
