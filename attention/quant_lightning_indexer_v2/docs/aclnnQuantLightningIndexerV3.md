# aclnnQuantLightningIndexerV3

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

- 算子功能：`aclnnQuantLightningIndexerV3`是[aclnnQuantLightningIndexerV2](aclnnQuantLightningIndexerV2.md)的两级TopK候选源(source)模式扩展接口。除位置级TopK结果外，新增candidateTopkBlocks、candidateBlockSize两个属性，以及candidateBlockIndicesOut、candidateBlockLengthOut两个输出，用于向quant_sparse_lightning_indexer提供候选逻辑块及每行有效候选数。

- 两级TopK候选源(source)模式：

  candidateTopkBlocks为2048、candidateBlockSize为8时，开启候选选择。算子将压缩后K位置空间中的相关性分数按candidateBlockSize划分为块，以块内可见token的最大分数作为块分数，选取候选块。该过程与位置级TopK共用相关性分数，不改变sparseIndicesOut、sparseValuesOut的计算定义。

  candidateBlockIndicesOut保存batch内的逻辑块号。块j对应压缩Key位置区间[j × candidateBlockSize, (j + 1) × candidateBlockSize)，有效块号写入每行前缀，其余槽位填-1。candidateBlockLengthOut保存每行实际有效候选数，后续计算只使用对应的有效前缀。

- 计算公式：

  $$
  out = \text{Top-}k\left\{[1]_{1\times g}@\left[(W@[1]_{1\times S_{k}})\odot\text{ReLU}\left(\left(Scale_Q@Scale_K^T\right)\odot\left(Q_{index}^{Quant}@{\left(K_{index}^{Quant}\right)}^T\right)\right)\right]\right\}
  $$

- 主要计算过程为：

  1. 将某个token对应的输入参数`q`（$Q_{index}^{Quant}\in\R^{g\times d}$）乘以给定上下文`k`（$K_{index}^{Quant}\in\R^{S_{k}\times d}$），得到相关性。
  2. 相关性结果与`q`和`k`对应的反量化系数`qDescale`（$Scale_Q$）和`kDescale`（$Scale_K^T$）相乘，通过激活函数$ReLU$过滤无效负相关信号后，得到当前Token与所有前序Token的相关性分数向量。
  3. 将其与权重系数`w`（$W$）相乘后，沿g方向求和，按分数选取前$Top-k$个位置，得到位置索引及可选的对应分数，作为Attention的输入。

- 与V2的关系：candidateTopkBlocks、candidateBlockSize均为-1时，关闭候选选择。在本接口支持的规格范围内，位置级计算与V2一致，两个候选输出均为空Tensor，shape为(0,)；也可传入nullptr，由接口内部创建空Tensor。关闭候选选择后，metadata仍必须由配套的aclnnQuantLightningIndexerV3Metadata生成。无需候选输出时，可直接使用V2接口。

## 函数原型

算子执行接口为[两段式接口](../../../docs/zh/context/two_phase_api.md)，必须先调用“aclnnQuantLightningIndexerV3GetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnQuantLightningIndexerV3”接口执行计算。

```Cpp
aclnnStatus aclnnQuantLightningIndexerV3GetWorkspaceSize(
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
    int64_t          topk,
    int64_t          quantMode,
    int64_t          maxSeqlenQ,
    char            *layoutQOptional,
    char            *layoutKOptional,
    int64_t          maskMode,
    int64_t          cmpRatio,
    int64_t          returnValue,
    int64_t          candidateTopkBlocks,
    int64_t          candidateBlockSize,
    const aclTensor *sparseIndicesOut,
    const aclTensor *sparseValuesOut,
    const aclTensor *candidateBlockIndicesOut,
    const aclTensor *candidateBlockLengthOut,
    uint64_t        *workspaceSize,
    aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnQuantLightningIndexerV3(
    void          *workspace,
    uint64_t       workspaceSize,
    aclOpExecutor *executor,
    aclrtStream    stream)
```

## aclnnQuantLightningIndexerV3GetWorkspaceSize

- **参数说明：**

  > [!NOTE]
  >
  > - q、k、w参数维度含义：B（Batch Size）表示输入样本批量大小、S（Sequence Length）表示输入样本序列长度、H（Head Size）表示hidden层的大小、N（Head Num）表示多头数、D（Head Dim）表示hidden层最小的单元尺寸，且满足D=H/N、T表示所有Batch输入样本序列长度的累加和。
  > - S1表示q shape中的S，S2表示k shape中的S，T1表示q shape中的T，T2表示k shape中的T，N1表示q shape中的N，N2表示k shape中的N。
  > - maxBlockNumPerSeq表示每个Batch中最大sequsedKOptional对应的block数量，S2_MAX表示sequsedKOptional中的最大值

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
      <td>公式中量化后的Query。</td>
      <td>不支持空tensor，N1支持32、64。</td>
      <td>INT8</td>
      <td>ND</td>
      <td>
          <ul>
                <li>layoutQOptional为BSND时，shape为(B,S1,N1,D)。</li>
                <li>layoutQOptional为TND时，shape为(T1,N1,D)。</li>
          </ul>
      </td>
      <td>x</td>
    </tr>
    <tr>
      <td>k</td>
      <td>输入</td>
      <td>公式中量化后的Key。</td>
      <td>
          <ul>
                <li>不支持空tensor。</li>
                <li>block_num为PageAttention时block总数，block_size为一个block的token数。</li>
          </ul>
      </td>
      <td>INT8</td>
      <td>ND</td>
      <td>
          <ul>
                <li>layoutKOptional为PA_BBND时，shape为(block_num, block_size, N2, D)，N2仅支持1。</li>
          </ul>
      </td>
      <td>支持0轴非连续</td>
    </tr>
    <tr>
      <td>w</td>
      <td>输入</td>
      <td>公式中的权重系数W。</td>
      <td>不支持空tensor。</td>
      <td>FLOAT16</td>
      <td>ND</td>
      <td>
          <ul>
                <li>layoutQOptional为BSND时，shape为(B,S1,N1)。</li>
                <li>layoutQOptional为TND时，shape为(T1,N1)。</li>
          </ul>
      </td>
      <td>x</td>
    </tr>
    <tr>
      <td>qDescale</td>
      <td>输入</td>
      <td>公式中Query的反量化系数。</td>
      <td>不支持空tensor。</td>
      <td>FLOAT16</td>
      <td>ND</td>
      <td>shape与w保持一致。</td>
      <td>x</td>
    </tr>
    <tr>
      <td>kDescale</td>
      <td>输入</td>
      <td>公式中Key的反量化系数。</td>
      <td>不支持空tensor。</td>
      <td>FLOAT16</td>
      <td>ND</td>
      <td>layoutKOptional为PA_BBND时，shape为(block_num,block_size,N2)。</td>
      <td>支持0轴非连续</td>
    </tr>
    <tr>
      <td>cuSeqlensQOptional</td>
      <td>输入</td>
      <td>每个Batch中，Query的有效token数（TND场景使用cu_seqlens格式）。</td>
      <td>
          <ul>
                <li>当layoutQOptional为TND时，该入参必须传入，且该入参元素的数量为B+1，该入参中每个元素的值表示当前batch与之前所有batch的token数总和，即前缀和。</li>
                <li>当layoutQOptional为BSND时，不允许传入。</li>
          </ul>
      </td>
      <td>INT32</td>
      <td>ND</td>
      <td>(B+1,)</td>
      <td>x</td>
    </tr>
    <tr>
      <td>cuSeqlensKOptional</td>
      <td>输入</td>
      <td>每个Batch中，Key的有效token数（TND场景使用cu_seqlens格式）。</td>
      <td>layoutKOptional为PA_BBND时，不允许传入。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>(B+1,)</td>
      <td>x</td>
    </tr>
    <tr>
      <td>sequsedQOptional</td>
      <td>输入</td>
      <td>每个Batch中，Query的有效token数。</td>
      <td>该入参中每个Batch的有效token数不超过q中的维度S大小且不小于0。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>(B,)</td>
      <td>x</td>
    </tr>
    <tr>
      <td>sequsedKOptional</td>
      <td>输入</td>
      <td>每个Batch中，Key的有效token数。</td>
      <td>
          <ul>
                <li>该入参中每个Batch的有效token数不超过k中的维度S大小且不小于0。</li>
                <li>当layoutKOptional为PA_BBND时，该入参必须传入。</li>
          </ul>
      </td>
      <td>INT32</td>
      <td>ND</td>
      <td>(B,)</td>
      <td>x</td>
    </tr>
    <tr>
      <td>cmpResidualKOptional</td>
      <td>输入</td>
      <td>压缩场景下Key的残余长度。</td>
      <td>maskMode为3且cmpRatio不为1时必须传入，需满足0 <= cmpResidualKOptional[i] < cmpRatio。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>(B,)</td>
      <td>x</td>
    </tr>
    <tr>
      <td>blockTableOptional</td>
      <td>输入</td>
      <td>表示PageAttention中KV存储使用的block映射表。</td>
      <td>
          <ul>
                <li>不支持空tensor。</li>
                <li>PageAttention场景下，blockTableOptional必须为二维，第一维长度需要等于B，第二维长度不能小于maxBlockNumPerSeq。</li>
          </ul>
      </td>
      <td>INT32</td>
      <td>ND</td>
      <td>(B, S2_MAX/block_size)</td>
      <td>x</td>
    </tr>
    <tr>
      <td>outputIdxOffsetOptional</td>
      <td>输入</td>
      <td>输出索引的偏移量。</td>
      <td>当前不支持，传入nullptr。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>layoutQOptional为BSND时shape为(B,S1,N2)，layoutQOptional为TND时shape为(T1,N2)。</td>
      <td>x</td>
    </tr>
    <tr>
      <td>metadataOptional</td>
      <td>输入</td>
      <td>aclnnQuantLightningIndexerV3Metadata接口传入的分核信息。</td>
      <td>
          <ul>
                <li>包含使用核数、分块大小以及每个核处理数据的起始点等内容。</li>
                <li>shape大小为[1024]，当前不支持传空。</li>
          </ul>
      </td>
      <td>INT32</td>
      <td>ND</td>
      <td>(1024,)</td>
      <td>x</td>
    </tr>
    <tr>
      <td>topk</td>
      <td>输入</td>
      <td>topK阶段需要保留的Key token索引数量。</td>
      <td>支持[1, 2048]。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>quantMode</td>
      <td>输入</td>
      <td>量化模式。</td>
      <td>仅支持2（Per-Token-Head量化）。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>maxSeqlenQ</td>
      <td>输入</td>
      <td>Query的最大序列长度。</td>
      <td>取值范围≥-1，显式给定时不小于实际有效长度。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layoutQOptional</td>
      <td>输入</td>
      <td>用于标识输入Query的数据排布格式。</td>
      <td>
          <ul>
                <li>支持BSND、TND。</li>
          </ul>
      </td>
      <td>STRING</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layoutKOptional</td>
      <td>输入</td>
      <td>用于标识输入Key的数据排布格式。</td>
      <td>
          <ul>
                <li>仅支持PA_BBND。</li>
          </ul>
      </td>
      <td>STRING</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>maskMode</td>
      <td>输入</td>
      <td>表示sparse的模式。</td>
      <td>
          <ul>
                <li>0代表defaultMask模式。</li>
                <li>3代表rightDownCausal模式的mask，对应以右顶点为划分的下三角场景。</li>
          </ul>
      </td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cmpRatio</td>
      <td>输入</td>
      <td>k的压缩倍数。</td>
      <td>
          <ul>
                <li>仅支持2的幂次方且范围为[1, 128]，即1/2/4/8/16/32/64/128。</li>
          </ul>
      </td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>returnValue</td>
      <td>输入</td>
      <td>表示是否输出sparseValuesOut。</td>
      <td>
          <ul>
                <li>1表示输出，0表示不输出，当前仅支持0。</li>
          </ul>
      </td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>candidateTopkBlocks</td>
      <td>输入</td>
      <td>两级TopK候选源(source)模式的候选块数量，即块级Top-k需要保留的候选块个数。</td>
      <td><ul><li>-1表示关闭candidate模式。</li><li>开启时仅支持2048。</li></ul></td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>candidateBlockSize</td>
      <td>输入</td>
      <td>候选块大小，即压缩后K空间中每个候选块包含的位置数。</td>
      <td><ul><li>candidateTopkBlocks为-1时，必须为-1。</li><li>candidateTopkBlocks为2048时，必须为8。</li><li>需与下游消费算子quant_sparse_lightning_indexer的配置保持一致。</li></ul></td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>sparseIndicesOut</td>
      <td>输出</td>
      <td>公式中的Indices输出。</td>
      <td>不支持空tensor。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>
          <ul>
                <li>layoutQOptional为"BSND"时输出shape为[B, S1, N2, topk]。</li>
                <li>layoutQOptional为"TND"时输出shape为[T1, N2, topk]。</li>
          </ul>
      </td>
      <td>x</td>
    </tr>
    <tr>
      <td>sparseValuesOut</td>
      <td>输出</td>
      <td>公式中的Indices输出对应的value值。</td>
      <td>当前returnValue仅支持0，输出shape为(0,)的空tensor。</td>
      <td>BFLOAT16</td>
      <td>ND</td>
      <td>(0,)</td>
      <td>x</td>
    </tr>
    <tr>
      <td>candidateBlockIndicesOut</td>
      <td>输出</td>
      <td>两级TopK候选源(source)模式的候选块索引输出。</td>
      <td><ul><li>candidateTopkBlocks开启时必须传入实际tensor。</li><li>关闭时可传nullptr（接口内部以空tensor占位）。</li><li>块号为batch内相对块号（压缩K空间），有效前缀内块号不重复，无效槽位填-1。</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td><ul><li>开启且layoutQOptional为"BSND"时输出shape为[B, S1, N2, candidateTopkBlocks]。</li><li>开启且layoutQOptional为"TND"时输出shape为[T1, N2, candidateTopkBlocks]。</li><li>关闭时输出shape为(0,)。</li></ul></td>
      <td>x</td>
    </tr>
    <tr>
      <td>candidateBlockLengthOut</td>
      <td>输出</td>
      <td>每行实际有效候选块数。</td>
      <td><ul><li>candidateTopkBlocks开启时必须传入实际tensor，取值范围为[0, candidateTopkBlocks]。</li><li>关闭时可传nullptr（接口内部以空tensor占位）。</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td><ul><li>开启且layoutQOptional为"BSND"时输出shape为[B, S1, N2]。</li><li>开启且layoutQOptional为"TND"时输出shape为[T1, N2]。</li><li>关闭时输出shape为(0,)。</li></ul></td>
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

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

  第一段接口会完成入参校验，出现以下场景时报错：

    <table style="undefined;table-layout: fixed;width: 1155px"><colgroup>
    <col style="width: 319px">
    <col style="width: 144px">
    <col style="width: 671px">
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
                <td>q、k、w、qDescale、kDescale、cuSeqlensQOptional、cuSeqlensKOptional、sequsedQOptional、sequsedKOptional、cmpResidualKOptional、blockTableOptional、metadataOptional、layoutQOptional、layoutKOptional、topk、quantMode、maskMode、cmpRatio、returnValue、candidateTopkBlocks、candidateBlockSize、sparseIndicesOut、sparseValuesOut、candidateBlockIndicesOut、candidateBlockLengthOut的数据类型、数据格式、shape或属性取值不在支持的范围内。</td>
            </tr>
        </tbody>
    </table>

## aclnnQuantLightningIndexerV3

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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnQuantLightningIndexerV3GetWorkspaceSize获取。</td>
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

- headdim支持128。
- block_size取值为16的倍数，最大支持1024。
- 当传入的参数layoutQOptional为TND时，必须传入cuSeqlensQOptional，如果也传入sequsedQOptional，应保证由sequsedQOptional传入的各个batch的q长度不超过根据cuSeqlensQOptional计算出的各个batch的q序列长度。当某个batch由sequsedQOptional传入的q序列长度seqlen1小于由cuSeqlensQOptional计算出的q长度seqlen2时，会启用TND Padding功能，将该batch的seqlen2与seqlen1差值部分的q输出的sparseIndicesOut和sparseValuesOut全部置为无效值。部分长序列场景下，如果需要填充的无效数据过多，由于硬件限制可能会导致aicore执行超时，可以通过(seqlen2 - seqlen1) * topk来计算需要填充的数据量，建议将这个数据量控制在4亿以内。
- **确定性说明：** aclnnQuantLightningIndexerV3默认确定性实现。
- **Batch一致性说明：** <term>Atlas A3系列产品</term>默认支持Batch一致性。

- candidateTopkBlocks、candidateBlockSize只支持(-1, -1)或(2048, 8)，需与配套aclnnQuantLightningIndexerV3Metadata接口的配置保持一致。
- metadataOptional必须由配套aclnnQuantLightningIndexerV3Metadata接口生成，序列信息、head配置、topk、quantMode、布局、maskMode、cmpRatio及candidate参数必须与本接口一致。
- candidateBlockLengthOut为每行实际有效候选块数，消费侧仅使用candidateBlockIndicesOut每行对应的有效前缀。

- 所有输入、输出Tensor须位于同一NPU。

## 调用示例

Q使用TND布局，K使用PA_BBND布局，quantMode=2，candidateTopkBlocks=2048，candidateBlockSize=8。先生成metadata，再在同一Stream中执行QLI V3；returnValue=0。

```cpp
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <numeric>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_quant_lightning_indexer_v3_metadata.h"
#include "aclnnop/aclnn_quant_lightning_indexer_v3.h"

#define CHECK_RET(expr)                                      \
    do {                                                     \
        int ret = (expr);                                     \
        if (ret != ACL_SUCCESS) {                            \
            std::printf("%s failed: %d\n", #expr, ret);        \
            return ret;                                      \
        }                                                    \
    } while (0)

struct Resources {
    aclrtStream stream;
    std::vector<aclTensor *> tensors;
    std::vector<void *> allocations;

    explicit Resources(aclrtStream inputStream) : stream(inputStream) {}
    ~Resources()
    {
        // 异常返回时也先等待已提交任务，随后释放Tensor描述和Device内存。
        aclrtSynchronizeStream(stream);
        for (auto tensor : tensors) {
            aclDestroyTensor(tensor);
        }
        for (auto address : allocations) {
            aclrtFree(address);
        }
    }

    int Allocate(uint64_t bytes, void **address)
    {
        CHECK_RET(aclrtMalloc(address, std::max<uint64_t>(bytes, 1), ACL_MEM_MALLOC_HUGE_FIRST));
        allocations.push_back(*address);
        return ACL_SUCCESS;
    }

    template <typename T>
    int CreateTensor(const std::vector<T> &host, const std::vector<int64_t> &shape,
                     aclDataType dtype, aclTensor **tensor, void **address)
    {
        uint64_t bytes = host.size() * sizeof(T);
        CHECK_RET(Allocate(bytes, address));
        if (bytes != 0) {
            CHECK_RET(aclrtMemcpy(*address, bytes, host.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE));
        }
        std::vector<int64_t> strides(shape.size(), 1);
        for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
            strides[i] = shape[i + 1] * strides[i + 1];
        }
        *tensor = aclCreateTensor(shape.data(), shape.size(), dtype, strides.data(), 0,
                                 ACL_FORMAT_ND, shape.data(), shape.size(), *address);
        if (*tensor == nullptr) {
            std::printf("aclCreateTensor failed\n");
            return 1;
        }
        tensors.push_back(*tensor);
        return ACL_SUCCESS;
    }
};

int RunExample(aclrtStream stream)
{
    Resources resources(stream);
    constexpr int64_t batch = 2;
    constexpr int64_t queryHeads = 32;
    constexpr int64_t keyHeads = 1;
    constexpr int64_t headDim = 128;
    constexpr int64_t keyLength = 2048;
    constexpr int64_t topk = 2048;
    constexpr int64_t candidateCapacity = 2048;
    constexpr int64_t candidateBlockSize = 8;
    char layoutQ[] = "TND";
    char layoutK[] = "PA_BBND";

    // 每个batch一个Query，Key有效长度均为2048；metadata使用固定1024个INT32元素。
    aclTensor *cuQ = nullptr;
    aclTensor *usedK = nullptr;
    aclTensor *metadata = nullptr;
    void *cuQAddress = nullptr;
    void *usedKAddress = nullptr;
    void *metadataAddress = nullptr;
    CHECK_RET(resources.CreateTensor<int32_t>({0, 1, 2}, {batch + 1}, ACL_INT32, &cuQ, &cuQAddress));
    CHECK_RET(resources.CreateTensor<int32_t>({keyLength, keyLength}, {batch}, ACL_INT32,
                                            &usedK, &usedKAddress));
    CHECK_RET(resources.CreateTensor<int32_t>(std::vector<int32_t>(1024), {1024}, ACL_INT32,
                                            &metadata, &metadataAddress));

    // 先生成metadata；后续计算使用完全相同的序列信息及属性。
    uint64_t metadataWorkspaceSize = 0;
    aclOpExecutor *metadataExecutor = nullptr;
    CHECK_RET(aclnnQuantLightningIndexerV3MetadataGetWorkspaceSize(
        cuQ, nullptr, nullptr, usedK, nullptr, queryHeads, keyHeads, headDim,
        topk, 2, batch, 1, keyLength, layoutQ, layoutK, 0, 1,
        candidateCapacity, candidateBlockSize, metadata, &metadataWorkspaceSize, &metadataExecutor));
    void *metadataWorkspace = nullptr;
    if (metadataWorkspaceSize != 0) {
        CHECK_RET(resources.Allocate(metadataWorkspaceSize, &metadataWorkspace));
    }
    CHECK_RET(aclnnQuantLightningIndexerV3Metadata(
        metadataWorkspace, metadataWorkspaceSize, metadataExecutor, stream));

    constexpr int64_t pageSize = 128;
    constexpr int64_t pagesPerBatch = keyLength / pageSize;
    constexpr int64_t pages = batch * pagesPerBatch;
    aclTensor *q = nullptr;
    aclTensor *k = nullptr;
    aclTensor *w = nullptr;
    aclTensor *qScale = nullptr;
    aclTensor *kScale = nullptr;
    aclTensor *blockTable = nullptr;
    aclTensor *sparseIndices = nullptr;
    aclTensor *sparseValues = nullptr;
    aclTensor *candidateIndices = nullptr;
    aclTensor *candidateLength = nullptr;
    void *address = nullptr;
    void *indicesAddress = nullptr;
    void *lengthAddress = nullptr;

    // A3采用INT8的Q/K及FP16的权重和反量化系数。0x3c00是FP16数值1。
    CHECK_RET(resources.CreateTensor<int8_t>(std::vector<int8_t>(batch * queryHeads * headDim, 1),
        {batch, queryHeads, headDim}, ACL_INT8, &q, &address));
    CHECK_RET(resources.CreateTensor<int8_t>(std::vector<int8_t>(pages * pageSize * keyHeads * headDim, 1),
        {pages, pageSize, keyHeads, headDim}, ACL_INT8, &k, &address));
    CHECK_RET(resources.CreateTensor<uint16_t>(std::vector<uint16_t>(batch * queryHeads, 0x3c00),
        {batch, queryHeads}, ACL_FLOAT16, &w, &address));
    CHECK_RET(resources.CreateTensor<uint16_t>(std::vector<uint16_t>(batch * queryHeads, 0x3c00),
        {batch, queryHeads}, ACL_FLOAT16, &qScale, &address));
    CHECK_RET(resources.CreateTensor<uint16_t>(std::vector<uint16_t>(pages * pageSize * keyHeads, 0x3c00),
        {pages, pageSize, keyHeads}, ACL_FLOAT16, &kScale, &address));
    std::vector<int32_t> pageIds(pages);
    std::iota(pageIds.begin(), pageIds.end(), 0);
    CHECK_RET(resources.CreateTensor<int32_t>(pageIds, {batch, pagesPerBatch}, ACL_INT32,
        &blockTable, &address));

    // returnValue=0，稀疏值输出为shape (0,)；候选容量不等于实际有效候选数。
    CHECK_RET(resources.CreateTensor<int32_t>(std::vector<int32_t>(batch * topk),
        {batch, keyHeads, topk}, ACL_INT32, &sparseIndices, &indicesAddress));
    CHECK_RET(resources.CreateTensor<uint16_t>({}, {0}, ACL_BF16, &sparseValues, &address));
    CHECK_RET(resources.CreateTensor<int32_t>(std::vector<int32_t>(batch * candidateCapacity),
        {batch, keyHeads, candidateCapacity}, ACL_INT32, &candidateIndices, &address));
    CHECK_RET(resources.CreateTensor<int32_t>(std::vector<int32_t>(batch),
        {batch, keyHeads}, ACL_INT32, &candidateLength, &lengthAddress));

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    CHECK_RET(aclnnQuantLightningIndexerV3GetWorkspaceSize(
        q, k, w, qScale, kScale, cuQ, nullptr, nullptr, usedK, nullptr, blockTable,
        nullptr, metadata, topk, 2, 1, layoutQ, layoutK, 0, 1, 0,
        candidateCapacity, candidateBlockSize, sparseIndices, sparseValues,
        candidateIndices, candidateLength, &workspaceSize, &executor));
    void *workspace = nullptr;
    if (workspaceSize != 0) {
        CHECK_RET(resources.Allocate(workspaceSize, &workspace));
    }
    // 与metadata在同一Stream执行，保证先生成分核信息再消费。
    CHECK_RET(aclnnQuantLightningIndexerV3(workspace, workspaceSize, executor, stream));
    CHECK_RET(aclrtSynchronizeStream(stream));
    std::vector<int32_t> hostLengths(batch);
    CHECK_RET(aclrtMemcpy(hostLengths.data(), batch * sizeof(int32_t), lengthAddress,
                         batch * sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST));
    for (int64_t i = 0; i < batch; ++i) {
        std::printf("row %ld candidate_block_length: %d\n", static_cast<long>(i), hostLengths[i]);
    }
    return ACL_SUCCESS;
}

int main()
{
    constexpr int32_t deviceId = 0;
    int ret = aclInit(nullptr);
    if (ret != ACL_SUCCESS) {
        return ret;
    }
    ret = aclrtSetDevice(deviceId);
    if (ret != ACL_SUCCESS) {
        aclFinalize();
        return ret;
    }
    aclrtStream stream = nullptr;
    ret = aclrtCreateStream(&stream);
    if (ret == ACL_SUCCESS) {
        ret = RunExample(stream);
        aclrtDestroyStream(stream);
    }
    aclrtResetDevice(deviceId);
    aclFinalize();
    return ret;
}
```
