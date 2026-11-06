# aclnnLightningIndexerV3

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

- 算子功能：aclnnLightningIndexerV3是[aclnnLightningIndexerV2](./aclnnLightningIndexerV2.md)的candidate（两级TopK候选源）扩展接口，在V2原型基础上新增`candidateTopkBlocks`、`candidateBlockSize`两个属性与`candidateBlockIndices`、`candidateBlockLength`两个输出。基础计算过程与V2一致：基于一系列操作得到每一个token对应的top-k个位置索引。

- 两级TopK候选源（source）模式：

  当属性`candidateTopkBlocks`传入非-1（-1为关闭）时，算子在位置级Top-k之外并行执行块级候选选择：将每个token在压缩后K位置空间上的相关性分数按`candidateBlockSize`个位置划分为一块（块内取最大分数，尾块以-inf补齐，最新可达token所在块强制入选），按分数选取前`candidateTopkBlocks`个块号写入`candidateBlockIndices`输出，供下游aclnnSparseLightningIndexer（consumer模式）两级TopK消费。候选块号为batch内相对块号（块j覆盖压缩位置区间[j×candidateBlockSize, (j+1)×candidateBlockSize)），无效槽位填-1，不叠加`outputIdxOffset`，槽位顺序不作为消费约束（按集合消费）。该模式不影响`sparseIndices`/`sparseValues`的计算结果。`candidateBlockLength`为预留接口，恒输出shape为(0,)的空tensor。

- 与V2的兼容关系：`candidateTopkBlocks`传入-1（关闭）时，本接口行为与aclnnLightningIndexerV2完全一致（candidate两个输出为shape (0,)的空tensor，可传nullptr由接口内部占位）。无需candidate能力的调用方建议直接使用aclnnLightningIndexerV2。

## 函数原型

算子执行接口为[两段式接口](../../../docs/zh/context/two_phase_api.md)，必须先调用“aclnnLightningIndexerV3GetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnLightningIndexerV3”接口执行计算。

```Cpp
aclnnStatus aclnnLightningIndexerV3GetWorkspaceSize(
    const aclTensor     *q,
    const aclTensor     *k,
    const aclTensor     *w,
    const aclTensor     *cuSeqlensQOptional,
    const aclTensor     *cuSeqlensKOptional,
    const aclTensor     *sequsedQOptional,
    const aclTensor     *sequsedKOptional,
    const aclTensor     *cmpResidualKOptional,
    const aclTensor     *blockTableOptional,
    const aclTensor     *outputIdxOffsetOptional,
    const aclTensor     *metadataOptional,
    int64_t              topk,
    int64_t              maxSeqlenQ,
    char                *layoutQOptional,
    char                *layoutKOptional,
    int64_t              maskMode,
    int64_t              cmpRatio,
    int64_t              returnValue,
    int64_t              candidateTopkBlocks,
    int64_t              candidateBlockSize,
    const aclTensor     *sparseIndices,
    const aclTensor     *sparseValues,
    const aclTensor     *candidateBlockIndices,
    const aclTensor     *candidateBlockLength,
    uint64_t            *workspaceSize,
    aclOpExecutor       **executor)
```

```Cpp
aclnnStatus aclnnLightningIndexerV3(
    void             *workspace,
    uint64_t          workspaceSize,
    aclOpExecutor    *executor,
    aclrtStream       stream)
```

## aclnnLightningIndexerV3GetWorkspaceSize

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1601px"><colgroup>
  <col style="width: 240px">
  <col style="width: 132px">
  <col style="width: 232px">
  <col style="width: 330px">
  <col style="width: 233px">
  <col style="width: 119px">
  <col style="width: 215px">
  <col style="width: 100px">
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
    <td>q（aclTensor*）</td>
    <td>输入</td>
    <td>公式中的输入Q。支持的shape为：
        <ul><li>layoutQOptional为BSND时，shape为(B,S1,N1,D)。</li><li>layoutQOptional为TND时，shape为(T1,N1,D)。</li></ul></td>
    <td>不支持空tensor。</td>
    <td>BFLOAT16、FLOAT16</td>
    <td>ND</td>
    <td>3-4</td>
    <td>x</td>
    </tr>
    <tr>
    <td>k（aclTensor*）</td>
    <td>输入</td>
    <td>公式中的输入K。支持的shape为：
        <ul><li>layoutKOptional为PA_BBND时，shape为(block_num, block_size, N2, D)。</li><li>layoutKOptional为BSND时，shape为(B, S2, N2, D)。</li><li>layoutKOptional为TND时，shape为(T2, N2, D)。</li></ul></td>
    <td><ul><li>不支持空tensor。</li><li>block_num为PageAttention时block总数，block_size为一个block的token数。</li></ul></td>
    <td>BFLOAT16、FLOAT16</td>
    <td>ND</td>
    <td>3-4</td>
    <td>✓</td>
    </tr>
    <tr>
    <td>w（aclTensor*）</td>
    <td>输入</td>
    <td>公式中的输入W。支持的shape为：
        <ul><li>layoutQOptional为BSND时，shape为(B,S1,N1)。</li><li>layoutQOptional为TND时，shape为(T1,N1)。</li></ul></td>
    <td>不支持空tensor。</td>
    <td>FLOAT</td>
    <td>ND</td>
    <td>2-3</td>
    <td>x</td>
    </tr>
    <tr>
    <td>cuSeqlensQOptional（aclTensor*）</td>
    <td>输入</td>
    <td>当前Batch及前序Batch中q的有效token数的累加和</td>
    <td><ul><li>shape约束为长度为B+1的一维tensor。</li><li>不能出现负值。</li></ul></td>
    <td>INT32</td>
    <td>ND</td>
    <td>1</td>
    <td>x</td>
    </tr>
    <tr>
    <td>cuSeqlensKOptional（aclTensor*）</td>
    <td>输入</td>
    <td>当前Batch及前序Batch中k的有效token数的累加和。</td>
    <td><ul><li>shape约束为长度为B+1的一维tensor。</li><li>不能出现负值。</li></ul></td>
    <td>INT32</td>
    <td>ND</td>
    <td>1</td>
    <td>x</td>
    </tr>
    <tr>
    <td>sequsedQOptional（aclTensor*）</td>
    <td>输入</td>
    <td>不同Batch中q的真实使用长度。</td>
    <td><ul><li>shape约束为长度为B的一维tensor。</li><li>不能出现负值。</li></ul></td>
    <td>INT32</td>
    <td>ND</td>
    <td>1</td>
    <td>x</td>
    </tr>
    <tr>
    <td>sequsedKOptional（aclTensor*）</td>
    <td>输入</td>
    <td>不同Batch中k的真实使用长度。</td>
    <td><ul><li>shape约束为长度为B的一维tensor。</li><li>不能出现负值。</li><li>layoutKOptional为PA_BBND时，该参数必须传入。</li></ul></td>
    <td>INT32</td>
    <td>ND</td>
    <td>1</td>
    <td>x</td>
    </tr>
    <tr>
    <td>cmpResidualKOptional（aclTensor*）</td>
    <td>输入</td>
    <td>表示k压缩前token数量除以cmpRatio的余数。</td>
    <td><ul><li>不支持空tensor。</li><li>需要在maskMode等于3，cmpRatio不等于1的场景下使用。</li><li>该参数每一个元素的值都应小于压缩率cmpRatio。</li></ul></td>
    <td>INT32</td>
    <td>ND</td>
    <td>1</td>
    <td>x</td>
    </tr>
    <tr>
    <td>blockTableOptional（aclTensor*）</td>
    <td>输入</td>
    <td>表示PageAttention中KV存储使用的block映射表。</td>
    <td><ul><li>不支持空tensor。</li><li>PageAttention场景下，block_table必须为二维，第一维长度需要等于B，第二维长度不能小于maxBlockNumPerSeq（maxBlockNumPerSeq为每个batch中最大的序列长度对应的block数量）</li></ul></td>
    <td>INT32</td>
    <td>ND</td>
    <td>2</td>
    <td>x</td>
    </tr>
    <tr>
    <td>outputIdxOffsetOptional（aclTensor*）</td>
    <td>输入</td>
    <td>表示topK结果输出索引所需要加上的偏移。支持的shape为：
        <ul><li>layoutQ为"BSND"时shape为[B, S1, N2]。</li><li>layoutQ为"TND"时shape为[T1, N2]。</li></ul></td>
    <td><ul><li>值必须大于等于0。</li><li>加上偏移后，topK index不能超过int32最大值。</li><li>candidate模式下偏移不叠加到candidateBlockIndices。</li></ul></td>
    <td>INT32</td>
    <td>ND</td>
    <td>2-3</td>
    <td>x</td>
    </tr>
    <tr>
    <td>metadataOptional（aclTensor*）</td>
    <td>输入</td>
    <td>LightningIndexerV2Metadata算子传入的分核信息，包含使用核数、分块大小以及每个核处理数据的起始点等内容。shape固定为[2048]。</td>
    <td><ul><li>不支持空tensor。</li></ul></td>
    <td>INT32</td>
    <td>ND</td>
    <td>1</td>
    <td>x</td>
    </tr>
    <tr>
    <td>topk（int64_t）</td>
    <td>输入</td>
    <td>topK阶段需要保留的Key token索引数量。</td>
    <td><ul><li>当前支持[1, 8192]。</li><li>candidate模式开启（candidateTopkBlocks不等于-1）时仅支持[1, 2048]。</li></ul></td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>maxSeqlenQ（int64_t）</td>
    <td>输入</td>
    <td>q的最大序列长度。</td>
    <td><ul><li>当前支持[-1]或大于等于[0]。</li><li>-1表示任意可能长度。</li><li>建议值是-1。</li></ul>
    </td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>layoutQ（char*）</td>
    <td>输入</td>
    <td>用于标识输入q的数据排布格式。</td>
    <td><ul><li>当前支持BSND、TND。</li><li>建议值是BSND。</li></ul></td>
    <td>STRING</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>layoutK（char*）</td>
    <td>输入</td>
    <td>用于标识输入k的数据排布格式。</td>
    <td><ul><li>当前支持PA_BBND、BSND、TND。</li><li>建议值是BSND。</li></ul></td>
    <td>STRING</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>maskMode（int64_t）</td>
    <td>输入</td>
    <td>表示mask的模式。</td>
    <td><ul><li>mask_mode为0时，代表defaultMask模式。</li><li>mask_mode为3时，代表rightDownCausal模式的mask，对应以右顶点为划分的下三角场景。</li><li>建议值是0。</li></ul></td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>cmpRatio（int64_t）</td>
    <td>输入</td>
    <td>用于稀疏计算，表示k的压缩倍数。</td>
    <td><ul><li>支持[1, 128]。</li></ul></td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>returnValue（int64_t）</td>
    <td>输入</td>
    <td>代表是否需要返回Indices对应的Values值。</td>
    <td><ul><li>0代表不返回（输出shape为(0,)的空tensor），1代表返回值。</li><li>值域为{0,1}。</li><li>returnValue=1时sparseValues输出位置级topk真实分数（shape与sparseIndices相同）。</li></ul></td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>candidateTopkBlocks（int64_t）</td>
    <td>输入</td>
    <td>两级TopK候选源模式的候选块数量，即块级Top-k需要保留的候选块个数。</td>
    <td><ul><li>-1表示关闭candidate模式，行为与aclnnLightningIndexerV2完全一致。</li><li>开启时支持(0, 2048]且必须是64的倍数。</li><li>开启时要求topk≤2048。</li></ul></td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>candidateBlockSize（int64_t）</td>
    <td>输入</td>
    <td>候选块大小，即压缩后K空间中每个候选块包含的位置数。</td>
    <td><ul><li>支持[2, 64]内的2的幂（2、4、8、16、32、64）。</li><li>candidateTopkBlocks关闭时同样校验该取值范围。</li><li>需与下游消费算子aclnnSparseLightningIndexer的配置保持一致。</li></ul></td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>sparseIndices（aclTensor*）</td>
    <td>输出</td>
    <td>公式中的Indices输出。支持的shape为：
        <ul><li>layoutQ为"BSND"时输出shape为[B, S1, N2, topk]。</li><li>layoutQ为"TND"时输出shape为[T1, N2, topk]。</li></ul></td>
    <td>不支持空tensor。</td>
    <td>INT32</td>
    <td>ND</td>
    <td>3-4</td>
    <td>x</td>
    </tr>
    <tr>
    <td>sparseValues（aclTensor*）</td>
    <td>输出</td>
    <td>公式中的Indices对应的Values输出。支持的shape为：
        <ul><li>returnValue为1且layoutQ为"BSND"时输出shape为[B, S1, N2, topk]。</li><li>returnValue为1且layoutQ为"TND"时输出shape为[T1, N2, topk]。</li></ul></td>
    <td>returnValue为1时输出对应值；为0时输出shape为(0,)的空tensor。</td>
    <td>FLOAT</td>
    <td>ND</td>
    <td>3-4</td>
    <td>x</td>
    </tr>
    <tr>
    <td>candidateBlockIndices（aclTensor*）</td>
    <td>输出</td>
    <td>两级TopK候选源模式的候选块索引输出。支持的shape为：
        <ul><li>开启且layoutQ为"BSND"时输出shape为[B, S1, N2, candidateTopkBlocks]。</li><li>开启且layoutQ为"TND"时输出shape为[T1, N2, candidateTopkBlocks]。</li><li>关闭时输出shape为(0,)。</li></ul></td>
    <td><ul><li>candidateTopkBlocks开启时必须传入实际tensor。</li><li>candidateTopkBlocks关闭时可传nullptr（接口内部以空tensor占位）。</li><li>块号为batch内相对块号（压缩K空间），无效槽位填-1，不叠加outputIdxOffset；消费侧按集合消费，不依赖槽位顺序。</li></ul></td>
    <td>INT32</td>
    <td>ND</td>
    <td>3-4</td>
    <td>x</td>
    </tr>
    <tr>
    <td>candidateBlockLength（aclTensor*）</td>
    <td>输出</td>
    <td>预留接口，kernel不消费。</td>
    <td>恒输出shape为(0,)的空tensor，可传nullptr（接口内部以空tensor占位）。</td>
    <td>INT32</td>
    <td>ND</td>
    <td>1</td>
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

  - q、k、w参数维度含义：B（Batch Size）表示输入样本批量大小、S（Sequence Length）表示输入样本序列长度、H（Head Size）表示hidden层的大小、N（Head Num）表示多头数、D（Head Dim）表示hidden层最小的单元尺寸，且满足D=H/N、T表示所有Batch输入样本序列长度的累加和。
  - 使用S1和S2分别表示q和k的输入样本序列长度，N1和N2分别表示q和k对应的多头数，k表示最后选取的索引个数。参数q中的D和参数k中的D值相等为128。T1和T2分别表示q和k的输入样本序列长度的累加和。

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
      <td>如果传入参数是必选输入，输出或者必选属性，且是空指针，则返回161001。candidateTopkBlocks开启（非-1）时candidateBlockIndices为空指针同样返回161001。</td>
      </tr>
      <tr>
      <td>ACLNN_ERR_PARAM_INVALID</td>
      <td>161002</td>
      <td><ul><li>q、k、w、cuSeqlensQ、cuSeqlensK、sequsedQ、sequsedK、cmpResidualK、blockTable、outputIdxOffset、metadata、sparseIndices、sparseValues、candidateBlockIndices、candidateBlockLength的数据类型和数据格式不在支持的范围内。</li><li>属性取值非法，如：topk、cmpRatio、maskMode、returnValue、maxSeqlenQ超出支持范围（returnValue仅支持0或1）；candidateTopkBlocks开启时不在(0, 2048]内或不是64的倍数；candidateTopkBlocks开启时topk大于2048；candidateBlockSize不为[2, 64]内的2的幂。</li></ul></td>
      </tr>
      </tbody>
    </table>

## aclnnLightningIndexerV3

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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnLightningIndexerV3GetWorkspaceSize获取。</td>
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

- 确定性计算：
  - aclnnLightningIndexerV3默认确定性实现。
- 参数q的N支持1~64，k的N支持1。
- headdim支持128。
- pa_kv_cache支持0轴非连续；pa_block_size支持[16, 1024]，且是16的倍数。
- 参数q、k的数据类型应保持一致。
- sparseIndices无效部分填-1；sparseValues无效部分填-inf；candidateBlockIndices无效槽位填-1。
- 传入的cmpResidualKOptional中每一个元素的值都应小于压缩率cmpRatio。
- candidate（两级TopK候选源）模式的candidateBlockSize须为[2, 64]内的2的幂（candidateTopkBlocks关闭时同样校验），且需与下游消费算子aclnnSparseLightningIndexer的配置保持一致（该一致性为调用方约束，框架层无法跨算子校验）；candidateBlockLength为预留接口，恒输出空tensor。

<!-- npu="A3,910b" id7 -->
- <term>Atlas A3系列产品</term>、<term>Atlas A2系列产品</term>：
  - 支持本接口，且支持开启candidate（两级TopK候选源）模式。
  - topk取值范围当前仅支持[1, 2048]，以及3072、4096、5120、6144、7168、8192。
  - 当前不支持sequsedQOptional、outputIdxOffsetOptional、maxSeqlenQ功能，不建议传入这些参数。
  - 当layoutKOptional为PA_BBND时，必须传入sequsedKOptional；当layoutK为BSND或TND时，sequsedKOptional可选传入。
  - 两级TopK候选源（source）模式仅在本系列产品上支持开启：candidateTopkBlocks传入非-1即开启；开启时topk仅支持[1, 2048]、returnValue已放开（值域{0,1}，rv=1时sparseValues输出位置级topk真实分数，shape与sparseIndices相同），且必须传入实际的candidateBlockIndices输出tensor（shape为[B, S1, N2, candidateTopkBlocks]或[T1, N2, candidateTopkBlocks]）。
<!-- end id7 -->
<!-- npu="950" id8 -->
- <term>Ascend 950PR&950DT系列产品</term>：
  - 支持本接口；candidate关闭（candidateTopkBlocks传入-1）时，接口行为与aclnnLightningIndexerV2一致。
  - 两级TopK候选源（source）模式在本系列产品上暂不支持开启：candidateTopkBlocks传入非-1时会被host侧tiling校验拒绝并返回错误。
<!-- end id8 -->

## 调用示例

完整示例（与[examples/test_aclnn_lightning_indexer_v3.cpp](../examples/test_aclnn_lightning_indexer_v3.cpp)保持一致，candidate source模式，仅Ascend 910B/910_93支持）：

```Cpp
/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include "securec.h"
#include "acl/acl.h"
#include "aclnnop/aclnn_lightning_indexer_v3.h"
#include "aclnn/opdev/platform.h"

using namespace std;

namespace {

#define CHECK_RET(cond) ((cond) ? true : (false))

#define LOG_PRINT(message, ...) \
    do { \
        (void)printf(message, ##__VA_ARGS__); \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream)
{
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
                    aclDataType dataType, aclTensor** tensor)
{
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
    void* keyDeviceAddr = nullptr;
    void* weightsDeviceAddr = nullptr;
    void* cmpResidualKDeviceAddr = nullptr;
    void* sparseIndicesDeviceAddr = nullptr;
    void* sparseValuesDeviceAddr = nullptr;
    // candidate source 模式输出：on 时为 [B,S1,N2,candTopkBlocks] 实 tensor（block_length 恒空）
    void* candidateSrcIndicesDeviceAddr = nullptr;

    aclTensor* queryTensor = nullptr;
    aclTensor* keyTensor = nullptr;
    aclTensor* weightsTensor = nullptr;
    aclTensor* cmpResidualKTensor = nullptr;
    aclTensor* sparseIndicesTensor = nullptr;
    aclTensor* sparseValuesTensor = nullptr;
    aclTensor* candidateSrcIndicesTensor = nullptr;
};

int InitializeTensors(TensorResources& resources)
{
    int64_t B = 2;
    int64_t S1 = 64;
    int64_t N1 = 16;
    int64_t D = 128;
    int64_t N2 = 1;
    int64_t topk = 32;
    int64_t cmpRatio = 4;
    int64_t S2Orig = 130;
    int64_t S2Compressed = S2Orig / cmpRatio;
    int64_t resK = S2Orig % cmpRatio;
    constexpr int64_t CAND_TOPK_BLOCKS = 64; // (0, 2048] 且 64 的倍数

    std::vector<int64_t> queryShape = {B, S1, N1, D};
    std::vector<int64_t> keyShape = {B, S2Compressed, N2, D};
    std::vector<int64_t> weightsShape = {B, S1, N1};
    std::vector<int64_t> cmpResidualKShape = {B};
    std::vector<int64_t> sparseIndicesShape = {B, S1, N2, topk};
    std::vector<int64_t> sparseValuesShape = {B, S1, N2, topk};
    std::vector<int64_t> candidateIndicesShape = {B, S1, N2, CAND_TOPK_BLOCKS};

    int64_t queryShapeSize = GetShapeSize(queryShape);
    int64_t keyShapeSize = GetShapeSize(keyShape);
    int64_t weightsShapeSize = GetShapeSize(weightsShape);
    int64_t sparseIndicesShapeSize = GetShapeSize(sparseIndicesShape);
    int64_t sparseValuesShapeSize = GetShapeSize(sparseValuesShape);
    int64_t candidateIndicesSize = GetShapeSize(candidateIndicesShape);

    std::vector<uint16_t> queryHostData(queryShapeSize, 0x3C00);
    std::vector<uint16_t> keyHostData(keyShapeSize, 0x3C00);
    std::vector<float> weightsHostData(weightsShapeSize, 1.0f);
    std::vector<int32_t> cmpResidualKHostData(B, static_cast<int32_t>(resK));
    std::vector<int32_t> sparseIndicesHostData(sparseIndicesShapeSize, 0);
    std::vector<float> sparseValuesHostData(sparseValuesShapeSize, 0.0f);
    std::vector<int32_t> candidateIndicesHostData(candidateIndicesSize, 0);

    int ret = CreateAclTensor(queryHostData, queryShape, &resources.queryDeviceAddr, aclDataType::ACL_FLOAT16,
                              &resources.queryTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }

    ret = CreateAclTensor(keyHostData, keyShape, &resources.keyDeviceAddr, aclDataType::ACL_FLOAT16,
                          &resources.keyTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }

    ret = CreateAclTensor(weightsHostData, weightsShape, &resources.weightsDeviceAddr, aclDataType::ACL_FLOAT,
                          &resources.weightsTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }

    ret = CreateAclTensor(cmpResidualKHostData, cmpResidualKShape, &resources.cmpResidualKDeviceAddr,
                          aclDataType::ACL_INT32, &resources.cmpResidualKTensor);
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

    ret = CreateAclTensor(candidateIndicesHostData, candidateIndicesShape, &resources.candidateSrcIndicesDeviceAddr,
                          aclDataType::ACL_INT32, &resources.candidateSrcIndicesTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }

    return ACL_SUCCESS;
}

// candidate (two-level topk) source 模式调用示例：candidate_topk_blocks=64（(0,2048] 且 64 的倍数）
// 时开启，candidate_block_indices 输出候选块索引（块号或 -1、无序、batch 内相对块号），
// candidate_block_length 为预留接口恒空（传 nullptr 由接口内部以空 tensor 兜底）。
// candidate 仅 ascend910b/ascend910_93（DAV_2201）支持，ascend950（DAV_3510）上启用会被
// host 校验拒绝，调用方需按 arch 门控。
int ExecuteLightningIndexerV3CandidateSource(TensorResources& resources, aclrtStream stream, void** workspaceAddr,
                                             uint64_t* workspaceSize)
{
    int64_t topk = 32;
    int64_t maxSeqlenQ = -1;
    int64_t maskMode = 3;
    int64_t cmpRatio = 4;
    int64_t returnValue = 0; // candidate 模式已放开 rv：0=恒空占位；1=输出位置级 topk 真实分
    int64_t candidateTopkBlocks = 64; // source on：(0, 2048] 且 64 的倍数
    int64_t candidateBlockSize = 8;   // [2, 64] 且 2 的幂（缺省 8）
    constexpr const char layoutQueryStr[] = "BSND";
    constexpr const char layoutKeyStr[] = "BSND";
    constexpr size_t layoutQueryLen = sizeof(layoutQueryStr);
    constexpr size_t layoutKeyLen = sizeof(layoutKeyStr);
    char layoutQuery[layoutQueryLen];
    char layoutKey[layoutKeyLen];
    errno_t memcpyRet = memcpy_s(layoutQuery, sizeof(layoutQuery), layoutQueryStr, layoutQueryLen);
    if (!CHECK_RET(memcpyRet == 0)) {
        LOG_PRINT("memcpy_s layoutQuery failed. ERROR: %d\n", memcpyRet);
        return -1;
    }
    memcpyRet = memcpy_s(layoutKey, sizeof(layoutKey), layoutKeyStr, layoutKeyLen);
    if (!CHECK_RET(memcpyRet == 0)) {
        LOG_PRINT("memcpy_s layoutKey failed. ERROR: %d\n", memcpyRet);
        return -1;
    }
    aclOpExecutor* executor;

    int ret = aclnnLightningIndexerV3GetWorkspaceSize(
        resources.queryTensor, resources.keyTensor, resources.weightsTensor, nullptr, nullptr, nullptr, nullptr,
        resources.cmpResidualKTensor, nullptr, nullptr, nullptr /* metadata: arch22 kernel 不消费，传 nullptr 即可 */,
        topk, maxSeqlenQ, layoutQuery, layoutKey, maskMode, cmpRatio, returnValue, candidateTopkBlocks,
        candidateBlockSize, resources.sparseIndicesTensor, resources.sparseValuesTensor,
        resources.candidateSrcIndicesTensor, nullptr /* candidate_block_length: 预留，可传 nullptr */, workspaceSize,
        &executor);

    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclnnLightningIndexerV3GetWorkspaceSize(candidate source) failed. ERROR: %d\n", ret);
        return ret;
    }

    if (*workspaceSize > 0ULL) {
        ret = aclrtMalloc(workspaceAddr, *workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (!CHECK_RET(ret == ACL_SUCCESS)) {
            LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
            return ret;
        }
    }

    ret = aclnnLightningIndexerV3(*workspaceAddr, *workspaceSize, executor, stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclnnLightningIndexerV3(candidate source) failed. ERROR: %d\n", ret);
        return ret;
    }

    return ACL_SUCCESS;
}

int PrintOutResult(const char* name, std::vector<int64_t>& shape, void** deviceAddr)
{
    auto size = GetShapeSize(shape);
    std::vector<int32_t> resultData(size, 0);
    auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), *deviceAddr,
                           size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret);
        return ret;
    }
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("%s result[%ld] is: %d\n", name, i, resultData[i]);
    }
    return ACL_SUCCESS;
}

void CleanupResources(TensorResources& resources, void* workspaceAddr, aclrtStream stream, int32_t deviceId)
{
    if (resources.queryTensor) {
        aclDestroyTensor(resources.queryTensor);
    }
    if (resources.keyTensor) {
        aclDestroyTensor(resources.keyTensor);
    }
    if (resources.weightsTensor) {
        aclDestroyTensor(resources.weightsTensor);
    }
    if (resources.cmpResidualKTensor) {
        aclDestroyTensor(resources.cmpResidualKTensor);
    }
    if (resources.sparseIndicesTensor) {
        aclDestroyTensor(resources.sparseIndicesTensor);
    }
    if (resources.sparseValuesTensor) {
        aclDestroyTensor(resources.sparseValuesTensor);
    }
    if (resources.candidateSrcIndicesTensor) {
        aclDestroyTensor(resources.candidateSrcIndicesTensor);
    }

    if (resources.queryDeviceAddr) {
        aclrtFree(resources.queryDeviceAddr);
    }
    if (resources.keyDeviceAddr) {
        aclrtFree(resources.keyDeviceAddr);
    }
    if (resources.weightsDeviceAddr) {
        aclrtFree(resources.weightsDeviceAddr);
    }
    if (resources.cmpResidualKDeviceAddr) {
        aclrtFree(resources.cmpResidualKDeviceAddr);
    }
    if (resources.sparseIndicesDeviceAddr) {
        aclrtFree(resources.sparseIndicesDeviceAddr);
    }
    if (resources.sparseValuesDeviceAddr) {
        aclrtFree(resources.sparseValuesDeviceAddr);
    }
    if (resources.candidateSrcIndicesDeviceAddr) {
        aclrtFree(resources.candidateSrcIndicesDeviceAddr);
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

int main()
{
    int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    TensorResources resources = {};
    void* workspaceAddr = nullptr;
    uint64_t workspaceSize = 0;
    int64_t B = 2;
    int64_t S1 = 64;
    int64_t N2 = 1;
    int64_t topk = 32;
    int64_t candidateTopkBlocks = 64; // source 示例：(0, 2048] 且 64 的倍数
    std::vector<int64_t> sparseIndicesShape = {B, S1, N2, topk};
    std::vector<int64_t> candidateIndicesShape = {B, S1, N2, candidateTopkBlocks};
    int ret = ACL_SUCCESS;

    ret = Init(deviceId, &stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("Init acl failed. ERROR: %d\n", ret);
        return ret;
    }

    // candidate（两级 TopK source 模式）仅 ascend910b/ascend910_93（arch22）支持，
    // ascend950（arch35）暂不支持。平台信息须在设备初始化后读取（设备初始化前
    // GetCurNpuArch() 返回无效值，门控将恒命中早退导致示例空跑）。
    const NpuArch npuArch = op::GetCurrentPlatformInfo().GetCurNpuArch();
    if (npuArch != NpuArch::DAV_2201) {
        LOG_PRINT("skip: candidate source mode requires DAV_2201 (ascend910b/ascend910_93), current arch = %u.\n",
                  static_cast<uint32_t>(npuArch));
        CleanupResources(resources, workspaceAddr, stream, deviceId);
        return 0;
    }

    ret = InitializeTensors(resources);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("InitializeTensors failed. ERROR: %d\n", ret);
        CleanupResources(resources, workspaceAddr, stream, deviceId);
        return ret;
    }

    ret = ExecuteLightningIndexerV3CandidateSource(resources, stream, &workspaceAddr, &workspaceSize);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("ExecuteLightningIndexerV3CandidateSource failed. ERROR: %d\n", ret);
        CleanupResources(resources, workspaceAddr, stream, deviceId);
        return ret;
    }

    ret = aclrtSynchronizeStream(stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
        CleanupResources(resources, workspaceAddr, stream, deviceId);
        return ret;
    }

    PrintOutResult("sparse_indices", sparseIndicesShape, &resources.sparseIndicesDeviceAddr);
    PrintOutResult("candidate_block_indices", candidateIndicesShape, &resources.candidateSrcIndicesDeviceAddr);

    CleanupResources(resources, workspaceAddr, stream, deviceId);
    return 0;
}
```
