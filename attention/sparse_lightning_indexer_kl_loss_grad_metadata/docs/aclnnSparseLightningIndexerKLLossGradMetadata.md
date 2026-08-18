# aclnnSparseLightningIndexerKLLossGradMetadata

[📄 查看源码](https://gitcode.com/cann/ops-transformer/tree/master/attention/sparse_lightning_indexer_kl_loss_grad_metadata)

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
- <term>Atlas 200I/500 A2 推理产品</term>：x
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas 推理系列产品</term>：x
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas 训练系列产品</term>：x
<!-- end id6 -->

## 功能说明

- 算子功能：该算子为AICPU算子，是aclnnSparseLightningIndexerKLLossGrad算子的前置算子。根据aclnnSparseLightningIndexerKLLossGrad算子的输入shape、layout、mask和压缩比例信息，计算并输出分核切分metadata。输出结果可作为aclnnSparseLightningIndexerKLLossGrad算子的metadataOptional输入，减少主算子tiling阶段对host array的访问。

  **该算子不建议单独使用，建议与aclnnSparseLightningIndexerKLLossGrad算子配合使用，形成完整的工作流。**
    1. 接收主算子的shape信息，包括batchSize、maxSeqlenQ、maxSeqlenK、numHeadsQ、numHeadsK、headDim、topk、layout和mask信息。
    2. 根据每个q对应的有效sparse长度估算负载，并将B/S1合轴后的任务均衡切分到可用AIC核上。
    3. 输出metadata后，后续作为aclnnSparseLightningIndexerKLLossGrad算子的metadataOptional输入使用。

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/two_phase_api.md)，必须先调用"aclnnSparseLightningIndexerKLLossGradMetadataGetWorkspaceSize"获取workspace大小，再调用"aclnnSparseLightningIndexerKLLossGradMetadata"执行计算。

``` cpp
aclnnStatus aclnnSparseLightningIndexerKLLossGradMetadataGetWorkspaceSize(
    const aclTensor *cuSeqLensQOptional,
    const aclTensor *cuSeqLensKOptional,
    const aclTensor *seqUsedQOptional,
    const aclTensor *seqUsedKOptional,
    const aclTensor *cmpResidualKOptional,
    int64_t batchSize,
    int64_t maxSeqlenQ,
    int64_t maxSeqlenK,
    int64_t numHeadsQ,
    int64_t numHeadsK,
    int64_t headDim,
    int64_t topk,
    char *layoutQOptional,
    char *layoutKOptional,
    int64_t maskMode,
    int64_t cmpRatio,
    const aclTensor *metadata,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);
```

``` cpp
aclnnStatus aclnnSparseLightningIndexerKLLossGradMetadata(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);
```

## aclnnSparseLightningIndexerKLLossGradMetadataGetWorkspaceSize

- **参数说明**

  <table style="undefined;table-layout: fixed; width: 1600px"><colgroup>
  <col style="width: 150px">
  <col style="width: 100px">
  <col style="width: 350px">
  <col style="width: 150px">
  <col style="width: 70px">
  <col style="width: 70px">
  <col style="width: 190px">
  <col style="width: 80px">
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
      <td>cuSeqLensQOptional（aclTensor*）</td>
      <td>输入</td>
      <td>表示不同batch中q的累积sequence length。</td>
      <td><ul><li>支持空Tensor</li><li>shape固定为(B+1, )。</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>1维</td>
      <td>√</td>
    </tr>
    <tr>
      <td>cuSeqLensKOptional（aclTensor*）</td>
      <td>输入</td>
      <td>表示不同batch中k的累积sequence length。</td>
      <td><ul><li>支持空Tensor</li><li>shape固定为(B+1, )。</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>1维</td>
      <td>√</td>
    </tr>
    <tr>
      <td>seqUsedQOptional（aclTensor*）</td>
      <td>输入</td>
      <td>表示不同batch中q实际参与运算的sequence length。</td>
      <td><ul><li>支持空Tensor。</li><li>shape固定为(B, )。</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>1维</td>
      <td>√</td>
    </tr>
    <tr>
      <td>seqUsedKOptional（aclTensor*）</td>
      <td>输入</td>
      <td>表示不同batch中k实际参与运算的sequence length。</td>
      <td><ul><li>支持空Tensor。</li><li>shape固定为(B, )。</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>1维</td>
      <td>√</td>
    </tr>
    <tr>
      <td>cmpResidualKOptional（aclTensor*）</td>
      <td>输入</td>
      <td>表示不同batch中k的sequence length与cmpRatio相关的残差。</td>
      <td><ul><li>支持空Tensor。</li><li>shape固定为(B, )。</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>1维</td>
      <td>√</td>
    </tr>
    <tr>
      <td>batchSize（int64_t）</td>
      <td>输入</td>
      <td>表示batch数量。</td>
      <td><ul><li>支持非负数。TND场景可填0，并通过cuSeqLensQOptional推导。</li><li>建议值为0。</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>maxSeqlenQ（int64_t）</td>
      <td>输入</td>
      <td>表示q的最大sequence length。</td>
      <td><ul><li>支持非负数。BSND场景必须为正数。</li><li>建议值为0。</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>maxSeqlenK（int64_t）</td>
      <td>输入</td>
      <td>表示k的最大sequence length。</td>
      <td><ul><li>支持非负数。BSND场景必须为正数。</li><li>建议值为0。</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>numHeadsQ（int64_t）</td>
      <td>输入</td>
      <td>表示q的head个数。</td>
      <td>必须为正数，并且能被numHeadsK整除，当前支持[1, 128]。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>numHeadsK（int64_t）</td>
      <td>输入</td>
      <td>表示k的head个数。</td>
      <td>必须为正数，当前仅支持1。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>headDim（int64_t）</td>
      <td>输入</td>
      <td>表示q/k的head dimension。</td>
      <td>必须为正数，当前仅支持128。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>topk（int64_t）</td>
      <td>输入</td>
      <td>表示从k中筛选出的关键token个数。</td>
      <td>必须为正数，当前支持[1, 2048]和4096、8192。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layoutQOptional（char*）</td>
      <td>输入</td>
      <td>表示q侧的排列格式。</td>
      <td><ul><li>支持 BSND、TND。</li><li>建议值为BSND。</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layoutKOptional（char*）</td>
      <td>输入</td>
      <td>表示k侧的排列格式。</td>
      <td><ul><li>支持 BSND、TND。</li><li>建议值为BSND。</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>maskMode（int64_t）</td>
      <td>输入</td>
      <td>表示sparse mask模式。</td>
      <td><ul><li>0: No mask。</li><li>3: rightDownCausal模式的mask，对应以右顶点为划分的下三角场景。</li><li>建议值为0。</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cmpRatio（int64_t）</td>
      <td>输入</td>
      <td>表示k的压缩率。</td>
      <td><ul><li>取值范围[1，128]。</li><li>建议值1，表示无压缩。</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>metadata（aclTensor*）</td>
      <td>输出</td>
      <td>表示负载均衡结果输出。</td>
      <td>输出结果作为aclnnSparseLightningIndexerKLLossGrad的metadataOptional输入，shape固定为(64, )。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>1维</td>
      <td>x</td>
    </tr>
    <tr>
      <td>workspaceSize（uint64_t*）</td>
      <td>输出</td>
      <td>返回需要在Device侧申请的workspace大小。</td>
      <td>当前实现返回0。</td>
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

  <ul>
    <!-- npu="950" id7 -->
    <li><term>Ascend 950PR/Ascend 950DT</term> ：topk仅支持[1, 2048]。</li>
    <!-- end id7 -->
    <!-- npu="A3" id8 -->
    <li><term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> ：不支持seqUsedQOptional、seqUsedKOptional、cmpResidualKOptional，numHeadsQ仅支持8/16/32/64，topk仅支持512/1024/2048/4096/8192。</li>
    <!-- end id8 -->
    <!-- npu="910b" id9 -->
    <li><term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> ：不支持seqUsedQOptional、seqUsedKOptional、cmpResidualKOptional，numHeadsQ仅支持8/16/32/64，topk仅支持512/1024/2048/4096/8192。</li>
    <!-- end id9 -->
  </ul>

- **返回值：**

    返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

    第一段接口完成入参校验，出现以下场景时报错：

    <table style="undefined;table-layout: fixed; width: 1134px"><colgroup>
    <col style="width: 319px">
    <col style="width: 144px">
    <col style="width: 671px">
    </colgroup>
    <thead>
      <tr>
        <th>返回值</th>
        <th>错误码</th>
        <th>描述</th>
      </tr></thead>
    <tbody>
      <tr>
        <td>ACLNN_ERR_INNER_CREATE_EXECUTOR</td>
        <td>561101</td>
        <td>创建aclOpExecutor失败。</td>
      </tr>
      <tr>
        <td>ACLNN_ERR_INNER_NULLPTR</td>
        <td>561103</td>
        <td>参数workspaceSize、executor是空指针，或参数cuSeqLensQOptional、cuSeqLensKOptional、seqUsedQOptional、seqUsedKOptional、cmpResidualKOptional进行Contiguous处理后为空指针。</td>
      </tr>
      <tr>
        <td rowspan="2">ACLNN_ERR_PARAM_INVALID</td>
        <td rowspan="2">161002</td>
        <td>参数cuSeqLensQOptional、cuSeqLensKOptional、seqUsedQOptional、seqUsedKOptional、cmpResidualKOptional、batchSize、maxSeqlenQ、maxSeqlenK、numHeadsQ、numHeadsK、headDim、topk、layoutQOptional、layoutKOptional、maskMode、cmpRatio的规格不在支持范围内。</td>
      </tr>
    </tbody>
    </table>

## aclnnSparseLightningIndexerKLLossGradMetadata

- **参数说明：**

    <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
    <col style="width: 168px">
    <col style="width: 128px">
    <col style="width: 854px">
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
        <td>在Device侧申请的workspace大小，由第一段接口aclnnSparseLightningIndexerKLLossGradMetadataGetWorkspaceSize获取。</td>
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

    返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

  - aclnnSparseLightningIndexerKLLossGradMetadata为确定性实现，确定性计算配置不会改变其输出规则。
  - B（Batch）表示输入样本批量大小，q为配套的aclnnSparseLightningIndexerKLLossGrad算子的入参，S1表示layoutQOptional=BSND时，q shape中的S轴的大小。
  - 参数cuSeqlensQOptional、cuSeqlensKOptional要求其值为当前Batch与前序Batch有效token数的累加值，第一个元素固定为0，后一个元素的值必须大于等于前一个元素的值。
  - 参数sequsedQOptional、sequsedKOptional要求其值表示每个Batch中的有效token数。
  - 参数cmpResidualKOptional需满足cmpResidualKOptional[i] < cmpRatio。
  - layoutQOptional、layoutKOptional须相同。
  - numHeadsQ必须能被numHeadsK整除。
  <!-- npu="950" id10 -->
  - Ascend 950PR/Ascend 950DT约束：
    - layoutQOptional=BSND场景
      - maxSeqlenQ必须传入S1的值。
    - layoutQOptional=TND场景
      - cuSeqlensQOptional必须传入。
  <!-- end id10 -->
  <!-- npu="A3" id11 -->
  - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>约束：
    - BSND场景
      - 必传batchSize、maxSeqlenQ、maxSeqlenK和topk参数，以获取shape信息。
    - TND场景
      - 必传cuSeqLensQOptional、cuSeqLensKOptional和topk参数，以获取正确shape信息。
      - 当batchSize为0时，通过cuSeqLensQOptional的shape推导batch。
  <!-- end id11 -->
  <!-- npu="910b" id12 -->
  - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>约束：
    - BSND场景
      - 必传batchSize、maxSeqlenQ、maxSeqlenK和topk参数，以获取shape信息。
    - TND场景
      - 必传cuSeqLensQOptional、cuSeqLensKOptional和topk参数，以获取正确shape信息。
      - 当batchSize为0时，通过cuSeqLensQOptional的shape推导batch。
  <!-- end id12 -->

<details>
<summary><a id="Mask"></a>Mask</summary>
    &nbsp;&nbsp;<table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
        <col style="width: 165px">
        <col style="width: 625px">
        <col style="width: 360px">
        </colgroup>
        <thead>
            <tr>
                <th>mask_mode</th>
                <th>含义</th>
                <th>备注</th>
            </tr>
        </thead>
        <tbody>
        <tr>
            <td>0</td>
            <td>无mask。</td>
            <td>支持</td>
        </tr>
        <tr>
            <td>3</td>
            <td>rightDownCausal模式的mask，对应以右顶点为划分的下三角场景。</td>
            <td>支持</td>
        </tr>
        </tbody>
    </table>
</details>

<details>
<summary><a id="特殊约束"></a>特殊约束</summary>

  <!-- npu="950" id13 -->
  - Ascend 950PR/Ascend 950DT约束：
      - Batch取值规则
        - layoutQOptional为BSND时，优先通过sequsedQOptional的shape推导batch，sequsedQOptional未传入则通过batch_size获取batch数。
        - layoutQOptional为TND时，优先通过sequsedQOptional的shape推导batch，sequsedQOptional未传入则通过cuSeqlensQOptional的shape推导batch。
      - q Seqlen取值规则
        - layoutQOptional为BSND时，优先通过sequsedQOptional中的元素获取seqlen，sequsedQOptional未传入则通过maxSeqlenQ获取seqlen。
        - layoutQOptional为TND时，优先通过sequsedQOptional中的元素获取seqlen，sequsedQOptional未传入则通过cuSeqlensQOptional中的元素获取seqlen。
  <!-- end id13 -->
  <!-- npu="A3" id14 -->
  - Atlas A3 训练系列产品/Atlas A3 推理系列产品约束：
    - Batch取值规则
      - 如果batchSize大于0，优先使用batchSize。
      - 如果batchSize小于等于0，且layoutQOptional为TND，则通过cuSeqLensQOptional的shape推导batch。
      - 如果batchSize小于等于0，且layoutQOptional为BSND，则报错。
    - Seqlen取值规则
      - TND场景下，通过cuSeqLensQOptional和cuSeqLensKOptional计算每个batch的实际q/k长度。
      - BSND场景下，通过maxSeqlenQ和maxSeqlenK获取q/k长度。
    - layout约束
      - layoutQOptional必须为BSND或TND。
      - layoutKOptional支持BSND和TND，建议与layoutQOptional保持一致。
    - head约束
      - numHeadsQ、numHeadsK和headDim必须为正数。
      - numHeadsQ必须能被numHeadsK整除。
    - sparse约束
      - topk必须为正数。
      - cmpRatio取值范围为[0, 128]。
      - maskMode当前仅支持0和3。
  <!-- end id14 -->
  <!-- npu="910b" id15 -->
  - Atlas A2 训练系列产品/Atlas A2 推理系列产品约束：
    - Batch取值规则
      - 如果batchSize大于0，优先使用batchSize。
      - 如果batchSize小于等于0，且layoutQOptional为TND，则通过cuSeqLensQOptional的shape推导batch。
      - 如果batchSize小于等于0，且layoutQOptional为BSND，则报错。
    - Seqlen取值规则
      - TND场景下，通过cuSeqLensQOptional和cuSeqLensKOptional计算每个batch的实际q/k长度。
      - BSND场景下，通过maxSeqlenQ和maxSeqlenK获取q/k长度。
    - layout约束
      - layoutQOptional必须为BSND或TND。
      - layoutKOptional支持BSND和TND，建议与layoutQOptional保持一致。
    - head约束
      - numHeadsQ、numHeadsK和headDim必须为正数。
      - numHeadsQ必须能被numHeadsK整除。
    - sparse约束
      - topk必须为正数。
      - cmpRatio取值范围为[0, 128]。
      - maskMode当前仅支持0和3。
  <!-- end id15 -->

</details>

<details>
<summary><a id="Metadata"></a>Metadata输出布局</summary>

  metadata输出为INT32 Tensor，当前shape固定为(64,)，字段布局如下。
  <ul>
  <!-- npu="950" id16 -->
  <li><term>Ascend 950PR/Ascend 950DT</term> ：
  <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
    <col style="width: 180px">
    <col style="width: 150px">
    <col style="width: 820px">
    </colgroup>
    <thead>
        <tr>
        <th>字段</th>
        <th>index</th>
        <th>说明</th>
        </tr>
    </thead>
    <tbody>
        <tr>
        <td>totalNum</td>
        <td>0</td>
        <td>B/S1合轴后的任务总行数，即所有batch中q的sequence length之和。</td>
        </tr>
        <tr>
        <td>formerCoreProcessNum</td>
        <td>1</td>
        <td>满载核（前部核）每个核处理的任务行数，等于ceil(totalNum / aicCoreNum)。</td>
        </tr>
        <tr>
        <td>remainCoreProcessNum</td>
        <td>2</td>
        <td>尾部核每个核处理的任务行数，等于formerCoreProcessNum - 1。</td>
        </tr>
        <tr>
        <td>remainCoreNum</td>
        <td>3</td>
        <td>尾部核数量，即处理较少任务的核数，等于formerCoreProcessNum × aicCoreNum - totalNum。</td>
        </tr>
        <tr>
        <td>usedCoreNum</td>
        <td>4</td>
        <td>实际使用的AIC核数，当totalNum < aicCoreNum时取totalNum，否则取aicCoreNum。</td>
        </tr>
    </tbody>
  </table>
  </li>
  <!-- end id16 -->
  <!-- npu="A3" id17 -->
  <li><term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> ：
  <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
    <col style="width: 180px">
    <col style="width: 150px">
    <col style="width: 820px">
    </colgroup>
    <thead>
        <tr>
        <th>字段</th>
        <th>index</th>
        <th>说明</th>
        </tr>
    </thead>
    <tbody>
        <tr>
        <td>coreNum</td>
        <td>0</td>
        <td>主kernel实际使用的cube core数量。</td>
        </tr>
        <tr>
        <td>totalSize</td>
        <td>1</td>
        <td>B/S1合轴后的任务总行数。</td>
        </tr>
        <tr>
        <td>splitFactorSize</td>
        <td>2</td>
        <td>均分场景下的初始分核粒度。</td>
        </tr>
        <tr>
        <td>reserved</td>
        <td>3-7</td>
        <td>预留字段，当前置0。</td>
        </tr>
        <tr>
        <td>bS1Index</td>
        <td>8-32</td>
        <td>每个cube core的起始B/S1合轴索引，非活跃core填充totalSize。</td>
        </tr>
    </tbody>
  </table>
  </li>
  <!-- end id17 -->
  <!-- npu="910b" id18 -->
  <li><term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> ：
  <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
    <col style="width: 180px">
    <col style="width: 150px">
    <col style="width: 820px">
    </colgroup>
    <thead>
        <tr>
        <th>字段</th>
        <th>index</th>
        <th>说明</th>
        </tr>
    </thead>
    <tbody>
        <tr>
        <td>coreNum</td>
        <td>0</td>
        <td>主kernel实际使用的cube core数量。</td>
        </tr>
        <tr>
        <td>totalSize</td>
        <td>1</td>
        <td>B/S1合轴后的任务总行数。</td>
        </tr>
        <tr>
        <td>splitFactorSize</td>
        <td>2</td>
        <td>均分场景下的初始分核粒度。</td>
        </tr>
        <tr>
        <td>reserved</td>
        <td>3-7</td>
        <td>预留字段，当前置0。</td>
        </tr>
        <tr>
        <td>bS1Index</td>
        <td>8-32</td>
        <td>每个cube core的起始B/S1合轴索引，非活跃core填充totalSize。</td>
        </tr>
    </tbody>
  </table>
  </li>
  <!-- end id18 -->
  </ul>

</details>

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参见[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。

``` cpp
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <iostream>
#include <string>
#include <utility>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_sparse_lightning_indexer_kl_loss_grad_metadata.h"

#define CHECK_LOG_RET(cond, ret_val, fmt, ...)      \
    do {                                            \
        if (!(cond)) {                              \
            printf(fmt "\n", ##__VA_ARGS__);        \
            return (ret_val);                       \
        }                                           \
    } while (0)

constexpr uint32_t SLI_METADATA_MAX_CORE_NUM = 25;
constexpr uint32_t SLI_METADATA_HEADER_SIZE = 8;
constexpr uint32_t SLI_METADATA_SIZE = 64;

constexpr uint32_t GRAD_METADATA_SIZE = 5;
constexpr uint32_t TOTAL_NUM = 0;
constexpr uint32_t FORMER_CORE_PROCESS_NUM = 1;
constexpr uint32_t REMAIN_CORE_PROCESS_NUM = 2;
constexpr uint32_t REMAIN_CORE_NUM = 3;
constexpr uint32_t USED_CORE_NUM = 4;

struct SliGradKLLossMetaData {
    int32_t coreNum;
    int32_t totalSize;
    int32_t splitFactorSize;
    int32_t reserved[SLI_METADATA_HEADER_SIZE - 3];
    int32_t bS1Index[SLI_METADATA_MAX_CORE_NUM];
};

struct ScopeGuard
{
    explicit ScopeGuard(std::function<void()> onExitScope) : m_exitFunc(std::move(onExitScope)),
        m_isDismissed(false) {}
    ScopeGuard(const ScopeGuard&) = delete;
    ScopeGuard& operator=(const ScopeGuard&) = delete;

    ~ScopeGuard()
    {
        if (!m_isDismissed) {
            m_exitFunc();
        }
    }

    void Dismiss()
    {
        m_isDismissed = true;
    }

    std::function<void()> m_exitFunc;
    bool m_isDismissed;
};

struct Tensor {
    void *hostAddr { nullptr };
    void *deviceAddr { nullptr };
    aclTensor *data { nullptr };
};

struct ArgScenario {
    bool hasCuSeq { true };
};

struct ArgContext {
    Tensor cuSeqLensQOptional {};
    Tensor cuSeqLensKOptional {};
    Tensor seqUsedQOptional {};
    Tensor seqUsedKOptional {};
    Tensor cmpResidualKOptional {};
    Tensor metadata {};
    int64_t batchSize { 0 };
    int64_t maxSeqlenQ { 0 };
    int64_t maxSeqlenK { 0 };
    int64_t numHeadsQ { 8 };
    int64_t numHeadsK { 1 };
    int64_t headDim { 128 };
    int64_t topk { 512 };
    char *layoutQOptional { nullptr };
    char *layoutKOptional { nullptr };
    int64_t maskMode { 0 };
    int64_t cmpRatio { 4 };
};

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

aclnnStatus Init(int32_t deviceId, aclrtStream* stream)
{
    auto ret = aclInit(nullptr);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "aclInit failed. ERROR: %d", ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "aclrtSetDevice failed. ERROR: %d", ret);
    ret = aclrtCreateStream(stream);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "aclrtCreateStream failed. ERROR: %d", ret);
    return ACL_SUCCESS;
}

void Finalize(int32_t deviceId, aclrtStream stream)
{
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
}

aclnnStatus CreateTensor(aclDataType dataType, const std::vector<int64_t> &shape, Tensor &tensor)
{
    auto size = GetShapeSize(shape) * aclDataTypeSize(dataType);
    auto ret = aclrtMallocHost(&(tensor.hostAddr), size);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "aclrtMallocHost failed. ERROR: %d", ret);
    memset(tensor.hostAddr, 0, size);

    ret = aclrtMalloc(&(tensor.deviceAddr), size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "aclrtMalloc failed. ERROR: %d", ret);
    tensor.data = aclCreateTensor(shape.data(), shape.size(), dataType, nullptr, 0, aclFormat::ACL_FORMAT_ND,
        shape.data(), shape.size(), tensor.deviceAddr);

    ret = aclrtMemcpy(tensor.deviceAddr, size, tensor.hostAddr, size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "aclrtMemcpy failed. ERROR: %d", ret);
    return ACL_SUCCESS;
}

void SetInt32TensorData(Tensor &tensor, const std::vector<int32_t> &hostData)
{
    auto size = hostData.size() * sizeof(int32_t);
    memcpy(tensor.hostAddr, hostData.data(), size);
    aclrtMemcpy(tensor.deviceAddr, size, tensor.hostAddr, size, ACL_MEMCPY_HOST_TO_DEVICE);
}

void DestroyTensor(Tensor &tensor)
{
    if (tensor.data != nullptr) {
        aclDestroyTensor(tensor.data);
        tensor.data = nullptr;
    }
    if (tensor.deviceAddr != nullptr) {
        aclrtFree(tensor.deviceAddr);
        tensor.deviceAddr = nullptr;
    }
    if (tensor.hostAddr != nullptr) {
        aclrtFreeHost(tensor.hostAddr);
        tensor.hostAddr = nullptr;
    }
}

void DestroyArgs(ArgContext &context)
{
    DestroyTensor(context.metadata);
    DestroyTensor(context.cuSeqLensQOptional);
    DestroyTensor(context.cuSeqLensKOptional);
    DestroyTensor(context.seqUsedQOptional);
    DestroyTensor(context.seqUsedKOptional);
    DestroyTensor(context.cmpResidualKOptional);

    if (context.layoutQOptional != nullptr) {
        free(context.layoutQOptional);
        context.layoutQOptional = nullptr;
    }
    if (context.layoutKOptional != nullptr) {
        free(context.layoutKOptional);
        context.layoutKOptional = nullptr;
    }
}

aclnnStatus CreateArgs(const ArgScenario &scenario, ArgContext &context)
{
    ScopeGuard argsGuard([&] { DestroyArgs(context); });
    aclnnStatus ret;

    int64_t batchSize = 1;
    context.maxSeqlenQ = 16;
    context.maxSeqlenK = 4;
    context.layoutQOptional = (char *)malloc(sizeof(char) * 16);
    context.layoutKOptional = (char *)malloc(sizeof(char) * 16);
    strcpy(context.layoutQOptional, scenario.hasCuSeq ? "TND" : "BSND");
    strcpy(context.layoutKOptional, scenario.hasCuSeq ? "TND" : "BSND");

    ret = CreateTensor(aclDataType::ACL_INT32, { SLI_METADATA_SIZE }, context.metadata);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "Create metadata failed. Error: %d", ret);

    if (scenario.hasCuSeq) {
        ret = CreateTensor(aclDataType::ACL_INT32, { batchSize + 1 }, context.cuSeqLensQOptional);
        CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "Create cuSeqLensQOptional failed. Error: %d", ret);
        ret = CreateTensor(aclDataType::ACL_INT32, { batchSize + 1 }, context.cuSeqLensKOptional);
        CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "Create cuSeqLensKOptional failed. Error: %d", ret);
        SetInt32TensorData(context.cuSeqLensQOptional, { 0, static_cast<int32_t>(context.maxSeqlenQ) });
        SetInt32TensorData(context.cuSeqLensKOptional, { 0, static_cast<int32_t>(context.maxSeqlenK) });
        context.batchSize = 0;
    } else {
        context.batchSize = batchSize;
    }

    argsGuard.Dismiss();
    return ACL_SUCCESS;
}

void PrintMetadata(const SliGradKLLossMetaData &metadata)
{
    const char *socName = aclrtGetSocName();
    std::string socVersion = (socName != nullptr) ? std::string(socName) : std::string();
    bool isA5 = socVersion.find("Ascend910") == std::string::npos;
    if (isA5) {
        const int32_t *data = reinterpret_cast<const int32_t *>(&metadata);
        printf("TOTAL_NUM               : %d\n", data[TOTAL_NUM]);
        printf("FORMER_CORE_PROCESS_NUM : %d\n", data[FORMER_CORE_PROCESS_NUM]);
        printf("REMAIN_CORE_PROCESS_NUM : %d\n", data[REMAIN_CORE_PROCESS_NUM]);
        printf("REMAIN_CORE_NUM         : %d\n", data[REMAIN_CORE_NUM]);
        printf("USED_CORE_NUM           : %d\n", data[USED_CORE_NUM]);
        return;
    }
    printf("coreNum          : %d\n", metadata.coreNum);
    printf("totalSize        : %d\n", metadata.totalSize);
    printf("splitFactorSize  : %d\n", metadata.splitFactorSize);
    for (uint32_t i = 0; i < std::min<uint32_t>(metadata.coreNum, SLI_METADATA_MAX_CORE_NUM); ++i) {
        printf("bS1Index[%u]      : %d\n", i, metadata.bS1Index[i]);
    }
}

int main() {
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "Init acl failed. ERROR: %d", ret);
    ScopeGuard sysGuard([&] { Finalize(deviceId, stream); });

    ArgScenario scenario {};
    scenario.hasCuSeq = true;
    ArgContext context {};
    ret = CreateArgs(scenario, context);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "Create input arguments failed. ERROR: %d", ret);
    ScopeGuard argsGuard([&] { DestroyArgs(context); });

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    void *workspaceAddr = nullptr;
    ret = aclnnSparseLightningIndexerKLLossGradMetadataGetWorkspaceSize(
        context.cuSeqLensQOptional.data, context.cuSeqLensKOptional.data, context.seqUsedQOptional.data,
        context.seqUsedKOptional.data, context.cmpResidualKOptional.data, context.batchSize, context.maxSeqlenQ,
        context.maxSeqlenK, context.numHeadsQ, context.numHeadsK, context.headDim, context.topk, context.layoutQOptional,
        context.layoutKOptional, context.maskMode, context.cmpRatio, context.metadata.data, &workspaceSize, &executor);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret,
        "aclnnSparseLightningIndexerKLLossGradMetadataGetWorkspaceSize failed. ERROR: %d", ret);

    if (workspaceSize > static_cast<uint64_t>(0)) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "allocate workspace failed. ERROR: %d", ret);
    }
    ScopeGuard workspaceGuard([&] {
        if (workspaceAddr != nullptr) {
            aclrtFree(workspaceAddr);
            workspaceAddr = nullptr;
        }
    });

    ret = aclnnSparseLightningIndexerKLLossGradMetadata(workspaceAddr, workspaceSize, executor, stream);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret,
        "aclnnSparseLightningIndexerKLLossGradMetadata failed. ERROR: %d", ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "aclrtSynchronizeStream failed. ERROR: %d", ret);

    SliGradKLLossMetaData result {};
    ret = aclrtMemcpy(&result, sizeof(result), context.metadata.deviceAddr, sizeof(result), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "aclrtMemcpy failed. ERROR: %d", ret);
    PrintMetadata(result);

    return 0;
}
```
