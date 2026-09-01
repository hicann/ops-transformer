# aclnnSparseFlashMlaGradMetadata

[📄 查看源码](https://gitcode.com/cann/ops-transformer/tree/master/attention/sparse_flash_mla_grad_metadata)

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR/Ascend 950DT</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：x
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：x
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

- 接口功能：该接口为AI CPU算子接口，是aclnnSparseFlashMlaGrad算子的前置算子接口。根据aclnnSparseFlashMlaGrad算子接口的输入信息，计算并输出负载均衡结果。输出结果可以作为aclnnSparseFlashMlaGrad算子接口的输入，减少aclnnSparseFlashMlaGrad算子接口的执行耗时。

  **该算子不建议单独使用，建议与aclnnSparseFlashMlaGrad算子配合使用，形成完整的工作流。**
    1. 接受aclnnSparseFlashMlaGrad算子接口输入数据shape信息，包含batchSize、qSeqlen、kSeqlen、mask。通过对输入分块并模拟计算耗时，均匀分配分块到可用核上，以降低aclnnSparseFlashMlaGrad算子的整体计算耗时，并提高硬件利用率。
    2. 分配结果输出后，后续作为输入供aclnnSparseFlashMlaGrad算子使用。
    3. 分配结果包含每个AIC核基本块的起始点和终止点，以及每个AIV核的FD任务信息。详细内容可以参考[调用示例](#调用示例)。

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/two_phase_api.md)，必须先调用"aclnnSparseFlashMlaGradMetadataGetWorkspaceSize"获取workspace大小，再调用"aclnnSparseFlashMlaGradMetadata"执行计算。

``` cpp
aclnnStatus aclnnSparseFlashMlaGradMetadataGetWorkspaceSize(
    const aclTensor   *cuSeqlensQOptional,
    const aclTensor   *cuSeqlensOriKvOptional,
    const aclTensor   *cuSeqlensCmpKvOptional,
    const aclTensor   *sequsedQOptional,
    const aclTensor   *sequsedOriKvOptional,
    const aclTensor   *sequsedCmpKvOptional,
    const aclTensor   *cmpResidualKvOptional,
    const aclTensor   *oriTopkLengthOptional,
    const aclTensor   *cmpTopkLengthOptional,
    int64_t            numHeadsQ,
    int64_t            numHeadsKv,
    int64_t            headDim,
    int64_t            batchSize,
    int64_t            maxSeqlenQ,
    int64_t            maxSeqlenOriKv,
    int64_t            maxSeqlenCmpKv,
    int64_t            oriTopk,
    int64_t            cmpTopk,
    int64_t            cmpRatio,
    int64_t            oriMaskMode,
    int64_t            cmpMaskMode,
    int64_t            oriWinLeft,
    int64_t            oriWinRight,
    const char        *layoutQOptional,
    const char        *layoutKvOptional,
    bool               hasOriKv,
    bool               hasCmpKv,
    const aclTensor   *metadata,
    uint64_t          *workspaceSize,
    aclOpExecutor    **executor)
```

``` cpp
aclnnStatus aclnnSparseFlashMlaGradMetadata(
    void              *workspace,
    uint64_t           workspaceSize,
    aclOpExecutor     *executor,
    aclrtStream        stream)
```

## aclnnSparseFlashMlaGradMetadataGetWorkspaceSize

- **参数说明**

  <table style="undefined;table-layout: fixed; width: 1450px"><colgroup>
  <col style="width: 240px">
  <col style="width: 120px">
  <col style="width: 333px">
  <col style="width: 160px">
  <col style="width: 100px">
  <col style="width: 100px">
  <col style="width: 170px">
  <col style="width: 140px">
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
      <td>cuSeqlensQOptional（aclTensor*）</td>
      <td>输入</td>
      <td>表示不同Batch中q的有效Sequence Length。</td>
      <td><ul><li>支持空Tensor。</li><li>shape固定为(B+1, )。</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>1维</td>
      <td>√</td>
    </tr>
    <tr>
      <td>cuSeqlensOriKvOptional（aclTensor*）</td>
      <td>输入</td>
      <td>表示不同Batch中oriKvOptional的有效Sequence Length。</td>
      <td><ul><li>支持空Tensor。</li><li>shape固定为(B+1, )。</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>1维</td>
      <td>√</td>
    </tr>
    <tr>
      <td>cuSeqlensCmpKvOptional（aclTensor*）</td>
      <td>输入</td>
      <td>表示不同Batch中cmpKvOptional的有效Sequence Length。</td>
      <td><ul><li>支持空Tensor。</li><li>shape固定为(B+1, )。</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>1维</td>
      <td>√</td>
    </tr>
    <tr>
      <td>sequsedQOptional（aclTensor*）</td>
      <td>输入</td>
      <td>表示不同Batch中q实际参与运算的Sequence Length。</td>
      <td><ul><li>支持空Tensor。</li><li>shape固定为(B, )。</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>1维</td>
      <td>√</td>
    </tr>
    <tr>
      <td>sequsedOriKvOptional（aclTensor*）</td>
      <td>输入</td>
      <td>表示不同Batch中oriKvOptional实际参与运算的Sequence Length。</td>
      <td><ul><li>支持空Tensor。</li><li>shape固定为(B, )。</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>1维</td>
      <td>√</td>
    </tr>
    <tr>
      <td>sequsedCmpKvOptional（aclTensor*）</td>
      <td>输入</td>
      <td>表示不同Batch中cmpKvOptional实际参与运算的Sequence Length。</td>
      <td><ul><li>支持空Tensor。</li><li>shape固定为(B, )。</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>1维</td>
      <td>√</td>
    </tr>
    <tr>
      <td>cmpResidualKvOptional（aclTensor*）</td>
      <td>输入</td>
      <td>表示不同Batch中cmpKvOptional压缩后Sequence Length的余数，配合cmpRatio实现cmpKvOptional部分的mask和负载计算。</td>
      <td><ul><li>支持空Tensor。</li><li>cmpRatio不为1，且mask为3场景下必传。</li><li>shape固定为(B, )。</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>1维</td>
      <td>√</td>
    </tr>
    <tr>
      <td>oriTopkLengthOptional（aclTensor*）</td>
      <td>输入</td>
      <td>表示不同q token对应的oriKvOptional部分关键稀疏token的个数。</td>
      <td><ul><li>支持空Tensor。</li><li>shape为(B, S1, N2)或(T1, N2)。</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>2维、3维</td>
      <td>√</td>
    </tr>
    <tr>
      <td>cmpTopkLengthOptional（aclTensor*）</td>
      <td>输入</td>
      <td>表示不同q token对应的cmpKvOptional部分关键稀疏token的个数。</td>
      <td><ul><li>支持空Tensor。</li><li>shape为(B, S1, N2)或(T1, N2)。</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>2维、3维</td>
      <td>√</td>
    </tr>
    <tr>
      <td>numHeadsQ（int64_t）</td>
      <td>输入</td>
      <td>表示q的head个数。</td>
      <td>当前支持[1, 128]。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>numHeadsKv（int64_t）</td>
      <td>输入</td>
      <td>oriKvOptional、cmpKvOptional对应的多头数。</td>
      <td>当前仅支持1。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>headDim（int64_t）</td>
      <td>输入</td>
      <td>注意力头的维度。</td>
      <td>当前仅支持512。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>batchSize（int64_t）</td>
      <td>输入</td>
      <td>表示Batch数量。</td>
      <td><ul><li>支持非负数。</li><li>建议值为0。</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>maxSeqlenQ（int64_t）</td>
      <td>输入</td>
      <td>表示q的最长Sequence Length。</td>
      <td><ul><li>支持非负数。</li><li>建议值为0。</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>maxSeqlenOriKv（int64_t）</td>
      <td>输入</td>
      <td>表示oriKvOptional的最长Sequence Length。</td>
      <td><ul><li>支持非负数。</li><li>建议值为0。</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>maxSeqlenCmpKv（int64_t）</td>
      <td>输入</td>
      <td>表示cmpKvOptional的最长Sequence Length。</td>
      <td><ul><li>支持非负数。</li><li>建议值为0。</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>oriTopk（int64_t）</td>
      <td>输入</td>
      <td>表示oriKvOptional中筛选出的关键稀疏token的个数。0表示非稀疏场景。</td>
      <td><ul><li>支持非负数。</li><li>建议值为0。</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cmpTopk（int64_t）</td>
      <td>输入</td>
      <td>表示cmpKvOptional中筛选出的关键稀疏token的个数。0表示非稀疏场景。</td>
      <td><ul><li>支持非负数。</li><li>建议值为0。</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cmpRatio（int64_t）</td>
      <td>输入</td>
      <td>表示对cmpKvOptional的压缩率。</td>
      <td><ul><li>当前支持[1, 128]。</li><li>建议值1，表示无压缩。</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>oriMaskMode（int64_t）</td>
      <td>输入</td>
      <td>表示q和oriKvOptional计算的mask模式。</td>
      <td><ul><li>0表示No mask。</li><li>3表示rightDownCausal模式。</li><li>4表示sliding window模式，对应由oriWinLeft和oriWinRight划分的窗口场景。</li><li>建议值为0。</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cmpMaskMode（int64_t）</td>
      <td>输入</td>
      <td>表示q和cmpKvOptional计算的mask模式。</td>
      <td><ul><li>0表示No mask。</li><li>3表示rightDownCausal模式，对应以右顶点为划分的下三角场景。</li><li>建议值为0。</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>oriWinLeft（int64_t）</td>
      <td>输入</td>
      <td>表示q和oriKvOptional计算中q对过去token计算的数量。</td>
      <td><ul><li>取值范围≥-1，-1表示无穷大。</li><li>建议值为-1。</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>oriWinRight（int64_t）</td>
      <td>输入</td>
      <td>表示q和oriKvOptional计算中q对未来token计算的数量。</td>
      <td><ul><li>取值范围≥-1，-1表示无穷大。</li><li>建议值为-1。</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layoutQOptional（char*）</td>
      <td>输入</td>
      <td>表示q的排列格式。</td>
      <td><ul><li>支持 BSND、TND。</li><li>建议值为BSND。</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layoutKvOptional（char*）</td>
      <td>输入</td>
      <td>表示oriKvOptional、cmpKvOptional的排列格式。</td>
      <td><ul><li>支持 BSND、TND。</li><li>建议值为BSND。</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>hasOriKv（bool）</td>
      <td>输入</td>
      <td>用于标识是否含有oriKvOptional。</td>
      <td><ul><li>true: 含有oriKvOptional。</li><li>false: 不含有oriKvOptional。</li><li>建议值为true。</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>hasCmpKv（bool）</td>
      <td>输入</td>
      <td>用于标识是否含有cmpKvOptional。</td>
      <td><ul><li>true: 含有cmpKvOptional。</li><li>false: 不含有cmpKvOptional。</li><li>建议值为true。</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>metadata（aclTensor*）</td>
      <td>输出</td>
      <td>表示负载均衡结果输出。</td>
      <td>shape固定为(1024, )</td>
      <td>INT32</td>
      <td>ND</td>
      <td>1维</td>
      <td>×</td>
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
        <td>参数workspaceSize、executor是空指针，或参数cuSeqlensQOptional、cuSeqlensOriKvOptional、cuSeqlensCmpKvOptional、sequsedQOptional、sequsedOriKvOptional、sequsedCmpKvOptional、cmpResidualKvOptional、oriTopkLengthOptional、cmpTopkLengthOptional进行Contiguous处理后为空指针。</td>
      </tr>
      <tr>
        <td rowspan="2">ACLNN_ERR_PARAM_INVALID</td>
        <td rowspan="2">161002</td>
        <td>参数cuSeqlensQOptional、cuSeqlensOriKvOptional、cuSeqlensCmpKvOptional、sequsedQOptional、sequsedOriKvOptional、sequsedCmpKvOptional、cmpResidualKvOptional、oriTopkLengthOptional、cmpTopkLengthOptional、numHeadsQ、numHeadsKv、headDim、batchSize、maxSeqlenQ、maxSeqlenOriKv、maxSeqlenCmpKv、oriTopk、cmpTopk、cmpRatio、oriMaskMode、cmpMaskMode、oriWinLeft、oriWinRight、layoutQOptional、layoutKvOptional、hasOriKv、hasCmpKv的规格不在支持范围内。</td>
      </tr>
    </tbody>
    </table>

## aclnnSparseFlashMlaGradMetadata

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
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>workspace</td>
            <td>输入</td>
            <td>在Device侧申请的workspace内存地址。</td>
        </tr>
        <tr>
            <td>workspaceSize</td>
            <td>输入</td>
            <td>在Device侧申请的workspace大小，由第一段接口aclnnSparseFlashMlaGradMetadataGetWorkspaceSize获取。</td>
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

  - aclnnSparseFlashMlaGradMetadata默认确定性实现。
  - B（Batch）表示输入样本批量大小，q、oriKvOptional、cmpKvOptional为配套的aclnnSparseFlashMlaGrad算子的入参，S1表示layoutQOptional=BSND时，q shape中的S轴的大小，T1表示layoutQOptional=TND时，q shape中的T轴的大小，S2表示layoutKvOptional=BSND时，oriKvOptional shape中的S轴的大小，S3表示layoutKvOptional=BSND时，cmpKvOptional shape中的S轴的大小，N2表示oriKvOptional、cmpKvOptional shape中的N轴的大小。
  - 参数cuSeqlensQOptional、cuSeqlensOriKvOptional、cuSeqlensCmpKvOptional要求其值为当前Batch与前序Batch有效token数的累加值，第一个元素固定为0，后一个元素的值必须大于等于前一个元素的值。
  - 参数sequsedQOptional、sequsedOriKvOptional、sequsedCmpKvOptional要求其值表示每个Batch中的有效token数。
  - layoutQOptional、layoutKvOptional须相同。
  - 参数cmpResidualKvOptional需满足cmpResidualKvOptional[i] < cmpRatio。
  - numHeadsQ必须能被numHeadsKv整除。
  - hasOriKv为true时，oriTopk大于0认为oriKvOptional部分是稀疏的，oriTopk为0则认为oriKvOptional部分是非稀疏的。
  - hasCmpKv为true时，cmpTopk大于0认为cmpKvOptional部分是稀疏的，cmpTopk为0则认为cmpKvOptional部分是非稀疏的。
  - hasOriKv为true，oriTopk不为0且oriMaskMode为0时，oriTopkLengthOptional必须传入，此时取oriMaskMode规则与oriTopkLengthOptional元素的最小值作为当前q token对应的oriKvOptional的有效seqlen，其他oriKvOptional稀疏场景取oriMaskMode规则与oriTopk的最小值作为当前q token对应的oriKvOptional的有效seqlen。
  - hasCmpKv为true，cmpTopk不为0且cmpMaskMode为0时，cmpTopkLengthOptional必须传入，此时取cmpMaskMode规则与cmpTopkLengthOptional元素的最小值作为当前q token对应的cmpKvOptional的有效seqlen，其他cmpKvOptional稀疏场景取cmpMaskMode规则与cmpTopk的最小值作为当前q token对应的cmpKvOptional的有效seqlen。
  - layoutQOptional=BSND场景
    - maxSeqlenQ必须传入S1的值。
  - layoutKvOptional=BSND场景
    - hasOriKv为true时，maxSeqlenOriKv必须传入S2的值。
    - hasCmpKv为true时，maxSeqlenCmpKv必须传入S3的值。
  - layoutQOptional=TND场景
    - cuSeqlensQOptional必须传入。
  - layoutKvOptional=TND场景
    - hasOriKv为true时，cuSeqlensOriKvOptional必须传入。
    - hasCmpKv为true时，cuSeqlensCmpKvOptional必须传入。
  - Batch取值规则
    - layoutQOptional为BSND时，优先通过sequsedQOptional的shape推导batch，sequsedQOptional未传入则通过batchSize获取batch数。
    - layoutQOptional为TND时，优先通过sequsedQOptional的shape推导batch，sequsedQOptional未传入则通过cuSeqlensQOptional的shape推导batch。
  - q Seqlen取值规则
    - layoutQOptional为BSND时，优先通过sequsedQOptional中的元素获取seqlen，sequsedQOptional未传入则通过maxSeqlenQ获取seqlen。
    - layoutQOptional为TND时，优先通过sequsedQOptional中的元素获取seqlen，sequsedQOptional未传入则通过cuSeqlensQOptional中的元素获取seqlen。
  - oriKvOptional Seqlen取值规则
    - layoutKvOptional为BSND时，优先通过sequsedOriKvOptional中的元素获取seqlen，sequsedOriKvOptional未传入则通过maxSeqlenOriKv获取seqlen。
    - layoutKvOptional为TND时，优先通过sequsedOriKvOptional中的元素获取seqlen，sequsedOriKvOptional未传入则通过cuSeqlensOriKvOptional中的元素获取seqlen。
  - cmpKvOptional Seqlen取值规则
    - layoutKvOptional为BSND时，优先通过sequsedCmpKvOptional中的元素获取seqlen，sequsedCmpKvOptional未传入则通过maxSeqlenCmpKv获取seqlen。
    - layoutKvOptional为TND时，优先通过sequsedCmpKvOptional中的元素获取seqlen，sequsedCmpKvOptional未传入则通过cuSeqlensCmpKvOptional中的元素获取seqlen。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。

``` cpp
#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include <limits>
#include <functional>
#include <utility>
#include "acl/acl.h"
#include "aclnnop/aclnn_sparse_flash_mla_grad_metadata.h"

#define CHECK_LOG_RET(cond, ret_val, fmt, ...)      \
    do {                                            \
        if (!(cond)) {                              \
            printf(fmt "\n", ##__VA_ARGS__);        \
            return (ret_val);                       \
        }                                           \
    } while (0)

// Constants
constexpr uint32_t AIC_CORE_MAX_NUM = 36;
constexpr uint32_t AIV_CORE_MAX_NUM = 72;
constexpr uint32_t SMLAG_METADATA_TOTAL_SIZE = 1024;
using SMLAG_METADATA_T = int32_t;
constexpr uint32_t GRAD_METADATA_SIZE = 7;

// Grad Metadata Index Definitions
constexpr uint32_t TOTAL_NUM = 0;
constexpr uint32_t FORMER_CORE_PROCESS_NUM = 1;
constexpr uint32_t REMAIN_CORE_PROCESS_NUM = 2;
constexpr uint32_t REMAIN_CORE_NUM = 3;
constexpr uint32_t USED_CORE_NUM = 4;
constexpr uint32_t MAX_ORI_KV_SIZE = 5;
constexpr uint32_t MAX_CMP_KV_SIZE = 6;

struct SmlagMetadata {
    uint32_t gradMetadata[GRAD_METADATA_SIZE];
};

struct ScopeGuard
{
    explicit ScopeGuard(std::function<void()> onExitScope) : m_exitFunc(std::move(onExitScope)),
        m_isDismissed(false) {}
    // 禁止拷贝
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
    bool hasCuSeq { false };
    bool hasSeqused { false };
};

struct ArgContext {
    // required input
    int64_t numHeadsQ { 0 };
    int64_t numHeadsKv { 0 };
    int64_t headDim { 0 };
    // optional input
    Tensor cuSeqlensQOptional {};
    Tensor cuSeqlensOriKvOptional {};
    Tensor cuSeqlensCmpKvOptional {};
    Tensor sequsedQOptional {};
    Tensor sequsedOriKvOptional {};
    Tensor sequsedCmpKvOptional {};
    Tensor cmpResidualKvOptional {};
    Tensor oriTopkLengthOptional {};
    Tensor cmpTopkLengthOptional {};
    int64_t batchSize { 0 };
    int64_t maxSeqlenQ { 0 };
    int64_t maxSeqlenOriKv { 0 };
    int64_t maxSeqlenCmpKv { 0 };
    int64_t oriTopk { 0 };
    int64_t cmpTopk { 0 };
    int64_t cmpRatio { 0 };
    int64_t oriMaskMode { 0 };
    int64_t cmpMaskMode { 0 };
    int64_t oriWinLeft { -1 };
    int64_t oriWinRight { -1 };
    char *layoutQOptional { nullptr };
    char *layoutKvOptional { nullptr };
    bool hasOriKv { true };
    bool hasCmpKv { true };
    // output
    Tensor metadata {};
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
    // 固定写法，初始化
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
    // 调用aclrtMallocHost申请host侧内存
    auto ret = aclrtMallocHost(&(tensor.hostAddr), size);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "aclrtMallocHost failed. ERROR: %d", ret);
    memset(tensor.hostAddr, 0, size);
    // 调用aclrtMalloc申请device侧内存
    ret = aclrtMalloc(&(tensor.deviceAddr), size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "aclrtMalloc failed. ERROR: %d", ret);
    // 调用aclCreateTensor接口创建aclTensor
    tensor.data = aclCreateTensor(shape.data(), shape.size(), dataType, nullptr, 0, aclFormat::ACL_FORMAT_ND,
        shape.data(), shape.size(), tensor.deviceAddr);
    CHECK_LOG_RET(tensor.data != nullptr, ACL_ERROR_FAILURE, "aclCreateTensor failed");
    // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
    ret = aclrtMemcpy(tensor.deviceAddr, size, tensor.hostAddr, size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "aclrtMemcpy failed. ERROR: %d", ret);
    return ACL_SUCCESS;
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
    DestroyTensor(context.cuSeqlensQOptional);
    DestroyTensor(context.cuSeqlensOriKvOptional);
    DestroyTensor(context.cuSeqlensCmpKvOptional);
    DestroyTensor(context.sequsedQOptional);
    DestroyTensor(context.sequsedOriKvOptional);
    DestroyTensor(context.sequsedCmpKvOptional);
    DestroyTensor(context.cmpResidualKvOptional);
    DestroyTensor(context.oriTopkLengthOptional);
    DestroyTensor(context.cmpTopkLengthOptional);

    if (context.layoutQOptional != nullptr) {
        free(context.layoutQOptional);
        context.layoutQOptional = nullptr;
    }
    if (context.layoutKvOptional != nullptr) {
        free(context.layoutKvOptional);
        context.layoutKvOptional = nullptr;
    }
}

aclnnStatus CreateArgs(const ArgScenario &scenario, ArgContext &context)
{
    ScopeGuard argsGuard([&] { DestroyArgs(context); });
    aclnnStatus ret;

    context.numHeadsQ = 64;
    context.numHeadsKv = 1;
    context.headDim = 512;
    ret = CreateTensor(aclDataType::ACL_INT32, { SMLAG_METADATA_TOTAL_SIZE }, context.metadata);     // 1024: Fix size
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "Create metadata failed. Error: %d", ret);
    context.oriTopk = 0;
    context.cmpTopk = 0;
    context.cmpRatio = 128;
    context.oriMaskMode = 4;
    context.cmpMaskMode = 3;
    context.oriWinLeft = 127;
    context.oriWinRight = 0;
    context.layoutQOptional = (char *)malloc(sizeof(char) * 16);
    context.layoutKvOptional = (char *)malloc(sizeof(char) * 16);
    CHECK_LOG_RET(context.layoutQOptional != nullptr, ACL_ERROR_FAILURE, "Create layoutQOptional failed");
    CHECK_LOG_RET(context.layoutKvOptional != nullptr, ACL_ERROR_FAILURE, "Create layoutKvOptional failed");
    strcpy(context.layoutQOptional, "BSND");                // BSND,TND
    strcpy(context.layoutKvOptional, "BSND");               // BSND,TND
    context.hasOriKv = true;
    context.hasCmpKv = true;

    context.batchSize = 4;
    context.maxSeqlenOriKv = 1024;
    context.maxSeqlenCmpKv = 1024;
    context.maxSeqlenQ = 1024;

    if (scenario.hasCuSeq) {
        // (B+1,), first element is always 0
        ret = CreateTensor(aclDataType::ACL_INT32, { context.batchSize + 1 }, context.cuSeqlensQOptional);
        CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "Create cuSeqlensQOptional failed. Error: %d", ret);
        ret = CreateTensor(aclDataType::ACL_INT32, { context.batchSize + 1 }, context.cuSeqlensOriKvOptional);
        CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "Create cuSeqlensOriKvOptional failed. Error: %d", ret);
        ret = CreateTensor(aclDataType::ACL_INT32, { context.batchSize + 1 }, context.cuSeqlensCmpKvOptional);
        CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "Create cuSeqlensCmpKvOptional failed. Error: %d", ret);
    }

    if (scenario.hasSeqused) {
        // (B,)
        ret = CreateTensor(aclDataType::ACL_INT32, { context.batchSize }, context.sequsedQOptional);
        CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "Create sequsedQOptional failed. Error: %d", ret);
        ret = CreateTensor(aclDataType::ACL_INT32, { context.batchSize }, context.sequsedOriKvOptional);
        CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "Create sequsedOriKvOptional failed. Error: %d", ret);
        ret = CreateTensor(aclDataType::ACL_INT32, { context.batchSize }, context.sequsedCmpKvOptional);
        CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "Create sequsedCmpKvOptional failed. Error: %d", ret);
    }

    if (context.hasCmpKv && context.cmpRatio != 1 && context.cmpMaskMode == 3) {
        // (B,)
        ret = CreateTensor(aclDataType::ACL_INT32, { context.batchSize }, context.cmpResidualKvOptional);
        CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "Create cmpResidualKvOptional failed. Error: %d", ret);
    }

    argsGuard.Dismiss();
    return ACL_SUCCESS;
}

int main() {
    // 1. （固定写法）device/stream初始化，参考对外接口列表
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "Init acl failed. ERROR: %d", ret);
    ScopeGuard sysGuard([&] { Finalize(deviceId, stream); });

    // 2. 构造输入与输出，需要根据API的接口定义构造
    ArgScenario scenario {};
    scenario.hasCuSeq = false;
    scenario.hasSeqused = false;
    ArgContext context {};
    ret = CreateArgs(scenario, context);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "Create input arguments failed. ERROR: %d", ret);
    ScopeGuard argsGuard([&] { DestroyArgs(context); });

    // 3. 调用CANN算子库API，需要修改为具体的API
    // 调用aclnnSparseFlashMlaGradMetadata第一段接口
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    void *workspaceAddr = nullptr;
    ret = aclnnSparseFlashMlaGradMetadataGetWorkspaceSize(
        context.cuSeqlensQOptional.data, context.cuSeqlensOriKvOptional.data, context.cuSeqlensCmpKvOptional.data,
        context.sequsedQOptional.data, context.sequsedOriKvOptional.data, context.sequsedCmpKvOptional.data,
        context.cmpResidualKvOptional.data, context.oriTopkLengthOptional.data, context.cmpTopkLengthOptional.data,
        context.numHeadsQ, context.numHeadsKv, context.headDim, context.batchSize, context.maxSeqlenQ,
        context.maxSeqlenOriKv, context.maxSeqlenCmpKv, context.oriTopk, context.cmpTopk, context.cmpRatio,
        context.oriMaskMode, context.cmpMaskMode, context.oriWinLeft, context.oriWinRight, context.layoutQOptional,
        context.layoutKvOptional, context.hasOriKv, context.hasCmpKv, context.metadata.data, &workspaceSize, &executor);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret,
        "aclnnSparseFlashMlaGradMetadataGetWorkspaceSize failed. ERROR: %d\n", ret);

    if (workspaceSize > static_cast<uint64_t>(0)) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "allocate workspace failed. ERROR: %d\n", ret);
    }
    ScopeGuard workspaceGuard([&] {
        if (workspaceAddr != nullptr) {
            aclrtFree(workspaceAddr);
            workspaceAddr = nullptr;
        }
    });

    // 调用aclnnSparseFlashMlaGradMetadata第二段接口
    ret = aclnnSparseFlashMlaGradMetadata(workspaceAddr, workspaceSize, executor, stream);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "aclnnSparseFlashMlaGradMetadata failed. ERROR: %d\n", ret);

    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "aclrtSynchronizeStream failed. ERROR: %d\n", ret);

    // 5. 打印输出
    SmlagMetadata result {};
    ret = aclrtMemcpy(&result, sizeof(result), context.metadata.deviceAddr, sizeof(result), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "aclrtMemcpy failed. ERROR: %d\n", ret);

    for (uint32_t i = 0; i < GRAD_METADATA_SIZE; ++i) {
        printf("AIC Core%u\n", i);
        printf("    Total Num                 : %u\n", result.gradMetadata[TOTAL_NUM]);
        printf("    Former Core Process Num   : %u\n", result.gradMetadata[FORMER_CORE_PROCESS_NUM]);
        printf("    Remain Core Process Num   : %u\n", result.gradMetadata[REMAIN_CORE_PROCESS_NUM]);
        printf("    Remain Core Num           : %u\n", result.gradMetadata[REMAIN_CORE_NUM]);
        printf("    Used Core Num             : %u\n", result.gradMetadata[USED_CORE_NUM]);
        printf("    Max Ori Kv Size           : %u\n", result.gradMetadata[MAX_ORI_KV_SIZE]);
        printf("    Max Cmp Kv Size           : %u\n", result.gradMetadata[MAX_CMP_KV_SIZE]);
    }

    return 0;
}
```
