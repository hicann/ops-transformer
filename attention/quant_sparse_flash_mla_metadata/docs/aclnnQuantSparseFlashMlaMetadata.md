# aclnnQuantSparseFlashMlaMetadata

[📄 查看源码](https://gitcode.com/cann/ops-transformer/tree/master/attention/quant_sparse_flash_mla_metadata)

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR/Ascend 950DT</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：不支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：不支持
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

- 接口功能：`aclnnQuantSparseFlashMlaMetadata`是`aclnnQuantSparseFlashMla`算子的前置算子，用于后续Attention计算生成负载均衡的任务划分方案。本算子不执行实际的Attention计算，而是根据输入参数在AI CPU计算出每个AI Core应处理的Attention计算起止范围，从而最大化计算资源的利用率，避免各Core间负载不均衡的问题。

  **该算子不建议单独使用，建议与aclnnQuantSparseFlashMla算子配合使用，形成完整的工作流。**
    1. 接受aclnnQuantSparseFlashMla算子接口输入数据shape信息，包含batchSize、qSeqen、kvSeqlen、mask。通过对输入分块并模拟计算耗时，均匀分配分块到可用核上，以降低aclnnQuantSparseFlashMla算子的整体计算耗时，并提高硬件利用率。
    2. 分配结果输出后，后续作为输入供aclnnQuantSparseFlashMla算子使用。
    3. 分配结果包含每个AIC核基本块的起始点和终止点，已经每个AIV核的FD任务信息。详细内容可以参考[调用示例](#调用示例)。

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/two_phase_api.md)，必须先调用"aclnnQuantSparseFlashMlaMetadataGetWorkspaceSize"获取workspace大小，在调用"aclnnQuantSparseFlashMlaMetadata"执行计算

``` cpp
aclnnStatus aclnnQuantSparseFlashMlaMetadataGetWorkspaceSize(
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
    int64_t            quantMode,
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
aclnnStatus aclnnQuantSparseFlashMlaMetadata(
    void              *workspace,
    uint64_t           workspaceSize,
    aclOpExecutor     *executor,
    aclrtStream        stream)
```

## aclnnQuantSparseFlashMlaMetadataGetWorkspaceSize

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
      <td>表示不同Batch中Query的有效Sequence Length。</td>
      <td><ul><li>支持空Tensor</li><li>layoutQOptional为TND场景下必传。</li><li>第一个值为额外值并固定为0。</li><li>shape固定为(B+1, )。</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>1维</td>
      <td>√</td>
    </tr>
    <tr>
      <td>cuSeqlensOriKvOptional（aclTensor*）</td>
      <td>输入</td>
      <td>表示不同Batch中oriKvOptional的有效Sequence Length。</td>
      <td><ul><li>支持空Tensor。</li><li>layoutKvOptional为TND场景下必传。</li><li>第一个值为额外值并固定为0。</li><li>shape固定为(B+1, )。</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>1维</td>
      <td>√</td>
    </tr>
    <tr>
      <td>cuSeqlensCmpKvOptional（aclTensor*）</td>
      <td>输入</td>
      <td>表示不同Batch中cmpKvOptional的有效Sequence Length。</td>
      <td><ul><li>支持空Tensor。</li><li>layoutKvOptional为TND场景下必传。</li><li>第一个值为额外值并固定为0。</li><li>shape固定为(B+1, )。</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>1维</td>
      <td>√</td>
    </tr>
    <tr>
      <td>sequsedQOptional（aclTensor*）</td>
      <td>输入</td>
      <td>表示不同Batch中Query实际参与运算的Sequence Length。</td>
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
      <td><ul><li>支持空Tensor。</li><li>cmpRatio不为1场景下必传。</li><li>shape固定为(B, )。</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>1维</td>
      <td>√</td>
    </tr>
    <tr>
      <td>oriTopkLengthOptional（aclTensor*）</td>
      <td>输入</td>
      <td>用于标识oriSparseIndicesOptional实际参与计算的长度。</td>
      <td>
        <ul>
          <li>支持空Tensor。</li>
          <li>oriTopk!=0且oriMaskMode=0的场景下必传。</li>
          <li>layoutQ为BSND时：(B, S1, N2)</li>
          <li>layoutQ为TND时：(T1, N2)</li>
        </ul>
      </td>
      <td>INT32</td>
      <td>ND</td>
      <td>2维、3维</td>
      <td>√</td>
    </tr>
    <tr>
      <td>cmpTopkLengthOptional（aclTensor*）</td>
      <td>输入</td>
      <td>用于标识cmpSparseIndicesOptional实际参与计算的长度。</td>
      <td>
        <ul>
          <li>支持空Tensor。</li>
          <li>cmpTopk!=0且cmpMaskMode=0的场景下必传。</li>
          <li>layoutQ为BSND时：(B, S1, N2)</li>
          <li>layoutQ为TND时：(T1, N2)</li>
        </ul>
      </td>
      <td>INT32</td>
      <td>ND</td>
      <td>2维、3维</td>
      <td>√</td>
    </tr>
    <tr>
      <td>numHeadsQ（int64_t）</td>
      <td>输入</td>
      <td>表示Query的head个数。</td>
      <td>范围支持1到128。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>numHeadsKv（int64_t）</td>
      <td>输入</td>
      <td>Key和Value对应的多头数。</td>
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
      <td>quantMode（int64_t）</td>
      <td>输入</td>
      <td>表示量化模式。</td>
      <td><ul><li>1: Q、K、V为per-tensor量化，Q、K、V数据类型为HIFLOAT8。</li></ul></td>
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
      <td>表示Query的最长Sequence Length。</td>
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
      <td><ul><li>当前仅支持0。</li><li>建议值为0。</li></ul></td>
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
      <td><ul><li>当前支持1到128。</li><li>建议值1，表示无压缩。</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>oriMaskMode（int64_t）</td>
      <td>输入</td>
      <td>表示q和oriKvOptional计算的mask模式。</td>
      <td><ul><li>当前支持：<br/>0: No Mask。<br/>3: RightDownCausal模式。<br/>4: Band模式。</li><li>建议值为0。</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cmpMaskMode（int64_t）</td>
      <td>输入</td>
      <td>表示q和cmpKvOptional计算的mask模式。</td>
      <td><ul><li>当前支持：<br/>0: No Mask。<br/>3: RightDownCausal模式。</li><li>建议值为0。</li></ul></td>
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
      <td>表示Query的排列格式。</td>
      <td><ul><li>支持 BSND、TND。</li><li>建议值为BSND。</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layoutKvOptional（char*）</td>
      <td>输入</td>
      <td>表示Key的排列格式。</td>
      <td><ul><li>支持 BSND、TND、PA_BBND。</li><li>建议值为BSND。</li></ul></td>
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
      <td>-</td>
      <td>INT32</td>
      <td>ND</td>
      <td>1维，shape固定为(1024)</td>
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
        <td>参数cuSeqlensQOptional、cuSeqlensOriKvOptional、cuSeqlensCmpKvOptional、sequsedQOptional、sequsedOriKvOptional、sequsedCmpKvOptional、cmpResidualKvOptional、oriTopkLengthOptional、cmpTopkLengthOptional、numHeadsQ、numHeadsKv、headDim、quantMode、batchSize、maxSeqlenQ、maxSeqlenOriKv、maxSeqlenCmpKv、oriTopk、cmpTopk、cmpRatio、oriMaskMode、cmpMaskMode、oriWinLeft、oriWinRight、layoutQOptional、layoutKvOptional、hasOriKv、hasCmpKv的规格不在支持范围内。</td>
      </tr>
    </tbody>
    </table>

## aclnnQuantSparseFlashMlaMetadata

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
            <td>在Device侧申请的workspace内存地址</td>
        </tr>
        <tr>
            <td>workspaceSize</td>
            <td>输入</td>
            <td>在Device侧申请的workspace大小，由第一段接口aclnnQuantSparseFlashMlaMetadataGetWorkspaceSize获取</td>
        </tr>
        <tr>
            <td>executor</td>
            <td>输入</td>
            <td>op执行器，包含了算子计算流程</td>
        </tr>
        <tr>
            <td>stream</td>
            <td>输入</td>
            <td>指定执行任务的Stream</td>
        </tr>
    </tbody>
    </table>

- **返回值：**

    返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

  - aclnnQuantSparseFlashMlaMetadata默认确定性实现。
  - B（Batch）表示输入样本批量大小。
  - Batch取值规则
    - 优先获取sequsedQOptional中的Batch信息。
    - 如果未传入sequsedQOptional，且layoutQOptional为TND和传入了cuSeqlensQOptional，则获取cuSeqlensQOptional中的Batch信息。
    - 除上所述，使用batchSize。
  - Query Sequence Length取值规则
    - 优先获取sequsedQOptional中的Sequence Length信息。
    - 如果未传入sequsedQOptional，且layoutQOptional为TND和传入了cuSeqlensQOptional，则获取cuSeqlensQOptional中的Sequence Length信息。
    - 除上所述，使用maxSeqlenQ。
    - oriKvOptional、cmpKvOptional Sequence Length与Query的获取规则一致。
  - BSND场景
    - 当传入的layoutQOptional为"BSND"时，在未传入sequsedQOptional的情况下，必传maxSeqlenQ参数。
    - 当传入的layoutKvOptional为"BSND"时，若hasOriKv为true，在未传入sequsedOriKvOptional的情况下，必传maxSeqlenOriKv参数；若hasCmpKv为true，在未传入sequsedCmpKvOptional的情况下，必传maxSeqlenCmpKv参数。
  - TND场景
    - 当传入的layoutQOptional为"TND"时，必传cuSeqlensQOptional参数。
    - 当传入的layoutKvOptional为"TND"时，若hasOriKv为true，必传cuSeqlensOriKvOptional；若hasCmpKv为true，必传cuSeqlensCmpKvOptional参数。
  - TopkLength参数
    - oriTopkLengthOptional、cmpTopkLengthOptional表示ori/cmp sparseIndices实际参与计算的长度。其值不能大于sparseIndices的最后一维大小，且当sequsedQOptional传入时，topkLength对应有效部分的值需要大于等于0。
    - 当oriMaskMode=0且oriSparseIndicesOptional不为空时，oriTopkLengthOptional必须传入。
    - 当cmpMaskMode=0且cmpSparseIndicesOptional不为空时，cmpTopkLengthOptional必须传入。

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
#include "aclnnop/aclnn_quant_sparse_flash_mla_metadata.h"

#define CHECK_LOG_RET(cond, ret_val, fmt, ...)                                                                         \
    do {                                                                                                               \
        if (!(cond)) {                                                                                                 \
            printf(fmt "\n", ##__VA_ARGS__);                                                                           \
            return (ret_val);                                                                                          \
        }                                                                                                              \
    } while (0)

// 参考 quant_sparse_flash_mla_metadata.h
constexpr uint32_t AIC_CORE_MAX_NUM = 36;
constexpr uint32_t AIV_CORE_MAX_NUM = 72;
constexpr uint32_t MQSMLA_METADATA_TOTAL_SIZE = 1024;
constexpr uint32_t FA_METADATA_SIZE = 9;
constexpr uint32_t FD_METADATA_SIZE = 8;

// FA Metadata Index Definitions
constexpr uint32_t FA_CORE_ENABLE_INDEX = 0;
constexpr uint32_t FA_BN2_START_INDEX = 1;
constexpr uint32_t FA_M_START_INDEX = 2;
constexpr uint32_t FA_S2_START_INDEX = 3;
constexpr uint32_t FA_BN2_END_INDEX = 4;
constexpr uint32_t FA_M_END_INDEX = 5;
constexpr uint32_t FA_S2_END_INDEX = 6;
constexpr uint32_t FA_FIRST_FD_DATA_WORKSPACE_IDX_INDEX = 7;
constexpr uint32_t FA_S2_MAX_NUM = 8;

// FD Metadata Index Definitions
constexpr uint32_t FD_CORE_ENABLE_INDEX = 0;
constexpr uint32_t FD_BN2_IDX_INDEX = 1;
constexpr uint32_t FD_M_IDX_INDEX = 2;
constexpr uint32_t FD_WORKSPACE_IDX_INDEX = 3;
constexpr uint32_t FD_WORKSPACE_NUM_INDEX = 4;
constexpr uint32_t FD_M_START_INDEX = 5;
constexpr uint32_t FD_M_NUM_INDEX = 6;

struct MqsmlaMetadata {
    uint32_t faMetadata[AIC_CORE_MAX_NUM][FA_METADATA_SIZE];
    uint32_t fdMetadata[AIV_CORE_MAX_NUM][FD_METADATA_SIZE];
};

struct ScopeGuard {
    explicit ScopeGuard(std::function<void()> onExitScope) : m_exitFunc(std::move(onExitScope)), m_isDismissed(false)
    {
    }
    // 禁止拷贝
    ScopeGuard(const ScopeGuard &) = delete;
    ScopeGuard &operator=(const ScopeGuard &) = delete;

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
    void *hostAddr{nullptr};
    void *deviceAddr{nullptr};
    aclTensor *data{nullptr};
};

struct ArgScenario {
    bool hasCuSeq{false};
    bool hasSeqused{false};
};

struct ArgContext {
    // required input
    int64_t numHeadsQ{0};
    int64_t numHeadsKv{0};
    int64_t headDim{0};
    int64_t quantMode{1};
    // optional input
    Tensor cuSeqlensQOptional{};
    Tensor cuSeqlensOriKvOptional{};
    Tensor cuSeqlensCmpKvOptional{};
    Tensor sequsedQOptional{};
    Tensor sequsedOriKvOptional{};
    Tensor sequsedCmpKvOptional{};
    Tensor cmpResidualKvOptional{};
    Tensor oriTopkLengthOptional{};
    Tensor cmpTopkLengthOptional{};
    int64_t batchSize{0};
    int64_t maxSeqlenQ{0};
    int64_t maxSeqlenOriKv{0};
    int64_t maxSeqlenCmpKv{0};
    int64_t oriTopk{0};
    int64_t cmpTopk{0};
    int64_t cmpRatio{0};
    int64_t oriMaskMode{0};
    int64_t cmpMaskMode{0};
    int64_t oriWinLeft{-1};
    int64_t oriWinRight{-1};
    char *layoutQOptional{nullptr};
    char *layoutKvOptional{nullptr};
    bool hasOriKv{true};
    bool hasCmpKv{true};
    // output
    Tensor metadata{};
};

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

aclnnStatus Init(int32_t deviceId, aclrtStream *stream)
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
    context.quantMode = 1;
    ret = CreateTensor(aclDataType::ACL_INT32, {MQSMLA_METADATA_TOTAL_SIZE}, context.metadata); // 1024: Fix size
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
    strcpy(context.layoutQOptional, "BSND");  // BSND,TND
    strcpy(context.layoutKvOptional, "BSND"); // BSND,TND,PA_BBND
    context.hasOriKv = true;
    context.hasCmpKv = true;

    context.batchSize = 4;
    context.maxSeqlenOriKv = 1024;
    context.maxSeqlenCmpKv = 1024;
    context.maxSeqlenQ = 1024;

    if (scenario.hasCuSeq) {
        // (B+1,), first element is always 0
        ret = CreateTensor(aclDataType::ACL_INT32, {context.batchSize + 1}, context.cuSeqlensQOptional);
        CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "Create cuSeqlensQOptional failed. Error: %d", ret);
        ret = CreateTensor(aclDataType::ACL_INT32, {context.batchSize + 1}, context.cuSeqlensOriKvOptional);
        CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "Create cuSeqlensOriKvOptional failed. Error: %d", ret);
        ret = CreateTensor(aclDataType::ACL_INT32, {context.batchSize + 1}, context.cuSeqlensCmpKvOptional);
        CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "Create cuSeqlensCmpKvOptional failed. Error: %d", ret);
    }

    if (scenario.hasSeqused) {
        // (B,)
        ret = CreateTensor(aclDataType::ACL_INT32, {context.batchSize}, context.sequsedQOptional);
        CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "Create sequsedQOptional failed. Error: %d", ret);
        ret = CreateTensor(aclDataType::ACL_INT32, {context.batchSize}, context.sequsedOriKvOptional);
        CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "Create sequsedOriKvOptional failed. Error: %d", ret);
        ret = CreateTensor(aclDataType::ACL_INT32, {context.batchSize}, context.sequsedCmpKvOptional);
        CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "Create sequsedCmpKvOptional failed. Error: %d", ret);
    }

    if (context.hasCmpKv && context.cmpRatio != 1 && context.cmpMaskMode == 3) {
        // (B,)
        ret = CreateTensor(aclDataType::ACL_INT32, {context.batchSize}, context.cmpResidualKvOptional);
        CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "Create cmpResidualKvOptional failed. Error: %d", ret);
    }

    argsGuard.Dismiss();
    return ACL_SUCCESS;
}

int main()
{
    // 1. （固定写法）device/stream初始化，参考对外接口列表
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "Init acl failed. ERROR: %d", ret);
    ScopeGuard sysGuard([&] { Finalize(deviceId, stream); });

    // 2. 构造输入与输出，需要根据API的接口定义构造
    ArgScenario scenario{};
    scenario.hasCuSeq = false;
    scenario.hasSeqused = false;
    ArgContext context{};
    ret = CreateArgs(scenario, context);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "Create input arguments failed. ERROR: %d", ret);
    ScopeGuard argsGuard([&] { DestroyArgs(context); });

    // 3. 调用CANN算子库API，需要修改为具体的API
    // 调用aclnnQuantSparseFlashMlaMetadata第一段接口
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    void *workspaceAddr = nullptr;
    ret = aclnnQuantSparseFlashMlaMetadataGetWorkspaceSize(
        context.cuSeqlensQOptional.data, context.cuSeqlensOriKvOptional.data, context.cuSeqlensCmpKvOptional.data,
        context.sequsedQOptional.data, context.sequsedOriKvOptional.data, context.sequsedCmpKvOptional.data,
        context.cmpResidualKvOptional.data, context.oriTopkLengthOptional.data, context.cmpTopkLengthOptional.data,
        context.numHeadsQ, context.numHeadsKv, context.headDim, context.quantMode, context.batchSize,
        context.maxSeqlenQ, context.maxSeqlenOriKv, context.maxSeqlenCmpKv, context.oriTopk, context.cmpTopk,
        context.cmpRatio, context.oriMaskMode, context.cmpMaskMode, context.oriWinLeft, context.oriWinRight,
        context.layoutQOptional, context.layoutKvOptional, context.hasOriKv, context.hasCmpKv, context.metadata.data,
        &workspaceSize, &executor);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "aclnnQuantSparseFlashMlaMetadataGetWorkspaceSize failed. ERROR: %d\n", ret);

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

    // 调用aclnnQuantSparseFlashMlaMetadata第二段接口
    ret = aclnnQuantSparseFlashMlaMetadata(workspaceAddr, workspaceSize, executor, stream);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "aclnnQuantSparseFlashMlaMetadata failed. ERROR: %d\n", ret);

    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "aclrtSynchronizeStream failed. ERROR: %d\n", ret);

    // 5. 打印输出
    MqsmlaMetadata result{};
    ret = aclrtMemcpy(&result, sizeof(result), context.metadata.deviceAddr, sizeof(result), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "aclrtMemcpy failed. ERROR: %d\n", ret);

    for (uint32_t i = 0; i < AIC_CORE_MAX_NUM; ++i) {
        printf("AIC Core%u\n", i);
        printf("    Core Enable : %u\n", result.faMetadata[i][FA_CORE_ENABLE_INDEX]);
        printf("    Start BN2   : %u\n", result.faMetadata[i][FA_BN2_START_INDEX]);
        printf("    Start M     : %u\n", result.faMetadata[i][FA_M_START_INDEX]);
        printf("    Start S2    : %u\n", result.faMetadata[i][FA_S2_START_INDEX]);
        printf("    End BN2     : %u\n", result.faMetadata[i][FA_BN2_END_INDEX]);
        printf("    End M       : %u\n", result.faMetadata[i][FA_M_END_INDEX]);
        printf("    End S2      : %u\n", result.faMetadata[i][FA_S2_END_INDEX]);
        printf("    First Worksapce Index : %u\n", result.faMetadata[i][FA_FIRST_FD_DATA_WORKSPACE_IDX_INDEX]);
        printf("    Max S2 Block Num : %u\n", result.faMetadata[i][FA_S2_MAX_NUM]);
    }
    for (uint32_t i = 0; i < AIV_CORE_MAX_NUM; ++i) {
        printf("AIV Core%u\n", i);
        printf("    Core Enable             : %u\n", result.fdMetadata[i][FD_CORE_ENABLE_INDEX]);
        printf("    FD Task BN2 Idx         : %u\n", result.fdMetadata[i][FD_BN2_IDX_INDEX]);
        printf("    FD Task M Idx           : %u\n", result.fdMetadata[i][FD_M_IDX_INDEX]);
        printf("    FD Task S2 Idx          : %u\n", result.fdMetadata[i][FD_WORKSPACE_IDX_INDEX]);
        printf("    FD Task Workspace Num   : %u\n", result.fdMetadata[i][FD_WORKSPACE_NUM_INDEX]);
        printf("    FD Subtask M Start      : %u\n", result.fdMetadata[i][FD_M_START_INDEX]);
        printf("    FD Subtask M Num        : %u\n", result.fdMetadata[i][FD_M_NUM_INDEX]);
    }

    return 0;
}
```
