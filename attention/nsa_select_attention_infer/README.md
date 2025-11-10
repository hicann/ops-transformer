# NsaSelectedAttentionInfer

# 产品支持情况

|产品      | 是否支持 |
|:----------------------------|:-----------:|
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>|      √     |
|<term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>|      √     |

# 功能说明

- 算子功能：Native Sparse Attention推理过程中，Selected Attention的计算。
- 计算公式：
  
  self-attention（自注意力）利用输入样本自身的关系构建了一种注意力模型。其原理是假设有一个长度为$n$的输入样本序列$x$，$x$的每个元素都是一个$d$维向量，可以将每个$d$维向量看作一个token embedding，将这样一条序列经过3个权重矩阵变换得到3个维度为$n*d$的矩阵。
  
  Selected Attention的计算由topk索引取数与attention计算融合而成，外加paged attention取kvCache。首先，通过$topkIndices$索引从$key$中取出$key_{topk}$，从$value$中取出$value_{topk}$，计算self_attention公式如下：
  
  $$
  Attention(query,key,value)=Softmax(\frac{query · key_{topk}^T}{\sqrt{d}})value_{topk}
  $$
  
  其中$query$和$key_{topk}^T$乘积代表输入$x$的注意力，为避免该值变得过大，通常除以$d$的开根号进行缩放，并对每行进行softmax归一化，与$value_{topk}$相乘后得到一个$n*d$的矩阵。

# 参数说明

<div style="overflow-x: auto;">
  <table style="table-layout: fixed; width: 902px; border-collapse: collapse; border: 1px solid #ccc;">
    <colgroup>
      <col style="width: 170px">  <!-- 参数名 -->
      <col style="width: 120px">  <!-- 输入/输出 -->
      <col style="width: 300px">  <!-- 描述 -->
      <col style="width: 212px">  <!-- 数据类型 -->
      <col style="width: 100px">  <!-- 数据格式 -->
    </colgroup>
    <thead>
      <tr>
        <th style="font-weight: bold;">参数名</th>
        <th style="font-weight: bold;">输入/输出/属性</th>
        <th style="font-weight: bold;">描述</th>
        <th style="font-weight: bold;">数据类型</th>
        <th style="font-weight: bold;">数据格式</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td style="white-space: nowrap;">query</td>
        <td>输入</td>
        <td>公式中的输入query。</td>
        <td>FLOAT16、BFLOAT16</td>
        <td>ND</td>
      </tr>
      <tr>
        <td style="white-space: nowrap;">key</td>
        <td>输入</td>
        <td>公式中的输入key。</td>
        <td>FLOAT16、BFLOAT16</td>
        <td>ND</td>
      </tr>
      <tr>
        <td style="white-space: nowrap;">value</td>
        <td>输入</td>
        <td>公式中的输入value。</td>
        <td>FLOAT16、BFLOAT16</td>
        <td>ND</td>
      </tr>
      <tr>
        <td style="white-space: nowrap;">topkIndices</td>
        <td>输入</td>
        <td>公式里的topk索引。</td>
        <td>INT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td style="white-space: nowrap;">output</td>
        <td>输出</td>
        <td>公式中attention的输出。</td>
        <td>FLOAT16、BFLOAT16</td>
        <td>ND</td>
      </tr>
    </tbody>
  </table>
</div>

# 约束说明

- 参数query中的N和numHeads值相等，key、value的N和numKeyValueHeads值相等，并且numHeads是numKeyValueHeads的倍数关系。
- 参数query中的D和key的D(H/numKeyValueHeads)值相等，value的D(H/numKeyValueHeads)和output的D值相等。
- query，key，value输入，功能使用限制如下：
  - 支持B轴小于等于3072。
  - 支持key/value的N轴小于等于256。
  - 支持query的N轴与key/value的N轴（H/D）的比值（即GQA中的group大小）小于等于16。
  - 支持query与Key的D轴等于192。
  - 支持value的D轴等于128。
  - 支持Key与Value的blockSize等于64或128。
  - 普通场景下仅支持query的S轴等于1。
  - 多token推理场景下，仅支持query的S轴最大等于4，并且此时要求每个batch单独的actualQSeqLen <= actualSelKvSeqLen。
  - 仅支持paged attention。
  - 仅支持selectBlockSize取值为16的整数倍，最大支持到128。
  - selectBlockCount上限满足selectBlockCount * selectBlockSize <= MaxKvSeqlen，MaxKvSeqlen = Max(actualSelKvSeqLenOptional)。

# 调用说明

| 调用方式  | 样例代码                                                                | 说明                                                                                          |
| ----------- | ------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| aclnn接口 | [test_aclnn_nsa_select_attention_infer](./examples/test_aclnn_nsa_select_attention_infer.cpp) | 通过[`aclnnNsaSelectedAttentionInfer`](./docs/aclnnNsaSelectedAttentionInfer.md)接口方式调用NsaCompressAttentionInfer算子。 |