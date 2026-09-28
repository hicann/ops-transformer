# DenseLightningIndexerSoftmaxLse

## 产品支持情况

|产品      | 是否支持 |
|:----------------------------|:-----------:|
|<term>Ascend 950PR&950DT系列产品</term>|      √     |
|<term>Atlas A3训练系列产品</term>|      √     |
|<term>Atlas A2训练系列产品</term>|      √     |
|<term>Atlas 200I/500 A2推理产品</term>|      ×     |
|<term>Atlas推理系列产品</term>|      ×     |
|<term>Atlas训练系列产品</term>|      ×     |

## 功能说明

- 算子功能：DenseLightningIndexerSoftmaxLse算子是DenseLightningIndexerGradKlLoss算子计算Softmax输入的一个分支算子。

- 计算公式：

$$
\text{res}=\text{AttentionMask}\left(\text{ReduceSum}\left(W\odot\text{ReLU}\left(Q_{index}@K_{index}^T\right)\right)\right)
$$

$$
\text{maxIndex}=\text{max}\left(res\right)
$$

$$
\text{sumIndex}=\text{ReduceSum}\left(\text{exp}\left(res-maxIndex\right)\right)
$$

maxIndex，sumIndex作为输出传递给算子DenseLightningIndexerGradKlLoss作为输入计算Softmax使用。

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
      <td>queryIndex</td>
      <td>输入</td>
      <td>lightningIndexer结构的输入queryIndex。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
     </tr>
     <tr>
      <td>keyIndex</td>
      <td>输入</td>
      <td>lightningIndexer结构的输入keyIndex。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
     </tr>
     <tr>
      <td>weights</td>
      <td>输入</td>
      <td>权重。</td>
      <td>FLOAT16、BFLOAT16、FLOAT32</td>
      <td>ND</td>
     </tr>
     <tr>
      <td>actualSeqLengthsQuery</td>
      <td>输入</td>
      <td>每个Batch中，Query的有效token数。</td>
      <td>INT64</td>
      <td>ND</td>
     </tr>
     <tr>
      <td>actualSeqLengthsKey</td>
      <td>输入</td>
      <td>每个Batch中，Key的有效token数。</td>
      <td>INT64</td>
      <td>ND</td>
     </tr>
     <tr>
      <td>layout</td>
      <td>输入</td>
      <td>layout格式。</td>
      <td>-</td>
      <td>-</td>
     </tr>
     <tr>
      <td>sparseMode</td>
      <td>输入</td>
      <td>sparse的模式。</td>
      <td>INT64</td>
      <td>-</td>
     </tr>
     <tr>
       <td>preTokens</td>
       <td>输入</td>
       <td>用于稀疏计算，表示Attention需要和前几个token计算关联。</td>
       <td>INT64</td>
       <td>-</td>
      </tr>
     <tr>
       <td>nextTokens</td>
       <td>输入</td>
       <td>用于稀疏计算，表示Attention需要和后几个token计算关联。</td>
       <td>INT64</td>
       <td>-</td>
     </tr>
     <tr>
      <td>softmaxMaxOut</td>
      <td>输出</td>
      <td>softmax计算使用的max值。</td>
      <td>FLOAT32</td>
      <td>ND</td>
     </tr>
     <tr>
      <td>softmaxSumOut</td>
      <td>输出</td>
      <td>softmax计算使用的sum值。</td>
      <td>FLOAT32</td>
      <td>ND</td>
     </tr>
     </tbody>
    </table>

## 约束说明

无

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn调用 | [test_aclnn_dense_lightning_indexer_softmax_lse](examples/test_aclnn_dense_lightning_indexer_softmax_lse.cpp) | 通过[aclnnDenseLightningIndexerSoftmaxLse](docs/aclnnDenseLightningIndexerSoftmaxLse.md)接口方式调用dense_lightning_indexer_softmax_lse算子。 |
