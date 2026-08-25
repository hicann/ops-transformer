# DenseLightningIndexerSoftmaxLseV2

## 产品支持情况

|产品      | 是否支持 |
|:----------------------------|:-----------:|
|<term>Ascend 950PR/Ascend 950DT</term>|      √     |
|<term>Atlas A3训练系列产品</term>|      ×     |
|<term>Atlas A2训练系列产品</term>|      ×     |
|<term>Atlas 200I/500 A2 推理产品</term>|      ×     |
|<term>Atlas 推理系列产品</term>|      ×     |
|<term>Atlas 训练系列产品</term>|      ×     |

## 功能说明

- 算子功能：DenseLightningIndexerSoftmaxLseV2算子是DenseLightningIndexerGradKlLoss算子计算Softmax输入的一个分支算子。相比DenseLightningIndexerSoftmaxLse，新增了压缩注意力（Compressed Attention）支持，并支持通过metadata前置算子进行分核负载均衡。

- 计算公式：

$$
\text{res}=\text{AttentionMask}\left(\text{ReduceSum}\left(W\odot\text{ReLU}\left(Q_{index}@K_{index}^T\right)\right)\right)
$$

$$
\text{lse}=\text{ReduceMax}\left(\text{res}\right)+\text{log}\left(\text{ReduceSum}\left(\text{exp}\left(\text{res}-\text{ReduceMax}\left(\text{res}\right)\right)\right)\right)
$$

lse作为输出传递给算子DenseLightningIndexerGradKlLoss作为输入计算Softmax使用。

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
      <td>weight</td>
      <td>输入</td>
      <td>权重张量。</td>
      <td>FLOAT32</td>
      <td>ND</td>
     </tr>
     <tr>
      <td>cuSeqLensQOptional</td>
      <td>可选输入</td>
      <td>当前Batch及前序Batch中q的有效token数的累加和。TND场景下必传，第一个值固定为0。支持空Tensor。</td>
      <td>INT32</td>
      <td>ND</td>
     </tr>
     <tr>
      <td>cuSeqLensKOptional</td>
      <td>可选输入</td>
      <td>当前Batch及前序Batch中k的有效token数的累加和。TND场景下必传，第一个值固定为0。支持空Tensor。</td>
      <td>INT32</td>
      <td>ND</td>
     </tr>
     <tr>
      <td>seqUsedQOptional</td>
      <td>可选输入</td>
      <td>不同Batch中q的实际使用长度。支持空Tensor。</td>
      <td>INT32</td>
      <td>ND</td>
     </tr>
     <tr>
      <td>seqUsedKOptional</td>
      <td>可选输入</td>
      <td>不同Batch中k的实际使用长度。支持空Tensor。</td>
      <td>INT32</td>
      <td>ND</td>
     </tr>
     <tr>
      <td>cmpResidualKOptional</td>
      <td>可选输入</td>
      <td>表示k的sequence length与cmpRatio相关的残差。当maskMode=3且cmpRatio>1时必须传入。支持空Tensor。</td>
      <td>INT32</td>
      <td>ND</td>
     </tr>
     <tr>
      <td>metadataOptional</td>
      <td>输入</td>
      <td>前置AICPU算子输出的分核负载均衡信息。由aclnnDenseLightningIndexerSoftmaxLseV2Metadata算子输出。支持空Tensor。</td>
      <td>INT32</td>
      <td>ND</td>
     </tr>
     <tr>
      <td>layoutQ</td>
      <td>属性</td>
      <td>表示query侧的排列格式。支持BSND、TND，传空指针时为BSND。layoutQ与layoutK必须一致。</td>
      <td>STRING</td>
      <td>-</td>
     </tr>
     <tr>
      <td>layoutK</td>
      <td>属性</td>
      <td>表示key侧的排列格式。支持BSND、TND，传空指针时为BSND。layoutK与layoutQ必须一致。</td>
      <td>STRING</td>
      <td>-</td>
     </tr>
     <tr>
      <td>maskMode</td>
      <td>属性</td>
      <td>表示mask的模式。0：No mask；3：rightDownCausal模式的mask。</td>
      <td>INT64</td>
      <td>-</td>
     </tr>
     <tr>
      <td>cmpRatio</td>
      <td>属性</td>
      <td>表示key的压缩倍数。取值范围[1, 128]，表示无压缩。</td>
      <td>INT64</td>
      <td>-</td>
     </tr>
     <tr>
      <td>softmaxLseOut</td>
      <td>输出</td>
      <td>softmax计算使用的LSE（log-sum-exp）值。</td>
      <td>FLOAT32</td>
      <td>ND</td>
     </tr>
     </tbody>
    </table>

## 约束说明

  - BSND场景下，必须传入batch_size和max_seqlen_q相关信息（通过tensor shape体现）。
  - TND场景下，必须传入cuSeqLensQ和cuSeqLensK。
  - 当maskMode=3且cmpRatio>1时，必须传入cmpResidualK，且cmpResidualK的每个值必须小于cmpRatio。
  - seqUsedQ的每个值不大于各Batch的实际seqlen_q。
  - seqUsedK的每个值不大于各Batch的实际seqlen_k。

  <table style="undefined;table-layout: fixed; width: 909px"><colgroup>
  <col style="width: 125px">
  <col style="width: 182px">
  <col style="width: 602px">
  </colgroup>
  <thead>
  <tr>
    <th>规格项</th>
    <th>规格</th>
    <th>规格说明</th>
  </tr>
  </thead>
  <tbody>
  <tr>
    <td>B</td>
    <td>1~256</td>
    <td>-</td>
  </tr>
  <tr>
    <td>S1、S2</td>
    <td>0~128K</td>
    <td>S1、S2支持不等长。</td>
  </tr>
  <tr>
    <td>N1</td>
    <td>1~128</td>
    <td>必须能被N2整除。</td>
  </tr>
  <tr>
    <td>N2</td>
    <td>1</td>
    <td>-</td>
  </tr>
  <tr>
    <td>D</td>
    <td>128</td>
    <td>-</td>
  </tr>
  <tr>
    <td>layout</td>
    <td>BSND/TND</td>
    <td>layoutQ与layoutK必须一致。</td>
  </tr>
  <tr>
    <td>cmpRatio</td>
    <td>1~128</td>
    <td>1表示无压缩。</td>
  </tr>
  <tr>
    <td>maskMode</td>
    <td>0、3</td>
    <td>0=No mask，3=rightDownCausal。</td>
  </tr>
  </tbody>
  </table>

## 调用说明

| 调用方式 | 调用样例                                                                          | 说明                                                                  |
|--------------|-------------------------------------------------------------------------------|--------------------------------------------------------------------|
| aclnn调用 | [test_aclnn_dense_lightning_indexer_softmax_lse_v2](examples/test_aclnn_dense_lightning_indexer_softmax_lse_v2.cpp) | 通过[aclnnDenseLightningIndexerSoftmaxLseV2](docs/aclnnDenseLightningIndexerSoftmaxLseV2.md)接口方式调用dense_lightning_indexer_softmax_lse_v2算子。 |
