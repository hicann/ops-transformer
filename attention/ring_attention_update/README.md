# aclnnRingAttentionUpdate

## 支持的产品型号

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3系列产品</term>     |    √     |
| <term>Atlas A2系列产品</term> |    √     |
| <term>Kirin X90处理器系列产品</term> | √ |
| <term>Kirin 9030处理器系列产品</term> | √ |

## 功能说明

- 算子功能：将两次FlashAttention的输出根据其不同的softmax的max和sum更新。

- 计算公式：

    $$
    softmax\_max = max(prev\_softmax\_max, cur\_softmax\_max)
    $$

    $$
    softmax\_sum = prev\_softmax\_sum * exp(prev\_softmax\_max - softmax\_max) + exp(cur\_softmax\_max - softmax\_max)
    $$

    $$
    attn\_out = prev\_attn\_out * exp(prev\_softmax\_max - softmax\_max) / softmax\_sum + cur\_attn\_out * exp(cur\_softmax\_max - softmax\_max) / softmax\_sum
    $$

## 参数说明

<table style="undefined;table-layout: fixed; width: 1576px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 310px">
  <col style="width: 212px">
  <col style="width: 100px">
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
      <td>prevAttnOut</td>
      <td>输入</td>
      <td>公式中的prev_attn_out。第一次FlashAttention的输出。输入shape和inputLayoutOptional属性保持一致。当输入数据排布inputLayoutOptional为TND时，D限制为64的倍数。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>prevSoftmaxMax</td>
      <td>输入</td>
      <td>公式中的prev_softmax_max，第一次FlashAttention的softmax的max结果，输入shape为(B,N,S,8)或(T,N,8)，最后一维8个数字相同，且需要为正数。此处B为batch size，N为head number，S为sequence length，T为time。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>prevSoftmaxSum</td>
      <td>输入</td>
      <td>公式中的prev_softmax_sum，第一次FlashAttention的softmax的sum结果，输入shape和prevSoftmaxMax保持一致，最后一维8个数字相同，且需要为正数。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>curAttnOut</td>
      <td>输入</td>
      <td>公式中的cur_attn_out，第二次FlashAttention的输出，数据类型和输入shape和prevAttnOut保持一致。当输入数据排布inputLayoutOptional为TND时，D限制为64的倍数。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>curSoftmaxMax</td>
      <td>输入</td>
      <td>公式中的cur_softmax_max，第二次FlashAttention的softmax的max结果，输入shape和prevSoftmaxMax保持一致，最后一维8个数字相同，且需要为正数。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>curSoftmaxSum</td>
      <td>输入</td>
      <td>公式中的cur_softmax_sum，第二次FlashAttention的softmax的sum结果，输入shape和prevSoftmaxMax保持一致，最后一维8个数字相同，且需要为正数。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>actualSeqQlenOptional</td>
      <td>可选输入</td>
      <td>从0开始的sequence length的累加，数据类型支持INT64。当数据排布inputLayoutOptional为TND时，需要传入该参数，这是一个从0开始递增至T的整数aclTensor。</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>inputLayoutOptional</td>
      <td>属性</td>
      <td>attn_out相关输入的数据排布。当前支持“TND”和“SBH”。</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>attnOutOut</td>
      <td>输出</td>
      <td>公式中的attn_out，通过两次结果更新后的输出，数据类型支持FLOAT16、FLOAT、BFLOAT16，数据类型和输出shape和prevAttnOut保持一致。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>softmaxMaxOut</td>
      <td>输出</td>
      <td>公式中的softmax_max，通过两次结果更新后的softmax的max，数据类型支持FLOAT，输出shape和prevSoftmaxMax保持一致。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>softmaxSumOut</td>
      <td>输出</td>
      <td>公式中的softmax_sum，通过两次结果更新后的softmax的sum，数据类型支持FLOAT，输出shape和prevSoftmaxMax保持一致。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- Kirin X90/Kirin 9030处理器系列产品: 不支持BFLOAT16。

## 约束说明

  - 当inputLayoutOptional为“TND”时，prevAttnOut的最后一个维度需要为64的倍数。
  - 当inputLayoutOptional为“TND”时，actualSeqQlenOptional为必填。
  - 当inputLayoutOptional为“TND”时，请注意N*D的大小，限制为：(N \* D)向上对齐64的结果 \* (attention的输入数据类型的大小 \* 6 + 8) + (N \* 8)向上对齐64的结果 \* 56 <= 192 \* 1024，数据类型大小：FLOAT为4，FLOAT16和BFLOAT16为2。若大小超过限制，会有相应拦截信息出现。

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                             |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| aclnn调用 | [test_aclnn_ring_attention_update](./examples/test_aclnn_ring_attention_update.cpp) | 通过[aclnnRingAttentionUpdate](./docs/aclnnRingAttentionUpdate.md)接口方式调用RingAttentionUpdate算子。    |
