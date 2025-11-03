## NsaCompressAttentionInfer

## 产品支持情况

|产品      | 是否支持 |
|:----------------------------|:-----------:|
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>|      √     |
|<term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>|      √     |

## 功能说明

* 算子功能：Native Sparse Attention推理过程中，Compress Attention的计算。
* 计算公式：

<center>

  $$
  P_{cmp}= Softmax(scale * query · key^T) \\
  attentionOut = P_{cmp} · value\\
  P_{slc}[j] = \sum\limits_{m=0}^{l'/d -1} \sum\limits_{n = 0}^{l/d -1} P_{cmp} [l'/d * j -m - n]\\
  P_{slc'} = \sum\limits_{h=1}^{H}  P_{slc} ^h\\
  topkIndices = topk(P_{slc'})\\
  $$

</center>

## 参数说明

</center>
<table class="tg"><thead>
  <tr>
    <th class="tg-0pky">参数名</th>
    <th class="tg-0lax">输入/输出/属性</th>
    <th class="tg-0lax">描述</th>
    <th class="tg-0lax">数据类型</th>
    <th class="tg-0lax">数据格式</th>
  </tr></thead>
<tbody>
  <tr>
    <td class="tg-0lax">query</td>
    <td class="tg-0lax">输入</td>
    <td class="tg-0lax">公式中的输入query。</td>
    <td class="tg-0lax">FLOAT16、BFLOAT16</td>
    <td class="tg-0lax">ND</td>
  </tr>
  <tr>
    <td class="tg-0lax">key</td>
    <td class="tg-0lax">输入</td>
    <td class="tg-0lax">公式中的输入key。</td>
    <td class="tg-0lax">FLOAT16、BFLOAT16</td>
    <td class="tg-0lax">ND</td>
  </tr>
  <tr>
    <td class="tg-0lax">value</td>
    <td class="tg-0lax">输入</td>
    <td class="tg-0lax">公式中的输入value。</td>
    <td class="tg-0lax">FLOAT16、BFLOAT16</td>
    <td class="tg-0lax">ND</td>
  </tr>
  <tr>
    <td class="tg-0lax">scale</td>
    <td class="tg-0lax">输入</td>
    <td class="tg-0lax">公式中的输入scale，代表attention计算的缩放系数。</td>
    <td class="tg-0lax">DOUBLE</td>
    <td class="tg-0lax">-</td>
  </tr>
  <tr>
    <td class="tg-0lax">l'</td>
    <td class="tg-0lax">属性</td>
    <td class="tg-0lax">公式中的输入l'，代表select阶段的block大小。</td>
    <td class="tg-0lax">INT64</td>
    <td class="tg-0lax">-</td>
  </tr>
  <tr>
    <td class="tg-0lax">l</td>
    <td class="tg-0lax">属性</td>
    <td class="tg-0lax">公式中的输入l，代表compress阶段的block大小。</td>
    <td class="tg-0lax">INT64</td>
    <td class="tg-0lax">-</td>
  </tr>
  <tr>
    <td class="tg-0lax">d</td>
    <td class="tg-0lax">属性</td>
    <td class="tg-0lax">公式中的输入d，代表两次压缩间的滑窗间隔大小。</td>
    <td class="tg-0lax">INT64</td>
    <td class="tg-0lax">-</td>
  </tr>
  <tr>
    <td class="tg-0lax">attentionOut</td>
    <td class="tg-0lax">输出</td>
    <td class="tg-0lax">公式中的attentionOut，attention计算的结果。</td>
    <td class="tg-0lax">FLOAT16、BFLOAT16</td>
    <td class="tg-0lax">ND</td>
  </tr>
  <tr>
    <td class="tg-0lax">topkIndices</td>
    <td class="tg-0lax">输出</td>
    <td class="tg-0lax">公式中的topkIndices，重要性得分最高的几个block的索引。</td>
    <td class="tg-0lax">INT32</td>
    <td class="tg-0lax">-</td>
  </tr>
</tbody></table>

## 约束说明

* 参数query中的N和numHeads值相等，key、value的N和numKeyValueHeads值相等，并且numHeads是numKeyValueHeads的倍数关系。
* 参数query中的D和key的D(H/numKeyValueHeads)值相等，value的D(H/numKeyValueHeads)和output的D值相等。
* query，key，value输入，功能使用限制如下：
  * 支持B轴小于等于3072。
  * 支持key/value的N轴小于等于256。
  * 支持query的N轴与key/value的N轴（H/D）的比值（即GQA中的group大小）小于等于16。
  * 支持query与key的D轴等于192。
  * 支持value的D轴等于128。
  * 支持key与value的blockSize等于64或128。
  * 普通场景下仅支持query的S轴等于1。
  * 多token推理场景下，仅支持query的S轴最大等于4，并且此时要求每个batch单独的actualQSeqLen<=actualSelKvSeqLen。
  * 仅支持paged attention。
  * 仅支持selectBlockSize取值为16的整数倍，最大支持到128。
  * selectBlockCount上限满足selectBlockCount*selectBlockSize<=MaxKvSeqlen，MaxKvSeqlen=Max(actualSelKvSeqLenOptional)。

## 调用说明

| 调用方式  | 样例代码                                                                | 说明                                                                                          |
| ----------- | ------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| aclnn接口 | [test_aclnn_nsa_compress_attention_infer](./examples/test_aclnn_nsa_compress_attention_infer.cpp) | 通过[`aclnnNsaCompressAttentionInfer`](./docs/aclnnNsaCompressAttentionInfer.md)接口方式调用NsaCompressAttentionInfer算子。 |