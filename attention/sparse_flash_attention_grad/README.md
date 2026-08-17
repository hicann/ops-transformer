# SparseFlashAttentionGrad

## 产品支持情况

|产品      | 是否支持 |
|:----------------------------|:-----------:|
|<term>Ascend 950PR/Ascend 950DT</term>|      √     |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>|      √     |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>|      √     |
|<term>Atlas 200I/500 A2 推理产品</term>|      ×     |
|<term>Atlas 推理系列产品</term>|      ×     |
|<term>Atlas 训练系列产品</term>|      ×     |

## 功能说明

- 算子功能：根据topkIndices对key和value选取大小为selectedBlockSize的数据重排，接着进行训练场景下计算注意力的反向输出。

- 计算公式：根据传入的topkIndice对keyIn和value选取数量为selectedBlockCount个大小为selectedBlockSize的数据重排，公式如下：

  $$
   selectedKey\text{ }=\text{ }Gather \left( key,topkIndices \left[ i \left]  \left) ,\text{ }0\text{ } < =i < \text{ }selectBlockCount\right. \right. \right. \right.
  $$

  $$
   selectedValue\text{ }=\text{ }Gather \left( value,topkIndices \left[ i \left]  \left) ,\text{ }0\text{ } < =i < \text{ }selectBlockCount\right. \right. \right. \right.
  $$

<div style="padding-left:40px;">

  阶段1：根据矩阵乘法导数规则，计算$dP$和$dV$:

</div>

  $$
   dP\mathop{{}}\nolimits_{{t,:}}=dO\mathop{{}}\nolimits_{{t,:}}\text{@}V\mathop{{}}\nolimits^{{T}}
  $$

  $$
   dV \left[ u \left] =P\mathop{{}}\nolimits_{{T}}^{{t,:}}\text{@}dO\mathop{{}}\nolimits_{{t,:}}\right. \right.
  $$

<div style="padding-left:40px;">

   阶段2：计算$dS$:

</div>

  $$
   d\mathop{{S}}\nolimits_{{t,:}}= \left[ P\mathop{{}}\nolimits_{{t,:}}@ \left( dP\mathop{{}}\nolimits_{{t,:}}-FlashSoftmaxGrad \left( dO,O \left)  \left)  \right] \right. \right. \right. \right.
  $$

<div style="padding-left:40px;">

   阶段3：计算$dQ$与$dK$:

</div>

  $$
   d\mathop{{Q}}\nolimits_{{t,:}}=d\mathop{{S}}\nolimits_{{t,:}}@K \left[ u \left] \mathop{{}}\nolimits_{{:t,:}}/\sqrt{{d\mathop{{}}\nolimits_{{k,:}}}}\right. \right.
  $$

  $$
   dK \left[ u \left] \mathop{{}}\nolimits_{{:t,:}}=dS\mathop{{}}\nolimits_{{t,:t}}\mathop{{}}\nolimits^{{T}}\text{@}Q/\sqrt{{d\mathop{{}}\nolimits_{{t,:}}}}\right. \right.
  $$

## 参数说明

<table style="undefined;table-layout: fixed; width: 1080px">
    <colgroup>
        <col style="width: 170px">
        <col style="width: 120px">
        <col style="width: 300px">
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
        <td>query</td>
        <td>输入</td>
        <td>attention结构的输入Q。</td>
        <td>BFLOAT16、FLOAT16</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>key</td>
        <td>输入</td>
        <td>attention结构的输入K。</td>
        <td>BFLOAT16、FLOAT16</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>value</td>
        <td>输入</td>
        <td>attention结构的输入v。</td>
        <td>BFLOAT16、FLOAT16</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>sparseIndices</td>
        <td>输入</td>
        <td>稀疏场景下选择的权重较高的注意力索引。</td>
        <td>INT32</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>dOut</td>
        <td>输入</td>
        <td>注意力输出矩阵的梯度。</td>
        <td>BFLOAT16、FLOAT16</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>out</td>
        <td>输入</td>
        <td>注意力输出矩阵。</td>
        <td>BFLOAT16、FLOAT16</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>softmaxMax</td>
        <td>输入</td>
        <td>注意力正向计算的中间输出。</td>
        <td>FLOAT32</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>softmaxSum</td>
        <td>输入</td>
        <td>注意力正向计算的中间输出。</td>
        <td>FLOAT32</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>actualSeqLengthsQueryOptional</td>
        <td>输入</td>
        <td>每个Batch中，Query的有效token数。</td>
        <td>INT32</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>actualSeqLengthskvOptional</td>
        <td>输入</td>
        <td>每个Batch中，Key、value的有效token数。</td>
        <td>INT32</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>queryRopeOptional</td>
        <td>输入</td>
        <td>MLA rope部分：Query位置编码的输出。</td>
        <td>BFLOAT16、FLOAT16</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>keyRopeOptional</td>
        <td>输入</td>
        <td>MLA rope部分：Key位置编码的输出。</td>
        <td>BFLOAT16、FLOAT16</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>scaleValue</td>
        <td>属性</td>
        <td>缩放系数。</td>
        <td>FLOAT32</td>
        <td>-</td>
    </tr>
    <tr>
        <td>sparseBlockSize</td>
        <td>属性</td>
        <td>选择的块的大小。</td>
        <td>INT64</td>
        <td>-</td>
    </tr>
    <tr>
        <td>layout</td>
        <td>属性</td>
        <td>layout格式。</td>
        <td>STRING</td>
        <td>-</td>
    </tr>
    <tr>
        <td>sparseMode</td>
        <td>属性</td>
        <td>sparse的模式。</td>
        <td>INT64</td>
        <td>-</td>
    </tr>
    <tr>
    <td>preTokens</td>
        <td>属性</td>
        <td>Attention算子里，对S矩阵的滑窗起始位置。</td>
        <td>INT64</td>
        <td>-</td>
    </tr>
    <tr>
    <td>nextTokens</td>
        <td>属性</td>
        <td>Attention算子里，对S矩阵的滑窗终止位置。</td>
        <td>INT64</td>
        <td>-</td>
    </tr>
    <tr>
    <td>deterministic</td>
        <td>属性</td>
        <td>确定性计算。</td>
        <td>BOOL</td>
        <td>-</td>
    </tr>
    <tr>
        <td>dQuery</td>
        <td>输出</td>
        <td>表示query的梯度。</td>
        <td>BFLOAT16、FLOAT16</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>dKey</td>
        <td>输出</td>
        <td>表示key的梯度。</td>
        <td>BFLOAT16、FLOAT16</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>dValue</td>
        <td>输出</td>
        <td>表示value的梯度。</td>
        <td>BFLOAT16、FLOAT16</td>
        <td>ND</td>
    </tr>
      <tr>
        <td>dQueryRopeOptional</td>
        <td>输出</td>
        <td>表示queryRope的梯度。</td>
        <td>BFLOAT16、FLOAT16</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>dKeyRopeOptional</td>
        <td>输出</td>
        <td>表示keyRope的梯度。</td>
        <td>BFLOAT16、FLOAT16</td>
        <td>ND</td>
    </tr>
    </tbody>
</table>

## 约束说明

- 参数query中的D和key、value的D值相等为512，参数query_rope中的Dr和key_rope的Dr值相等为64。
- 参数query、key、value的数据类型必须保持一致。
- 当前只支持value和key完全一致的场景。
- 当前仅支持sparseMode=0或3（无mask或以右顶点为划分的下三角场景）
- 仅支持BSND或TND layout；关于数据shape的约束如下：
  - B：取值范围1~256。
  - S1支持1~128K，S2支持1~1M；S1、S2支持不等长。
  - N1支持1/2/4/8/16/32/64/128。
    - <term>Ascend 950PR/Ascend 950DT</term>：
      - 额外还支持48、24、12、6、3。
  - N2：仅支持1。
  - D：仅支持512。
  - Drope：仅支持64。
  - topk：1024、2048、3072、4096、5120、6144、7168、8192。
    - <term>不建议topk * sparseBlockSize超过100k，由于内部算法硬件限制可能会导致oom。</term>
- 确定性计算：
  - SparseFlashAttentionGrad默认非确定性实现，支持通过aclrtCtxSetSysParamOpt开启确定性。

## 调用示例

<table class="tg"><thead>
  <tr>
    <th class="tg-0pky">调用方式</th>
    <th class="tg-0pky">样例代码</th>
    <th class="tg-0pky">说明</th>
  </tr></thead>
<tbody>
  <tr>
    <td class="tg-9wq8" rowspan="6">aclnn接口</td>
    <td class="tg-0pky">
    <a href="./examples/test_aclnn_sparse_flash_attention_grad.cpp">test_aclnn_sparse_flash_attention_grad
    </a>
    </td>
    <td class="tg-lboi" rowspan="6">
    通过
    <a href="./docs/aclnnSparseFlashAttentionGrad.md">aclnnSparseFlashAttentionGrad
    </a>
    接口方式调用算子
    </td>
  </tr>
</tbody></table>
