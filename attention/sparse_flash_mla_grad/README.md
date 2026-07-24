# SparseFlashMlaGrad

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

- 算子功能：计算`SparseFlashMla`训练场景下注意力的反向输出，支持Sliding Window Attention、Compressed Attention以及Sparse Compressed Attention。

- 计算公式：

    阶段一：根据不同cmp_ratio场景，对输入ori_kv与cmp_kv进行选择

    * 当cmp_ratio = 1 (SWA)：

    $$
    selectedKv\text{ }=\text{ }orikv
    $$

    * 当cmp_ratio = 4 (SCFA)：

    $$
    selectedKv\text{ }=concat(oriKv, \text{ }Gather \left( cmpkv,topkIndices \left[ i \left]  \left)) ,\text{ }0\text{ } < =i < \text{ }selectBlockCount\right. \right. \right. \right.
    $$

    * else (CFA):

    $$
    selectedKv\text{ }=concat(oriKv, \text{ }cmpkv)
    $$

    阶段二：计算P、dP、dS

    $$
    P = SimpleSoftmax(Mask(Q \text{ }@\text{ } selectedKv^{{T}} \cdot \text{ } scale), lse)
    $$

    $$
    dP = dO \text{ }@\text{ } selectedKv^{{T}}
    $$

    $$
    dS = P \times (dP\text{ } -\text{ } SoftmaxGrad(dO, O))
    $$

    阶段三：计算dQ, dKV, dSinks

    $$
    dQ = dS \text{ } @ \text{ } selectedKv \text{ }  \cdot \text{ } scale
    $$

    $$
    dKV = dS^{{T}} \text{ } @ \text{ } Q \text{ } \cdot \text{ } scale + P^{{T}}  @ \text{ } dO
    $$

    $$
    dSinks = ReduceSum(-P \text{ }\times\text{ } dP \text{ }\times\text{ } SimpleSoftmax(sinks, lse), dim=-1)
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
        <td>lse</td>
        <td>输入</td>
        <td>注意力正向计算的输出lse，计算公式详见正向文档。</td>
        <td>FLOAT32</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>oriKvOptional</td>
        <td>输入</td>
        <td>attention结构的原始输入K(V)。</td>
        <td>BFLOAT16、FLOAT16</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>cmpKvOptional</td>
        <td>输入</td>
        <td>经过Compressor压缩后的K(V)。</td>
        <td>BFLOAT16、FLOAT16</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>oriSparseIndicesOptional</td>
        <td>输入</td>
        <td>稀疏场景下选择的oriKvOptional中权重较高的注意力索引。</td>
        <td>INT32</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>cmpSparseIndicesOptional</td>
        <td>输入</td>
        <td>稀疏场景下选择的cmpKvOptional中权重较高的注意力索引。</td>
        <td>INT32</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>cuSeqlensQOptional</td>
        <td>输入</td>
        <td>每个Batch中，Query的有效token数。</td>
        <td>INT32</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>cuSeqlensOriKvOptional</td>
        <td>输入</td>
        <td>每个Batch中，oriKvOptional的有效token数。</td>
        <td>INT32</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>cuSeqlensCmpKvOptional</td>
        <td>输入</td>
        <td>每个Batch中，cmpKvOptional的有效token数。</td>
        <td>INT32</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>sequsedQOptional</td>
        <td>输入</td>
        <td>表示不同batch中query实际参与运算的token数。</td>
        <td>INT32</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>sequsedOriKvOptional</td>
        <td>输入</td>
        <td>表示不同batch中oriKvOptional实际参与运算的token数。</td>
        <td>INT32</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>sequsedCmpKvOptional</td>
        <td>输入</td>
        <td>表示不同batch中cmpKvOptional实际参与运算的token数。</td>
        <td>INT32</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>cmpResidualKvOptional</td>
        <td>输入</td>
        <td>表示每个batch S2 // cmpRatio后的余数。</td>
        <td>INT32</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>oriTopkLengthOptional</td>
        <td>输入</td>
        <td>表示每行query对应的oriKvOptional实际可选的topk长度。</td>
        <td>INT32</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>cmpTopkLengthOptional</td>
        <td>输入</td>
        <td>表示每行query对应的cmpKvOptional实际可选的topk长度。</td>
        <td>INT32</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>sinksOptional</td>
        <td>输入</td>
        <td>注意力下沉tensor。</td>
        <td>FLOAT32</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>metadataOptional</td>
        <td>输入</td>
        <td>表示tiling下沉的aicpu算子输出结果。</td>
        <td>INT32</td>
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
        <td>cmpRatio</td>
        <td>属性</td>
        <td>表示对oriKvOptional的压缩率。</td>
        <td>INT64</td>
        <td>-</td>
    </tr>
    <tr>
        <td>oriMaskMode</td>
        <td>属性</td>
        <td>表示query和oriKvOptional计算的mask模式。</td>
        <td>INT64</td>
        <td>-</td>
    </tr>
    <tr>
        <td>cmpMaskMode</td>
        <td>属性</td>
        <td>表示query和cmpKvOptional计算的mask模式。</td>
        <td>INT64</td>
        <td>-</td>
    </tr>
    <tr>
        <td>oriWinLeft</td>
        <td>属性</td>
        <td>表示query和oriKvOptional计算中query对过去token计算的数量。</td>
        <td>INT64</td>
        <td>-</td>
    </tr>
    <tr>
        <td>oriWinRight</td>
        <td>属性</td>
        <td>表示query和oriKvOptional计算中query对未来token计算的数量。</td>
        <td>INT64</td>
        <td>-</td>
    </tr>
    <tr>
        <td>layoutQOptional</td>
        <td>属性</td>
        <td>表示输入query的数据排布格式。</td>
        <td>STRING</td>
        <td>-</td>
    </tr>
    <tr>
        <td>layoutKvOptional</td>
        <td>属性</td>
        <td>表示输入ori_kv和cmp_kv的数据排布格式。</td>
        <td>STRING</td>
        <td>-</td>
    </tr>
    <tr>
        <td>dQueryOut</td>
        <td>输出</td>
        <td>表示query的梯度。</td>
        <td>BFLOAT16、FLOAT16</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>dOriKvOutOptional</td>
        <td>输出</td>
        <td>表示oriKvOptional的梯度。</td>
        <td>BFLOAT16、FLOAT16</td>
        <td>ND</td>
    </tr>
    <tr>  
        <td>dCmpKvOptional</td>
        <td>输出</td>
        <td>表示cmpKvOptional的梯度。</td>
        <td>BFLOAT16、FLOAT16</td>
        <td>ND</td>
    </tr>
    <tr>  
        <td>dSinksOutOptional</td>
        <td>输出</td>
        <td>表示sinksOptional的梯度。</td>
        <td>FLOAT32</td>
        <td>ND</td>
    </tr>
    <tr>  
        <td>oriSoftmaxL1NormOptional</td>
        <td>输出</td>
        <td>表示query与oriKvOptional计算得出的softmax结果。</td>
        <td>FLOAT32</td>
        <td>ND</td>
    </tr>
    <tr>  
        <td>cmpSoftmaxL1NormOptional</td>
        <td>输出</td>
        <td>表示query与cmpKvOptional计算得出的softmax结果。</td>
        <td>FLOAT32</td>
        <td>ND</td>
    </tr>
    </tbody>
</table>

- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：暂不支持sequsedQOptional、sequsedOriKvOptional、sequsedCmpKvOptional、oriTopkLengthOptional、cmpTopkLengthOptional、metadataOptional字段。

## 约束说明

- 仅支持BSND或TND layout；关于数据shape的约束如下：
  - B：泛化支持。
  - S1、S2：泛化支持；S1、S2支持不等长。
  - N1：支持1~128。
  - N2：仅支持1。
  - D：仅支持512。

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
    <a href="./examples/test_aclnn_sparse_flash_mla_grad.cpp">test_aclnn_sparse_flash_mla_grad
    </a>
    </td>
    <td class="tg-lboi" rowspan="6">
    通过
    <a href="./docs/aclnnSparseFlashMlaGrad.md">aclnnSparseFlashMlaGrad
    </a>
    接口方式调用算子
    </td>
  </tr>
</tbody></table>
