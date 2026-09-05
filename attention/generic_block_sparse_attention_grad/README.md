# GenericBlockSparseAttentionGrad

## 产品支持情况

|产品      | 是否支持 |
|:----------------------------|:-----------:|
|<term>Ascend 950PR/Ascend 950DT</term>|      √     |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>|      ×     |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>|      ×     |
|<term>Atlas 200I/500 A2 推理产品</term>|      ×     |
|<term>Atlas 推理系列产品</term>|      ×     |
|<term>Atlas 训练系列产品</term>|      ×     |

## 功能说明

+ 算子功能：GenericBlockSparseAttentionGrad是通用块稀疏注意力的反向计算算子。依据`rsvdBlockIdx`/`rsvdBlockCount`（稀疏块索引表）定义的索引，仅在被选中的KV块上计算和传播梯度，支持动态、可变长的分块稀疏模式。调用前须先通过`aclnnGenericBlockSparseAttentionGradMetadata`生成分核`metadata`。
+ 计算公式：

$$
P = SimpleSoftmax(Mask(Q @ selectedK^{T} \cdot scale), lse)
$$

$$
dP = dO @ selectedV^{T}
$$

$$
dS = P \odot (dP - SoftmaxGrad(dO, O))
$$

$$
dQ = dS @ selectedK \cdot scale
$$

$$
dK = dS^{T} @ Q \cdot scale
$$

$$
dV = P^{T} @ dO
$$

## 参数说明

<table class="tg"><thead>
  <tr>
    <th class="tg-0pky">参数名</th>
    <th class="tg-0pky">输入/输出/属性</th>
    <th class="tg-0pky">描述</th>
    <th class="tg-0pky">数据类型</th>
    <th class="tg-0pky">数据格式</th>
  </tr></thead>
<tbody>
  <tr>
    <td class="tg-0pky">query</td>
    <td class="tg-0pky">输入</td>
    <td class="tg-0pky">attention结构的输入Q。</td>
    <td class="tg-0pky">FLOAT16、BFLOAT16</td>
    <td class="tg-0pky">ND</td>
  </tr>
  <tr>
    <td class="tg-0pky">key</td>
    <td class="tg-0pky">输入</td>
    <td class="tg-0pky">attention结构的输入K。</td>
    <td class="tg-0pky">FLOAT16、BFLOAT16</td>
    <td class="tg-0pky">ND</td>
  </tr>
  <tr>
    <td class="tg-0pky">value</td>
    <td class="tg-0pky">输入</td>
    <td class="tg-0pky">attention结构的输入V，shape与key相同。</td>
    <td class="tg-0pky">FLOAT16、BFLOAT16</td>
    <td class="tg-0pky">ND</td>
  </tr>
  <tr>
    <td class="tg-0pky">dout</td>
    <td class="tg-0pky">输入</td>
    <td class="tg-0pky">注意力输出矩阵的梯度。</td>
    <td class="tg-0pky">FLOAT16、BFLOAT16</td>
    <td class="tg-0pky">ND</td>
  </tr>
  <tr>
    <td class="tg-0pky">out</td>
    <td class="tg-0pky">输入</td>
    <td class="tg-0pky">注意力输出矩阵。</td>
    <td class="tg-0pky">FLOAT16、BFLOAT16</td>
    <td class="tg-0pky">ND</td>
  </tr>
  <tr>
    <td class="tg-0pky">lse</td>
    <td class="tg-0pky">输入</td>
    <td class="tg-0pky">注意力正向计算的输出lse。</td>
    <td class="tg-0pky">FLOAT32</td>
    <td class="tg-0pky">ND</td>
  </tr>
  <tr>
    <td class="tg-0pky">rsvdBlockIdx</td>
    <td class="tg-0pky">输入</td>
    <td class="tg-0pky">稀疏块索引数组，指定每个KV块选择的Q块/token索引。</td>
    <td class="tg-0pky">INT32</td>
    <td class="tg-0pky">ND</td>
  </tr>
  <tr>
    <td class="tg-0pky">rsvdBlockCount</td>
    <td class="tg-0pky">输入</td>
    <td class="tg-0pky">指定每个KV块实际选择的Q数量。</td>
    <td class="tg-0pky">INT32</td>
    <td class="tg-0pky">ND</td>
  </tr>
  <tr>
    <td class="tg-0pky">metadata</td>
    <td class="tg-0pky">输入</td>
    <td class="tg-0pky">由GenericBlockSparseAttentionGradMetadata生成的分核信息。</td>
    <td class="tg-0pky">INT64</td>
    <td class="tg-0pky">ND</td>
  </tr>
  <tr>
    <td class="tg-0pky">attenMaskOptional</td>
    <td class="tg-0pky">可选输入</td>
    <td class="tg-0pky">atten_mask，当前暂不支持，应传nullptr。</td>
    <td class="tg-0pky">BOOL</td>
    <td class="tg-0pky">ND</td>
  </tr>
  <tr>
    <td class="tg-0pky">cuSeqLengthsOptional</td>
    <td class="tg-0pky">可选输入</td>
    <td class="tg-0pky">TND layout下query的累积序列长度。</td>
    <td class="tg-0pky">INT64</td>
    <td class="tg-0pky">ND</td>
  </tr>
  <tr>
    <td class="tg-0pky">cuSeqLengthsKvOptional</td>
    <td class="tg-0pky">可选输入</td>
    <td class="tg-0pky">TND layout下key/value的累积序列长度。</td>
    <td class="tg-0pky">INT64</td>
    <td class="tg-0pky">ND</td>
  </tr>
  <tr>
    <td class="tg-0pky">sequsedQOptional</td>
    <td class="tg-0pky">可选输入</td>
    <td class="tg-0pky">各batch中query的实际序列长度。</td>
    <td class="tg-0pky">INT32</td>
    <td class="tg-0pky">ND</td>
  </tr>
  <tr>
    <td class="tg-0pky">sequsedKvOptional</td>
    <td class="tg-0pky">可选输入</td>
    <td class="tg-0pky">各batch中kv的实际序列长度。</td>
    <td class="tg-0pky">INT32</td>
    <td class="tg-0pky">ND</td>
  </tr>
  <tr>
    <td class="tg-0pky">blockShape</td>
    <td class="tg-0pky">属性</td>
    <td class="tg-0pky">稀疏块形状[blockShapeX, blockShapeY]，当前支持[1, 128]。</td>
    <td class="tg-0pky">INT64</td>
    <td class="tg-0pky">-</td>
  </tr>
  <tr>
    <td class="tg-0pky">isPackedGqa</td>
    <td class="tg-0pky">属性</td>
    <td class="tg-0pky">同一group内qHead是否共享稀疏pattern，当前仅支持1。</td>
    <td class="tg-0pky">INT64</td>
    <td class="tg-0pky">-</td>
  </tr>
  <tr>
    <td class="tg-0pky">qInputLayout</td>
    <td class="tg-0pky">属性</td>
    <td class="tg-0pky">query侧layout格式，支持TND/BNSD/BSND。</td>
    <td class="tg-0pky">STRING</td>
    <td class="tg-0pky">-</td>
  </tr>
  <tr>
    <td class="tg-0pky">kvInputLayout</td>
    <td class="tg-0pky">属性</td>
    <td class="tg-0pky">key/value侧layout格式，须与qInputLayout一致。</td>
    <td class="tg-0pky">STRING</td>
    <td class="tg-0pky">-</td>
  </tr>
  <tr>
    <td class="tg-0pky">scaleValue</td>
    <td class="tg-0pky">属性</td>
    <td class="tg-0pky">缩放系数，建议值为1/sqrt(D)。</td>
    <td class="tg-0pky">FLOAT</td>
    <td class="tg-0pky">-</td>
  </tr>
  <tr>
    <td class="tg-0pky">maskType</td>
    <td class="tg-0pky">属性</td>
    <td class="tg-0pky">mask模式，当前仅支持1（RIGHT_DOWN_CAUSAL）。</td>
    <td class="tg-0pky">INT64</td>
    <td class="tg-0pky">-</td>
  </tr>
  <tr>
    <td class="tg-0pky">softmaxPrecision</td>
    <td class="tg-0pky">属性</td>
    <td class="tg-0pky">Softmax精度级别，取值0或1，当前实现传0。</td>
    <td class="tg-0pky">INT64</td>
    <td class="tg-0pky">-</td>
  </tr>
  <tr>
    <td class="tg-0pky">windowSizeLeft</td>
    <td class="tg-0pky">属性</td>
    <td class="tg-0pky">滑窗向前包含token数，不使能时须为-1。</td>
    <td class="tg-0pky">INT64</td>
    <td class="tg-0pky">-</td>
  </tr>
  <tr>
    <td class="tg-0pky">windowSizeRight</td>
    <td class="tg-0pky">属性</td>
    <td class="tg-0pky">滑窗向后包含token数，不使能时须为-1。</td>
    <td class="tg-0pky">INT64</td>
    <td class="tg-0pky">-</td>
  </tr>
  <tr>
    <td class="tg-0pky">dQuery</td>
    <td class="tg-0pky">输出</td>
    <td class="tg-0pky">query的梯度。</td>
    <td class="tg-0pky">FLOAT16、BFLOAT16</td>
    <td class="tg-0pky">ND</td>
  </tr>
  <tr>
    <td class="tg-0pky">dKey</td>
    <td class="tg-0pky">输出</td>
    <td class="tg-0pky">key的梯度。</td>
    <td class="tg-0pky">FLOAT16、BFLOAT16</td>
    <td class="tg-0pky">ND</td>
  </tr>
  <tr>
    <td class="tg-0pky">dValue</td>
    <td class="tg-0pky">输出</td>
    <td class="tg-0pky">value的梯度。</td>
    <td class="tg-0pky">FLOAT16、BFLOAT16</td>
    <td class="tg-0pky">ND</td>
  </tr>
</tbody></table>

## 约束说明

* <term>Ascend 950PR/Ascend 950DT</term>：支持FLOAT16、BFLOAT16的query/key/value/dout/out/dQuery/dKey/dValue，且数据类型保持一致；lse为FLOAT32。
* 须先调用GenericBlockSparseAttentionGradMetadata生成metadata，再调用本算子。
* qInputLayout与kvInputLayout须相同，取值TND/BNSD/BSND；TND布局下须传入对应cuSeqLengths。
* HeadDim固定为128；N1/N2取值范围[1, 128]，且N1 % N2 == 0。
* blockShape当前仅支持[1, 128]；isPackedGqa当前仅支持1；maskType当前仅支持1。
* windowSizeLeft/windowSizeRight不使能时必须为-1；attenMaskOptional当前应传nullptr。
* 默认为非确定性实现，暂不支持确定性实现。

## 调用说明

| 调用方式  | 样例代码                                                                | 说明                                                                                          |
| ----------- | ------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| aclnn接口 | [test_aclnn_generic_block_sparse_attention_grad](./examples/test_aclnn_generic_block_sparse_attention_grad.cpp) | 通过[aclnnGenericBlockSparseAttentionGrad](./docs/aclnnGenericBlockSparseAttentionGrad.md)接口方式调用GenericBlockSparseAttentionGrad算子。 |
