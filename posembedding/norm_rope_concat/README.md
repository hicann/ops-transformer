# NormRopeConcat

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|<term>Ascend 950PR&950DT系列产品</term>|      √     |
|<term>Atlas A3系列产品</term>|      √     |
|<term>Atlas A2系列产品</term>|      √     |
|<term>Atlas 200I/500 A2推理产品</term>|      ×     |
|<term>Atlas推理系列产品</term>|      ×     |
|<term>Atlas训练系列产品</term>|      ×     |

## 功能说明

- 算子功能:（多模态）transformer注意力机制中，针对query、key和Value实现归一化（Norm）、旋转位置编码（Rope）、特征拼接（Concat）：

    - 归一化（Norm）当前支持层归一化（LayerNorm）、带仿射变换参数层归一化（AFFINE LayerNorm）、均方根归一化（RmsNorm）和带仿射变换参数均方根归一化（AFFINE RmsNorm）类型。
    - 旋转位置编码（Rope）支持Interleave和Half类型。
    - 特征拼接（Concat）支持在sequence维度上进行拼接，拼接有顺序区别。

- 计算公式（以Query（视频）和EncoderQuery（文本）为例）：

	  $$
    hiddenState_q = \text{Norm}(query, normQueryWeight, normQueryBias, eps) \\
    hiddenState_{eq} = \text{Norm}(encoderQuery, normEncoderQueryWeight, normEncoderQueryBias, eps) \\
    concatedHiddenState = \text{Concat}(hiddenState_q, hiddenState_{eq}) \\
    transposedHiddenState = \text{Transpose}(concatedHiddenState, (0, 2, 1, 3)) \\
    hiddenState = \text{RoPE}(concatedHiddenState, ropeSin, ropeCos)
    $$

- 说明：

    1. 输入输出布局如下：输入`query`的shape为`(B, S, N, D)`，输出`hiddenState`的shape为`(B, N, S, D)`，其中
    B为batch，S为sequenceLen，N为headNum，D为headDim。
    2. Norm有五种模式(`normType`)：`NONE(0), LAYER_NORM(1), LAYER_NORM_AFFINE(2), RMS_NORM(3), RMS_NORM_AFFINE(4)`，其中：
        当`normType = NONE`时：

        $$
        hiddenState_q = query
        $$

        当`normType = LAYER_NORM`时

        $$
        queryMean_{b,s,n} = \frac{1}{D}\sum_{i=0}^{D}query_{b,s,n} \\
        queryVar_{b,s,n} = \frac{1}{D}\sum_{i=0}^{D}(query-queryMean_{b,s,n})^2 \\
        queryRstd_{b,s,n}=  \frac{1}{\sqrt{queryVar_{b,s,n}+\epsilon}} \\
        hiddenState_q = (query-queryMean)*queryRstd$$
        当`normType = LAYER_NORM_AFFINE`时，在上面的基础上

        $$
        hiddenState_q = normQueryWeight*hiddenState_q + normQueryBias
        $$

        当`normType = RMS_NORM`时：

        $$
        queryMs = \frac{1}{D}\sum_{i=0}^{D}(query_{b,s,n})^2 \\
        queryRms = \frac{1}{\sqrt{queryMs+\epsilon}} \\
        hiddenState_q = query * queryRms
        $$

        当`normType = RMS_NORM_AFFINE`时，在上面的基础上

        $$
        hiddenState_q = normQueryWeight*hiddenState_q
        $$

    3. Concat指在sequence维度上进行拼接，拼接有顺序区别(`concatOrder`)，当`concatOrder=0`时，$hiddenState_q$在$hiddenState_{eq}$前，当`concatOrder=1`时，$hiddenState_q$在$hiddenState_{eq}$后。
    4. RoPE有三种模式(`ropeType`):`NONE(0), INTERLEAVE(1), HALF(2)`，其中当`ropeType=NONE`时直接输出不做变换，其余情况参考如下:

        ```python
          def image_rotary_emb(hidden_states, rope_sin, rope_cos, mode=1):
              out = torch.empty_like(hidden_states)
              if mode == 1: # interleave
                  x = hidden_states.view(*hidden_states.shape[:-1], -1, 2)
                  x1, x2 = x[..., 0], x[..., 1]
                  rotated_x = torch.stack([-x2, x1],dim=-1).flatten(3)
                  out = hidden_states.float() * rope_cos + rotated_x.float()*rope_sin
                  return out.type_as(hidden_states)
              else: # half
                  x1, x2 = hidden_states.reshape(*hidden_states.shape[:-1], 2, -1).unbind(-2)
                  rotated_x = torch.cat([-x2, x1],dim=-1)
                  out = hidden_states.float() * rope_cos + rotated_x.float()*rope_sin
                  return out.type_as(hidden_states)
        ```

    5. RoPE的输入`ropeSin`的shape为`(seqRope, D)`，其中

    $$
    seqRope <= min(seqQuery+seqEncoderQuery, seqKey+seqEncoderKey)
    $$

    6. 当场景为训练时，会输出`queryMean, queryRstd, encoderQueryMean, encoderQueryRstd`供后续反向使用。

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
      <td>query</td>
      <td>输入</td>
      <td>表示注意力机制中的Query</td>
      <td>FLOAT16、BFLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>key</td>
      <td>输入</td>
      <td>表示注意力机制中的Key</td>
      <td>FLOAT16、BFLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>value</td>
      <td>输入</td>
      <td>表示注意力机制中的Value</td>
      <td>FLOAT16、BFLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>encoderQuery</td>
      <td>输入</td>
      <td>表示注意力机制中的Query，来自EncoderHiddenState，可以为空指针</td>
      <td>FLOAT16、BFLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>encoderKey</td>
      <td>输入</td>
      <td>表示注意力机制中的Key，来自EncoderHiddenState，可以为空指针</td>
      <td>FLOAT16、BFLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>encoderValue</td>
      <td>输入</td>
      <td>表示注意力机制中的Value，来自EncoderHiddenState，可以为空指针</td>
      <td>FLOAT16、BFLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>normQueryWeight</td>
      <td>输入</td>
      <td>表示LayerNorm的仿射变换参数，作用在Query上</td>
      <td>FLOAT16、BFLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>normQueryBias</td>
      <td>输入</td>
      <td>表示LayerNorm的仿射变换参数，作用在Query上</td>
      <td>FLOAT16、BFLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>normKeyWeight</td>
      <td>输入</td>
      <td>表示LayerNorm的仿射变换参数，作用在Key上</td>
      <td>FLOAT16、BFLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>normKeyBias</td>
      <td>输入</td>
      <td>表示LayerNorm的仿射变换参数，作用在Key上</td>
      <td>FLOAT16、BFLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>normAddedQueryWeight</td>
      <td>输入</td>
      <td>表示LayerNorm的仿射变换参数，作用在encoderQuery上</td>
      <td>FLOAT16、BFLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>normAddedQueryBias</td>
      <td>输入</td>
      <td>表示LayerNorm的仿射变换参数，作用在encoderQuery上</td>
      <td>FLOAT16、BFLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>normAddedKeyWeight</td>
      <td>输入</td>
      <td>表示LayerNorm的仿射变换参数，作用在encoderKey上</td>
      <td>FLOAT16、BFLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>normAddedKeyBias</td>
      <td>输入</td>
      <td>表示LayerNorm的仿射变换参数，作用在encoderKey上</td>
      <td>FLOAT16、BFLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>ropeSin</td>
      <td>输入</td>
      <td>表示RoPE的正弦编码</td>
      <td>FLOAT16、BFLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>ropeCos</td>
      <td>输入</td>
      <td>表示RoPE的余弦编码</td>
      <td>FLOAT16、BFLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>normType</td>
      <td>属性</td>
      <td>表示作用在q，k上的正则化类型，0: 不做正则化，1: LayerNorm, 2: LayerNormAffine, 3: RmsNorm, 4: RmsNormAffine</td>
      <td>int64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>normAddedType</td>
      <td>属性</td>
      <td>表示作用在encoderQuery，encoderKey上的正则化类型，0: 不做正则化，1: LayerNorm, 2: LayerNormAffine, 3: RmsNorm, 4: RmsNormAffine</td>
      <td>int64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>ropeType</td>
      <td>属性</td>
      <td>表示RoPE的模式，int64类型，0: 不做RoPE，1: Interleave, 2: Half</td>
      <td>int64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>concatOrder</td>
      <td>属性</td>
      <td>表示拼接的顺序，int64类型，0: query在前，1: query在后</td>
      <td>int64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>eps</td>
      <td>属性</td>
      <td>表示正则化中的epsilon值</td>
      <td>float32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>isTraining</td>
      <td>属性</td>
      <td>表示是否为训练阶段，决定是否输出反向使用的值</td>
      <td>bool</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>queryOutput</td>
      <td>输出</td>
      <td>表示注意力机制中的query输出</td>
      <td>FLOAT16、BFLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>keyOutput</td>
      <td>输出</td>
      <td>表示注意力机制中的key输出</td>
      <td>FLOAT16、BFLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>valueOutput</td>
      <td>输出</td>
      <td>表示注意力机制中的value输出</td>
      <td>FLOAT16、BFLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>normQueryMean</td>
      <td>输出</td>
      <td>LayerNorm中的query均值输出，用于反向</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>normQueryRstd</td>
      <td>输出</td>
      <td>LayerNorm中的query标准差输出，用于反向</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>normKeyMean</td>
      <td>输出</td>
      <td>LayerNorm中的key均值输出，用于反向</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>normKeyRstd</td>
      <td>输出</td>
      <td>LayerNorm中的key标准差输出，用于反向</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>normEncoderQueryMean</td>
      <td>输出</td>
      <td>LayerNorm中的encoderQuery均值输出，用于反向</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>normEncoderQueryRstd</td>
      <td>输出</td>
      <td>LayerNorm中的encoderQuery标准差输出，用于反向</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>normEncoderKeyMean</td>
      <td>输出</td>
      <td>LayerNorm中的encoderKey均值输出，用于反向</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>normEncoderKeyRstd</td>
      <td>输出</td>
      <td>LayerNorm中的encoderKey标准差输出，用于反向</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- 确定性计算：
  - aclnnNormRopeConcat默认确定性实现。
- query、key、value、encoderQuery、encoderKey、encoderValue数据类型需一致。
- headDim长度在[1~1024]间，且为偶数。
- seqRope长度大小在[1~Min(seqQuery+seqEncoderQuery, seqKey+seqEncoderKey)]之间。

## 调用说明

| 调用方式           | 调用样例                                                   | 说明                                                                                                                    |
|----------------|-------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------|
| aclnn调用 | [test_aclnn_norm_rope_concat.cpp](./examples/test_aclnn_norm_rope_concat.cpp)                     |通过[aclnnNormRopeConcat](./docs/aclnnNormRopeConcat.md)接口方式调用NormRopeConcat算子。                   |
