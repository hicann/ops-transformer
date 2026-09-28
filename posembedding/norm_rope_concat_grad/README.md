# NormRopeConcatGrad

## 产品支持情况

|产品      | 是否支持 |
|:----------------------------|:-----------:|
|<term>Ascend 950PR&950DT系列产品</term>|      √     |
|<term>Atlas A3系列产品</term>|      √     |
|<term>Atlas A2系列产品</term>|      √     |
|<term>Atlas 200I/500 A2推理产品</term>|      ×     |
|<term>Atlas推理系列产品</term>|      ×     |
|<term>Atlas训练系列产品</term>|      ×     |

## 功能说明

- 算子功能：（多模态）transformer注意力机制中，针对query、key和Value实现归一化（Norm）、旋转位置编码（Rope）、特征拼接（Concat）融合算子功能反向推导。

- 计算公式：
  - **LayerNorm反向推导：**
    $$
        \frac{\partial L}{\partial x} = \text{rstd} \cdot \Bigg( \frac{\partial L}{\partial y} - \text{Mean}\left( \frac{\partial L}{\partial y} \right) - \hat{x} \cdot \text{Mean}\left( \frac{\partial L}{\partial y} \odot \hat{x} \right) \Bigg)  \quad \quad \quad \quad \quad \quad \quad \text{[Mean over headDim dimension]}
    $$

  - **LayerNorm（带仿射变换参数）反向推导：**
    $$
        \left\{
        \begin{aligned}
        \frac{\partial L}{\partial \beta} &= \sum_{B, S, H} \frac{\partial L}{\partial y}, &\quad \text{[Sum over batch, seq, headNum dimensions]} \\
        \frac{\partial L}{\partial \gamma} &= \sum_{B, S, H} \frac{\partial L}{\partial y} \odot \hat{x}, &\quad \text{[Element-wise product accumulation]} \\
        \frac{\partial L}{\partial x} &= \text{rstd} \cdot \Bigg( \frac{\partial L}{\partial \hat{x}} - \text{Mean}\left( \frac{\partial L}{\partial \hat{x}} \right) - \hat{x} \cdot \text{Mean}\left( \frac{\partial L}{\partial \hat{x}} \odot \hat{x} \right) \Bigg) &\quad \text{[Mean over headDim dimension]} \\
        \end{aligned}
        \right\}
    $$

  - **其中（μ为均值，σ^2为方差）:**
    $$
      \hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}, \quad \quad \frac{\partial L}{\partial \hat{x}} = \frac{\partial L}{\partial y} \odot \gamma, \quad \quad \text{rstd} = \frac{1}{\sqrt{\sigma^2 + \epsilon}}
    $$

  - **Rope-Interleave反向推导：**
    $$
        \frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \cdot \text{cos} + Interleave({\frac{\partial L}{\partial y} \cdot \text{sin}}) \odot  \text{negMask}
    $$

  - **Rope-Half反向推导：**
    $$
        \frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \cdot \text{cos} + Half({\frac{\partial L}{\partial y} \cdot \text{sin}}) \odot  \text{negMask}
    $$

  - **其中Interleave()表示headDim维度奇数与偶数位置交替重组，Half()表示headDim维度后半和前一半元素交替重组，例如x = [0,1,2,3,4,5,6,7],则Interleave(x) = [1,0,3,2,5,4,7,6]，Half(x)=[4,0,5,1,6,2,7,3]；negMask为headDim长度，偶数位为1，奇数位为-1，即(1, -1, 1, -1, 1, ...)**

- **参数说明：**
    <table style="undefined;table-layout: fixed; width: 1300px"><colgroup>
      <col style="width: 210px">
      <col style="width: 150px">
      <col style="width: 616px">
      <col style="width: 188px">
      <col style="width: 125px">
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
          <td>gradQueryOutput</td>
          <td>输入</td>
          <td>网络层对query和encoderQuery正向输出结果的反向梯度值，对应公式中的y，encoderQuery参数为nullptr时seqEncoderQuery值为0，headDim长度大小需在[1~1024]间且为偶数，shape为[batch, headNum, seqQuery+seqEncoderQuery, headDim]。</td>
          <td>FLOAT32、FLOAT16、BFLOAT16</td>
          <td>ND</td>
        </tr>
        <tr>
          <td>gradKeyOutput</td>
          <td>输入</td>
          <td>网络层对key和encoderKey正向输出结果的反向梯度值，对应公式中的y，encoderKey参数为nullptr时seqEncoderKey值为0，数据类型与参数gradQueryOutput保持一致，shape为[batch, headNum, seqKey+seqEncoderKey, headDim]。</td>
          <td>FLOAT32、FLOAT16、BFLOAT16</td>
          <td>ND</td>
        </tr>
        <tr>
          <td>gradValueOutput</td>
          <td>输入</td>
          <td>网络层对value和encoderValue正向输出结果的反向梯度值，对应公式中的y，encoderValue参数为nullptr时seqEncoderValue值为0，seqValue长度大小与seqKey一致，seqEncoderValue长度大小与seqEncoderKey一致，数据类型与参数gradQueryOutput保持一致，shape为[batch, headNum, seqValue+seqEncoderValue, headDim]。</td>
          <td>FLOAT32、FLOAT16、BFLOAT16</td>
          <td>ND</td>
        </tr>
        <tr>
          <td>query</td>
          <td>输入</td>
          <td>正向输入的query（多模态中图片Query），对应公式中的x，数据类型与参数gradQueryOutput保持一致，shape为[batch, seqQuery, headNum, headDim]。</td>
          <td>FLOAT32、FLOAT16、BFLOAT16</td>
          <td>ND</td>
        </tr>
        <tr>
          <td>key</td>
          <td>输入</td>
          <td>正向输入的key（多模态中图片Key），对应公式中的x，数据类型与参数gradQueryOutput保持一致，shape为[batch, seqKey, headNum,headDim]。</td>
          <td>FLOAT32、FLOAT16、BFLOAT16</td>
          <td>ND</td>
        </tr>
        <tr>
          <td>encoderQuery</td>
          <td>可选输入</td>
          <td>正向输入的encoderQuery（多模态中文本Query），对应公式中的x，当文本Query参与训练时进行传入，数据类型与参数gradQueryOutput保持一致，shape为[batch, seqEncoderQuery, headNum, headDim]。</td>
          <td>FLOAT32、FLOAT16、BFLOAT16</td>
          <td>ND</td>
        </tr>
        <tr>
          <td>encoderKey</td>
          <td>可选输入</td>
          <td>正向输入的encoderKey（多模态中文本Key），对应公式中的x，当文本Key参与训练时进行传入，数据类型与参数gradQueryOutput保持一致，shape为[batch, seqEncoderKey, headNum, headDim]。</td>
          <td>FLOAT32、FLOAT16、BFLOAT16</td>
          <td>ND</td>
        </tr>
        <tr>
          <td>normQueryWeight</td>
          <td>可选输入</td>
          <td>正向query进行归一化操作的权重值，对应公式中的γ，当图片Query、Key进行带仿射LayerNorm归一化操作时传入，数据类型与参数gradQueryOutput保持一致，shape为[headDim]。</td>
          <td>FLOAT32、FLOAT16、BFLOAT16</td>
          <td>ND</td>
        </tr>
        <tr>
          <td>normQueryMean</td>
          <td>可选输入</td>
          <td>正向query进行归一化操作时输出的均值，对应公式中的μ，当图片Query、Key进行归一化操作时传入，shape为[batch, seqQuery, headNum, 1]。</td>
          <td>FLOAT32</td>
          <td>ND</td>
        </tr>
        <tr>
          <td>normQueryRstd</td>
          <td>可选输入</td>
          <td>正向query进行归一化操作时输出的方差相关项，对应公式中的rstd，当图片Query、Key进行归一化操作时传入，shape为[batch, seqQuery, headNum, 1]。</td>
          <td>FLOAT32</td>
          <td>ND</td>
        </tr>
        <tr>
          <td>normKeyWeight</td>
          <td>可选输入</td>
          <td>正向key进行归一化操作的权重值，对应公式中的γ，当图片Query、Key进行带仿射LayerNorm归一化操作时传入，数据类型与参数gradQueryOutput保持一致，shape为[headDim]。</td>
          <td>FLOAT32、FLOAT16、BFLOAT16</td>
          <td>ND</td>
        </tr>
        <tr>
          <td>normKeyMean</td>
          <td>可选输入</td>
          <td>正向key进行归一化操作时输出的均值，对应公式中的μ，当图片Query、Key进行归一化操作时传入，shape为[batch, seqKey, headNum, 1]。</td>
          <td>FLOAT32</td>
          <td>ND</td>
        </tr>
        <tr>
          <td>normKeyRstd</td>
          <td>可选输入</td>
          <td>正向key进行归一化操作时输出的方差相关项rstd，对应公式中的rstd，当图片Query、Key进行归一化操作时传入，shape为[batch, seqKey, headNum, 1]。</td>
          <td>FLOAT32</td>
          <td>ND</td>
        </tr>
        <tr>
          <td>normAddQueryWeight</td>
          <td>可选输入</td>
          <td>正向encoderQuery进行归一化操作的权重值，对应公式中的γ，当文本Query、Key进行带仿射LayerNorm归一化操作时传入，数据类型与参数gradQueryOutput保持一致，shape为[headDim]。</td>
          <td>FLOAT32、FLOAT16、BFLOAT16</td>
          <td>ND</td>
        </tr>
        <tr>
          <td>normAddQueryMean</td>
          <td>可选输入</td>
          <td>正向encoderQuery进行归一化操作时输出的均值，对应公式中的μ，当文本Query、Key进行归一化操作时传入，shape为[batch, seqEncoderQuery, headNum, 1]。</td>
          <td>FLOAT32</td>
          <td>ND</td>
        </tr>
            <tr>
          <td>normAddQueryRstd</td>
          <td>可选输入</td>
          <td>正向encoderQuery进行归一化操作时输出的方差相关项，对应公式中的rstd，当文本Query、Key进行归一化操作时传入，shape为[batch, seqEncoderQuery, headNum, 1]。</td>
          <td>FLOAT32</td>
          <td>ND</td>
        </tr>
        <tr>
          <td>normAddKeyWeight</td>
          <td>可选输入</td>
          <td>正向encoderKey进行归一化操作的权重值，对应公式中的γ，当文本Query、Key进行带仿射LayerNorm归一化操作时传入，数据类型与参数gradQueryOutput保持一致，shape为[headDim]。</td>
          <td>FLOAT32、FLOAT16、BFLOAT16</td>
          <td>ND</td>
        </tr>
        <tr>
          <td>normAddKeyMean</td>
          <td>可选输入</td>
          <td>正向encoderKey进行归一化操作时输出的均值，对应公式中的μ，当文本Query、Key进行归一化操作时传入，shape为[batch, seqEncoderKey, headNum, 1]。</td>
          <td>FLOAT32</td>
          <td>ND</td>
        </tr>
        <tr>
          <td>normAddKeyRstd</td>
          <td>可选输入</td>
          <td>正向encoderKey进行归一化操作时输出的方差相关项，对应公式中的rstd，当文本Query、Key进行归一化操作时传入，shape为[batch, seqEncoderKey, headNum, 1]。</td>
          <td>FLOAT32</td>
          <td>ND</td>
        </tr>
        <tr>
          <td>ropeSin</td>
          <td>可选输入</td>
          <td>公式中正向输入进行旋转位置编码操作的sin值，当图片或文本Query、Key进行旋转位置编码操作时传入，seqRope长度大小需在[1~min(seqQuery+seqEncoderQuery, seqKey+seqEncoderKey)]之间，数据类型与参数gradQueryOutput保持一致，shape为[seqRope, headDim]。</td>
          <td>FLOAT32、FLOAT16、BFLOAT16</td>
          <td>ND</td>
        </tr>
        <tr>
          <td>ropeCos</td>
          <td>可选输入</td>
          <td>公式中正向输入进行旋转位置编码操作的cos值，当图片或文本Query、Key进行旋转位置编码操作时传入，数据类型与参数gradQueryOutput保持一致，shape为[seqRope, headDim]。</td>
          <td>FLOAT32、FLOAT16、BFLOAT16</td>
          <td>ND</td>
        </tr>
        <tr>
          <td>normType</td>
          <td>可选属性</td>
          <td>指定query、key归一化操作类型，0：不进行归一化操作，1：层归一化操作，2：带仿射变换参数层归一化操作，用户不特意指定时建议传入为0。</td>
          <td>INT64</td>
          <td>-</td>
        </tr>
        <tr>
          <td>normAddedType</td>
          <td>可选属性</td>
          <td>指定encoderQuery、encoderKey归一化操作类型，0：不进行归一化操作，1：层归一化操作，2：带仿射变换参数层归一化操作，用户不特意指定时建议传入为0。</td>
          <td>INT64</td>
          <td>-</td>
        </tr>
        <tr>
          <td>ropeType</td>
          <td>可选属性</td>
          <td>指定query与encoderQuery、key与encoderKey进行Concat后的旋转位置编码操作类型，0：不进行旋转位置编码操作，1：Interleave类型旋转位置编码，2：Half类型旋转位置编码，用户不特意指定时建议传入为0。</td>
          <td>INT64</td>
          <td>-</td>
        </tr>
        <tr>
          <td>concatOrder</td>
          <td>可选属性</td>
          <td>指定query与encoderQuery、key与encoderKey、value与encoderValue的Concat操作叠加顺序，以query为例，0：[query, encoderQuery]，1：[encoderQuery, query]，用户不特意指定时建议传入为0。</td>
          <td>INT64</td>
          <td>-</td>
        </tr>
        <tr>
          <td>gradQuery</td>
          <td>输出</td>
          <td>公式中网络层对正向输入query的反向梯度值，数据类型与参数gradQueryOutput保持一致，shape为[batch, seqQuery, headNum, headDim]。</td>
          <td>FLOAT32、FLOAT16、BFLOAT16</td>
          <td>ND</td>
        </tr>
        <tr>
          <td>gradKey</td>
          <td>输出</td>
          <td>公式中网络层对正向输入key的反向梯度值，数据类型与参数gradQueryOutput保持一致，shape为[batch, seqKey, headNum, headDim]。</td>
          <td>FLOAT32、FLOAT16、BFLOAT16</td>
          <td>ND</td>
        </tr>
        <tr>
          <td>gradValue</td>
          <td>输出</td>
          <td>公式中网络层对正向输入value的反向梯度值，数据类型与参数gradQueryOutput保持一致，shape为[batch, seqValue, headNum, headDim]。</td>
          <td>FLOAT32、FLOAT16、BFLOAT16</td>
          <td>ND</td>
        </tr>
        <tr>
          <td>gradEncoderQuery</td>
          <td>可选输出</td>
          <td>公式中网络层对正向输入encoderQuery的反向梯度值，当文本Query参与训练时输出，数据类型与参数gradQueryOutput保持一致，shape为[batch, seqEncoderQuery, headNum, headDim]。</td>
          <td>FLOAT32、FLOAT16、BFLOAT16</td>
          <td>ND</td>
        </tr>
        <tr>
          <td>gradEncoderKey</td>
          <td>可选输出</td>
          <td>公式中网络层对正向输入encoderKey的反向梯度值，当文本Key参与训练时输出，数据类型与参数gradQueryOutput保持一致，shape为[batch, seqEncoderKey, headNum, headDim]。</td>
          <td>FLOAT32、FLOAT16、BFLOAT16</td>
          <td>ND</td>
        </tr>
        <tr>
          <td>gradEncoderValue</td>
          <td>可选输出</td>
          <td>公式中网络层对正向输入encoderValue的反向梯度值，当文本Value参与训练时输出，数据类型与参数gradQueryOutput保持一致，shape为[batch, seqEncoderValue, headNum, headDim]。</td>
          <td>FLOAT32、FLOAT16、BFLOAT16</td>
          <td>ND</td>
        </tr>
        <tr>
          <td>gradNormQueryWeight</td>
          <td>可选输出</td>
          <td>公式中网络层对正向输入query进行归一化操作的γ权重反向梯度值，当图片Query、Key进行带仿射LayerNorm归一化操作时输出，数据类型与参数gradQueryOutput保持一致，shape为[headDim]。</td>
          <td>FLOAT32、FLOAT16、BFLOAT16</td>
          <td>ND</td>
        </tr>
        <tr>
          <td>gradNormQueryBias</td>
          <td>可选输出</td>
          <td>公式中网络层对正向输入query进行归一化操作的β偏移反向梯度值，当图片Query、Key进行带仿射LayerNorm归一化操作时输出，数据类型与参数gradQueryOutput保持一致，shape为[headDim]。</td>
          <td>FLOAT32、FLOAT16、BFLOAT16</td>
          <td>ND</td>
        </tr>
        <tr>
          <td>gradNormKeyWeight</td>
          <td>可选输出</td>
          <td>公式中网络层对正向输入key进行归一化操作的γ权重反向梯度值，当图片Query、Key进行带仿射LayerNorm归一化操作时输出，数据类型与参数gradQueryOutput保持一致，shape为[headDim]。</td>
          <td>FLOAT32、FLOAT16、BFLOAT16</td>
          <td>ND</td>
        </tr>
        <tr>
          <td>gradNormKeyBias</td>
          <td>可选输出</td>
          <td>公式中网络层对正向输入key进行归一化操作的β偏移反向梯度值，当图片Query、Key进行带仿射LayerNorm归一化操作时输出，数据类型与参数gradQueryOutput保持一致，shape为[headDim]。</td>
          <td>FLOAT32、FLOAT16、BFLOAT16</td>
          <td>ND</td>
        </tr>
        <tr>
          <td>gradNormAddedQueryWeight</td>
          <td>可选输出</td>
          <td>公式中网络层对正向输入encoderQuery进行归一化操作的γ权重反向梯度值，当文本Query、Key进行带仿射LayerNorm归一化操作时输出，数据类型与参数gradQueryOutput保持一致，shape为[headDim]。</td>
          <td>FLOAT32、FLOAT16、BFLOAT16</td>
          <td>ND</td>
        </tr>
        <tr>
          <td>gradNormAddedQueryBias</td>
          <td>可选输出</td>
          <td>公式中网络层对正向输入encoderQuery进行归一化操作的β偏移反向梯度值，当文本Query、Key进行带仿射LayerNorm归一化操作时输出，数据类型与参数gradQueryOutput保持一致，shape为[headDim]。</td>
          <td>FLOAT32、FLOAT16、BFLOAT16</td>
          <td>ND</td>
        </tr>
        <tr>
          <td>gradNormAddedKeyWeight</td>
          <td>可选输出</td>
          <td>公式中网络层对正向输入encoderKey进行归一化操作的γ权重反向梯度值，当文本Query、Key进行带仿射LayerNorm归一化操作时输出，数据类型与参数gradQueryOutput保持一致，shape为[headDim]。</td>
          <td>FLOAT32、FLOAT16、BFLOAT16</td>
          <td>ND</td>
        </tr>
        <tr>
          <td>gradNormAddedKeyBias</td>
          <td>可选输出</td>
          <td>公式中网络层对正向输入encoderKey进行归一化操作的β偏移反向梯度值，当文本Query、Key进行带仿射LayerNorm归一化操作时输出，数据类型与参数gradQueryOutput保持一致，shape为[headDim]。</td>
          <td>FLOAT32、FLOAT16、BFLOAT16</td>
          <td>ND</td>
        </tr>
      </tbody></table>

## 约束说明

无。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_norm_rope_concat_backward.cpp](examples/test_aclnn_norm_rope_concat_backward.cpp) | 通过[aclnnNormRopeConcatBackward](docs/aclnnNormRopeConcatBackward.md)接口方式调用NormRopeConcatGrad算子。 |
