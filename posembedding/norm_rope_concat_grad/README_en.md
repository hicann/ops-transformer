# NormRopeConcatGrad

English | [简体中文](./README.md)

## Product Support

|Product      | Supported |
|:----------------------------|:-----------:|
|<term>Ascend 950PR&950DT products</term>|      √     |
|<term>Atlas A3 products</term>|      √     |
|<term>Atlas A2 products</term>|      √     |
|<term>Atlas 200I/500 A2 inference products</term>|      ×     |
|<term>Atlas inference products</term>|      ×     |
|<term>Atlas training products</term>|      ×     |

## Function Description

- Operator function: implements the backward computation of the fused operator that performs normalization (Norm), rotary position embedding (Rope), and feature concatenation (Concat) on query, key, and Value in the (multimodal) transformer attention mechanism.

- Computation formula:
  - **LayerNorm backward:**
    $$
        \frac{\partial L}{\partial x} = \text{rstd} \cdot \Bigg( \frac{\partial L}{\partial y} - \text{Mean}\left( \frac{\partial L}{\partial y} \right) - \hat{x} \cdot \text{Mean}\left( \frac{\partial L}{\partial y} \odot \hat{x} \right) \Bigg)  \quad \quad \quad \quad \quad \quad \quad \text{[Mean over headDim dimension]}
    $$

  - **LayerNorm (with affine transformation parameters) backward:**
    $$
        \left\{
        \begin{aligned}
        \frac{\partial L}{\partial \beta} &= \sum_{B, S, H} \frac{\partial L}{\partial y}, &\quad \text{[Sum over batch, seq, headNum dimensions]} \\
        \frac{\partial L}{\partial \gamma} &= \sum_{B, S, H} \frac{\partial L}{\partial y} \odot \hat{x}, &\quad \text{[Element-wise product accumulation]} \\
        \frac{\partial L}{\partial x} &= \text{rstd} \cdot \Bigg( \frac{\partial L}{\partial \hat{x}} - \text{Mean}\left( \frac{\partial L}{\partial \hat{x}} \right) - \hat{x} \cdot \text{Mean}\left( \frac{\partial L}{\partial \hat{x}} \odot \hat{x} \right) \Bigg) &\quad \text{[Mean over headDim dimension]} \\
        \end{aligned}
        \right\}
    $$

  - **Where (μ is the mean and σ^2 is the variance):**
    $$
      \hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}, \quad \quad \frac{\partial L}{\partial \hat{x}} = \frac{\partial L}{\partial y} \odot \gamma, \quad \quad \text{rstd} = \frac{1}{\sqrt{\sigma^2 + \epsilon}}
    $$

  - **Rope-Interleave backward:**
    $$
        \frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \cdot \text{cos} + Interleave({\frac{\partial L}{\partial y} \cdot \text{sin}}) \odot  \text{negMask}
    $$

  - **Rope-Half backward:**
    $$
        \frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \cdot \text{cos} + Half({\frac{\partial L}{\partial y} \cdot \text{sin}}) \odot  \text{negMask}
    $$

  - **Where Interleave() indicates that elements at odd and even positions along the headDim dimension are reorganized alternately, and Half() indicates that elements in the second half and the first half along the headDim dimension are reorganized alternately. For example, if x = [0,1,2,3,4,5,6,7], then Interleave(x) = [1,0,3,2,5,4,7,6] and Half(x)=[4,0,5,1,6,2,7,3]. negMask has the length of headDim, where even positions are 1 and odd positions are -1, that is, (1, -1, 1, -1, 1, ...)**

- **Parameter Description:**
    <table style="undefined;table-layout: fixed; width: 1300px"><colgroup>
      <col style="width: 210px">
      <col style="width: 150px">
      <col style="width: 616px">
      <col style="width: 188px">
      <col style="width: 125px">
      </colgroup>
      <thead>
        <tr>
          <th>Parameter</th>
          <th>Input/Output/Attribute</th>
          <th>Description</th>
          <th>Data Type</th>
          <th>Data Format</th>
        </tr></thead>
      <tbody>
        <tr>
          <td>gradQueryOutput</td>
          <td>Input</td>
          <td>Backward gradient value of the forward outputs of query and encoderQuery at the network layer, corresponding to y in the formula. When the encoderQuery parameter is nullptr, the value of seqEncoderQuery is 0. The length of headDim must be an even number in the range of [1~1024]. The shape is [batch, headNum, seqQuery+seqEncoderQuery, headDim].</td>
          <td>FLOAT32, FLOAT16, BFLOAT16</td>
          <td>ND</td>
        </tr>
        <tr>
          <td>gradKeyOutput</td>
          <td>Input</td>
          <td>Backward gradient value of the forward outputs of key and encoderKey at the network layer, corresponding to y in the formula. When the encoderKey parameter is nullptr, the value of seqEncoderKey is 0. The data type must be the same as that of gradQueryOutput. The shape is [batch, headNum, seqKey+seqEncoderKey, headDim].</td>
          <td>FLOAT32, FLOAT16, BFLOAT16</td>
          <td>ND</td>
        </tr>
        <tr>
          <td>gradValueOutput</td>
          <td>Input</td>
          <td>Backward gradient value of the forward outputs of value and encoderValue at the network layer, corresponding to y in the formula. When the encoderValue parameter is nullptr, the value of seqEncoderValue is 0. The length of seqValue must be the same as that of seqKey, and the length of seqEncoderValue must be the same as that of seqEncoderKey. The data type must be the same as that of gradQueryOutput. The shape is [batch, headNum, seqValue+seqEncoderValue, headDim].</td>
          <td>FLOAT32, FLOAT16, BFLOAT16</td>
          <td>ND</td>
        </tr>
        <tr>
          <td>query</td>
          <td>Input</td>
          <td>Forward input query (image Query in multimodal scenarios), corresponding to x in the formula. The data type must be the same as that of gradQueryOutput. The shape is [batch, seqQuery, headNum, headDim].</td>
          <td>FLOAT32, FLOAT16, BFLOAT16</td>
          <td>ND</td>
        </tr>
        <tr>
          <td>key</td>
          <td>Input</td>
          <td>Forward input key (image Key in multimodal scenarios), corresponding to x in the formula. The data type must be the same as that of gradQueryOutput. The shape is [batch, seqKey, headNum,headDim].</td>
          <td>FLOAT32, FLOAT16, BFLOAT16</td>
          <td>ND</td>
        </tr>
        <tr>
          <td>encoderQuery</td>
          <td>Optional input</td>
          <td>Forward input encoderQuery (text Query in multimodal scenarios), corresponding to x in the formula. This parameter is passed in when the text Query participates in training. The data type must be the same as that of gradQueryOutput. The shape is [batch, seqEncoderQuery, headNum, headDim].</td>
          <td>FLOAT32, FLOAT16, BFLOAT16</td>
          <td>ND</td>
        </tr>
        <tr>
          <td>encoderKey</td>
          <td>Optional input</td>
          <td>Forward input encoderKey (text Key in multimodal scenarios), corresponding to x in the formula. This parameter is passed in when the text Key participates in training. The data type must be the same as that of gradQueryOutput. The shape is [batch, seqEncoderKey, headNum, headDim].</td>
          <td>FLOAT32, FLOAT16, BFLOAT16</td>
          <td>ND</td>
        </tr>
        <tr>
          <td>normQueryWeight</td>
          <td>Optional input</td>
          <td>Weight value for the normalization of the forward query, corresponding to γ in the formula. This parameter is passed in when image Query and Key are normalized by LayerNorm with affine transformation. The data type must be the same as that of gradQueryOutput. The shape is [headDim].</td>
          <td>FLOAT32, FLOAT16, BFLOAT16</td>
          <td>ND</td>
        </tr>
        <tr>
          <td>normQueryMean</td>
          <td>Optional input</td>
          <td>Mean output during the normalization of the forward query, corresponding to μ in the formula. This parameter is passed in when image Query and Key are normalized. The shape is [batch, seqQuery, headNum, 1].</td>
          <td>FLOAT32</td>
          <td>ND</td>
        </tr>
        <tr>
          <td>normQueryRstd</td>
          <td>Optional input</td>
          <td>Variance-related item output during the normalization of the forward query, corresponding to rstd in the formula. This parameter is passed in when image Query and Key are normalized. The shape is [batch, seqQuery, headNum, 1].</td>
          <td>FLOAT32</td>
          <td>ND</td>
        </tr>
        <tr>
          <td>normKeyWeight</td>
          <td>Optional input</td>
          <td>Weight value for the normalization of the forward key, corresponding to γ in the formula. This parameter is passed in when image Query and Key are normalized by LayerNorm with affine transformation. The data type must be the same as that of gradQueryOutput. The shape is [headDim].</td>
          <td>FLOAT32, FLOAT16, BFLOAT16</td>
          <td>ND</td>
        </tr>
        <tr>
          <td>normKeyMean</td>
          <td>Optional input</td>
          <td>Mean output during the normalization of the forward key, corresponding to μ in the formula. This parameter is passed in when image Query and Key are normalized. The shape is [batch, seqKey, headNum, 1].</td>
          <td>FLOAT32</td>
          <td>ND</td>
        </tr>
        <tr>
          <td>normKeyRstd</td>
          <td>Optional input</td>
          <td>Variance-related item rstd output during the normalization of the forward key, corresponding to rstd in the formula. This parameter is passed in when image Query and Key are normalized. The shape is [batch, seqKey, headNum, 1].</td>
          <td>FLOAT32</td>
          <td>ND</td>
        </tr>
        <tr>
          <td>normAddQueryWeight</td>
          <td>Optional input</td>
          <td>Weight value for the normalization of the forward encoderQuery, corresponding to γ in the formula. This parameter is passed in when text Query and Key are normalized by LayerNorm with affine transformation. The data type must be the same as that of gradQueryOutput. The shape is [headDim].</td>
          <td>FLOAT32, FLOAT16, BFLOAT16</td>
          <td>ND</td>
        </tr>
        <tr>
          <td>normAddQueryMean</td>
          <td>Optional input</td>
          <td>Mean output during the normalization of the forward encoderQuery, corresponding to μ in the formula. This parameter is passed in when text Query and Key are normalized. The shape is [batch, seqEncoderQuery, headNum, 1].</td>
          <td>FLOAT32</td>
          <td>ND</td>
        </tr>
            <tr>
          <td>normAddQueryRstd</td>
          <td>Optional input</td>
          <td>Variance-related item output during the normalization of the forward encoderQuery, corresponding to rstd in the formula. This parameter is passed in when text Query and Key are normalized. The shape is [batch, seqEncoderQuery, headNum, 1].</td>
          <td>FLOAT32</td>
          <td>ND</td>
        </tr>
        <tr>
          <td>normAddKeyWeight</td>
          <td>Optional input</td>
          <td>Weight value for the normalization of the forward encoderKey, corresponding to γ in the formula. This parameter is passed in when text Query and Key are normalized by LayerNorm with affine transformation. The data type must be the same as that of gradQueryOutput. The shape is [headDim].</td>
          <td>FLOAT32, FLOAT16, BFLOAT16</td>
          <td>ND</td>
        </tr>
        <tr>
          <td>normAddKeyMean</td>
          <td>Optional input</td>
          <td>Mean output during the normalization of the forward encoderKey, corresponding to μ in the formula. This parameter is passed in when text Query and Key are normalized. The shape is [batch, seqEncoderKey, headNum, 1].</td>
          <td>FLOAT32</td>
          <td>ND</td>
        </tr>
        <tr>
          <td>normAddKeyRstd</td>
          <td>Optional input</td>
          <td>Variance-related item rstd output during the normalization of the forward encoderKey, corresponding to rstd in the formula. This parameter is passed in when text Query and Key are normalized. The shape is [batch, seqEncoderKey, headNum, 1].</td>
          <td>FLOAT32</td>
          <td>ND</td>
        </tr>
        <tr>
          <td>ropeSin</td>
          <td>Optional input</td>
          <td>Sin value used for the rotary position embedding of the forward input in the formula. This parameter is passed in when image or text Query and Key undergo rotary position embedding. The length of seqRope must be in the range of [1~min(seqQuery+seqEncoderQuery, seqKey+seqEncoderKey)]. The data type must be the same as that of gradQueryOutput. The shape is [seqRope, headDim].</td>
          <td>FLOAT32, FLOAT16, BFLOAT16</td>
          <td>ND</td>
        </tr>
        <tr>
          <td>ropeCos</td>
          <td>Optional input</td>
          <td>Cos value used for the rotary position embedding of the forward input in the formula. This parameter is passed in when image or text Query and Key undergo rotary position embedding. The data type must be the same as that of gradQueryOutput. The shape is [seqRope, headDim].</td>
          <td>FLOAT32, FLOAT16, BFLOAT16</td>
          <td>ND</td>
        </tr>
        <tr>
          <td>normType</td>
          <td>Optional attribute</td>
          <td>Specifies the normalization type of query and key. 0: no normalization. 1: LayerNorm. 2: LayerNorm with affine transformation parameters. If you do not specify this parameter, the recommended value is 0.</td>
          <td>INT64</td>
          <td>-</td>
        </tr>
        <tr>
          <td>normAddedType</td>
          <td>Optional attribute</td>
          <td>Specifies the normalization type of encoderQuery and encoderKey. 0: no normalization. 1: LayerNorm. 2: LayerNorm with affine transformation parameters. If you do not specify this parameter, the recommended value is 0.</td>
          <td>INT64</td>
          <td>-</td>
        </tr>
        <tr>
          <td>ropeType</td>
          <td>Optional attribute</td>
          <td>Specifies the rotary position embedding type of query and encoderQuery, and key and encoderKey after Concat. 0: no rotary position embedding. 1: Interleave rotary position embedding. 2: Half rotary position embedding. If you do not specify this parameter, the recommended value is 0.</td>
          <td>INT64</td>
          <td>-</td>
        </tr>
        <tr>
          <td>concatOrder</td>
          <td>Optional attribute</td>
          <td>Specifies the stacking order of the Concat operations on query and encoderQuery, key and encoderKey, and value and encoderValue. Taking query as an example, 0: [query, encoderQuery], 1: [encoderQuery, query]. If you do not specify this parameter, the recommended value is 0.</td>
          <td>INT64</td>
          <td>-</td>
        </tr>
        <tr>
          <td>gradQuery</td>
          <td>Output</td>
          <td>Backward gradient value of the network layer for the forward input query in the formula. The data type must be the same as that of gradQueryOutput. The shape is [batch, seqQuery, headNum, headDim].</td>
          <td>FLOAT32, FLOAT16, BFLOAT16</td>
          <td>ND</td>
        </tr>
        <tr>
          <td>gradKey</td>
          <td>Output</td>
          <td>Backward gradient value of the network layer for the forward input key in the formula. The data type must be the same as that of gradQueryOutput. The shape is [batch, seqKey, headNum, headDim].</td>
          <td>FLOAT32, FLOAT16, BFLOAT16</td>
          <td>ND</td>
        </tr>
        <tr>
          <td>gradValue</td>
          <td>Output</td>
          <td>Backward gradient value of the network layer for the forward input value in the formula. The data type must be the same as that of gradQueryOutput. The shape is [batch, seqValue, headNum, headDim].</td>
          <td>FLOAT32, FLOAT16, BFLOAT16</td>
          <td>ND</td>
        </tr>
        <tr>
          <td>gradEncoderQuery</td>
          <td>Optional output</td>
          <td>Backward gradient value of the network layer for the forward input encoderQuery in the formula. This parameter is output when the text Query participates in training. The data type must be the same as that of gradQueryOutput. The shape is [batch, seqEncoderQuery, headNum, headDim].</td>
          <td>FLOAT32, FLOAT16, BFLOAT16</td>
          <td>ND</td>
        </tr>
        <tr>
          <td>gradEncoderKey</td>
          <td>Optional output</td>
          <td>Backward gradient value of the network layer for the forward input encoderKey in the formula. This parameter is output when the text Key participates in training. The data type must be the same as that of gradQueryOutput. The shape is [batch, seqEncoderKey, headNum, headDim].</td>
          <td>FLOAT32, FLOAT16, BFLOAT16</td>
          <td>ND</td>
        </tr>
        <tr>
          <td>gradEncoderValue</td>
          <td>Optional output</td>
          <td>Backward gradient value of the network layer for the forward input encoderValue in the formula. This parameter is output when the text Value participates in training. The data type must be the same as that of gradQueryOutput. The shape is [batch, seqEncoderValue, headNum, headDim].</td>
          <td>FLOAT32, FLOAT16, BFLOAT16</td>
          <td>ND</td>
        </tr>
        <tr>
          <td>gradNormQueryWeight</td>
          <td>Optional output</td>
          <td>Backward gradient value of the γ weight used for the normalization of the forward input query at the network layer in the formula. This parameter is output when image Query and Key are normalized by LayerNorm with affine transformation. The data type must be the same as that of gradQueryOutput. The shape is [headDim].</td>
          <td>FLOAT32, FLOAT16, BFLOAT16</td>
          <td>ND</td>
        </tr>
        <tr>
          <td>gradNormQueryBias</td>
          <td>Optional output</td>
          <td>Backward gradient value of the β offset used for the normalization of the forward input query at the network layer in the formula. This parameter is output when image Query and Key are normalized by LayerNorm with affine transformation. The data type must be the same as that of gradQueryOutput. The shape is [headDim].</td>
          <td>FLOAT32, FLOAT16, BFLOAT16</td>
          <td>ND</td>
        </tr>
        <tr>
          <td>gradNormKeyWeight</td>
          <td>Optional output</td>
          <td>Backward gradient value of the γ weight used for the normalization of the forward input key at the network layer in the formula. This parameter is output when image Query and Key are normalized by LayerNorm with affine transformation. The data type must be the same as that of gradQueryOutput. The shape is [headDim].</td>
          <td>FLOAT32, FLOAT16, BFLOAT16</td>
          <td>ND</td>
        </tr>
        <tr>
          <td>gradNormKeyBias</td>
          <td>Optional output</td>
          <td>Backward gradient value of the β offset used for the normalization of the forward input key at the network layer in the formula. This parameter is output when image Query and Key are normalized by LayerNorm with affine transformation. The data type must be the same as that of gradQueryOutput. The shape is [headDim].</td>
          <td>FLOAT32, FLOAT16, BFLOAT16</td>
          <td>ND</td>
        </tr>
        <tr>
          <td>gradNormAddedQueryWeight</td>
          <td>Optional output</td>
          <td>Backward gradient value of the γ weight used for the normalization of the forward input encoderQuery at the network layer in the formula. This parameter is output when text Query and Key are normalized by LayerNorm with affine transformation. The data type must be the same as that of gradQueryOutput. The shape is [headDim].</td>
          <td>FLOAT32, FLOAT16, BFLOAT16</td>
          <td>ND</td>
        </tr>
        <tr>
          <td>gradNormAddedQueryBias</td>
          <td>Optional output</td>
          <td>Backward gradient value of the β offset used for the normalization of the forward input encoderQuery at the network layer in the formula. This parameter is output when text Query and Key are normalized by LayerNorm with affine transformation. The data type must be the same as that of gradQueryOutput. The shape is [headDim].</td>
          <td>FLOAT32, FLOAT16, BFLOAT16</td>
          <td>ND</td>
        </tr>
        <tr>
          <td>gradNormAddedKeyWeight</td>
          <td>Optional output</td>
          <td>Backward gradient value of the γ weight used for the normalization of the forward input encoderKey at the network layer in the formula. This parameter is output when text Query and Key are normalized by LayerNorm with affine transformation. The data type must be the same as that of gradQueryOutput. The shape is [headDim].</td>
          <td>FLOAT32, FLOAT16, BFLOAT16</td>
          <td>ND</td>
        </tr>
        <tr>
          <td>gradNormAddedKeyBias</td>
          <td>Optional output</td>
          <td>Backward gradient value of the β offset used for the normalization of the forward input encoderKey at the network layer in the formula. This parameter is output when text Query and Key are normalized by LayerNorm with affine transformation. The data type must be the same as that of gradQueryOutput. The shape is [headDim].</td>
          <td>FLOAT32, FLOAT16, BFLOAT16</td>
          <td>ND</td>
        </tr>
      </tbody></table>

## Constraints

None.

## Calling Instructions

| Calling Method   | Calling Sample           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn invocation  | [test_aclnn_norm_rope_concat_backward.cpp](examples/test_aclnn_norm_rope_concat_backward.cpp) | Calls the NormRopeConcatGrad operator through the [aclnnNormRopeConcatBackward](docs/aclnnNormRopeConcatBackward_en.md) API. |
