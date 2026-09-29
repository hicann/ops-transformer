# aclnnNormRopeConcatBackward

## Supported Products

|Product     | Supported|
|:----------------------------|:-----------:|
|Ascend 950PR/Ascend 950DT|      √     |
|<term>Atlas A3 training products/Atlas A3 inference products</term>|      √     |
|<term>Atlas A2 training products/Atlas A2 inference products</term>|      √     |
|<term>Atlas 200I/500 A2 inference products</term>|      ×     |
|<term>Atlas inference products</term>|      ×     |
|<term>Atlas training products</term>|      ×     |

## Function

- Description: Implements backward propagation of normalization (Norm), Rotary Position Embedding (RoPE), and feature concatenation (Concat) for `query`, `key`, and `value` in the (multi-modal) transformer attention mechanism.

    - Currently, Norm supports layer normalization (LayerNorm) and layer normalization with affine transformation parameters (AFFINE LayerNorm).
    - RoPE supports the Interleave and Half types.

- Formulas:

    - **Backward propagation of LayerNorm:**
    $$
        \frac{\partial L}{\partial x} = \text{rstd} \cdot \Bigg( \frac{\partial L}{\partial y} - \text{Mean}\left( \frac{\partial L}{\partial y} \right) - \hat{x} \cdot \text{Mean}\left( \frac{\partial L}{\partial y} \odot \hat{x} \right) \Bigg)  \quad \quad \quad \quad \quad \quad \quad \text{[Mean over headDim dimension]}
    $$

    - **Backward propagation of LayerNorm (with affine transformation parameters):**
    $$
        \left\{
        \begin{aligned}
        \frac{\partial L}{\partial \beta} &= \sum_{B, S, H} \frac{\partial L}{\partial y}, &\quad \text{[Sum over batch, seq, headNum dimensions]} \\
        \frac{\partial L}{\partial \gamma} &= \sum_{B, S, H} \frac{\partial L}{\partial y} \odot \hat{x}, &\quad \text{[Element-wise product accumulation]} \\
        \frac{\partial L}{\partial x} &= \text{rstd} \cdot \Bigg( \frac{\partial L}{\partial \hat{x}} - \text{Mean}\left( \frac{\partial L}{\partial \hat{x}} \right) - \hat{x} \cdot \text{Mean}\left( \frac{\partial L}{\partial \hat{x}} \odot \hat{x} \right) \Bigg) &\quad \text{[Mean over headDim dimension]} \\
        \end{aligned}
        \right\}
    $$

    - **Where (μ is the mean value, and σ<sup>2</sup> is the variance):**
    $$
      \hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}, \quad \quad \frac{\partial L}{\partial \hat{x}} = \frac{\partial L}{\partial y} \odot \gamma, \quad \quad \text{rstd} = \frac{1}{\sqrt{\sigma^2 + \epsilon}}
    $$

    - **Backward propagation of RoPE (Interleave):**
    $$
        \frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \cdot \text{cos} + Interleave({\frac{\partial L}{\partial y} \cdot \text{sin}}) \odot  \text{negMask}
    $$

    - **Backward propagation of RoPE (Half):**
    $$
        \frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \cdot \text{cos} + Half({\frac{\partial L}{\partial y} \cdot \text{sin}}) \odot  \text{negMask}
    $$

    - Interleave() indicates that the odd and even positions in the headDim dimension are alternately rearranged, and Half() indicates that the second half and the first half of the elements in the headDim dimension are alternately rearranged. For example, if x = [0,1,2,3,4,5,6,7], then Interleave(x) = [1,0,3,2,5,4,7,6] and Half(x) = [4,0,5,1,6,2,7,3]. negMask is the length of headDim. The even bit is 1 and the odd bit is -1, that is, (1, -1, 1, -1, 1,...).

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnNormRopeConcatBackwardGetWorkspaceSize` is called to obtain the input parameters and compute the required workspace size based on the process. Then, `aclnnNormRopeConcatBackward` is called to perform computation.

```Cpp
aclnnStatus aclnnNormRopeConcatBackwardGetWorkspaceSize(
    const aclTensor *gradQueryOutput, 
    const aclTensor *gradKeyOutput, 
    const aclTensor *gradValueOutput, 
    const aclTensor *query, 
    const aclTensor *key, 
    const aclTensor *encoderQuery, 
    const aclTensor *encoderKey, 
    const aclTensor *normQueryWeight, 
    const aclTensor *normQueryMean, 
    const aclTensor *normQueryRstd, 
    const aclTensor *normKeyWeight, 
    const aclTensor *normKeyMean, 
    const aclTensor *normKeyRstd, 
    const aclTensor *normAddedQueryWeight, 
    const aclTensor *normAddedQueryMean, 
    const aclTensor *normAddedQueryRstd, 
    const aclTensor *normAddedKeyWeight, 
    const aclTensor *normAddedKeyMean, 
    const aclTensor *normAddedKeyRstd, 
    const aclTensor *ropeSin, 
    const aclTensor *ropeCos, 
    int64_t          normType, 
    int64_t          normAddedType, 
    int64_t          ropeType, 
    int64_t          concatOrder, 
    const aclTensor *gradQuery, 
    const aclTensor *gradKey, 
    const aclTensor *gradValue, 
    const aclTensor *gradEncoderQuery, 
    const aclTensor *gradEncoderKey, 
    const aclTensor *gradEncoderValue, 
    const aclTensor *gradNormQueryWeight, 
    const aclTensor *gradNormQueryBias, 
    const aclTensor *gradNormKeyWeight, 
    const aclTensor *gradNormKeyBias, 
    const aclTensor *gradNormAddedQueryWeight, 
    const aclTensor *gradNormAddedQueryBias, 
    const aclTensor *gradNormAddedKeyWeight, 
    const aclTensor *gradNormAddedKeyBias, 
    uint64_t        *workspaceSize, 
    aclOpExecutor   **executor)
```

```cpp
aclnnStatus aclnnNormRopeConcatBackward(
    void          *workspace, 
    uint64_t       workspaceSize, 
    aclOpExecutor *executor, 
    aclrtStream    stream)
```

## aclnnNormRopeConcatBackwardGetWorkspaceSize

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1488px"><colgroup>
    <col style="width: 210px">
    <col style="width: 121px">
    <col style="width: 253px">
    <col style="width: 262px">
    <col style="width: 213px">
    <col style="width: 115px">
    <col style="width: 169px">
    <col style="width: 145px">
    </colgroup>
    <thead>
      <tr>
        <th>Name</th>
        <th>Input/Output</th>
        <th>Description</th>
        <th>Usage Notes</th>
        <th>Data Type</th>
        <th>Data Format</th>
        <th>Dimension (Shape)</th>
        <th>Non-contiguous Tensor</th>
      </tr></thead>
    <tbody>
      <tr>
          <td>gradQueryOutput</td>
          <td>Input</td>
          <td>Backward gradient value of the forward output for <code>query</code> and <code>encoderQuery</code>, corresponding to <code>y</code> in the formulas.</td>
          <td>When <code>encoderQuery</code> is <code>nullptr</code>, the value of <code>seqEncoderQuery</code> is <code>0</code>. The value of <code>headDim</code> must be an even number in the range of [1, 1024].</td>
          <td>FLOAT32, FLOAT16, BFLOAT16</td>
          <td>ND</td>
          <td>[batch, headNum, seqQuery+seqEncoderQuery, headDim]</td>
          <td>√</td>
      </tr>
      <tr>
          <td>gradKeyOutput</td>
          <td>Input</td>
          <td>Backward gradient value of the forward output for <code>key</code> and <code>encoderKey</code>, corresponding to <code>y</code> in the formulas.</td>
          <td>When <code>encoderKey</code> is <code>nullptr</code>, the value of <code>seqEncoderKey</code> is <code>0</code>. The data type of this parameter must be the same as that of <code>gradQueryOutput</code>.</td>
          <td>FLOAT32, FLOAT16, BFLOAT16</td>
          <td>ND</td>
          <td>[batch, headNum, seqKey+seqEncoderKey, headDim]</td>
          <td>√</td>
      </tr>
      <tr>
          <td>gradValueOutput</td>
          <td>Input</td>
          <td>Backward gradient value of the forward output for <code>value</code> and <code>encoderValue</code>, corresponding to <code>y</code> in the formulas.</td>
          <td>When <code>encoderValue</code> is <code>nullptr</code>, the value of <code>seqEncoderValue</code> is <code>0</code>. The value of <code>seqValue</code> must be the same as that of <code>seqKey</code> and the value of <code>seqEncoderValue</code> must be the same as that of <code>seqEncoderKey</code>. The data type of this parameter must be the same as that of <code>gradQueryOutput</code>.</td>
          <td>FLOAT32, FLOAT16, BFLOAT16</td>
          <td>ND</td>
          <td>[batch, headNum, seqValue+seqEncoderValue, headDim]</td>
          <td>√</td>
      </tr>
      <tr>
          <td>query</td>
          <td>Input</td>
          <td>Forward input <code>query</code> (image query in the multi-modal model), corresponding to <code>x</code> in the formulas.</td>
          <td>The data type must be the same as that of <code>gradQueryOutput</code>.</td>
          <td>FLOAT32, FLOAT16, BFLOAT16</td>
          <td>ND</td>
          <td>[batch, seqQuery, headNum, headDim]</td>
          <td>√</td>
      </tr>
      <tr>
          <td>key</td>
          <td>Input</td>
          <td>Forward input <code>key</code> (image key in the multi-modal model), corresponding to <code>x</code> in the formulas.</td>
          <td>The data type must be the same as that of <code>gradQueryOutput</code>.</td>
          <td>FLOAT32, FLOAT16, BFLOAT16</td>
          <td>ND</td>
          <td>[batch, seqKey, headNum,headDim]</td>
          <td>√</td>
      </tr>
      <tr>
          <td>encoderQuery</td>
          <td>Optional input</td>
          <td>Forward input <code>encoderQuery</code> (text query in the multi-modal model), corresponding to <code>x</code> in the formulas.</td>
          <td>This parameter is passed when the text query is used for training. Its data type must be the same as that of <code>gradQueryOutput</code>.</td>
          <td>FLOAT32, FLOAT16, BFLOAT16</td>
          <td>ND</td>
          <td>[batch, seqEncoderQuery, headNum, headDim]</td>
          <td>√</td>
      </tr>
      <tr>
          <td>encoderKey</td>
          <td>Optional input</td>
          <td>Forward input <code>encoderKey</code> (text key in the multi-modal model), corresponding to <code>x</code> in the formulas.</td>
          <td>This parameter is passed when the text key is used for training. Its data type must be the same as that of <code>gradQueryOutput</code>.</td>
          <td>FLOAT32, FLOAT16, BFLOAT16</td>
          <td>ND</td>
          <td>[batch, seqEncoderKey, headNum, headDim]</td>
          <td>√</td>
      </tr>
      <tr>
          <td>normQueryWeight</td>
          <td>Optional input</td>
          <td>Weight value for normalizing the forward input <code>query</code>, corresponding to <code>γ</code> in the formulas.</td>
          <td>This parameter is passed when LayerNorm with affine transformation is performed on the image query and key. Its data type must be the same as that of <code>gradQueryOutput</code>.</td>
          <td>FLOAT32, FLOAT16, BFLOAT16</td>
          <td>ND</td>
          <td>[headDim]</td>
          <td>√</td>
      </tr>
      <tr>
          <td>normQueryMean</td>
          <td>Optional input</td>
          <td>Mean value output when the forward input <code>query</code> is normalized, corresponding to <code>μ</code> in the formula.</td>
          <td>This parameter is passed when normalization is performed on the image query and key.</td>
          <td>FLOAT32</td>
          <td>ND</td>
          <td>[batch, seqQuery, headNum, 1]</td>
          <td>√</td>
      </tr>
      <tr>
          <td>normQueryRstd</td>
          <td>Optional input</td>
          <td>Variance-related item output when the forward input <code>query</code> is normalized, corresponding to <code>rstd</code> in the formulas.</td>
          <td>This parameter is passed when normalization is performed on the image query and key.</td>
          <td>FLOAT32</td>
          <td>ND</td>
          <td>[batch, seqQuery, headNum, 1]</td>
          <td>√</td>
      </tr>
      <tr>
          <td>normKeyWeight</td>
          <td>Optional input</td>
          <td>Weight value for normalizing the forward input <code>key</code>, corresponding to <code>γ</code> in the formulas. This parameter is passed when LayerNorm with affine transformation is performed on the image query and key.</td>
          <td>The data type must be the same as that of <code>gradQueryOutput</code>.</td>
          <td>FLOAT32, FLOAT16, BFLOAT16</td>
          <td>ND</td>
          <td>[headDim]</td>
          <td>√</td>
      </tr>
      <tr>
          <td>normKeyMean</td>
          <td>Optional input</td>
          <td>Mean value output when the forward input <code>key</code> is normalized, corresponding to <code>μ</code> in the formula.</td>
          <td>This parameter is passed when normalization is performed on the image query and key.</td>
          <td>FLOAT32</td>
          <td>ND</td>
          <td>[batch, seqKey, headNum, 1]</td>
          <td>√</td>
      </tr>
      <tr>
          <td>normKeyRstd</td>
          <td>Optional input</td>
          <td>Variance-related item output when the forward input <code>key</code> is normalized, corresponding to <code>rstd</code> in the formulas.</td>
          <td>This parameter is passed when normalization is performed on the image query and key.</td>
          <td>FLOAT32</td>
          <td>ND</td>
          <td>[batch, seqKey, headNum, 1]</td>
          <td>√</td>
      </tr>
      <tr>
          <td>normAddQueryWeight</td>
          <td>Optional input</td>
          <td>Weight value for normalizing the forward input <code>encoderQuery</code>, corresponding to <code>γ</code> in the formulas.</td>
          <td>This parameter is passed when LayerNorm with affine transformation is performed on the text query and key. Its data type must be the same as that of <code>gradQueryOutput</code>.</td>
          <td>FLOAT32, FLOAT16, BFLOAT16</td>
          <td>ND</td>
          <td>[headDim]</td>
          <td>√</td>
      </tr>
      <tr>
          <td>normAddQueryMean</td>
          <td>Optional input</td>
          <td>Mean value output when the forward input <code>encoderQuery</code> is normalized, corresponding to <code>μ</code> in the formula.</td>
          <td>This parameter is passed when normalization is performed on the text query and key.</td>
          <td>FLOAT32</td>
          <td>ND</td>
          <td>[batch, seqEncoderQuery, headNum, 1]</td>
          <td>√</td>
      </tr>
      <tr>
          <td>normAddQueryRstd</td>
          <td>Optional input</td>
          <td>Variance-related item output when the forward input <code>encoderQuery</code> is normalized, corresponding to <code>rstd</code> in the formulas.</td>
          <td>This parameter is passed when normalization is performed on the text query and key.</td>
          <td>FLOAT32</td>
          <td>ND</td>
          <td>[batch, seqEncoderQuery, headNum, 1]</td>
          <td>√</td>
      </tr>
      <tr>
          <td>normAddKeyWeight</td>
          <td>Optional input</td>
          <td>Weight value for normalizing the forward input <code>encoderKey</code>, corresponding to <code>γ</code> in the formulas.</td>
          <td>This parameter is passed when LayerNorm with affine transformation is performed on the text query and key. Its data type must be the same as that of <code>gradQueryOutput</code>.</td>
          <td>FLOAT32, FLOAT16, BFLOAT16</td>
          <td>ND</td>
          <td>[headDim]</td>
          <td>√</td>
      </tr>
      <tr>
          <td>normAddKeyMean</td>
          <td>Optional input</td>
          <td>Mean value output when the forward input <code>encoderKey</code> is normalized, corresponding to <code>μ</code> in the formula.</td>
          <td>This parameter is passed when normalization is performed on the text query and key.</td>
          <td>FLOAT32</td>
          <td>ND</td>
          <td>[batch, seqEncoderKey, headNum, 1]</td>
          <td>√</td>
      </tr>
      <tr>
          <td>normAddKeyRstd</td>
          <td>Optional input</td>
          <td>Variance-related item output when the forward input <code>encoderKey</code> is normalized, corresponding to <code>rstd</code> in the formulas.</td>
          <td>This parameter is passed when normalization is performed on the text query and key.</td>
          <td>FLOAT32</td>
          <td>ND</td>
          <td>[batch, seqEncoderKey, headNum, 1]</td>
          <td>√</td>
      </tr>
      <tr>
          <td>ropeSin</td>
          <td>Optional input</td>
          <td>Forward input sine value for performing RoPE in the formulas.</td>
          <td>This parameter is passed when RoPE is performed on the image/text query and key. Its data type must be the same as that of <code>gradQueryOutput</code>. The value of <code>seqRope</code> must be within the range of [1, min(<code>seqQuery</code> + <code>seqEncoderQuery</code>, <code>seqKey</code> + <code>seqEncoderKey</code>)].</td>
          <td>FLOAT32, FLOAT16, BFLOAT16</td>
          <td>ND</td>
          <td>[seqRope, headDim]</td>
          <td>√</td>
      </tr>
      <tr>
          <td>ropeCos</td>
          <td>Optional input</td>
          <td>Forward input cosine value for performing RoPE in the formulas.</td>
          <td>This parameter is passed when RoPE is performed on the image/text query and key. Its data type must be the same as that of <code>gradQueryOutput</code>.</td>
          <td>FLOAT32, FLOAT16, BFLOAT16</td>
          <td>ND</td>
          <td>[seqRope, headDim]</td>
          <td>√</td>
      </tr>
      <tr>
          <td>normType</td>
          <td>Optional input</td>
          <td>Normalization type for <code>query</code> and <code>key</code>.</td>
          <td><code>0</code>: no normalization; <code>1</code>: layer normalization; <code>2</code>: layer normalization with affine transformation parameters. If the value is not specified, <code>0</code> is recommended.</td>
          <td>INT64</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
      </tr>
      <tr>
          <td>normAddedType</td>
          <td>Optional input</td>
          <td>Normalization type for <code>encoderQuery</code> and <code>encoderKey</code>.</td>
          <td><code>0</code>: no normalization; <code>1</code>: layer normalization; <code>2</code>: layer normalization with affine transformation parameters. If the value is not specified, <code>0</code> is recommended.</td>
          <td>INT64</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
      </tr>
      <tr>
          <td>ropeType</td>
          <td>Optional input</td>
          <td>RoPE type after <code>query</code> and <code>encoderQuery</code>, and <code>key</code> and <code>encoderKey</code> are concatenated.</td>
          <td><code>0</code>: no RoPE; <code>1</code>: Interleave type; <code>2</code>: Half type. If the value is not specified, <code>0</code> is recommended.</td>
          <td>INT64</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
      </tr>
      <tr>
          <td>concatOrder</td>
          <td>Optional input</td>
          <td>Concatenation order for <code>query</code> and <code>encoderQuery</code>, <code>key</code> and <code>encoderKey</code>, and <code>value</code> and <code>encoderValue</code>.</td>
          <td>Use <code>query</code> as an example: <code>0</code>: [<code>query</code>, <code>encoderQuery</code>]; <code>1</code>: [<code>encoderQuery</code>, <code>query</code>]. If the value is not specified, <code>0</code> is recommended.</td>
          <td>INT64</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
      </tr>
      <tr>
          <td>gradQuery</td>
          <td>Output</td>
          <td>Backward gradient value of the forward input <code>query</code> in the formulas.</td>
          <td>The data type must be the same as that of <code>gradQueryOutput</code>.</td>
          <td>FLOAT32, FLOAT16, BFLOAT16</td>
          <td>ND</td>
          <td>[batch, seqQuery, headNum, headDim]</td>
          <td>x</td>
      </tr>
      <tr>
          <td>gradKey</td>
          <td>Output</td>
          <td>Backward gradient value of the forward input <code>key</code> in the formulas.</td>
          <td>The data type must be the same as that of <code>gradQueryOutput</code>.</td>
          <td>FLOAT32, FLOAT16, BFLOAT16</td>
          <td>ND</td>
          <td>[batch, seqKey, headNum, headDim]</td>
          <td>x</td>
      </tr>
      <tr>
          <td>gradValue</td>
          <td>Output</td>
          <td>Backward gradient value of the forward input <code>value</code> in the formulas.</td>
          <td>The data type must be the same as that of <code>gradQueryOutput</code>.</td>
          <td>FLOAT32, FLOAT16, BFLOAT16</td>
          <td>ND</td>
          <td>[batch, seqValue, headNum, headDim]</td>
          <td>x</td>
      </tr>
      <tr>
          <td>gradEncoderQuery</td>
          <td>Optional output</td>
          <td>Backward gradient value of the forward input <code>encoderQuery</code> in the formulas.</td>
          <td>This parameter is output when the text query is used for training. Its data type must be the same as that of <code>gradQueryOutput</code>.</td>
          <td>FLOAT32, FLOAT16, BFLOAT16</td>
          <td>ND</td>
          <td>[batch, seqEncoderQuery, headNum, headDim]</td>
          <td>x</td>
      </tr>
      <tr>
          <td>gradEncoderKey</td>
          <td>Optional output</td>
          <td>Backward gradient value of the forward input <code>encoderKey</code> in the formulas.</td>
          <td>This parameter is output when the text key is used for training. Its data type must be the same as that of <code>gradQueryOutput</code>.</td>
          <td>FLOAT32, FLOAT16, BFLOAT16</td>
          <td>ND</td>
          <td>[batch, seqEncoderKey, headNum, headDim]</td>
          <td>x</td>
      </tr>
      <tr>
          <td>gradEncoderValue</td>
          <td>Optional output</td>
          <td>Backward gradient value of the forward input <code>encoderValue</code> in the formulas.</td>
          <td>This parameter is output when the text value is used for training. Its data type must be the same as that of <code>gradQueryOutput</code>.</td>
          <td>FLOAT32, FLOAT16, BFLOAT16</td>
          <td>ND</td>
          <td>[batch, seqEncoderValue, headNum, headDim]</td>
          <td>x</td>
      </tr>
      <tr>
          <td>gradNormQueryWeight</td>
          <td>Optional output</td>
          <td>Backward gradient value of the weight (<code>γ</code>) for normalizing the forward input <code>query</code> in the formulas.</td>
          <td>This parameter is output when LayerNorm with affine transformation is performed on the image query and key. Its data type must be the same as that of <code>gradQueryOutput</code>.</td>
          <td>FLOAT32, FLOAT16, BFLOAT16</td>
          <td>ND</td>
          <td>[headDim]</td>
          <td>x</td>
      </tr>
      <tr>
          <td>gradNormQueryBias</td>
          <td>Optional output</td>
          <td>Backward gradient value of the bias (<code>β</code>) for normalizing the forward input <code>query</code> in the formulas.</td>
          <td>This parameter is output when LayerNorm with affine transformation is performed on the image query and key. Its data type must be the same as that of <code>gradQueryOutput</code>.</td>
          <td>FLOAT32, FLOAT16, BFLOAT16</td>
          <td>ND</td>
          <td>[headDim]</td>
          <td>x</td>
      </tr>
      <tr>
          <td>gradNormKeyWeight</td>
          <td>Optional output</td>
          <td>Backward gradient value of the weight (<code>γ</code>) for normalizing the forward input <code>key</code> in the formulas.</td>
          <td>This parameter is output when LayerNorm with affine transformation is performed on the image query and key. Its data type must be the same as that of <code>gradQueryOutput</code>.</td>
          <td>FLOAT32, FLOAT16, BFLOAT16</td>
          <td>ND</td>
          <td>[headDim]</td>
          <td>x</td>
      </tr>
      <tr>
          <td>gradNormKeyBias</td>
          <td>Optional output</td>
          <td>Backward gradient value of the bias (<code>β</code>) for normalizing the forward input <code>key</code> in the formulas.</td>
          <td>This parameter is output when LayerNorm with affine transformation is performed on the image query and key. Its data type must be the same as that of <code>gradQueryOutput</code>.</td>
          <td>FLOAT32, FLOAT16, BFLOAT16</td>
          <td>ND</td>
          <td>[headDim]</td>
          <td>x</td>
      </tr>
      <tr>
          <td>gradNormAddedQueryWeight</td>
          <td>Optional output</td>
          <td>Backward gradient value of the weight (<code>γ</code>) for normalizing the forward input <code>encoderQuery</code> in the formulas.</td>
          <td>This parameter is output when LayerNorm with affine transformation is performed on the text query and key. Its data type must be the same as that of <code>gradQueryOutput</code>.</td>
          <td>FLOAT32, FLOAT16, BFLOAT16</td>
          <td>ND</td>
          <td>[headDim]</td>
          <td>x</td>
      </tr>
      <tr>
          <td>gradNormAddedQueryBias</td>
          <td>Optional output</td>
          <td>Backward gradient value of the bias (<code>β</code>) for normalizing the forward input <code>encoderQuery</code> in the formulas.</td>
          <td>This parameter is output when LayerNorm with affine transformation is performed on the text query and key. Its data type must be the same as that of <code>gradQueryOutput</code>.</td>
          <td>FLOAT32, FLOAT16, BFLOAT16</td>
          <td>ND</td>
          <td>[headDim]</td>
          <td>x</td>
      </tr>
      <tr>
          <td>gradNormAddedKeyWeight</td>
          <td>Optional output</td>
          <td>Backward gradient value of the weight (<code>γ</code>) for normalizing the forward input <code>encoderKey</code> in the formulas.</td>
          <td>This parameter is output when LayerNorm with affine transformation is performed on the text query and key. Its data type must be the same as that of <code>gradQueryOutput</code>.</td>
          <td>FLOAT32, FLOAT16, BFLOAT16</td>
          <td>ND</td>
          <td>[headDim]</td>
          <td>x</td>
      </tr>
      <tr>
          <td>gradNormAddedKeyBias</td>
          <td>Optional output</td>
          <td>Backward gradient value of the bias (<code>β</code>) for normalizing the forward input <code>encoderKey</code> in the formulas.</td>
          <td>This parameter is output when LayerNorm with affine transformation is performed on the text query and key. Its data type must be the same as that of <code>gradQueryOutput</code>.</td>
          <td>FLOAT32, FLOAT16, BFLOAT16</td>
          <td>ND</td>
          <td>[headDim]</td>
          <td>x</td>
      </tr>
      <tr>
          <td>workspaceSize</td>
          <td>Output</td>
          <td>Size of the workspace required to be allocated on the device.</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
      </tr>
      <tr>
          <td>executor</td>
          <td>Output</td>
          <td>Operator executor, containing the operator computation process.</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
      </tr>
    </tbody>
  </table>

- **Returns**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API implements input parameter verification. The following errors may be thrown.
  <table style="undefined;table-layout: fixed;width: 1202px"><colgroup>
  <col style="width: 262px">
  <col style="width: 121px">
  <col style="width: 819px">
  </colgroup>
  <thead>
    <tr>
      <th>Return</th>
      <th>Error Code</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>Null pointers exist in the required computation input and output.</td>
    </tr>
    <tr>
      <td rowspan="2">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="2">161002</td>
      <td>The data type of the computation input or output is not supported.</td>
    </tr>
  </tbody>
  </table>

## aclnnNormRopeConcatBackward

- **Parameters**
  <table style="undefined;table-layout: fixed; width: 1202px"><colgroup>
  <col style="width: 262px">
  <col style="width: 121px">
  <col style="width: 819px">
  </colgroup>
  <thead>
    <tr>
      <th>Name</th>
      <th>Input/Output</th>
      <th>Description</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>workspace</td>
      <td>Input</td>
      <td>Address of the workspace to be allocated on the device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace to be allocated on the device, which is obtained by calling the first-phase API <code>aclnnNormRopeConcatBackwardGetWorkspaceSize</code>.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Operator executor, containing the operator computation process.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream for executing the task.</td>
    </tr>
  </tbody>
  </table>

- **Returns**

    `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - `aclnnNormRopeConcatBackward` defaults to a non-deterministic implementation. You can call `aclrtCtxSetSysParamOpt` to enable deterministic computation.

## Example

- Single-aclnn-operator calling

  The following is an example of aclnn single-operator calling. For details about the compilation and execution process, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

    ```c++
    #include <algorithm>
    #include <cstdint>
    #include <iostream>
    #include <vector>
    #include <sys/types.h>
    #include <sys/stat.h>
    #include <unistd.h>
    #include <fstream>
    #include <fcntl.h>

    #include "acl/acl.h"
    #include "aclnnop/aclnn_norm_rope_concat_grad.h"

    #define SUCCESS 0
    #define FAILED 1

    #define INFO_LOG(fmt, args...) fprintf(stdout, "[INFO]  " fmt "\n", ##args)
    #define WARN_LOG(fmt, args...) fprintf(stdout, "[WARN]  " fmt "\n", ##args)
    #define ERROR_LOG(fmt, args...) fprintf(stderr, "[ERROR]  " fmt "\n", ##args)

    #define CHECK_RET(cond, return_expr) \
    do {                               \
        if (!(cond)) {                   \
        return_expr;                   \
        }                                \
    } while (0)

    #define LOG_PRINT(message, ...)     \
    do {                              \
        printf(message, ##__VA_ARGS__); \
    } while (0)

    bool ReadFile(const std::string &filePath, size_t &fileSize, void *buffer, size_t bufferSize)
    {
        struct stat sBuf;
        int fileStatus = stat(filePath.data(), &sBuf);
        if (fileStatus == -1) {
            ERROR_LOG("failed to get file %s", filePath.c_str());
            return false;
        }
        if (S_ISREG(sBuf.st_mode) == 0) {
            ERROR_LOG("%s is not a file, please enter a file", filePath.c_str());
            return false;
        }

        std::ifstream file;
        file.open(filePath, std::ios::binary);
        if (!file.is_open()) {
            ERROR_LOG("Open file failed. path = %s", filePath.c_str());
            return false;
        }

        std::filebuf *buf = file.rdbuf();
        size_t size = buf->pubseekoff(0, std::ios::end, std::ios::in);
        if (size == 0) {
            ERROR_LOG("file size is 0");
            file.close();
            return false;
        }
        if (size > bufferSize) {
            ERROR_LOG("file size is larger than buffer size");
            file.close();
            return false;
        }
        buf->pubseekpos(0, std::ios::in);
        buf->sgetn(static_cast<char *>(buffer), size);
        fileSize = size;
        file.close();
        return true;
    }

    int64_t GetShapeSize(const std::vector<int64_t>& shape) {
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
    }

    int Init(int32_t deviceId, aclrtContext* context, aclrtStream* stream) {
        // (Fixed writing) Initialize AscendCL.
        auto ret = aclInit(nullptr);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
        ret = aclrtSetDevice(deviceId);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
        ret = aclrtCreateContext(context, deviceId);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateContext failed. ERROR: %d\n", ret); return ret);
        ret = aclrtSetCurrentContext(*context);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetCurrentContext failed. ERROR: %d\n", ret); return ret);
        ret = aclrtCreateStream(stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
        return 0;
    }

    template <typename T>
    int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                        aclDataType dataType, aclTensor** xOrResult) {
        auto size = GetShapeSize(shape) * sizeof(T);
        // Call aclrtMalloc to allocate memory on the device.
        auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
        // Call aclrtMemcpy to copy host data to the device memory.
        ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

        // Compute the strides of the contiguous xOrResult.
        std::vector<int64_t> strides(shape.size(), 1);
        for (int64_t i = shape.size() - 2; i >= 0; i--) {
            strides[i] = shape[i + 1] * strides[i + 1];
    }

    // Call aclCreateTensor to create an aclTensor.
        *xOrResult = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                                    shape.data(), shape.size(), *deviceAddr);
    return 0;
    }

    int main() {
        // 1. (Fixed writing) Initialize the device, context, and stream. For details, see the list of external AscendCL APIs.
        // Set the device ID (deviceId) based on the actual device.
        int32_t deviceId = 0;
        aclrtContext context;
        aclrtStream stream;
        auto ret = Init(deviceId, &context, &stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

        // 2. Construct the inputs and outputs based on the API definition.
        uint32_t batchSize = 1;
        uint32_t headNum = 10;
        uint32_t headDim = 6;
        uint32_t querySeq = 5;
        uint32_t keySeq = 5;
        uint32_t valueSeq = 5;
        uint32_t encoderQuerySeq = 3;
        uint32_t encoderKeySeq = 3;
        uint32_t encoderValueSeq = 3;
        uint32_t ropeSeq = 5;
        uint32_t ropeDim = 6;
        uint32_t normType = 2;
        uint32_t normAddedType = 2;
        uint32_t ropeType = 1;
        uint32_t concatOrder = 0;

        std::vector<int64_t> gradQueryOutputShape =  {batchSize, headNum, querySeq + encoderQuerySeq, headDim};
        std::vector<int64_t> gradKeyOutputShape =  {batchSize, headNum, keySeq + encoderKeySeq, headDim};
        std::vector<int64_t> gradValueOutputShape =  {batchSize, headNum, valueSeq + encoderValueSeq, headDim};
        std::vector<int64_t> queryShape =  {batchSize, querySeq, headNum, headDim};
        std::vector<int64_t> keyShape =  {batchSize, keySeq, headNum, headDim};
        std::vector<int64_t> encoderQueryShape =  {batchSize, encoderQuerySeq, headNum, headDim};
        std::vector<int64_t> encoderKeyShape =  {batchSize, encoderKeySeq, headNum, headDim};
        std::vector<int64_t> normQueryWeightShape =  {headDim};
        std::vector<int64_t> normQueryMeanShape =  {batchSize, querySeq, headNum, 1};
        std::vector<int64_t> normQueryRstdShape =  {batchSize, querySeq, headNum, 1};
        std::vector<int64_t> normKeyWeightShape =  {headDim};
        std::vector<int64_t> normKeyMeanShape =  {batchSize, keySeq, headNum, 1};
        std::vector<int64_t> normKeyRstdShape =  {batchSize, keySeq, headNum, 1};
        std::vector<int64_t> normAddedQueryWeightShape =  {headDim};
        std::vector<int64_t> normAddedQueryMeanShape =  {batchSize, encoderQuerySeq, headNum, 1};
        std::vector<int64_t> normAddedQueryRstdShape =  {batchSize, encoderQuerySeq, headNum, 1};
        std::vector<int64_t> normAddedKeyWeightShape =  {headDim};
        std::vector<int64_t> normAddedKeyMeanShape =  {batchSize, encoderKeySeq, headNum, 1};
        std::vector<int64_t> normAddedKeyRstdShape =  {batchSize, encoderKeySeq, headNum, 1};
        std::vector<int64_t> ropeSinShape =  {ropeSeq, headDim};
        std::vector<int64_t> ropeCosShape =  {ropeSeq, headDim};
        std::vector<int64_t> gradQueryShape =  {batchSize, querySeq, headNum, headDim};
        std::vector<int64_t> gradKeyShape =  {batchSize, keySeq, headNum, headDim};
        std::vector<int64_t> gradValueShape =  {batchSize, valueSeq, headNum, headDim};
        std::vector<int64_t> gradEncoderQueryShape =  {batchSize, encoderQuerySeq, headNum, headDim};
        std::vector<int64_t> gradEncoderKeyShape =  {batchSize, encoderKeySeq, headNum, headDim};
        std::vector<int64_t> gradEncoderValueShape =  {batchSize, encoderValueSeq, headNum, headDim};
        std::vector<int64_t> gradNormQueryWeightShape =  {headDim};
        std::vector<int64_t> gradNormQueryBiasShape =  {headDim};
        std::vector<int64_t> gradNormKeyWeightShape =  {headDim};
        std::vector<int64_t> gradNormKeyBiasShape =  {headDim};
        std::vector<int64_t> gradNormAddedQueryWeightShape =  {headDim};
        std::vector<int64_t> gradNormAddedQueryBiasShape =  {headDim};
        std::vector<int64_t> gradNormAddedKeyWeightShape =  {headDim};
        std::vector<int64_t> gradNormAddedKeyBiasShape =  {headDim};

        void* gradQueryOutputDeviceAddr =  nullptr;
        void* gradKeyOutputDeviceAddr =  nullptr;
        void* gradValueOutputDeviceAddr =  nullptr;
        void* queryDeviceAddr =  nullptr;
        void* keyDeviceAddr =  nullptr;
        void* encoderQueryDeviceAddr =  nullptr;
        void* encoderKeyDeviceAddr =  nullptr;
        void* normQueryWeightDeviceAddr =  nullptr;
        void* normQueryMeanDeviceAddr =  nullptr;
        void* normQueryRstdDeviceAddr =  nullptr;
        void* normKeyWeightDeviceAddr =  nullptr;
        void* normKeyMeanDeviceAddr =  nullptr;
        void* normKeyRstdDeviceAddr =  nullptr;
        void* normAddedQueryWeightDeviceAddr =  nullptr;
        void* normAddedQueryMeanDeviceAddr =  nullptr;
        void* normAddedQueryRstdDeviceAddr =  nullptr;
        void* normAddedKeyWeightDeviceAddr =  nullptr;
        void* normAddedKeyMeanDeviceAddr =  nullptr;
        void* normAddedKeyRstdDeviceAddr =  nullptr;
        void* ropeSinDeviceAddr =  nullptr;
        void* ropeCosDeviceAddr =  nullptr;
        void* gradQueryDeviceAddr =  nullptr;
        void* gradKeyDeviceAddr =  nullptr;
        void* gradValueDeviceAddr =  nullptr;
        void* gradEncoderQueryDeviceAddr =  nullptr;
        void* gradEncoderKeyDeviceAddr =  nullptr;
        void* gradEncoderValueDeviceAddr =  nullptr;
        void* gradNormQueryWeightDeviceAddr =  nullptr;
        void* gradNormQueryBiasDeviceAddr =  nullptr;
        void* gradNormKeyWeightDeviceAddr =  nullptr;
        void* gradNormKeyBiasDeviceAddr =  nullptr;
        void* gradNormAddedQueryWeightDeviceAddr =  nullptr;
        void* gradNormAddedQueryBiasDeviceAddr =  nullptr;
        void* gradNormAddedKeyWeightDeviceAddr =  nullptr;
        void* gradNormAddedKeyBiasDeviceAddr =  nullptr;

        aclTensor* gradQueryOutput =  nullptr;
        aclTensor* gradKeyOutput =  nullptr;
        aclTensor* gradValueOutput =  nullptr;
        aclTensor* query =  nullptr;
        aclTensor* key =  nullptr;
        aclTensor* encoderQuery =  nullptr;
        aclTensor* encoderKey =  nullptr;
        aclTensor* normQueryWeight =  nullptr;
        aclTensor* normQueryMean =  nullptr;
        aclTensor* normQueryRstd =  nullptr;
        aclTensor* normKeyWeight =  nullptr;
        aclTensor* normKeyMean =  nullptr;
        aclTensor* normKeyRstd =  nullptr;
        aclTensor* normAddedQueryWeight =  nullptr;
        aclTensor* normAddedQueryMean =  nullptr;
        aclTensor* normAddedQueryRstd =  nullptr;
        aclTensor* normAddedKeyWeight =  nullptr;
        aclTensor* normAddedKeyMean =  nullptr;
        aclTensor* normAddedKeyRstd =  nullptr;
        aclTensor* ropeSin =  nullptr;
        aclTensor* ropeCos =  nullptr;
        aclTensor* gradQuery =  nullptr;
        aclTensor* gradKey =  nullptr;
        aclTensor* gradValue =  nullptr;
        aclTensor* gradEncoderQuery =  nullptr;
        aclTensor* gradEncoderKey =  nullptr;
        aclTensor* gradEncoderValue =  nullptr;
        aclTensor* gradNormQueryWeight =  nullptr;
        aclTensor* gradNormQueryBias =  nullptr;
        aclTensor* gradNormKeyWeight =  nullptr;
        aclTensor* gradNormKeyBias =  nullptr;
        aclTensor* gradNormAddedQueryWeight =  nullptr;
        aclTensor* gradNormAddedQueryBias =  nullptr;
        aclTensor* gradNormAddedKeyWeight =  nullptr;
        aclTensor* gradNormAddedKeyBias =  nullptr;

        std::vector<float> gradQueryOutputHostData(batchSize * headNum * (querySeq + encoderQuerySeq) * headDim, 1.0);
        std::vector<float> gradKeyOutputHostData(batchSize * headNum * (keySeq + encoderKeySeq) * headDim, 1.0);
        std::vector<float> gradValueOutputHostData(batchSize * headNum * (valueSeq + encoderValueSeq) * headDim, 1.0);
        std::vector<float> queryHostData(batchSize * headNum * querySeq * headDim, 1.0);
        std::vector<float> keyHostData(batchSize * headNum * keySeq * headDim, 1.0);
        std::vector<float> encoderQueryHostData(batchSize * headNum * encoderQuerySeq * headDim, 1.0);
        std::vector<float> encoderKeyHostData(batchSize * headNum * encoderKeySeq * headDim, 1.0);
        std::vector<float> normQueryWeightHostData(headDim, 1.0);
        std::vector<float> normQueryMeanHostData(batchSize * headNum * querySeq * 1, 0.0);
        std::vector<float> normQueryRstdHostData(batchSize * headNum * querySeq * 1, 1.0);
        std::vector<float> normKeyWeightHostData(headDim, 1.0);
        std::vector<float> normKeyMeanHostData(batchSize * headNum * keySeq * 1, 0.0);
        std::vector<float> normKeyRstdHostData(batchSize * headNum * keySeq * 1, 1.0);
        std::vector<float> normAddedQueryWeightHostData(headDim, 1.0);
        std::vector<float> normAddedQueryMeanHostData(batchSize * headNum * encoderQuerySeq * 1, 1.0);
        std::vector<float> normAddedQueryRstdHostData(batchSize * headNum * encoderQuerySeq * 1, 1.0);
        std::vector<float> normAddedKeyWeightHostData(headDim, 1.0);
        std::vector<float> normAddedKeyMeanHostData(batchSize * headNum * encoderKeySeq * 1, 0.0);
        std::vector<float> normAddedKeyRstdHostData(batchSize * headNum * encoderKeySeq * 1, 1.0);
        std::vector<float> ropeSinHostData(ropeSeq * headDim, 1.0);
        std::vector<float> ropeCosHostData(ropeSeq * headDim, 1.0);
        std::vector<float> gradQueryHostData(batchSize * headNum * querySeq * headDim, 0.0);
        std::vector<float> gradKeyHostData(batchSize * headNum * keySeq * headDim, 0.0);
        std::vector<float> gradValueHostData(batchSize * headNum * valueSeq * headDim, 0.0);
        std::vector<float> gradEncoderQueryHostData(batchSize * headNum * encoderQuerySeq * headDim, 0.0);
        std::vector<float> gradEncoderKeyHostData(batchSize * headNum * encoderKeySeq * headDim, 0.0);
        std::vector<float> gradEncoderValueHostData(batchSize * headNum * encoderValueSeq * headDim, 0.0);
        std::vector<float> gradNormQueryWeightHostData(headDim, 0.0);
        std::vector<float> gradNormQueryBiasHostData(headDim, 0.0);
        std::vector<float> gradNormKeyWeightHostData(headDim, 0.0);
        std::vector<float> gradNormKeyBiasHostData(headDim, 0.0);
        std::vector<float> gradNormAddedQueryWeightHostData(headDim, 0.0);
        std::vector<float> gradNormAddedQueryBiasHostData(headDim, 0.0);
        std::vector<float> gradNormAddedKeyWeightHostData(headDim, 0.0);
        std::vector<float> gradNormAddedKeyBiasHostData(headDim, 0.0);

        ret = CreateAclTensor(gradQueryOutputHostData, gradQueryOutputShape, &gradQueryOutputDeviceAddr, aclDataType::ACL_FLOAT, &gradQueryOutput);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(gradKeyOutputHostData, gradKeyOutputShape, &gradKeyOutputDeviceAddr, aclDataType::ACL_FLOAT, &gradKeyOutput);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(gradValueOutputHostData, gradValueOutputShape, &gradValueOutputDeviceAddr, aclDataType::ACL_FLOAT, &gradValueOutput);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(queryHostData, queryShape, &queryDeviceAddr, aclDataType::ACL_FLOAT, &query);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(keyHostData, keyShape, &keyDeviceAddr, aclDataType::ACL_FLOAT, &key);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(encoderQueryHostData, encoderQueryShape, &encoderQueryDeviceAddr, aclDataType::ACL_FLOAT, &encoderQuery);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(encoderKeyHostData, encoderKeyShape, &encoderKeyDeviceAddr, aclDataType::ACL_FLOAT, &encoderKey);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(normQueryWeightHostData, normQueryWeightShape, &normQueryWeightDeviceAddr, aclDataType::ACL_FLOAT, &normQueryWeight);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(normQueryMeanHostData, normQueryMeanShape, &normQueryMeanDeviceAddr, aclDataType::ACL_FLOAT, &normQueryMean);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(normQueryRstdHostData, normQueryRstdShape, &normQueryRstdDeviceAddr, aclDataType::ACL_FLOAT, &normQueryRstd);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(normKeyWeightHostData, normKeyWeightShape, &normKeyWeightDeviceAddr, aclDataType::ACL_FLOAT, &normKeyWeight);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(normKeyMeanHostData, normKeyMeanShape, &normKeyMeanDeviceAddr, aclDataType::ACL_FLOAT, &normKeyMean);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(normKeyRstdHostData, normKeyRstdShape, &normKeyRstdDeviceAddr, aclDataType::ACL_FLOAT, &normKeyRstd);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(normAddedQueryWeightHostData, normAddedQueryWeightShape, &normAddedQueryWeightDeviceAddr, aclDataType::ACL_FLOAT, &normAddedQueryWeight);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(normAddedQueryMeanHostData, normAddedQueryMeanShape, &normAddedQueryMeanDeviceAddr, aclDataType::ACL_FLOAT, &normAddedQueryMean);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(normAddedQueryRstdHostData, normAddedQueryRstdShape, &normAddedQueryRstdDeviceAddr, aclDataType::ACL_FLOAT, &normAddedQueryRstd);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(normAddedKeyWeightHostData, normAddedKeyWeightShape, &normAddedKeyWeightDeviceAddr, aclDataType::ACL_FLOAT, &normAddedKeyWeight);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(normAddedKeyMeanHostData, normAddedKeyMeanShape, &normAddedKeyMeanDeviceAddr, aclDataType::ACL_FLOAT, &normAddedKeyMean);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(normAddedKeyRstdHostData, normAddedKeyRstdShape, &normAddedKeyRstdDeviceAddr, aclDataType::ACL_FLOAT, &normAddedKeyRstd);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(ropeSinHostData, ropeSinShape, &ropeSinDeviceAddr, aclDataType::ACL_FLOAT, &ropeSin);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(ropeCosHostData, ropeCosShape, &ropeCosDeviceAddr, aclDataType::ACL_FLOAT, &ropeCos);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(gradQueryHostData, gradQueryShape, &gradQueryDeviceAddr, aclDataType::ACL_FLOAT, &gradQuery);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(gradKeyHostData, gradKeyShape, &gradKeyDeviceAddr, aclDataType::ACL_FLOAT, &gradKey);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(gradValueHostData, gradValueShape, &gradValueDeviceAddr, aclDataType::ACL_FLOAT, &gradValue);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(gradEncoderQueryHostData, gradEncoderQueryShape, &gradEncoderQueryDeviceAddr, aclDataType::ACL_FLOAT, &gradEncoderQuery);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(gradEncoderKeyHostData, gradEncoderKeyShape, &gradEncoderKeyDeviceAddr, aclDataType::ACL_FLOAT, &gradEncoderKey);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(gradEncoderValueHostData, gradEncoderValueShape, &gradEncoderValueDeviceAddr, aclDataType::ACL_FLOAT, &gradEncoderValue);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(gradNormQueryWeightHostData, gradNormQueryWeightShape, &gradNormQueryWeightDeviceAddr, aclDataType::ACL_FLOAT, &gradNormQueryWeight);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(gradNormQueryBiasHostData, gradNormQueryBiasShape, &gradNormQueryBiasDeviceAddr, aclDataType::ACL_FLOAT, &gradNormQueryBias);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(gradNormKeyWeightHostData, gradNormKeyWeightShape, &gradNormKeyWeightDeviceAddr, aclDataType::ACL_FLOAT, &gradNormKeyWeight);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(gradNormKeyBiasHostData, gradNormKeyBiasShape, &gradNormKeyBiasDeviceAddr, aclDataType::ACL_FLOAT, &gradNormKeyBias);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(gradNormAddedQueryWeightHostData, gradNormAddedQueryWeightShape, &gradNormAddedQueryWeightDeviceAddr, aclDataType::ACL_FLOAT, &gradNormAddedQueryWeight);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(gradNormAddedQueryBiasHostData, gradNormAddedQueryBiasShape, &gradNormAddedQueryBiasDeviceAddr, aclDataType::ACL_FLOAT, &gradNormAddedQueryBias);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(gradNormAddedKeyWeightHostData, gradNormAddedKeyWeightShape, &gradNormAddedKeyWeightDeviceAddr, aclDataType::ACL_FLOAT, &gradNormAddedKeyWeight);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(gradNormAddedKeyBiasHostData, gradNormAddedKeyBiasShape, &gradNormAddedKeyBiasDeviceAddr, aclDataType::ACL_FLOAT, &gradNormAddedKeyBias);
        CHECK_RET(ret == ACL_SUCCESS, return ret);

        // 3. Call the CANN operator library API. Change the API name to the actual one.
        uint64_t workspaceSize = 0;
        aclOpExecutor* executor;
        // Call the first-phase API of aclnnGeGluBackward.
        ret = aclnnNormRopeConcatBackwardGetWorkspaceSize(
                gradQueryOutput, gradKeyOutput, gradValueOutput, query, key, encoderQuery, encoderKey,
                normQueryWeight, normQueryMean, normQueryRstd, normKeyWeight, normKeyMean, normKeyRstd,
                normAddedQueryWeight, normAddedQueryMean, normAddedQueryRstd, normAddedKeyWeight, normAddedKeyMean,
                normAddedKeyRstd, ropeSin, ropeCos, normType, normAddedType, ropeType, 
                concatOrder, gradQuery, gradKey, gradValue, gradEncoderQuery, gradEncoderKey, 
                gradEncoderValue, gradNormQueryWeight, gradNormQueryBias, gradNormKeyWeight, gradNormKeyBias,
                gradNormAddedQueryWeight, gradNormAddedQueryBias, gradNormAddedKeyWeight, gradNormAddedKeyBias, 
                & workspaceSize, & executor);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNormRopeConcatBackwardGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
        // Allocate device memory based on workspaceSize computed by the first-phase API.
        void* workspaceAddr = nullptr;
        if (workspaceSize > static_cast<uint64_t>(0)) {
            ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
        }
        // Call the second-phase API of aclnnGeGluBackward.
        ret = aclnnNormRopeConcatBackward(workspaceAddr, workspaceSize, executor, stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNormRopeConcatBackward failed. ERROR: %d\n", ret); return ret);

        // 4. (Fixed writing) Wait until the task execution is complete.
        ret = aclrtSynchronizeStream(stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

        // 5. Obtain the output value and copy the result from the device memory to the host. Modify the code based on the API definition.
        auto size = GetShapeSize(gradQueryShape);
        std::vector<float> gradQueryData(size, 0);
        ret = aclrtMemcpy(gradQueryData.data(), gradQueryData.size() * sizeof(gradQueryData[0]), gradQueryDeviceAddr,
                            size * sizeof(gradQueryData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy gradQuery result from device to host failed. ERROR: %d\n", ret); return ret);
        for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("gradQuery result[%ld] is: %f\n", i, gradQueryData[i]);
        }
        size = GetShapeSize(gradKeyShape);
        std::vector<float> gradKeyData(size, 0);
        ret = aclrtMemcpy(gradKeyData.data(), gradKeyData.size() * sizeof(gradKeyData[0]), gradKeyDeviceAddr,
                            size * sizeof(gradKeyData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy gradKey result from device to host failed. ERROR: %d\n", ret); return ret);
        for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("gradKey result[%ld] is: %f\n", i, gradKeyData[i]);
        }
        size = GetShapeSize(gradValueShape);
        std::vector<float> gradValueData(size, 0);
        ret = aclrtMemcpy(gradValueData.data(), gradValueData.size() * sizeof(gradValueData[0]), gradValueDeviceAddr,
                            size * sizeof(gradValueData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy gradValue result from device to host failed. ERROR: %d\n", ret); return ret);
        for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("gradValue result[%ld] is: %f\n", i, gradValueData[i]);
        }
        size = GetShapeSize(gradEncoderQueryShape);
        std::vector<float> gradEncoderQueryData(size, 0);
        ret = aclrtMemcpy(gradEncoderQueryData.data(), gradEncoderQueryData.size() * sizeof(gradEncoderQueryData[0]), gradEncoderQueryDeviceAddr,
                            size * sizeof(gradEncoderQueryData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy gradEncoderQuery result from device to host failed. ERROR: %d\n", ret); return ret);
        for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("gradEncoderQuery result[%ld] is: %f\n", i, gradEncoderQueryData[i]);
        }
        size = GetShapeSize(gradEncoderKeyShape);
        std::vector<float> gradEncoderKeyData(size, 0);
        ret = aclrtMemcpy(gradEncoderKeyData.data(), gradEncoderKeyData.size() * sizeof(gradEncoderKeyData[0]), gradEncoderKeyDeviceAddr,
                            size * sizeof(gradEncoderKeyData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy gradEncoderKey result from device to host failed. ERROR: %d\n", ret); return ret);
        for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("gradEncoderKey result[%ld] is: %f\n", i, gradEncoderKeyData[i]);
        }
        size = GetShapeSize(gradEncoderValueShape);
        std::vector<float> gradEncoderValueData(size, 0);
        ret = aclrtMemcpy(gradEncoderValueData.data(), gradEncoderValueData.size() * sizeof(gradEncoderValueData[0]), gradEncoderValueDeviceAddr,
                            size * sizeof(gradEncoderValueData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy gradEncoderValue result from device to host failed. ERROR: %d\n", ret); return ret);
        for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("gradEncoderValue result[%ld] is: %f\n", i, gradEncoderValueData[i]);
        }
        size = GetShapeSize(gradNormQueryWeightShape);
        std::vector<float> gradNormQueryWeightData(size, 0);
        ret = aclrtMemcpy(gradNormQueryWeightData.data(), gradNormQueryWeightData.size() * sizeof(gradNormQueryWeightData[0]), gradNormQueryWeightDeviceAddr,
                            size * sizeof(gradNormQueryWeightData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy gradNormQueryWeight result from device to host failed. ERROR: %d\n", ret); return ret);
        for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("gradNormQueryWeight result[%ld] is: %f\n", i, gradNormQueryWeightData[i]);
        }
        size = GetShapeSize(gradNormQueryBiasShape);
        std::vector<float> gradNormQueryBiasData(size, 0);
        ret = aclrtMemcpy(gradNormQueryBiasData.data(), gradNormQueryBiasData.size() * sizeof(gradNormQueryBiasData[0]), gradNormQueryBiasDeviceAddr,
                            size * sizeof(gradNormQueryBiasData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy gradNormQueryBias result from device to host failed. ERROR: %d\n", ret); return ret);
        for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("gradNormQueryBias result[%ld] is: %f\n", i, gradNormQueryBiasData[i]);
        }
        size = GetShapeSize(gradNormKeyWeightShape);
        std::vector<float> gradNormKeyWeightData(size, 0);
        ret = aclrtMemcpy(gradNormKeyWeightData.data(), gradNormKeyWeightData.size() * sizeof(gradNormKeyWeightData[0]), gradNormKeyWeightDeviceAddr,
                            size * sizeof(gradNormKeyWeightData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy gradNormKeyWeight result from device to host failed. ERROR: %d\n", ret); return ret);
        for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("gradNormKeyWeight result[%ld] is: %f\n", i, gradNormKeyWeightData[i]);
        }
        size = GetShapeSize(gradNormKeyBiasShape);
        std::vector<float> gradNormKeyBiasData(size, 0);
        ret = aclrtMemcpy(gradNormKeyBiasData.data(), gradNormKeyBiasData.size() * sizeof(gradNormKeyBiasData[0]), gradNormKeyBiasDeviceAddr,
                            size * sizeof(gradNormKeyBiasData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy gradNormKeyBias result from device to host failed. ERROR: %d\n", ret); return ret);
        for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("gradNormKeyBias result[%ld] is: %f\n", i, gradNormKeyBiasData[i]);
        }
        size = GetShapeSize(gradNormAddedQueryWeightShape);
        std::vector<float> gradNormAddedQueryWeightData(size, 0);
        ret = aclrtMemcpy(gradNormAddedQueryWeightData.data(), gradNormAddedQueryWeightData.size() * sizeof(gradNormAddedQueryWeightData[0]), gradNormAddedQueryWeightDeviceAddr,
                            size * sizeof(gradNormAddedQueryWeightData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy gradNormAddedQueryWeight result from device to host failed. ERROR: %d\n", ret); return ret);
        for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("gradNormAddedQueryWeight result[%ld] is: %f\n", i, gradNormAddedQueryWeightData[i]);
        }
        size = GetShapeSize(gradNormAddedQueryBiasShape);
        std::vector<float> gradNormAddedQueryBiasData(size, 0);
        ret = aclrtMemcpy(gradNormAddedQueryBiasData.data(), gradNormAddedQueryBiasData.size() * sizeof(gradNormAddedQueryBiasData[0]), gradNormAddedQueryBiasDeviceAddr,
                            size * sizeof(gradNormAddedQueryBiasData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy gradNormAddedQueryBias result from device to host failed. ERROR: %d\n", ret); return ret);
        for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("gradNormAddedQueryBias result[%ld] is: %f\n", i, gradNormAddedQueryBiasData[i]);
        }
        size = GetShapeSize(gradNormAddedKeyWeightShape);
        std::vector<float> gradNormAddedKeyWeightData(size, 0);
        ret = aclrtMemcpy(gradNormAddedKeyWeightData.data(), gradNormAddedKeyWeightData.size() * sizeof(gradNormAddedKeyWeightData[0]), gradNormAddedKeyWeightDeviceAddr,
                            size * sizeof(gradNormAddedKeyWeightData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy gradNormAddedKeyWeight result from device to host failed. ERROR: %d\n", ret); return ret);
        for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("gradNormAddedKeyWeight result[%ld] is: %f\n", i, gradNormAddedKeyWeightData[i]);
        }
        size = GetShapeSize(gradNormAddedKeyBiasShape);
        std::vector<float> gradNormAddedKeyBiasData(size, 0);
        ret = aclrtMemcpy(gradNormAddedKeyBiasData.data(), gradNormAddedKeyBiasData.size() * sizeof(gradNormAddedKeyBiasData[0]), gradNormAddedKeyBiasDeviceAddr,
                            size * sizeof(gradNormAddedKeyBiasData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy gradNormAddedKeyBias result from device to host failed. ERROR: %d\n", ret); return ret);
        for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("gradNormAddedKeyBias result[%ld] is: %f\n", i, gradNormAddedKeyBiasData[i]);
        }

        // 6. Release aclTensors and aclScalars. Modify the code based on the API definition.
        aclDestroyTensor(gradQueryOutput);
        aclDestroyTensor(gradKeyOutput);
        aclDestroyTensor(gradValueOutput);
        aclDestroyTensor(query);
        aclDestroyTensor(key);
        aclDestroyTensor(encoderQuery);
        aclDestroyTensor(encoderKey);
        aclDestroyTensor(normQueryWeight);
        aclDestroyTensor(normQueryMean);
        aclDestroyTensor(normQueryRstd);
        aclDestroyTensor(normKeyWeight);
        aclDestroyTensor(normKeyMean);
        aclDestroyTensor(normKeyRstd);
        aclDestroyTensor(normAddedQueryWeight);
        aclDestroyTensor(normAddedQueryMean);
        aclDestroyTensor(normAddedQueryRstd);
        aclDestroyTensor(normAddedKeyWeight);
        aclDestroyTensor(normAddedKeyMean);
        aclDestroyTensor(normAddedKeyRstd);
        aclDestroyTensor(ropeSin);
        aclDestroyTensor(ropeCos);
        aclDestroyTensor(gradQuery);
        aclDestroyTensor(gradKey);
        aclDestroyTensor(gradValue);
        aclDestroyTensor(gradEncoderQuery);
        aclDestroyTensor(gradEncoderKey);
        aclDestroyTensor(gradEncoderValue);
        aclDestroyTensor(gradNormQueryWeight);
        aclDestroyTensor(gradNormQueryBias);
        aclDestroyTensor(gradNormKeyWeight);
        aclDestroyTensor(gradNormKeyBias);
        aclDestroyTensor(gradNormAddedQueryWeight);
        aclDestroyTensor(gradNormAddedQueryBias);
        aclDestroyTensor(gradNormAddedKeyWeight);
        aclDestroyTensor(gradNormAddedKeyBias);

        // 7. Release device resources. Modify the code based on the API definition.
        aclrtFree(gradQueryOutputDeviceAddr);
        aclrtFree(gradKeyOutputDeviceAddr);
        aclrtFree(gradValueOutputDeviceAddr);
        aclrtFree(queryDeviceAddr);
        aclrtFree(keyDeviceAddr);
        aclrtFree(encoderQueryDeviceAddr);
        aclrtFree(encoderKeyDeviceAddr);
        aclrtFree(normQueryWeightDeviceAddr);
        aclrtFree(normQueryMeanDeviceAddr);
        aclrtFree(normQueryRstdDeviceAddr);
        aclrtFree(normKeyWeightDeviceAddr);
        aclrtFree(normKeyMeanDeviceAddr);
        aclrtFree(normKeyRstdDeviceAddr);
        aclrtFree(normAddedQueryWeightDeviceAddr);
        aclrtFree(normAddedQueryMeanDeviceAddr);
        aclrtFree(normAddedQueryRstdDeviceAddr);
        aclrtFree(normAddedKeyWeightDeviceAddr);
        aclrtFree(normAddedKeyMeanDeviceAddr);
        aclrtFree(normAddedKeyRstdDeviceAddr);
        aclrtFree(ropeSinDeviceAddr);
        aclrtFree(ropeCosDeviceAddr);
        aclrtFree(gradQueryDeviceAddr);
        aclrtFree(gradKeyDeviceAddr);
        aclrtFree(gradValueDeviceAddr);
        aclrtFree(gradEncoderQueryDeviceAddr);
        aclrtFree(gradEncoderKeyDeviceAddr);
        aclrtFree(gradEncoderValueDeviceAddr);
        aclrtFree(gradNormQueryWeightDeviceAddr);
        aclrtFree(gradNormQueryBiasDeviceAddr);
        aclrtFree(gradNormKeyWeightDeviceAddr);
        aclrtFree(gradNormKeyBiasDeviceAddr);
        aclrtFree(gradNormAddedQueryWeightDeviceAddr);
        aclrtFree(gradNormAddedQueryBiasDeviceAddr);
        aclrtFree(gradNormAddedKeyWeightDeviceAddr);
        aclrtFree(gradNormAddedKeyBiasDeviceAddr);
        if (workspaceSize > static_cast<uint64_t>(0)) {
            aclrtFree(workspaceAddr);
        }
        aclrtDestroyStream(stream);
        aclrtDestroyContext(context);
        aclrtResetDevice(deviceId);
        aclFinalize();
        return 0;
    }
    ```
