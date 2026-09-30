# aclnnFusedInferAttentionScoreV3

## Supported Products

|Product     | Supported|
|:----------------------------|:-----------:|
|<term>Atlas A3 training products/Atlas A3 inference products</term>|      ×     |
|<term>Atlas A2 training products/Atlas A2 inference products</term>|      √     |
|<term>Atlas 200I/500 A2 inference products</term>|      ×     |
|<term>Atlas inference accelerator cards</term>|      ×     |
|<term>Atlas training products</term>|      ×     |

## Function

- Adapts to the `FlashAttention` operator in the incremental (`IncreFlashAttention`) and full (`PromptFlashAttention`) inference scenarios. When `S` of the `Query` matrix is `1`, the `IncreFlashAttention` branch is executed. In other scenarios, the `PromptFlashAttention` branch is executed. Compared with `FusedInferAttentionScoreV2`, this API introduces new parameters: `queryRopeOptional`, `keyRopeOptional`, and `keyRopeAntiquantScaleOptional`.

- Formula:

  Self-attention constructs an attention model by leveraging the relationships within the input samples. The principle assumes there is an input sample sequence $x$ of length $n$, where each element of $x$ is a $d$-dimensional vector. Each $d$-dimensional vector can be regarded as a token embedding. Such a sequence is transformed by three weight matrices to obtain three $n × d$ matrices.

  The computation formula for self-attention is generally defined as follows, where $Q$, $K$, and $V$ are key attribute elements of the input sample, obtained through spatial transformation and unified into a single feature space. "Attention" in the formula and operator name is an abbreviation for "self-attention."

  $$
  Attention(Q,K,V)=Score(Q,K)V
  $$

  In this operator, the `Softmax` function is used, instead of the `Score` function. The self-attention computation formula is as follows:

  $$
  Attention(Q,K,V)=Softmax(\frac{QK^T}{\sqrt{d}})V
  $$

  The product of $Q$ and $K^T$ represents the attention to the input $x$. To prevent this value from becoming excessively large, it is typically scaled by dividing by the square root of $d$, followed by row-wise softmax normalization. The result is then multiplied by $V$ to produce an $n × d$ matrix.

## Function Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnFusedInferAttentionScoreV3GetWorkspaceSize` is called to obtain the input parameters and compute the required workspace size based on the process. Then, `aclnnFusedInferAttentionScoreV3` is called to perform computation.

```cpp
aclnnStatus aclnnFusedInferAttentionScoreV3GetWorkspaceSize(
    const aclTensor     *query, 
    const aclTensorList *key, 
    const aclTensorList *value, 
    const aclTensor     *pseShiftOptional,
    const aclTensor     *attenMaskOptional, 
    const aclIntArray   *actualSeqLengthsOptional,
    const aclIntArray   *actualSeqLengthsKvOptional, 
    const aclTensor     *deqScale1Optional,
    const aclTensor     *quantScale1Optional, 
    const aclTensor     *deqScale2Optional, 
    const aclTensor     *quantScale2Optional,
    const aclTensor     *quantOffset2Optional, 
    const aclTensor     *antiquantScaleOptional,
    const aclTensor     *antiquantOffsetOptional, 
    const aclTensor     *blockTableOptional,
    const aclTensor     *queryPaddingSizeOptional, 
    const aclTensor     *kvPaddingSizeOptional,
    const aclTensor     *keyAntiquantScaleOptional, 
    const aclTensor     *keyAntiquantOffsetOptional,
    const aclTensor     *valueAntiquantScaleOptional, 
    const aclTensor     *valueAntiquantOffsetOptional,
    const aclTensor     *keySharedPrefixOptional, 
    const aclTensor     *valueSharedPrefixOptional,
    const aclIntArray   *actualSharedPrefixLenOptional, 
    const aclTensor     *queryRopeOptional, 
    const aclTensor     *keyRopeOptional, 
    const aclTensor     *keyRopeAntiquantScaleOptional,
    int64_t              numHeads, 
    double               scaleValue, 
    int64_t              preTokens,
    int64_t              nextTokens, 
    char                *inputLayout, 
    int64_t              numKeyValueHeads, 
    int64_t              sparseMode, 
    int64_t              innerPrecise,
    int64_t              blockSize, 
    int64_t              antiquantMode, 
    bool                 softmaxLseFlag, 
    int64_t              keyAntiquantMode, 
    int64_t              valueAntiquantMode,
    const aclTensor     *attentionOut, 
    const aclTensor     *softmaxLse, 
    uint64_t            *workspaceSize, 
    aclOpExecutor       **executor)
```

```cpp
aclnnStatus aclnnFusedInferAttentionScoreV3(
     void                *workspace,
     uint64_t             workspaceSize,
     aclOpExecutor       *executor,
     const aclrtStream    stream)
```

## aclnnFusedInferAttentionScoreV3GetWorkspaceSize

- **Parameters**

    <div style="overflow-x: auto;">
    <table style="undefined;table-layout: fixed; width: 1497px"><colgroup> 
     <col style="width: 150px"> 
     <col style="width: 120px"> 
     <col style="width: 300px"> 
     <col style="width: 330px"> 
     <col style="width: 212px"> 
     <col style="width: 100px">  
     <col style="width: 140px">  
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
        <td>query</td>
        <td>Input</td>
        <td>Input <code>Q</code> in the formula.</td>
        <td>-</td>
        <td>FLOAT16, BFLOAT16, INT8</td>
        <td>ND</td>
        <td>See the <code>inputLayout</code> parameter.</td>
        <td>×</td>
      </tr>
      <tr>
        <td>key</td>
        <td>Input</td>
        <td>Input <code>K</code> in the formula.</td>
        <td>-</td>
        <td>FLOAT16, BFLOAT16, INT8, INT4 (INT32) </td>
        <td>ND</td>
        <td>See the <code>inputLayout</code> parameter.</td>
        <td>×</td>
      </tr>
      <tr>
        <td>value</td>
        <td>Input</td>
        <td>Input <code>V</code> in the formula.</td>
        <td>-</td>
        <td>FLOAT16, BFLOAT16, INT8, INT4 (INT32) </td>
        <td>ND</td>
        <td>See the <code>inputLayout</code> parameter.</td>
        <td>×</td>
      </tr>
      <tr>
        <td>pseShiftOptional</td>
        <td>Input</td>
        <td>Positional encoding.</td>
        <td><ul><li>Empty tensors are not supported.</li>
        <li>If this parameter is not used, pass <code>nullptr</code>.</li>
            <li>For details about the constraints, see <a href="#constraints">Constraints</a>.</li></ul></td>
        <td>FLOAT16, BFLOAT16</td>
        <td>ND</td>
        <td>Recommended shape input: <code>(B, Q_N, Q_S, KV_S)</code> or <code>(1, Q_N, Q_S, KV_S)</code></td>
        <td>×</td>
      </tr>
      <tr>
        <td>attenMaskOptional</td>
        <td>Input</td>
        <td>Mask matrix.</td>
        <td><ul><li>Empty tensors are not supported.</li>
        <li>If this parameter is not used, pass <code>nullptr</code>.</li>
            <li>For details about the constraints, see <a href="#constraints">Constraints</a>.</li></ul></td>
        <td>BOOL, INT8, UINT8</td>
        <td>ND</td>
        <td>
            <ul>
                <li>When <code>sparseMode</code> is set to <code>2</code>, <code>3</code>, or <code>4</code>, the shape of <code>attenMaskOptional</code> must be <code>(2048, 2048)</code>, <code>(1, 2048, 2048)</code>, or <code>(1, 1, 2048, 2048)</code>.</li>
                <li>When <code>sparseMode</code> is set to other values and <code>Q_S</code> is not <code>1</code>, the recommended input shapes are <code>(Q_S, KV_S)</code>, <code>(B, Q_S, KV_S)</code>, <code>(1, Q_S, KV_S)</code>, <code>(B, 1, Q_S, KV_S)</code>, and <code>(1, 1, Q_S, KV_S)</code>.</li>
                <li>When <code>sparseMode</code> is set to other values and <code>Q_S</code> is <code>1</code>, the recommended input shapes are <code>(B, KV_S)</code>, <code>(B, 1, KV_S)</code>, and <code>(B, 1, 1, KV_S)</code>.</li>
            </ul>
            </td>
        <td>×</td>
      </tr>
      <tr>
        <td>actualSeqLengthsOptional</td>
        <td>Input</td>
        <td>Valid sequence length of <code>query</code> in different batches.</td>
        <td><ul><li>If the sequence length is not specified, pass <code>nullptr</code>.</li>
            <li>For details about the constraints, see <a href="#constraints">Constraints</a>.</li></ul></td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>actualSeqLengthsKvOptional</td>
        <td>Input</td>
        <td>Valid sequence lengths of <code>key</code>/<code>value</code> in different batches.</td>
        <td><ul><li>If the sequence length is not specified, pass <code>nullptr</code>.</li>
            <li>For details about the constraints, see <a href="#constraints">Constraints</a>.</li></ul></td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>deqScale1Optional</td>
        <td>Input</td>
        <td>Dequantization factor after BMM1.</td>
          <td><ul><li>Empty tensors are not supported.</li>
          <li>Per-tensor is supported.</li>
              <li>If this parameter is not used, pass <code>nullptr</code>.</li>
              <li>For details about the constraints, see <a href="#constraints">Constraints</a>.</li></ul></td>
        <td>UINT64, FLOAT32</td>
        <td>ND</td>
        <td>See <a href="#INT8">Restrictions on the number of input parameters and input and output data formats related to INT8 quantization</a>.</td>
        <td>-</td>
      </tr>
      <tr>
        <td>quantScale1Optional</td>
        <td>Input</td>
        <td>Quantization factor before BMM2.</td>
        <td><ul><li>Empty tensors are not supported.</li>
        <li>Per-tensor is supported. </li>
            <li>If this parameter is not used, pass <code>nullptr</code>.</li>
            <li>For details about the constraints, see <a href="#constraints">Constraints</a>.</li></ul></td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>See <a href="#INT8">Restrictions on the number of input parameters and input and output data formats related to INT8 quantization</a>.</td>
        <td>-</td>
      </tr>
      <tr>
        <td>deqScale2Optional</td>
        <td>Input</td>
        <td>Dequantization factor after BMM2.</td>
        <td><ul><li>Empty tensors are not supported.</li>
        <li>Per-tensor is supported. </li>
           <li>If this parameter is not used, pass <code>nullptr</code>.</li>
            <li>For details about the constraints, see <a href="#constraints">Constraints</a>.</li></ul></td>
        <td>UINT64, FLOAT32</td>
        <td>ND</td>
        <td>See <a href="#INT8">Restrictions on the number of input parameters and input and output data formats related to INT8 quantization</a>.</td>
        <td>-</td>
      </tr>
      <tr>
        <td>quantScale2Optional</td>
        <td>Input</td>
        <td>Output quantization factor.</td>
        <td><ul><li>Empty tensors are not supported.</li>
        <li>Per-tensor and per-channel are supported. </li>
            <li>If this parameter is not used, pass <code>nullptr</code>.</li>
             <li>For details about the constraints, see <a href="#constraints">Constraints</a>.</li></ul></td>
        <td>FLOAT32, BFLOAT16</td>
        <td>ND</td>
        <td>When the output layout is <code>BSH</code>, it is recommended that the shape of <code>quantScale2</code> be set to <code>[1, 1, H]</code> or <code>[H]</code>. When the output layout is <code>BNSD</code>, it is recommended that the shape of <code>quantScale2</code> be set to <code>[1, N, 1, D]</code> or <code>[N, D]</code>. When the output layout is <code>BSND</code>, it is recommended that the shape of <code>quantScale2</code> be set to <code>[1, 1, N, D]</code> or <code>[N, D]</code>.</td>
        <td>-</td>
      </tr>
      <tr>
        <td>quantOffset2Optional</td>
        <td>Input</td>
        <td>Output quantization offset.</td>
        <td><ul><li>Empty tensors are not supported.</li>
        <li>Per-tensor and per-channel are supported. </li>
            <li>If this parameter is not used, pass <code>nullptr</code>.</li>
             <li>For details about the constraints, see <a href="#constraints">Constraints</a>.</li></ul></td>
        <td>FLOAT32, BFLOAT16</td>
        <td>ND</td>
        <td>Same as <code>quantScale2Optional</code>.</td>
        <td>-</td>
      </tr>
      <tr>
        <td>antiquantScaleOptional</td>
        <td>Input</td>
        <td>Fake-quantization factor.</td>
        <td><ul><li>Empty tensors are not supported.</li>
        <li>Per-tensor, per-channel, and per-token are supported. </li>
            <li>If this parameter is not used, pass <code>nullptr</code>.</li>
            <li>For details about the constraints, see <a href="#constraints">Constraints</a>.</li></ul></td>
        <td>FLOAT16, BFLOAT16, FLOAT32</td>
        <td>ND</td>
        <td>See <a href="#AntiQuant">Constraints on the fake-quantization parameters <code>antiquantScale</code> and <code>antiquantOffset</code></a>.</td>
        <td>-</td>
      </tr>
        <tr>
      <td>antiquantOffsetOptional</td>
        <td>Input</td>
        <td>Fake-quantization offset.</td>
        <td><ul><li>Empty tensors are not supported.</li>
        <li>Per-tensor, per-channel, and per-token are supported. </li>
            <li>The shape must be the same as that of <code>antiquantScaleOptional</code>.</li>
            <li>If this parameter is not used, pass <code>nullptr</code>.</li>
            <li>For details about the constraints, see <a href="#constraints">Constraints</a>.</li></ul></td>
        <td>FLOAT16, BFLOAT16, FLOAT32</td>
        <td>ND</td>
        <td>Same as <code>antiquantScaleOptional</code>.</td>
        <td>-</td>
      </tr> 
    <tr>
       <td>blockTableOptional</td>
        <td>Input</td>
        <td>Block mapping table used for KV storage in page attention.</td>
        <td><ul><li>Empty tensors are not supported.</li>
        <li>If this parameter is not used, pass <code>nullptr</code>.</li></ul></td>
        <td>INT32</td>
        <td>ND</td>
        <td>The length of the first dimension must be equal to <code>B</code>, and the length of the second dimension must be greater than or equal to <code>maxBlockNumPerSeq</code> (the maximum number of blocks corresponding to <code>actualSeqLengthsKv</code> in different batches).</td>
        <td>-</td>
      </tr>
      <tr> 
       <td>queryPaddingSizeOptional</td>
        <td>Input</td>
        <td>Whether the data in each batch of <code>query</code> is right-aligned and the number of right-aligned elements.</td>
        <td><ul><li>Empty tensors are not supported.</li>
        <li>This parameter is valid only when <code>Q_S</code> is greater than <code>1</code>. In other scenarios, it is invalid.</li>
            <li>If this parameter is not used, pass <code>nullptr</code>.</li></ul></td>
        <td>INT64</td>
        <td>ND</td>
        <td><code>(1)</code></td>
        <td>-</td>
      </tr>
      <tr> 
       <td>kvPaddingSizeOptional</td>
        <td>Input</td>
        <td>Whether the data in each batch of <code>key</code>/<code>value</code> is right-aligned and the number of right-aligned elements.</td>
        <td><ul><li>Empty tensors are not supported.</li>
        <li>If this parameter is not used, pass <code>nullptr</code>.</li></ul></td>
        <td>INT64</td>
        <td>ND</td>
        <td><code>(1)</code></td>
        <td>-</td>
      </tr>
      <tr> 
       <td>keyAntiquantScaleOptional</td>
        <td>Input</td>
        <td>Dequantization factor of <code>key</code> when the KV fake-quantization parameters are separated.</td>
        <td><ul><li>Empty tensors are not supported.</li>
        <li>If this parameter is not used, pass <code>nullptr</code>.</li>
            <li>For details about the constraints, see <a href="#constraints">Constraints</a>.</li></ul></td>
        <td>FLOAT16, BFLOAT16, FLOAT32</td>
        <td>ND</td>
        <td>See <a href="#constraints">Constraints</a>.</td>
        <td>-</td>
      </tr>
        <tr> 
       <td>keyAntiquantOffsetOptional</td>
        <td>Input</td>
        <td>Dequantization offset of <code>key</code> when the KV fake-quantization parameters are separated.</td>
        <td><ul><li>Empty tensors are not supported.</li>
        <li>The shape must be the same as that of <code>keyAntiquantScaleOptional</code>.</li>
            <li>If this parameter is not used, pass <code>nullptr</code>.</li>
            <li>For details about the constraints, see <a href="#constraints">Constraints</a>.</li></ul></td>
        <td>FLOAT16, BFLOAT16, FLOAT32</td>
        <td>ND</td>
        <td>See <a href="#constraints">Constraints</a>.</td>
        <td>-</td>
      </tr>
         <tr> 
       <td>valueAntiquantScaleOptional</td>
        <td>Input</td>
        <td>Dequantization factor of <code>value</code> when the KV fake-quantization parameters are separated.</td>
        <td><ul><li>Empty tensors are not supported.</li>
        <li>The shape must be the same as that of <code>valueAntiquantScaleOptional</code>.</li>
            <li>If this parameter is not used, pass <code>nullptr</code>.</li>
            <li>For details about the constraints, see <a href="#constraints">Constraints</a>.</li></ul></td>
        <td>FLOAT16, BFLOAT16, FLOAT32</td>
        <td>ND</td>
        <td>See <a href="#constraints">Constraints</a>.</td>
        <td>-</td>
      </tr>
         <tr> 
       <td>valueAntiquantOffsetOptional</td>
        <td>Input</td>
        <td>Dequantization offset of <code>value</code> when the KV fake-quantization parameters are separated.</td>
        <td><ul><li>Empty tensors are not supported.</li>
            <li>If this parameter is not used, pass <code>nullptr</code>.</li>
            <li>For details about the constraints, see <a href="#constraints">Constraints</a>.</li></ul></td>
        <td>FLOAT16, BFLOAT16, FLOAT32</td>
        <td>ND</td>
        <td>See <a href="#constraints">Constraints</a>.</td>
        <td>-</td>
      </tr>        
       <tr> 
       <td>keySharedPrefixOptional</td>
        <td>Input</td>
        <td>System prefix of <code>key</code> in the attention structure.</td>
        <td><ul><li>Empty tensors are not supported.</li>
            <li>If this parameter is not used, pass <code>nullptr</code>.</li>
            <li>For details about the constraints, see <a href="#constraints">Constraints</a>.</li></ul></td>
        <td>FLOAT16, BFLOAT16, INT8</td>
        <td>ND</td>
        <td>
            <ul>
                <li>When the input layout is <code>BSH</code>, the shape is <code>(1, prefix_S, H = KV_N × KV_D)</code>.</li>
                <li>When the input layout is <code>BSND</code>, the shape is <code>(1, prefix_S, KV_N, KV_D)</code>.</li>
                <li>When the input layout is <code>BNSD</code> or <code>BNSD_BSND</code>, the shape is <code>(1, KV_N, prefix_S, KV_D)</code>.</li>
            </ul>
        </td>
        <td>-</td>
      </tr>
       <tr> 
       <td>valueSharedPrefixOptional</td>
        <td>Input</td>
        <td>System prefix of <code>value</code> in the attention structure.</td>
        <td><ul><li>Empty tensors are not supported.</li>
            <li>If this parameter is not used, pass <code>nullptr</code>.</li>
            <li>For details about the constraints, see <a href="#constraints">Constraints</a>.</li></ul></td>
        <td>FLOAT16, BFLOAT16, INT8</td>
        <td>ND</td>
        <td>
            <ul>
                <li>When the input layout is <code>BSH</code>, the shape is <code>(1, prefix_S, H = KV_N × KV_D)</code>.</li>
                <li>When the input layout is <code>BSND</code>, the shape is <code>(1, prefix_S, KV_N, KV_D)</code>.</li>
                <li>When the input layout is <code>BNSD</code> or <code>BNSD_BSND</code>, the shape is <code>(1, KV_N, prefix_S, KV_D)</code>.</li>
            </ul>
        </td>
        <td>-</td>
    </tr>
      <tr> 
        <td>actualSharedPrefixLenOptional</td>
        <td>Input</td>
        <td>Valid sequence length of <code>keySharedPrefix</code>/<code>valueSharedPrefix</code>.</td>
        <td><ul><li>If this parameter is not used, pass <code>nullptr</code>, indicating that the sequence length is the same as <code>S</code> of <code>keySharedPrefix</code>/<code>valueSharedPrefix</code>.</li>
         <li>The valid sequence length in this input parameter must be less than or equal to that in <code>keySharedPrefix</code>/<code>valueSharedPrefix</code>.</li>
         <li>For details about the constraints, see <a href="#constraints">Constraints</a>.</li></ul></td>
        <td>INT64</td>
        <td>-</td>
        <td><code>(1)</code></td>
        <td>-</td>
      </tr>
      <tr>
        <td>queryRopeOptional</td>
        <td>Input</td>
        <td>Rope information of <code>query</code> in the MLA structure.</td>
        <td><ul><li>Empty tensors are not supported.</li>
                <li>For details about the constraints, see <a href="#constraints">Constraints</a>.</li></ul></td>
        <td>FLOAT16, BFLOAT16</td>
        <td>ND</td>
        <td> In <code>queryRope</code>, dimension <code>d</code> of the shape is <code>64</code>, and the values of other dimensions are the same as those of <code>query</code>.</td>
        <td>-</td>
      </tr>
      <tr>
        <td>keyRopeOptional</td>
        <td>Input</td>
        <td>Rope information of <code>key</code> in the MLA structure.</td>
        <td><ul><li>Empty tensors are not supported.</li>
                <li>For details about the constraints, see <a href="#constraints">Constraints</a>.</li></ul></td>
        <td>FLOAT16, BFLOAT16</td>
        <td>ND</td>
        <td> In <code>keyRope</code>, dimension <code>d</code> of the shape is <code>64</code>, and the values of other dimensions are the same as those of <code>key</code>.</td>
        <td>-</td>
      </tr>
       <tr>
        <td>keyRopeAntiquantScaleOptional</td>
        <td>Input</td>
        <td>Dequantization factor of the rope information of <code>key</code> in the MLA structure.</td>
        <td><ul><li>Empty tensors are not supported.</li>
                <li>This parameter is reserved and does not take effect in the current version. Pass <code>nullptr</code>.</li></ul></td>
        <td>FLOAT16, BFLOAT16</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>        
      <tr>
        <td>numHeads</td>
        <td>Input</td>
        <td>Number of heads in <code>query</code>.</td>
        <td>When the layout is <code>BNSD</code>, the parameter value must be the same as the N-axis shape value of <code>query</code> in the shape. Otherwise, an exception occurs.</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>scaleValue</td>
        <td>Input</td>
        <td>Reciprocal of the square root of <code>d</code> in the formula.</td>
        <td><ul><li>Its data type must be compatible with that of <code>query</code> according to the type deduction rules. </li>
            <li>If no specific value is required, <code>1.0</code> is recommended. </li></ul></td>
        <td>DOUBLE</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>preTokens</td>
        <td>Input</td>
        <td>Number of preceding tokens to associate in attention computation for sparse computation.</td>
          <td><ul><li>If no specific value is required, <code>2147483647</code> is recommended.</li>
              <li>This parameter is invalid when <code>Q_S</code> is <code>1</code>.</li></ul></td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>      
      <tr>
        <td>nextTokens</td>
        <td>Input</td>
        <td>Number of succeeding tokens to associate in attention computation.</td>
        <td><ul><li>If no specific value is required, <code>2147483647</code> is recommended.</li>
            <li>This parameter is invalid when <code>Q_S</code> is <code>1</code>.</li></ul></td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>inputLayout</td>
        <td>Input</td>
        <td>Layout of the input <code>query</code>, <code>key</code>, and <code>value</code>.</td>
        <td><ul><li>Currently, the supported layouts include: <code>BSH</code>, <code>BSND</code>, <code>BNSD</code>, <code>BNSD_BSND</code> (<code>BSND</code> for the output layout and <code>Q_S</code> greater than <code>1</code> when the input layout is <code>BNSD</code>), <code>BSH_NBSD</code>, <code>BSND_NBSD</code>, <code>BNSD_NBSD</code> (<code>Q_S</code> greater than <code>1</code> and less than or equal to <code>16</code> when the output layout is <code>NBSD</code>), <code>TND</code>, <code>TND_NTD</code>, and <code>NTD_TND</code> (TND-related scenarios subject to <a href="#constraints "> Constraints</a>). If no specific value is required, <code>BSH</code> is recommended.</li></ul>
            <ul><li>For details about the constraints, see <a href="#constraints">Constraints</a>.</li></ul></td>
        <td>CHAR</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>numKeyValueHeads</td>
        <td>Input</td>
        <td>Number of heads in <code>key</code> and <code>value</code>.</td>
        <td><ul><li>If no specific value is required, <code>0</code> is recommended, indicating that <code>key</code>/<code>value</code> and <code>query</code> have the same number of heads.</li></ul>
            <ul><li>For details about the constraints, see <a href="#constraints">Constraints</a>.</li></ul></td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>sparseMode</td>
        <td>Input</td>
        <td>Sparse mode.</td>
        <td>For details about the constraints, see <a href="#constraints">Constraints</a>.</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>innerPrecise</td>
        <td>Input</td>
        <td>Choice between the high-precision and high-performance modes.</td>
        <td>For details about the constraints, see <a href="#constraints">Constraints</a>.</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>blockSize</td>
        <td>Input</td>
        <td>Maximum number of tokens in each block for KV storage in page attention.</td>
        <td>The value can only be <code>0</code>.</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>antiquantMode</td>
        <td>Input</td>
        <td>Fake-quantization mode.</td>
          <td><ul><li><code>0</code> indicates per-channel (per-channel contains per-tensor).</li>
              <li><code>1</code> indicates per-token.</li>
              <li>If no specific value is required, <code>0</code> is recommended.</li>
              <li>When <code>Q_S</code> is <code>1</code>, an exception occurs if a value other than <code>0</code> or <code>1</code> is passed. This parameter is invalid when <code>Q_S</code> is greater than or equal to <code>2</code>.</li></ul></td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>softmaxLseFlag</td>
        <td>Input</td>
        <td>Whether to output <code>softmax_lse</code>.</td>
          <td><ul><li>S-axis outer splitting (augmented output) is supported.</li>
              <li>If no specific value is required, <code>false</code> is recommended.</li></ul></td>
        <td>BOOL</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>        
      <tr>
        <td>keyAntiquantMode</td>
        <td>Input</td>
        <td>Fake-quantization mode of <code>key</code>.</td>
        <td><ul>
            <li>If no specific value is required, <code>0</code> is recommended.</li>
            <li>For details about the constraints, see <a href="#constraints">Constraints</a>.</li>
            </ul></td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>valueAntiquantMode</td>
        <td>Input</td>
        <td>Fake-quantization mode of <code>value</code>.</td>
          <td><ul><li>The mode ID is the same as that of <code>keyAntiquantMode</code>.</li>
              <li>If no specific value is required, <code>0</code> is recommended.</li>
               <li>For details about the constraints, see <a href="#constraints">Constraints</a>.</li></ul></td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>        
      <tr>
        <td>attentionOut</td>
        <td>Output</td>
        <td>Output in the formula.</td>
        <td>-</td>
        <td>FLOAT16, BFLOAT16, INT8</td>
        <td>ND</td>
        <td>Dimension <code>D</code> of this parameter must be the same as that of <code>value</code>, and other dimensions must be the same as the shapes of <code>query</code>.</td>
        <td>-</td>
      </tr>
      <tr>
        <td>softmaxLse</td>
        <td>Output</td>
        <td>Result of <code>query</code>-<code>key</code> multiplication in ring attention.</td>
        <td>For details about the constraints, see <a href="#constraints">Constraints</a>.</td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>When <code>softmaxLseFlag</code> is <code>True</code>, the shape must be <code>[B, N, Q_S, 1]</code> in general. When <code>inputLayout</code> is <code>TND</code> or <code>NTD_TND</code>, the shape must be <code>[T, N, 1]</code>.</td>
        <td>-</td>
      </tr>
      <tr>
        <td>workspaceSize</td>
        <td>Output</td>
        <td>Size of the workspace to be allocated on the device.</td>
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
    </tbody></table>
    </div>

- **Returns**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).
  
  The first-phase API implements input parameter verification. The following errors may be thrown.
  
  <table style="undefined;table-layout: fixed; width: 1149px"><colgroup>
  <col style="width: 286px">
  <col style="width: 118px">
  <col style="width: 745px">
  </colgroup>
  <thead>
    <tr>
      <th>Return</th>
      <th>Error Code</th>
      <th>Description</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>The passed <code>query</code>, <code>key</code>, <code>value</code>, or <code>attentionOut</code> is a null pointer.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_PARAM_INVALID</td>
      <td>161002</td>
      <td>The data type or data format of <code>query</code>, <code>key</code>, <code>value</code>, <code>pseShift</code>, <code>attenMask</code>, or <code>attentionOut</code> is not supported.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_RUNTIME_ERROR</td>
      <td>361001</td>
      <td>An exception occurred when the NPU Runtime API was called.</td>
    </tr>
  </tbody>
  </table>

## aclnnFusedInferAttentionScoreV3

- **Parameters**
  
  <table style="undefined;table-layout: fixed; width: 1049px"><colgroup>
  <col style="width: 167px">
  <col style="width: 118px">
  <col style="width: 764px">
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
      <td>Size of the workspace to be allocated on the device, which is obtained by calling the first-phase API <code>aclnnPromptFlashAttentionV3GetWorkspaceSize</code>.</td>
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
  - `aclnnPromptFlashAttentionV3` defaults to a deterministic implementation.
- When this API is used together with PyTorch, ensure that the CANN package versions match the PyTorch package versions.

- Processing logic for a null input parameter: The operator checks whether `query` is a null pointer. If so, an error is reported. If `query` is not an empty tensor but `key` and `value` are empty tensors (that is, `S2` is `0`), `attentionOut` is filled with all zeros. If `attentionOut` is an empty tensor, the framework will process it. For other input parameters which support the passing of `nullptr` as described in the preceding parameter description, no processing is performed when they are null pointers.

- The shapes of the tensors corresponding to `key` and `value` must be identical. In non-contiguous scenarios, the batch size in the tensor lists of `key` and `value` can only be `1`, the number of elements must be equal to the batch size (`B`) of `query`, and the `N` and `D` dimensions must be the same. Due to the tensor list restrictions, `B` cannot be greater than `256` in non-contiguous scenarios.

- When the data type of `attenMask` is INT8 or UINT8, the value in the tensor must be `0` or `1`.

- <a id="INT8"></a>Restrictions on the number of input parameters and input and output data formats related to INT8 quantization:

  - If both the input and output are of the INT8 type, the input parameters `deqScale1`, `quantScale1`, `deqScale2`, and `quantScale2` must exist at the same time. `quantOffset2` is optional and defaults to `0` if not passed.
  - If the input is of the INT8 type and the output is of the FLOAT16 type, the input parameters `deqScale1`, `quantScale1`, and `deqScale2` must exist at the same time. If the input parameter `quantOffset2` or `quantScale2` exists (not `nullptr`), an error is reported and returned.
  - If the input is of the FLOAT16 or BFLOAT16 type and the output is of the INT8 type, the input parameter `quantScale2` must exist, and `quantOffset2` is optional (`0` is used if no value is passed). If the input parameter `deqScale1`, `quantScale1`, or `deqScale2` exists (not `nullptr`), an error is reported and returned.
  - The input parameters `quantScale2` and `quantOffset2` support both the per-tensor and per-channel data formats and the FLOAT32 and BFLOAT16 data types. If `quantOffset2` is passed, ensure that its type and shape are consistent with those of `quantScale2`. If the input is of the BFLOAT16 type, both FLOAT32 and BFLOAT16 are supported. Otherwise, only FLOAT32 is supported. In per-channel format, when the output layout is `BSH`, the product of all dimensions of `quantScale2` must be equal to `H`. For other layouts, the product must be equal to `N` × `D`. (When the output layout is `BSH`, it is recommended that the shape of `quantScale2` be set to `[1, 1, H]` or `[H]`. When the output layout is `BNSD`, it is recommended that the shape of `quantScale2` be set to `[1, N, 1, D]` or `[N, D]`. When the output layout is `BSND`, it is recommended that the shape of `quantScale2` be set to `[1, 1, N, D]` or `[N, D]`.)

- <a id="AntiQuant"></a>Constraints on the fake-quantization parameters `antiquantScale` and `antiquantOffset`:

  - Only the fake-quantization scenario where `kv_dtype` is INT8 is supported.
  - Per-channel mode: The shapes of the two parameters can be `\(2, N, 1, D\)`, `\(2, N, D\)`, or `\(2, H\)`, where `N` is `numKeyValueHeads`. The data type is the same as that of `query`, and `antiquantMode` is set to `0`.
  - Per-tensor mode: The shapes of the two parameters are `(2)`, the data type is the same as that of `query`, and `antiquantMode` is set to `0`.
  - Per-token mode: The shapes of the two parameters are `\(2, B, S\)`, the data type is fixed at FLOAT32, and `antiquantMode` is set to `1`.
  - In asymmetric quantization mode, both `antiquantScale` and `antiquantOffset` must be present.
  - In symmetric quantization mode, `antiquantOffset` can be `nullptr`. If `antiquantOffset` is `nullptr`, symmetric quantization is performed. Otherwise, asymmetric quantization is performed.

- Restrictions on the input parameters `query`, `key`, and `value` when the layout is `TND`, `TND_NTD`, or `NTD_TND`:

  - Both `actualSeqLengths` and `actualSeqLengthsKv` must be passed, and the number of elements in these input parameters is used as the batch size. The value of each element in these parameters indicates the sum of sequence lengths of the current batch and all previous batches. Therefore, the value of the next element must be greater than or equal to the value of the previous element.
  - <term>Atlas A2 training products/Atlas A2 inference products</term>:
    - The sparse mode can only be `0` without a mask, `3` with a mask, or `4` with a mask.
    - When dimension `d` of `query` is `512`:
      - The layout can be `TND` or `TND_NTD`.
      - Page attention must be enabled. In this case, the length of `actualSeqLengthsKv` is equal to the batch size of `key`/`value`, indicating the actual length of each batch. The value must be less than or equal to `KV_S`.
      - Dimension `s` for each batch of `query` can be set to a value from `1` to `16`.
      - Dimension `n` for `query` must be set to `32`/`64`/`128`, and dimension `n` for `key` and `value` must be `1`.
      - `queryRope` and `keyRope` must not be empty, and dimension `d` for both `queryRope` and `keyRope` must be set to `64`.
      - Left padding, tensor list, PSE, prefix, fake-quantization, full quantization, and post-quantization are not supported.
      - When the layout is `NTD_TND`, `softmaxLse` cannot be enabled.
    - When dimension `d` of `query` is not `512`:
      - When `queryRope` and `keyRope` are empty, if the layout is `TND`, `Q_D`, `K_D`, and `V_D` must be equal and less than or equal to `256`, or `Q_D` and `K_D` must be equal to `192` and `V_D` must be equal to `128`/`192`; if the layout is `NTD_TND`, `Q_D` and `K_D` must be equal to `128`/`192`, and `V_D` must be equal to `128`. When `queryRope` and `keyRope` are not null, `Q_D`, `K_D`, and `V_D` must be equal to `128`.
      - The layout can be `TND` or `NTD_TND`.
      - If the layout is `TND`, the data type can only be FLOAT16 or BFLOAT16. If the layout is `NTD_TND`, the data type can only be BFLOAT16.
      - If the layout is `TND`, when the head configuration is GQA/MQA (that is, the `numHeads` and `numKeyValueHeads` parameters must be completely passed, `numHeads` must be an integer multiple of `numKeyValueHeads`, and the two values must be different), the following constraints apply:
        - When the data type is FLOAT16, the sparse mode can be `0` without a mask or `3` with optimized `attentionMask`.
        - When the data type is BFLOAT16, the sparse mode can be `0` without a mask, `3` with optimized `attentionMask`, or `4` with optimized `attentionMask`.
        - `innerPrecise` can only be `0`, indicating the high-precision mode without invalid row correction.
        - Page attention is supported. The KV cache layout supports BnBsH `(BlockNum, BlockSize, H)`, where `H` is less than or equal to `65535` and `BlockSize` can only be `128`.
      - If the layout is `TND`, when the head configuration is MHA, the following constraints apply:
        - When the data type is FLOAT16, the sparse mode can only be `0` without a mask.
        - When the data type is FLOAT16, `innerPrecise` can only be `1`.
        - When the data type is FLOAT16, page attention is not supported.
        - When the data type is BFLOAT16, the sparse mode can be `0` without a mask, `3` with optimized `attentionMask`, or `4` with optimized `attentionMask`.
      - If the layout is `NTD_TND`, page attention is not supported.
      - When the sparse mode is `3`, `actualSeqLengths` must be less than `actualSeqLengthsKv` for each batch.
      - The sparse mode can be `4` with a mask. When the sparse mode is `4`, `preTokens` must be greater than or equal to negative `actualSeqLengths`, `nextTokens` must be greater than or equal to negative `actualSeqLengthsKv`, and the sum of `preTokens` and `nextTokens` must be greater than or equal to `0`.
      - Left padding, tensor list, PSE, page attention, prefix, fake-quantization, full quantization, and post-quantization are not supported.
      - The number of elements in `actualSeqLengths` and `actualSeqLengthsKv` must be less than or equal to `4096`.

- Constraints on `queryRope` and `keyRope` in the MLA structure:

  - The data type and format of `queryRope` must be the same as those of `query`.
  - The data type and format of `keyRope` must be the same as those of `key`.
  - Both `queryRope` and `keyRope` are configured, or neither of them is configured. Configuring only one of the parameters is not supported.
  - When `queryRope` and `keyRope` are input, only the following features are supported:
    - `dtype` can only be FP16 or BF16.
    - Dimension `d` in `query` can only be `512`/`128`.
    - <term>Atlas A2 training products/Atlas A2 inference products</term>:
      - When `Q_S` is greater than `1` (that is, in the MTP mode), the `actualSeqLengths` parameter can be configured only when `inputLayout` is `TND`. Other layouts are not supported.
      - When dimension `d` of `query` is `512`:
        - When configuring `queryRope`, ensure that dimension `s` in `query` is set to a value from `1` to `16` and dimension `n` is `1`, `2`, `4`, `8`, `16`, `32`, `64`, or `128`. In the shape of `queryRope`, dimension `d` is `64`, and the values of other dimensions are the same as those of `query`.
        - When configuring `keyRope`, ensure that dimension `n` in `key` is `1` and dimension `d` is `512`. In the shape of `keyRope`, dimension `d` is `64`, and the values of other dimensions are the same as those of `key`.
        - When `Q_S` is `1`, the sparse mode can only be `0` without a mask. When `Q_S` is greater than `1`, the sparse mode can only be `3` with a mask.
        - The ND and NZ inputs are supported for `key`, `value`, and `keyRope`. The input format for NZ is `[BlockNum, N, D/16, BlockSize, 16]`.
        - `inputLayout` can be `BSH`, `BSND`, `BNSD`, `BNSD_NBSD`, `BSND_NBSD`, `BSH_NBSD`, `TND`, or `TND_NTD`. In the NZ input format, `inputLayout` cannot be `BNSD` or `BNSD_NBSD`.
        - When page attention must be enabled, `BlockSize` can be `16` or `128`. In the NZ input format, `BlockSize` cannot be set to `16`.
        - Left padding, tensor list, PSE, prefix, fake-quantization, full quantization, and post-quantization are not supported.
      - When dimension `d` of `query` is `128`:
        - `inputLayout` can be `TND` or `NTD_TND`.
        - When configuring `queryRope`, ensure that dimension `d` in the shape of `queryRope` is `64`, and the values of other dimensions are the same as those of `query`.
        - When configuring `keyRope`, ensure that dimension `d` in the shape of `keyRope` is `64`, and the values of other dimensions are the same as those of `key`.
        - Other restrictions are the same as those when the layout is `TND` or `NTD_TND`.
        - Left padding, tensor list, PSE, page attention, prefix, fake-quantization, full quantization, and post-quantization are not supported.

- Restrictions on `numKeyValueHeads`: `numHeads` must be exactly divided by `numKeyValueHeads`. When the layout is `BSND`, `BNSD`, or `BNSD_BSND`, the value must be the same as the N-axis value of `key`/`value` in the shape. Otherwise, an exception occurs.

- Restrictions on `sparseMode`:

  <div style="overflow-x: auto;">
  <table style="table-layout: fixed; width: 1210px">
      <colgroup>
          <col style="width: 150px">
          <col style="width: 210px">
          <col style="width: 850px">
      </colgroup>
      <thead>
          <tr>
              <th>sparseMode</th>
              <th>Mode</th>
              <th>Description</th>
          </tr>
      </thead>
      <tbody>
          <tr>
              <td>0</td>
              <td>defaultMask</td>
              <td>
                  <ul style="margin: 0; padding-left: 20px;">
                      <li>If <code>attenMask</code> is not passed, the mask operation is not performed, and <code>preTokens</code> and <code>nextTokens</code> are ignored (internally set to <code>INT_MAX)</code>.</li>
                      <li>If <code>attenMask</code> is passed, a complete <code>attenMask</code> matrix (S1 × S2) needs to be passed, indicating that the portion between <code>preTokens</code> and <code>nextTokens</code> needs to be calculated.</li>
                  </ul>
              </td>
          </tr>
          <tr>
              <td>1</td>
              <td>allMask</td>
              <td>A complete <code>attenMask</code> matrix (S1 × S2) must be passed.</td>
          </tr>
          <tr>
              <td>2</td>
              <td>leftUpCausal</td>
              <td>An optimized <code>attenMask</code> matrix (2048 × 2048) must be passed.</td>
          </tr>
          <tr>
              <td>3</td>
              <td>rightDownCausal</td>
              <td>This corresponds to a lower-triangular matrix partitioned by the top-right vertex. In this case, an optimized <code>attenMask</code> matrix (2048 × 2048) needs to be passed.</td>
          </tr>
          <tr>
              <td>4</td>
              <td>band</td>
              <td>An optimized <code>attenMask</code> matrix (2048 × 2048) must be passed.</td>
          </tr>
          <tr>
              <td>5</td>
              <td>prefix</td>
              <td>This mode is not supported currently. If no specific value is required, <code>0</code> is recommended.</td>
          </tr>
          <tr>
              <td>6</td>
              <td>global</td>
              <td>This mode is not supported currently. If no specific value is required, <code>0</code> is recommended.</td>
          </tr>
          <tr>
              <td>7</td>
              <td>dilated</td>
              <td>This mode is not supported currently. If no specific value is required, <code>0</code> is recommended.</td>
          </tr>
          <tr>
              <td>8</td>
              <td>block_local</td>
              <td>This mode is not supported currently. If no specific value is required, <code>0</code> is recommended.</td>
          </tr>
          <tr>
              <td colspan="3" style="text-align: left; ">
                  <strong>Note:</strong> When <code>Q_S</code> is <code>1</code> and no rope input is provided, <code>sparseMode</code> is invalid.
              </td>
          </tr>
          </tbody>
      </table>
  </div>

- Restrictions on `innerPrecise`:

  - There are four modes (<code>0</code>, <code>1</code>, <code>2</code>, and <code>3</code>) in total, represented by 2-bit combinations. Bit 0 indicates whether to use the high-precision or high-performance mode, and bit 1 indicates whether to perform invalid row correction.

    <table style="undefined;table-layout: fixed; width: 600px"><colgroup>
    <col style="width: 200px">
    <col style="width: 200px">
      <col style="width: 200px">
    </colgroup><thead>
    <tr>
    <th>innerPrecise</th>
    <th>Mode</th>
    <th>Invalid Row Correction</th>
    </tr></thead>
    <tbody>
    <tr>
    <td>0</td>
    <td>High-precision</td>
    <td>×</td>
    </tr>
    <tr>
    <td>1</td>
    <td>High-performance</td>
    <td>×</td>
    </tr>
    <tr>
    <td>2</td>
    <td>High-precision</td>
    <td>√</td>
    </tr>
    <tr>
    <td>3</td>
    <td>High-performance</td>
    <td>√</td>
    </tr>
    </tbody>
    </table>

  - Note: The high-precision and high-performance modes are applicable to both BFLOAT16 and INT8. Invalid row correction takes effect for FLOAT16, BFLOAT16, and INT8. The values `0` and `1` are reserved. If the masks involved in the computation are all 1s, the precision may be affected. In this case, you can set this parameter to `2` or `3` to enable invalid row correction to improve the precision. However, this configuration deteriorates the performance. If the operator can determine that invalid rows exist, invalid row correction is automatically enabled, such as in scenarios where `sparseMode` is set to `3` and `Sq` is greater than `Skv`.

- Restrictions on `keyAntiquantMode`:

    <div style="overflow-x: auto;">
    <table style="table-layout: fixed; width: 830px">
        <colgroup>
            <col style="width: 230px">
            <col style="width: 600px">
        </colgroup>
        <thead>
            <tr>
                <th>keyAntiquantMode</th>
                <th>Description</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>0</td>
                <td>Per-channel (per-channel contains per-tensor.)</td>
            </tr>
            <tr>
                <td>1</td>
                <td>Per-token</td>
            </tr>
            <tr>
                <td>2</td>
                <td>Per-tensor + per-head</td>
            </tr>
            <tr>
                <td>3</td>
                <td>Per-token + per-head</td>
            </tr>
            <tr>
                <td>4</td>
                <td>Per-token + page attention, used to manage scale/offset</td>
            </tr>
            <tr>
                <td>5</td>
                <td>Per-token + per-head + page attention, used to manage scale/offset</td>
            </tr>
            <tr>
                <td>6</td>
                <td>Per-token-group</td>
            </tr>
            <tr>
                <td colspan="2" style="text-align: left; ">
                    <strong>Note:</strong> This parameter must be the same as <code>valueAntiquantMode</code> except when <code>keyAntiquantMode</code> is <code>0</code> and <code>valueAntiquantMode</code> is <code>1</code>.
                </td>
            </tr>
          </tbody>
        </table>
    </div>

  - <term>For Atlas A2 training products/Atlas A2 inference products</term>, if `Q_S` is `1`, passing a value other than `0`, `1`, `2`, `3`, `4`, or `5` will result in an execution error. If `Q_S` is greater than or equal to `2`, only the values `0` and `1` are supported. Other values will result in an execution error.

- Restrictions on `valueAntiquantMode`:

  - Except for the scenario where `keyAntiquantMode` is `0` and `valueAntiquantMode` is `1`, the value must be the same as that of `keyAntiquantMode`.
  - <term>For Atlas A2 training products/Atlas A2 inference products</term>, if `Q_S` is `1`, passing a value other than `0`, `1`, `2`, `3`, `4`, or `5` will result in an execution error. If `Q_S` is greater than or equal to `2`, only the values `0` and `1` are supported. Other values will result in an execution error.

- Restrictions on `softmaxLse`:

  - In the ring attention algorithm, the product of `query` and `key` is first processed to obtain `softmax_max`. This max value is subtracted from the product before calculating the exponential, which is then summed to yield `softmax_sum`. Finally, the log of `softmax_sum` is added back to `softmax_max` to obtain the final result.
  - When `softmaxLseFlag` is `True`, the shape must be `[B, N, Q_S, 1]` in general. When `inputLayout` is `TND` or `NTD_TND`, the shape must be `[T, N, 1]`.
  - When `softmaxLseFlag` is `False`, if the `softmaxLse` tensor is not `nullptr`, the tensor data is returned directly. If `softmaxLse` is `nullptr`, a tensor of shape `{1}` filled with zeros is returned.

- **When `Q_S` is greater than `1`**:

  - Restrictions on `query`, `key`, and `value`:

    - The B axis must be less than or equal to `65536`.

    - <term>For Atlas A2 training products/Atlas A2 inference products</term>, if the input type is INT8 and the D axis is not 32-byte aligned, the maximum value of the B axis is `128`. If the input type is FLOAT16 or BFLOAT16 and the D axis is not 16-byte aligned, the maximum value of the B axis is also `128`.

    - The N axis can be less than or equal to `256`, and the D axis can be less than or equal to `512`. If `inputLayout` is `BSH` or `BSND`, it is recommended that N × D be less than `65535`.

    - The S axis must be less than or equal to `20971520` (20M). In some long sequence scenarios, if the computation load is too large, the PFA operator execution may time out (an AI Core error is reported, and `errorStr` is `timeout or trap error`). In this case, S axis splitting is recommended. Note: The computation load is affected by parameters such as `B`, `S`, `N`, and `D`. Larger values indicate larger computation loads. The following lists some typical scenarios with long sequences (that is, the product of `B`, `S`, `N`, and `D` is large).

      <div style="overflow-x: auto;">
      <table style="undefined;table-layout: fixed; width: 930px"><colgroup>
      <col style="width: 130px">
      <col style="width: 130px">
      <col style="width: 230px">
      <col style="width: 130px">
      <col style="width: 130px">
      <col style="width: 180px">
      </colgroup><thead>
      <tr>
      <th>B</th>
      <th>Q_N</th>
      <th>Q_S</th>
      <th>D</th>
      <th>KV_N</th>
      <th>KV_S</th>
      </tr></thead>
      <tbody>
      <tr>
      <td>1</td>
      <td>20</td>
      <td>2097152</td>
      <td>256</td>
      <td>1</td>
      <td>2097152</td>
      </tr>
      <tr>
      <td>1</td>
      <td>2</td>
      <td>20971520</td>
      <td>256</td>
      <td>2</td>
      <td>20971520</td>
      </tr>
      <tr>
      <td>20</td>
      <td>1</td>
      <td>2097152</td>
      <td>256</td>
      <td>1</td>
      <td>2097152</td>
      </tr>
      <tr>
      <td>1</td>
      <td>10</td>
      <td>2097152</td>
      <td>512</td>
      <td>1</td>
      <td>2097152</td>
      </tr>
      </tbody>
      </table>
      </div>

    - Restrictions on the D axis: <term>For Atlas A2 training products/Atlas A2 inference products</term>, if `query`, `key`, `value`, or `attentionOut` is of the INT8 type, the D axis must be 32-byte aligned. If `query`, `key`, `value`, or `attentionOut` is of the INT4 type, the D axis must be 64-byte aligned. If they are all of the FLOAT16 or BFLOAT16 type, the D axis must be 16-byte aligned.

  - For the input parameter `actualSeqLengths`, the value must be a non-negative number.

    - <term>For Atlas A2 training products/Atlas A2 inference products</term>, the valid sequence length of each batch in this input parameter must be less than or equal to the sequence length of the corresponding batch in `query`. If the input length of `seqlen` is `1`, all batches use the same `seqlen`. If the input length is greater than or equal to the batch size, the first *N* elements (where *N* equals the batch size) of `seqlen` are used. Other lengths are not supported.  
  - For the input parameter `actualSeqLengthsKv`, the value must be a non-negative number.

    - <term>For Atlas A2 training products/Atlas A2 inference products</term>, the valid sequence length of each batch in this input parameter must be less than or equal to the sequence length of the corresponding batch in `key` or `value`. If the input length of `seqlenKv` is `1`, all batches use the same `seqlenKv`. If the input length is greater than or equal to the batch size, the first *N* elements (where *N* equals the batch size) of `seqlenKv` are used. Other lengths are not supported.  
  - Currently, `sparseMode` can only be set to `0`, `1`, `2`, `3`, or `4`. An error will be reported if it is set to other values.

    - When `sparseMode` is set to `0`, if `attenMask` is a null pointer or is passed in the left padding scenario, the input parameters `preTokens` and `nextTokens` are ignored.
    - When `sparseMode` is set to `2`, `3`, or `4`, the shape of `attenMask` must be `(S, S)`, `(1, S, S)`, or `(1, 1, S, S)`, where `S` must be fixed at `2048`. Additionally, the input`attenMask` must be a lower triangular matrix. If `attenMask` is not passed or the passed shape is incorrect, an error will be reported.
    - When `sparseMode` is set to `1`, `2`, or `3`, the input parameters `preTokens` and `nextTokens` are ignored, and their values are assigned based on related rules.
  - In the synthesis parameter scenario of KV cache dequantization, only when `query` is of the FLOAT16 type can `key` and `value` of the INT8 type be dequantized to FLOAT16. If the product of the data ranges of the input parameters `key` and `value` and the data range of the input parameter `antiquantScale` must be within the range of (–1, 1), the high-performance mode can ensure precision. Otherwise, the high-precision mode needs to be enabled to ensure precision.
  - Page attention scenario:

    - The prerequisite for enabling page attention is that `blockTable` exists and is valid, and `key` and `value` are arranged in a continuous memory based on the indexes in `blockTable`. In this scenario, `inputLayout` of `key` and `value` is invalid. `blockTable` is filled with block IDs. Currently, the validity of block IDs is not verified. You need to ensure the validity of block IDs.
      - <term>For Atlas A2 training products/Atlas A2 inference products</term>, the data types of `key` and `value` can be FLOAT16, BFLOAT16, or INT8.
    - `BlockSize` is a user-defined parameter. Its value affects the page attention performance. When page attention is enabled, the value of `BlockSize` must be a multiple of `128`, ranging from `128` to `512`. Generally, page attention can improve the throughput but deteriorate the performance.
    - In the page attention scenario, if the input KV cache layout is BnBsH `(BlockNum, BlockSize, H)` and the product of `KV_N` multiplied by `D` exceeds `65535`, an error will be reported due to hardware instruction constraints. This problem can be solved by enabling GQA (decreasing `KV_N`) or adjusting the KV cache layout to BnNBsD `(BlockNum, KV_N, BlockSize, D)`. When `inputLayout` of `query` is `BNSD` or `TND`, the KV cache layout can be BnBsH or BnNBsD. When `inputLayout` of `query` is `BSH` or `BSND`, the KV cache layout can only be BnBsH. The value of `BlockNum` cannot be less than the sum of blocks in each batch calculated based on `actualSeqLengthsKv` and `BlockSize`. The shapes of `key` and `value` must be the same.
    - Page attention fake-quantization scenario
      - <term>For Atlas A2 training products/Atlas A2 inference products</term>, the data type of `query` can be FLOAT16 or BFLOAT16, and the data types of `key` and `value` can be INT8.
    - Page attention full-quantization scenario
      - <term>For Atlas A2 training products/Atlas A2 inference products</term>, the data type of `query` cannot be INT8.
    - Page attention does not support the tensor list or left padding.
    - In the page attention scenario, `actualSeqLengthsKv` must be passed.
    - In the page attention scenario, `blockTable` must be two-dimensional. The length of the first dimension must be equal to `B`, and the length of the second dimension must be greater than or equal to `maxBlockNumPerSeq` (the maximum number of blocks corresponding to `actualSeqLengthsKv` in different batches).
    - When page attention is enabled, the input `KV_S` must be greater than or equal to `maxBlockNumPerSeq` × `BlockSize` in the following scenarios:
      - `attenMask` is passed, for example, when the mask shape is `(B, 1, Q_S, KV_S)`.
      - `pseShift` is passed, for example, when the `pseShift` shape is `(B, N, Q_S, KV_S)`.
  - Left padding for `query`:

    - The transfer start point of `query` is calculated as follows: `Q_S` – `queryPaddingSize` – `actualSeqLengths`. The transfer end point of `query` is calculated as follows: `Q_S` – `queryPaddingSize`. The transfer start point of `query` cannot be less than `0`, while the end point cannot be greater than `Q_S`. Otherwise, the result will not meet the expectation.
    - If `kvPaddingSize` is less than `0`, it will be set to `0`.
    - It must be enabled together with `actualSeqLengths`. Otherwise, the default scenario is right padding for `query`.
    - It does not support page attention and cannot be enabled together with `blockTable`.
    - It is not supported when `Q` is of type BF16/FP16 and `KV` is of type INT4.
  - Left padding for `kv`:

    - The transfer start point of `key` and `value` is calculated as follows: `KV_S` – `kvPaddingSize` – `actualSeqLengthsKv`. The transfer end point of `key` and `value` is calculated as follows: `KV_S` – `kvPaddingSize`. The transfer start point of `key` and `value` cannot be less than `0`, while the end point cannot be greater than `KV_S`. Otherwise, the result will not meet the expectation.
    - If `kvPaddingSize` is less than `0`, it will be set to `0`.
    - It must be enabled together with `actualSeqLengthsKv`. Otherwise, the default scenario is right padding for `kv`.
    - It does not support page attention and cannot be enabled together with `blockTable`.
    - It is not supported when `Q` is of type BF16/FP16 and `KV` is of type INT4.
  - When the output is of type INT8 and `quantScale2` and `quantOffset2` are per-channel, left padding, ring attention, or non-32-byte alignment of the D axis is not supported.
  - When the output is of type INT8, `sparse` cannot be `band` and `preTokens` or `nextTokens` cannot be negative.
  - Restrictions on `pseShift`:

    - This function is supported when the data type of `query` is FLOAT16, BFLOAT16, or INT8.
    - When the data type of `query` is FLOAT16 and `pseShift` exists, the high-precision mode is forcibly used. The corresponding restrictions are the same as those of the high-precision mode.
    - `Q_S` must be greater than or equal to `S` of `query`, and `KV_S` must be greater than or equal to `S` of `key`. In the prefix scenario, `KV_S` must be greater than or equal to the sum of `actualSharedPrefixLen` and `S` of `key`.
  - When the output is of type INT8, if the input parameter`quantOffset2` is a non-null pointer and a non-null tensor, and `sparseMode`, `preTokens`, and `nextTokens` meet the following conditions, certain rows of the matrix will not be involved in computation, resulting in a computation result error. In this scenario, the computation will be intercepted. (Solution: To prevent interception, perform post-quantization outside the FIA interface.)

    - When `sparseMode` is `0` and `attenMask` is a non-null pointer, interception occurs if for any batch: `actualSeqLengths` – `actualSeqLengthsKV` – `actualSharedPrefixLen` – `preTokens` > `0`, or `nextTokens` < `0`.
    - When `sparseMode` is `1` or `2`, interception does not occur.
    - When `sparseMode` is `3`, interception occurs if for any batch: `actualSeqLengthsKV` + `actualSharedPrefixLen` – `actualSeqLengths` < `0`.
    - When `sparseMode` is `4`, interception occurs if for any batch: `preTokens` < `0`, or `nextTokens` + `actualSeqLengthsKV` + `actualSharedPrefixLen` – `actualSeqLengths` < `0`.
  - Restrictions on prefix parameters:

    - Both `keySharedPrefix` and `valueSharedPrefix` must be either null or non-null.
    - If neither `keySharedPrefix` nor `valueSharedPrefix` is null, the dimensions and data types of `keySharedPrefix`, `valueSharedPrefix`, `key`, and `value` must be the same.
    - If neither `keySharedPrefix` nor `valueSharedPrefix` is null, the first dimension (batch) of the shape of `keySharedPrefix` must be `1`. When the layout is `BNSD` or `BSND`, the N and D axes must be the same as those of `key`. When the layout is `BSH`, the H axis must be the same as that of `key`. The same rules apply to `valueSharedPrefix`. `S` of `keySharedPrefix` and `valueSharedPrefix` must be the same.
    - When `actualSharedPrefixLen` exists, its shape must be `[1]`, and its value cannot be greater than `S` of `keySharedPrefix` and `valueSharedPrefix`.
    - The sum of `S` of the public prefix and `S` of `key` or `value` must meet the original restriction on `S` of `key` or `value`.
    - The prefix does not support page attention, left padding, or tensor list.
    - In the prefix scenario, when `sparse` is `0` or `1` and `attenMask` is passed, `S2` must be greater than or equal to the sum of `actualSharedPrefixLen` and `S` of `key`.
    - In the prefix scenario, the input `qkv` cannot be all INT8.
  - KV fake-quantization parameter separation:

    - <term>Atlas A2 training products/Atlas A2 inference products</term>:
      - `keyAntiquantMode` and `valueAntiquantMode` must be the same.
      - Both `keyAntiquantScale` and `valueAntiquantScale` must be either null or non-null. Both `keyAntiquantOffset` and `valueAntiquantOffset` must be either null or non-null.
      - If neither `keyAntiquantScale` nor `valueAntiquantScale` is null, their shapes must be the same. If neither `keyAntiquantOffset` nor `valueAntiquantOffset` is null, their shapes must be the same.
      - Only the per-token and per-channel modes are supported. In per-token mode, the shapes of the two parameters must be both `\(B, S\)`, and the data type is fixed at FLOAT32. In per-channel mode, the shapes of the two parameters must be `(N, D\)`, `(N, 1, D\)`, or `(H\)`, and the data type is fixed at BF16.
      - When both fake-quantization parameters and KV separation quantization parameters are passed, the KV separation quantization parameters take effect.
      - When `keyAntiquantScale` and `valueAntiquantScale` are non-null, `S` of `query` must be less than or equal to `16`.
      - When `keyAntiquantScale` and `valueAntiquantScale` are non-null, the data type of `query` must be BFLOAT16, the data types of `key` and `value` must be INT8, and the data type of the output must be BFLOAT16.
      - When `keyAntiquantScale` and `valueAntiquantScale` are non-null, the tensor list, left padding, and page attention functions are not supported.

- **When `Q_S` is equal to `1`:**

  - Restrictions on `query`, `key`, and `value`:
    - The B axis can be less than or equal to `65536`, the N axis can be less than or equal to `256`, and the D axis can be less than or equal to `512`.
    - The input types of `query`, `key`, and `value` cannot be all INT8.
    - In INT4 (INT32) fake-quantization scenarios, the aclnn single-operator call supports KV inputs in either INT4 format or INT4-packed INT32 format. Using `dynamicQuant` to generate INT4 data is recommended, which stores eight INT4 elements within one INT32.
    - In INT4 (INT32) fake-quantization scenarios, if KV INT4 values are packed into INT32 inputs, the `N`, `D`, or `H` dimensions of KV must be 1/8 of their actual values (the same applies to prefix).
    - Restrictions on the D axis for `key` and `value` in specific data types
      - <term>For Atlas A2 training products/Atlas A2 inference products</term>, when the input type of `key` and `value` is INT4 (INT32), the D axis must be 64-byte aligned (or 8-byte aligned for INT32).
  - For the input parameter `actualSeqLengths`, the value must be a non-negative number.
    - <term>For Atlas A2 training products/Atlas A2 inference products</term>, when `inputLayout` of `query` is not `TND`, this parameter is invalid if `Q_S` is `1`.  
  - For the input parameter `actualSeqLengthsKv`, the value must be a non-negative number.
    - <term>For Atlas A2 training products/Atlas A2 inference products</term>, the valid sequence length of each batch in this input parameter must be less than or equal to the sequence length of the corresponding batch in `key` or `value`. If the input length of `seqlenKv` is `1`, all batches use the same `seqlenKv`. If the input length is greater than or equal to the batch size, the first *N* elements (where *N* equals the batch size) of `seqlenKv` are used. Other lengths are not supported.  
  - Page attention scenario:
    - The prerequisite for enabling page attention is that `blockTable` exists and is valid, and `key` and `value` are arranged in a continuous memory based on the indexes in `blockTable`. In this scenario, `inputLayout` of `key` and `value` is invalid.
      - <term>For Atlas A2 training products/Atlas A2 inference products</term>, the data types of `key` and `value` can be FLOAT16, BFLOAT16, or INT8.
    - `BlockSize` is a user-defined parameter. Its value affects the page attention performance. When page attention is enabled, `BlockSize` must be set to a non-zero value, and the maximum value of `BlockSize` cannot exceed `512`. Generally, page attention can improve the throughput but deteriorate the performance.
      - <term>For Atlas A2 training products/Atlas A2 inference products</term>, if the input type of `key` and `value` is FLOAT16 or BFLOAT16, 16-byte alignment is required. If the input type of `key` and `value` is INT8, 32-byte alignment is required (128-byte alignment is recommended).
    - In the page attention scenario, when `inputLayout` of `query` is `BNSD` or `TND`, the KV cache layout can be BnBsH `(BlockNum, BlockSize, H)` or BnNBsD `(BlockNum, KV_N, BlockSize, D)`. When `inputLayout` of `query` is `BSH` or `BSND`, the KV cache layout can only be BnBsH. The value of `BlockNum` cannot be less than the sum of blocks in each batch calculated based on `actualSeqLengthsKv` and `BlockSize`. The shapes of `key` and `value` must be the same.
    - In the page attention scenario, the performance is generally better when the KV cache layout is BnNBsD than when it is BnBsH. Therefore, BnNBsD is recommended.
    - In the page attention scenario, if the input KV cache layout is BnBsH and `numKvHeads` × `headDim` exceeds 64 KB, an error will be reported due to hardware instruction constraints. This problem can be solved by enabling GQA (decreasing `numKvHeads`) or adjusting the KV cache layout to BnNBsD.
    - Page attention does not support the tensor list or left padding:
        - <term>For Atlas A2 training products/Atlas A2 inference products</term>, scenarios where `Q` is of type BF16/FP16 and `KV` is of type INT4 (INT32) are not supported.
    - In the page attention scenario, `actualSeqLengthsKv` must be passed.
    - In the page attention scenario, `blockTable` must be two-dimensional. The length of the first dimension must be equal to `B`, and the length of the second dimension must be greater than or equal to `maxBlockNumPerSeq` (the maximum number of blocks corresponding to `actualSeqLengthsKv` in each batch).
    - When page attention is enabled, the input `S` must be greater than or equal to `blockTable` second dimension × `BlockSize` in the following scenarios:
      - `attenMask` is enabled, for example, when the mask shape is `\(B, 1, 1, S\)`.
      - `pseShift` is enabled, for example, when the `pseShift` shape is `\(B, N, 1, S\)`.
      - When the fake-quantization per-token mode is enabled, the shape of the input parameters `antiquantScale` and `antiquantOffset` is both `\(2, B, S\)`.
      - Per-token + per-head mode: The shapes of the two parameters are both `\(B, N, S\)`, their data type is fixed at FLOAT32, and the data types of `key` and `value` are INT8 or INT4 (INT32).
      - Per-token-group mode: The shape of `antiquantScale` is `\(1, B, N, S, D/32\)`, and its data type is fixed at FLOAT8_E8M0. `antiquantOffset` is not supported. The data types of `key` and `value` are FLOAT4_E1M2 or FLOAT4_E2M1.
    - Left padding for `kv`:
      - <term>For Atlas A2 training products/Atlas A2 inference products</term>, scenarios where `Q` is of type BF16/FP16 and `KV` is of type INT4 (INT32) are not supported.
      - The transfer start point of `kvCache` is calculated as follows: `KV_S` – `kvPaddingSize` – `actualSeqLengths`. The transfer end point of `kvCache` is calculated as follows: `KV_S` – `kvPaddingSize`. If the transfer start point or end point is less than `0`, the returned data is all 0s.
      - If `kvPaddingSize` is less than `0`, it will be set to `0`.
      - It must be enabled together with `actualSeqLengths`. Otherwise, the default scenario is right padding for `kv`.
      - Page attention and tensor list are not supported. Otherwise, the default scenario is right padding for `kv`.
      - When it is enabled together with `attenMask`, ensure that the meaning of `attenMask` is correct, that is, invalid data can be correctly masked. Otherwise, precision issues may occur.
    - Restrictions on `pseShift`:
      - The data types of `pseShift` and `query` must be the same.
    - KV fake-quantization parameter separation:
      - Except when `keyAntiquantMode` is `0` and `valueAntiquantMode` is `1`, the values of `keyAntiquantMode` and `valueAntiquantMode` must be the same.
      - Both `keyAntiquantScale` and `valueAntiquantScale` must be either null or non-null. Both `keyAntiquantOffset` and `valueAntiquantOffset` must be either null or non-null.
      - If neither `keyAntiquantScale` nor `valueAntiquantScale` is null, their shapes must be the same, except when `keyAntiquantMode` is `0` and `valueAntiquantMode` is `1`. If neither `keyAntiquantOffset` nor `valueAntiquantOffset` is null, their shapes must be the same, except when `keyAntiquantMode` is `0` and `valueAntiquantMode` is `1`.
      - <term>Atlas A2 training products/Atlas A2 inference products</term>: The following eight modes are supported: per-channel, per-tensor, per-token, per-tensor + per-head, per-token + per-head, per-token + page attention to manage scale/offset, per-token + per-head + page attention to manage scale/offset, and per-channel for `key` + per-token for `value`. In the following description, `N` indicates `numKeyValueHeads`.
        - Per-channel mode: The shapes of the two parameters can be `\(1, N, 1, D\)`, `\(1, N, D\)`, or `\(1, H\)`, their data type is the same as that of `query`, and the data types of `key` and `value` are INT8 or INT4 (INT32).
        - Per-tensor mode: The shapes of the two parameters are both `\(1\)`, their data type is the same as that of `query`, and the data types of `key` and `value` are INT8.
        - Per-token mode: The shapes of the two parameters are both `\(1, B, S\)`, their data type is fixed at FLOAT32, and the data types of `key` and `value` are INT8 or INT4 (INT32).
        - Per-tensor + per-head mode: The shapes of the two parameters are both `\(N\)`, their data type is the same as that of `query`, and the data types of `key` and `value` are INT8.
        - Per-channel for `key` + per-token for `value` mode: In per-channel for `key`, the shapes of the two parameters can be `\(1, N, 1, D\)`, `\(1, N, D\)`, or `\(1, H\)`, and their data type is the same as that of `query`. In per-token for `value`, the shapes of the two parameters are both `\(1, B, S\)`, and their data type is fixed at FLOAT32. The data types of `key` and `value` are INT8 or INT4 (INT32). When the data types of `key` and `value` are INT8, only the data types of `query` and `attentionOut` can be FLOAT16.
      - Per-token + per-head mode: The shapes of the two parameters are both `(B, N, S)`, their data type is fixed at FLOAT32, and the data types of `key` and `value` are INT8 or INT4 (INT32).
      - Per-token + page attention to manage scale/offset: The shapes of the two parameters are both `(BlockNum, BlockSize)`, their data type is fixed at FLOAT32, and the data types of `key` and `value` are INT8.
      - Per-token + per-head + page attention to manage scale/offset: The shapes of the two parameters are both `(BlockNum, N, BlockSize)`, their data type is fixed at FLOAT32, and the data types of `key` and `value` are INT8.
      - When both fake-quantization parameters and KV separation quantization parameters are passed, the KV separation quantization parameters take effect.
      - In the INT4 (INT32) fake-quantization scenario where only KV fake-quantization parameter separation is supported, the following modes are supported:
        - Per-channel
        - Per-token
        - Per-token + per-head
        - Per-channel for `key` + per-token for `value`
      - Post-quantization is not supported in INT4 (INT32) fake-quantization scenarios.
    - Restrictions on prefix parameters:
      - Both `keySharedPrefix` and `valueSharedPrefix` must be either null or non-null.
      - If neither `keySharedPrefix` nor `valueSharedPrefix` is null, the dimensions and data types of `keySharedPrefix`, `valueSharedPrefix`, `key`, and `value` must be the same.
      - If neither `keySharedPrefix` nor `valueSharedPrefix` is null, the first dimension (batch) of the shape of `keySharedPrefix` must be `1`. When the layout is `BNSD` or `BSND`, the N and D axes must be the same as those of `key`. When the layout is `BSH`, the H axis must be the same as that of `key`. The same rules apply to `valueSharedPrefix`. `S` of `keySharedPrefix` and `valueSharedPrefix` must be the same.
      - When `actualSharedPrefixLen` exists, its shape must be `[1]`, and its value cannot be greater than `S` of `keySharedPrefix` and `valueSharedPrefix`.
      - The sum of `S` of the public prefix and `S` of `key` or `value` must meet the original restriction on `S` of `key` or `value`.

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```cpp
#include <iostream>
#include <vector>
#include <math.h>
#include <cstring>
#include "acl/acl.h"
#include "aclnn/opdev/fp16_t.h"
#include "aclnnop/aclnn_fused_infer_attention_score_v3.h"

using namespace std;

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

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
  int64_t shapeSize = 1;
  for (auto i : shape) {
      shapeSize *= i;
  }
  return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream) {
  // (Fixed writing) Initialize AscendCL.
  auto ret = aclInit(nullptr);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
  ret = aclrtSetDevice(deviceId);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
  ret = aclrtCreateStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
  return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                  aclDataType dataType, aclTensor** tensor) {
  auto size = GetShapeSize(shape) * aclDataTypeSize(dataType);
  // Call aclrtMalloc to allocate memory on the device.
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  // Call aclrtMemcpy to copy the data from the host to the device.
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // Calculate the strides of consecutive tensors.
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
      strides[i] = shape[i + 1] * strides[i + 1];
  }

  // Call aclCreateTensor to create an aclTensor.
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

int main() {
  // 1. (Fixed writing) Initialize the device and stream. For details, see the list of external AscendCL APIs.
  // Set the device ID in use.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the inputs and outputs based on the API definition.
  int32_t batchSize = 1;
  int32_t numHeads = 2;
  int32_t sequenceLengthQ = 1;
  int32_t headDims = 16;
  int32_t numKeyValueHeads = 2;
  int32_t sequenceLengthKV = 16;
  std::vector<int64_t> queryShape = {batchSize, numHeads, sequenceLengthQ, headDims};           // BNSD
  std::vector<int64_t> keyShape = {batchSize, numKeyValueHeads, sequenceLengthKV, headDims};    // BNSD
  std::vector<int64_t> valueShape = {batchSize, numKeyValueHeads, sequenceLengthKV, headDims};  // BNSD
  std::vector<int64_t> attenMaskShape = {batchSize, 1, sequenceLengthQ, sequenceLengthKV};      // B 1 S1 S2
  std::vector<int64_t> outShape = {batchSize, numHeads, sequenceLengthQ, headDims};             // BNSD
  void *queryDeviceAddr = nullptr;
  void *keyDeviceAddr = nullptr;
  void *valueDeviceAddr = nullptr;
  void *attenMaskDeviceAddr = nullptr;
  void *outDeviceAddr = nullptr;
  aclTensor *queryTensor = nullptr;
  aclTensor *keyTensor = nullptr;
  aclTensor *valueTensor = nullptr;
  aclTensor *attenMaskTensor = nullptr;
  aclTensor *outTensor = nullptr;
  int64_t queryShapeSize = GetShapeSize(queryShape);          // BNSD
  int64_t keyShapeSize = GetShapeSize(keyShape);              // BNSD
  int64_t valueShapeSize = GetShapeSize(valueShape);          // BNSD
  int64_t attenMaskShapeSize = GetShapeSize(attenMaskShape);  // B 1 S1 S2
  int64_t outShapeSize = GetShapeSize(outShape);              // BNSD
  std::vector<op::fp16_t> queryHostData(queryShapeSize, 1.0f);
  std::vector<op::fp16_t> keyHostData(keyShapeSize, 1.0f);
  std::vector<op::fp16_t> valueHostData(valueShapeSize, 1.0f);
  std::vector<int8_t> attenMaskHostData(attenMaskShapeSize, 0);
  std::vector<op::fp16_t> outHostData(outShapeSize, 1.0f);

  // Create a query aclTensor.
  ret = CreateAclTensor(queryHostData, queryShape, &queryDeviceAddr, aclDataType::ACL_FLOAT16, &queryTensor);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a key aclTensor.
  ret = CreateAclTensor(keyHostData, keyShape, &keyDeviceAddr, aclDataType::ACL_FLOAT16, &keyTensor);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  int kvTensorNum = 1;
  aclTensor *tensorsOfKey[kvTensorNum];
  tensorsOfKey[0] = keyTensor;
  auto tensorKeyList = aclCreateTensorList(tensorsOfKey, kvTensorNum);
  // Create a value aclTensor.
  ret = CreateAclTensor(valueHostData, valueShape, &valueDeviceAddr, aclDataType::ACL_FLOAT16, &valueTensor);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  aclTensor *tensorsOfValue[kvTensorNum];
  tensorsOfValue[0] = valueTensor;
  auto tensorValueList = aclCreateTensorList(tensorsOfValue, kvTensorNum);
  // Create an attenMask aclTensor.
  ret = CreateAclTensor(attenMaskHostData, attenMaskShape, &attenMaskDeviceAddr, aclDataType::ACL_BOOL, &attenMaskTensor);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT16, &outTensor);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  std::vector<int64_t> actualSeqlenVector = {sequenceLengthKV};
  auto actualSeqLengths = aclCreateIntArray(actualSeqlenVector.data(), actualSeqlenVector.size());

  double scaleValue = 1 / sqrt(headDims); // 1/sqrt(d)
  int64_t preTokens = 65535;
  int64_t nextTokens = 65535;
  string sLayerOut = "BNSD";
  char layerOut[sLayerOut.length()];
  strcpy(layerOut, sLayerOut.c_str());
  int64_t sparseMode = 0;
  int64_t innerPrecise = 1;
  int blockSize = 0;
  int antiquantMode = 0;
  bool softmaxLseFlag = false;
  int keyAntiquantMode = 0;
  int valueAntiquantMode = 0;
  // 3. Call the CANN operator library API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API.
  ret = aclnnFusedInferAttentionScoreV3GetWorkspaceSize(queryTensor, tensorKeyList, tensorValueList,  nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, numHeads, scaleValue, preTokens, nextTokens, layerOut, numKeyValueHeads, sparseMode, innerPrecise, blockSize, antiquantMode, softmaxLseFlag, keyAntiquantMode, valueAntiquantMode, outTensor, nullptr, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFusedInferAttentionScoreV3GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspaceSize calculated by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
      ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API.
  ret = aclnnFusedInferAttentionScoreV3(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFusedInferAttentionScoreV3 failed. ERROR: %d\n", ret); return ret);

  // 4. (Fixed writing) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device to the host. Modify the code based on the API definition.
  auto size = GetShapeSize(outShape);
  std::vector<op::fp16_t> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
      std::cout << "index: " << i << ": " << static_cast<float>(resultData[i]) << std::endl;
  }

  // 6. Release resources.
  aclDestroyTensor(queryTensor);
  aclDestroyTensor(keyTensor);
  aclDestroyTensor(valueTensor);
  aclDestroyTensor(attenMaskTensor);
  aclDestroyTensor(outTensor);
  aclDestroyIntArray(actualSeqLengths);
  aclrtFree(queryDeviceAddr);
  aclrtFree(keyDeviceAddr);
  aclrtFree(valueDeviceAddr);
  aclrtFree(attenMaskDeviceAddr);
  aclrtFree(outDeviceAddr);
  if (workspaceSize > 0) {
      aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
