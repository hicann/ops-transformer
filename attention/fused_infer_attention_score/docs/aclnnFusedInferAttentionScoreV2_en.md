# aclnnFusedInferAttentionScoreV2

## Applicable Products

|Product      | Supported or Not |
|:----------------------------|:-----------:|
|<term>Atlas A3 training products/Atlas A3 inference products</term>|      ×     |
|<term>Atlas A2 training products/Atlas A2 inference products</term>|      √     |
|<term>Atlas 200I/500 A2 inference products</term>|      ×     |
|<term>Atlas inference accelerator cards</term>|      ×     |
|<term>Atlas training products</term>|      ×     |

## Function Description

- A FlashAttention operator adapted for both full and incremental inference scenarios. It supports both full computation (`PromptFlashAttention`) and incremental computation (`IncreFlashAttention`). When `S` of the query matrix is 1, the IncreFlashAttention branch is executed. In other scenarios, the PromptFlashAttention branch is executed.

- Formula

  Self-attention constructs an attention model by leveraging the relationships within the input samples. The principle assumes an input sample sequence $x$ of length $n$, where each element of $x$ is a $d$-dimensional vector. Each $d$-dimensional vector can be regarded as a token embedding. Such a sequence is transformed by three weight matrices to produce three matrices of $n*d$.

  The calculation formula for self-attention is generally defined as follows, where $Q$, $K$, and $V$ are key attribute elements of the input sample, obtained through spatial transformation and unified into a single feature space. "Attention" in the formula and operator name is an abbreviation for "self-attention."

  $$
  Attention(Q,K,V)=Score(Q,K)V
  $$

  The score function in this operator employs the softmax function. The self-attention calculation formula is as follows:

  $$
  Attention(Q,K,V)=Softmax(\frac{QK^T}{\sqrt{d}})V
  $$

  The product of $Q$ and $K^T$ represents the attention to the input $x$. To prevent this value from becoming excessively large, it is typically scaled by dividing by the square root of $d$, followed by row-wise softmax normalization. The result is then multiplied by $V$ to produce an $n*d$ matrix.

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnFusedInferAttentionScoreV2GetWorkspaceSize` is called to obtain the input parameters and compute the required workspace size based on the process. Then, `aclnnFusedInferAttentionScoreV2` is called to perform computation.

```cpp
aclnnStatus aclnnFusedInferAttentionScoreV2GetWorkspaceSize(
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
aclnnStatus aclnnFusedInferAttentionScoreV2(
     void                *workspace,
     uint64_t             workspaceSize,
     aclOpExecutor       *executor,
     const aclrtStream    stream)
```

## aclnnFusedInferAttentionScoreV2GetWorkspaceSize

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
        <th>Parameter</th>
        <th>Input/Output</th>
        <th>Description</th>
        <th>Instruction</th>
        <th>Data Type</th>
        <th>Data Format</th>
        <th>Dimension (Shape)</th>
        <th>Non-contiguous Tensor</th>
      </tr></thead>
    <tbody>
      <tr>
        <td>query</td>
        <td>Input</td>
        <td>Input Q in the formula.</td>
        <td>-</td>
        <td>FLOAT16, BFLOAT16, INT8</td>
        <td>ND</td>
        <td>3-4</td>
        <td>×</td>
      </tr>
      <tr>
        <td>key</td>
        <td>Input</td>
        <td>Input K in the formula.</td>
        <td>-</td>
        <td>FLOAT16, BFLOAT16, INT8, INT4 (INT32), HIFLOAT8, FLOAT8_E4M3FN, FLOAT4_E1M2, FLOAT4_E2M1</td>
        <td>ND</td>
        <td>3-4</td>
        <td>×</td>
      </tr>
      <tr>
        <td>value</td>
        <td>Input</td>
        <td>Input V in the formula.</td>
        <td>-</td>
        <td>FLOAT16, BFLOAT16, INT8, INT4 (INT32), HIFLOAT8, FLOAT8_E4M3FN, FLOAT4_E1M2, FLOAT4_E2M1</td>
        <td>ND</td>
        <td>3-4</td>
        <td>×</td>
      </tr>
      <tr>
        <td>pseShiftOptional</td>
        <td>Input</td>
        <td>Positional encoding.</td>
        <td><ul><li>Empty tensors are not supported.</li>
        <li>If this parameter is not used, pass nullptr.</li>
            <li>For details about the constraints, see <a href="#constraints">Constraints</a>.</li></ul></td>
        <td>FLOAT16, BFLOAT16</td>
        <td>ND</td>
        <td>4</td>
        <td>×</td>
      </tr>
      <tr>
        <td>attenMaskOptional</td>
        <td>Input</td>
        <td>Mask matrix.</td>
        <td><ul><li>Empty tensors are not supported.</li>
        <li>If this parameter is not used, pass nullptr.</li>
            <li>For details about the constraints, see <a href="#constraints">Constraints</a>.</li></ul></td>
        <td>BOOL, INT8, UINT8</td>
        <td>ND</td>
        <td>2-4</td>
        <td>×</td>
      </tr>
      <tr>
        <td>actualSeqLengthsOptional</td>
        <td>Input</td>
        <td>Valid sequence length of queries in different batches.</td>
        <td><ul><li>If the sequence length is not specified, pass nullptr.</li></ul>
            <li>For details about the constraints, see <a href="#constraints">Constraints</a>.</li></td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>actualSeqLengthsKvOptional</td>
        <td>Input</td>
        <td>Valid sequence length of key and value in different batches.</td>
        <td><ul><li>If the sequence length is not specified, pass nullptr, indicating that the sequence length is the same as the S length of the key/value shape.</li>
            <li>Restriction: The valid sequence length of each batch in the input parameter must be less than or equal to that of the corresponding batch in key/value. If the input length of `seqlenKv` is 1, all batches use the same `seqlenKv`. If the input length is greater than or equal to the batch size, the first *N* elements (where *N* equals the batch size) of `seqlenKv` are used. Other lengths are not supported.</li>
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
              <li>If this parameter is not used, pass nullptr.</li>
              <li>For details about the constraints, see <a href="#constraints">Constraints</a>.</li></ul></td>
        <td>UINT64, FLOAT32</td>
        <td>ND</td>
        <td>1</td>
        <td>-</td>
      </tr>
      <tr>
        <td>quantScale1Optional</td>
        <td>Input</td>
        <td>Quantization factor before BMM2.</td>
        <td><ul><li>Empty tensors are not supported.</li>
        <li>Per-tensor is supported. </li>
            <li>If this parameter is not used, pass nullptr.</li>
            <li>For details about the constraints, see <a href="#constraints">Constraints</a>.</li></ul></td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>1</td>
        <td>-</td>
      </tr>
      <tr>
        <td>deqScale2Optional</td>
        <td>Input</td>
        <td>Dequantization factor after BMM2.</td>
        <td><ul><li>Empty tensors are not supported.</li>
        <li>Per-tensor is supported. </li>
           <li>If this parameter is not used, pass nullptr.</li>
            <li>For details about the constraints, see <a href="#constraints">Constraints</a>.</li></ul></td>
        <td>UINT64, FLOAT32</td>
        <td>ND</td>
        <td>1</td>
        <td>-</td>
      </tr>
      <tr>
        <td>quantScale2Optional</td>
        <td>Input</td>
        <td>Output quantization factor.</td>
        <td><ul><li>Empty tensors are not supported.</li>
        <li>Per-tensor and per-channel are supported. </li>
            <li>If this parameter is not used, pass nullptr.</li>
             <li>For details about the constraints, see <a href="#constraints">Constraints</a>.</li></ul></td>
        <td>FLOAT32, BFLOAT16</td>
        <td>ND</td>
        <td>1-4</td>
        <td>-</td>
      </tr>
      <tr>
        <td>quantOffset2Optional</td>
        <td>Input</td>
        <td>Output quantization offset.</td>
        <td><ul><li>Empty tensors are not supported.</li>
        <li>Per-tensor and per-channel are supported. </li>
            <li>If this parameter is not used, pass nullptr.</li>
             <li>For details about the constraints, see <a href="#constraints">Constraints</a>.</li></ul></td>
        <td>FLOAT32, BFLOAT16</td>
        <td>ND</td>
        <td>1-4</td>
        <td>-</td>
      </tr>
      <tr>
        <td>antiquantScaleOptional</td>
        <td>Input</td>
        <td>Fake-quantization factor.</td>
        <td><ul><li>Empty tensors are not supported.</li>
        <li>Per-tensor, per-channel, and per-token are supported. </li>
            <li>If this parameter is not used, pass nullptr.</li>
            <li>For details about the constraints, see <a href="#constraints">Constraints</a>.</li></ul></td>
        <td>FLOAT16, BFLOAT16, FLOAT32</td>
        <td>ND</td>
        <td>1-4</td>
        <td>-</td>
      </tr>
        <tr>
      <td>antiquantOffsetOptional</td>
        <td>Input</td>
        <td>Fake-quantization offset.</td>
        <td><ul><li>Empty tensors are not supported.</li>
        <li>Per-tensor, per-channel, and per-token are supported. </li>
            <li>The shape must be the same as that of antiquantScaleOptional.</li>
            <li>If this parameter is not used, pass nullptr.</li>
            <li>For details about the constraints, see <a href="#constraints">Constraints</a>.</li></ul></td>
        <td>FLOAT16, BFLOAT16, FLOAT32</td>
        <td>ND</td>
        <td>1-4</td>
        <td>-</td>
      </tr> 
    <tr>
       <td>blockTableOptional</td>
        <td>Input</td>
        <td>Block mapping table used for KV storage in page attention.</td>
        <td><ul><li>Empty tensors are not supported.</li>
        <li>If this parameter is not used, pass nullptr.</li></ul></td>
        <td>INT32</td>
        <td>ND</td>
        <td>2</td>
        <td>-</td>
      </tr>
      <tr> 
       <td>queryPaddingSizeOptional</td>
        <td>Input</td>
        <td>Whether the data in each batch of query is right-aligned and the number of right-aligned elements.</td>
        <td><ul><li>Empty tensors are not supported.</li>
        <li>This parameter is valid only when Q_S is greater than 1. In other scenarios, it is invalid.</li>
            <li>If this parameter is not used, pass nullptr.</li></ul></td>
        <td>INT64</td>
        <td>ND</td>
        <td>1</td>
        <td>-</td>
      </tr>
      <tr> 
       <td>kvPaddingSizeOptional</td>
        <td>Input</td>
        <td>Whether the data in each batch of key/value is right-aligned and the number of right-aligned elements.</td>
        <td><ul><li>Empty tensors are not supported.</li>
        <li>If this parameter is not used, pass nullptr.</li></ul></td>
        <td>INT64</td>
        <td>ND</td>
        <td>1</td>
        <td>-</td>
      </tr>
      <tr> 
       <td>keyAntiquantScaleOptional</td>
        <td>Input</td>
        <td>Dequantization factor of key when the KV fake-quantization parameters are separated.</td>
        <td><ul><li>Empty tensors are not supported.</li>
        <li>If this parameter is not used, pass nullptr.</li>
            <li>For details about the constraints, see <a href="#constraints">Constraints</a>.</li></ul></td>
        <td>FLOAT16, BFLOAT16, FLOAT32</td>
        <td>ND</td>
        <td>1-4</td>
        <td>-</td>
      </tr>
        <tr> 
       <td>keyAntiquantOffsetOptional</td>
        <td>Input</td>
        <td>Dequantization offset of key when the KV fake-quantization parameters are separated.</td>
        <td><ul><li>Empty tensors are not supported.</li>
        <li>The shape must be the same as that of keyAntiquantScaleOptional.</li>
            <li>If this parameter is not used, pass nullptr.</li>
            <li>For details about the constraints, see <a href="#constraints">Constraints</a>.</li></ul></td>
        <td>FLOAT16, BFLOAT16, FLOAT32</td>
        <td>ND</td>
        <td>1-4</td>
        <td>-</td>
      </tr>
         <tr> 
       <td>valueAntiquantScaleOptional</td>
        <td>Input</td>
        <td>Dequantization factor of value when the KV fake-quantization parameters are separated.</td>
        <td><ul>
            <li>Empty tensors are not supported.</li>
            <li>If this parameter is not used, pass nullptr.</li>
            <li>For details about the constraints, see <a href="#constraints">Constraints</a>.</li></ul></td>
        <td>FLOAT16, BFLOAT16, FLOAT32</td>
        <td>ND</td>
        <td>1-4</td>
        <td>-</td>
      </tr>
         <tr> 
       <td>valueAntiquantOffsetOptional</td>
        <td>Input</td>
        <td>Dequantization factor of value when the KV fake-quantization parameters are separated.</td>
        <td><ul>
            <li>Empty tensors are not supported.</li>
            <li>If this parameter is not used, pass nullptr.</li>
            <li>For details about the constraints, see <a href="#constraints">Constraints</a>.</li></ul></td>
        <td>FLOAT16, BFLOAT16, FLOAT32</td>
        <td>ND</td>
        <td>1-4</td>
        <td>-</td>
      </tr>        
       <tr> 
       <td>keySharedPrefixOptional</td>
        <td>Input</td>
        <td>System prefix of key in the attention structure.</td>
        <td><ul>
            <li>Empty tensors are not supported.</li>
            <li>If this parameter is not used, pass nullptr.</li>
            <li>For details about the constraints, see <a href="#constraints">Constraints</a>.</li></ul></td>
        <td>FLOAT16, BFLOAT16, INT8</td>
        <td>ND</td>
        <td>3-4</td>
        <td>-</td>
      </tr>
       <tr> 
       <td>valueSharedPrefixOptional</td>
        <td>Input</td>
        <td>System prefix of value in the attention structure.</td>
        <td><ul>
            <li>Empty tensors are not supported.</li>
            <li>If this parameter is not used, pass nullptr.</li>
            <li>For details about the constraints, see <a href="#constraints">Constraints</a>.</li></ul></td>
        <td>FLOAT16, BFLOAT16, INT8</td>
        <td>ND</td>
        <td>3-4</td>
        <td>-</td>
    </tr>
      <tr> 
        <td>actualSharedPrefixLenOptional</td>
        <td>Input</td>
        <td>Valid sequence length of keySharedPrefix/valueSharedPrefix.</td>
        <td><ul><li>If this function is not used, pass nullptr, indicating that the sequence length is the same as S of keySharedPrefix/valueSharedPrefix.</li>
         <li>The valid sequence length in this input parameter must be less than or equal to that in keySharedPrefix/valueSharedPrefix.</li>
         <li>For details about the constraints, see <a href="#constraints">Constraints</a>.</li></ul></td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>       
      <tr>
        <td>numHeads</td>
        <td>Input</td>
        <td>Number of query heads.</td>
        <td>In the BNSD scenario, the value must be the same as the N-axis value of query in the shape. Otherwise, an exception occurs.</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>scaleValue</td>
        <td>Input</td>
        <td>Reciprocal of the square root of d in the formula.</td>
        <td><ul><li>Its data type must be compatible with that of query according to the deduction relationship. </li>
            <li>If no specific value is required, 1.0 is recommended. </li></ul></td>
        <td>DOUBLE</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>preTokens</td>
        <td>Input</td>
        <td>Number of preceding tokens to associate in attention computation for sparse computation.</td>
          <td><ul><li>If no specific value is required, 2147483647 is recommended.</li>
              <li>This parameter is invalid when Q_S is 1.</li></ul></td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>      
      <tr>
        <td>nextTokens</td>
        <td>Input</td>
        <td>Number of succeeding tokens to associate in attention computation.</td>
        <td><ul><li>If no specific value is required, 2147483647 is recommended.</li>
            <li>This parameter is invalid when Q_S is 1.</li></ul></td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>inputLayout</td>
        <td>Input</td>
        <td>Layout of the input query, key, and value.</td>
        <td><ul><li>If no specific value is required, BSH is recommended.</li></ul>
            <ul><li>For details about the constraints, see <a href="#constraints">Constraints</a>.</li></ul></td>
        <td>CHAR</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>numKeyValueHeads</td>
        <td>Input</td>
        <td>Number of heads in key and value.</td>
        <td><ul><li>If no specific value is required, 0 is recommended, indicating that key, value, and query have the same number of heads.</li></ul>
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
        <td>Choice between high precision and high performance.</td>
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
        <td>The value can only be 0.</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>antiquantMode</td>
        <td>Input</td>
        <td>Fake-quantization mode.</td>
          <td><ul><li>The value 0 indicates per-channel (per-channel contains per-tensor).</li>
              <li>The value 1 indicates per-token.</li>
              <li>If no specific value is required, 0 is recommended.</li>
              <li>When Q_S is 1, an exception occurs if a value other than 0 or 1 is passed. This parameter is invalid when Q_S is greater than or equal to 2.</li></ul></td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>softmaxLseFlag</td>
        <td>Input</td>
        <td>Whether to output softmax_lse.</td>
          <td><ul><li>Supports S-axis outer splittingping (augmented output).</li></ul>
              <li>If no specific value is required, false is recommended.</li></td>
        <td>BOOL</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>        
      <tr>
        <td>keyAntiquantMode</td>
        <td>Input</td>
        <td>Fake-quantization mode of key.</td>
        <td><ul>
            <li>If no specific value is required, 0 is recommended.</li>
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
        <td>Fake-quantization mode of value.</td>
          <td><ul><li>The mode ID is the same as that of keyAntiquantMode.</li>
              <li>If no specific value is required, 0 is recommended.</li>
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
        <td>The D dimension of this input parameter must be the same as that of value, and other dimensions must be the same as the shape of the input query.</td>
        <td>FLOAT16, BFLOAT16, INT8</td>
        <td>ND</td>
        <td>3-4</td>
        <td>-</td>
      </tr>
      <tr>
        <td>softmaxLse</td>
        <td>Output</td>
        <td>Result of query-key multiplication in ring attention.</td>
        <td>For details about the constraints, see <a href="#constraints">Constraints</a>.</td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>4</td>
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
      <td>The passed query, key, value, or attentionOut is a null pointer.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_PARAM_INVALID</td>
      <td>161002</td>
      <td>The data type or data format of query, key, value, pseShift, attenMask, or attentionOut is not supported.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_RUNTIME_ERROR</td>
      <td>361001</td>
      <td>An exception occurred when the NPU Runtime API was called.</td>
    </tr>
  </tbody>
  </table>

## aclnnFusedInferAttentionScoreV2

- **Parameters**
  
  <table style="undefined;table-layout: fixed; width: 1049px"><colgroup>
  <col style="width: 167px">
  <col style="width: 118px">
  <col style="width: 764px">
  </colgroup>
  <thead>
    <tr>
      <th>Parameter</th>
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
      <td>Size of the workspace to be allocated on the device, which is obtained by calling the first-phase API aclnnPromptFlashAttentionV2GetWorkspaceSize.</td>
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

- Deterministic computation
  
  - `aclnnPromptFlashAttentionV2` defaults to a deterministic implementation.
- When this API is used together with PyTorch, ensure that the CANN package versions match the PyTorch package versions.

- Processing logic for a null input parameter: The operator checks whether `query` is a null pointer. If so, an error is reported. If `query` is not an empty tensor but `key` and `value` are empty tensors (that is, S2 is 0), `attentionOut` is filled with all zeros. If `attentionOut` is an empty tensor, the framework will process it. For other input parameters which support the passing of null pointers as described in the preceding parameter description, no processing is performed when they are null pointers.

- The shapes of the tensors corresponding to the `key` and `value` parameters must be identical. In non-contiguous scenarios, the batch size in the tensor lists of `key` and `value` can only be 1, and the number of elements must be equal to the batch size (B) of `query`. The N and D dimensions must be the same. Due to the tensor list restrictions, B cannot be greater than 256 in non-contiguous scenarios.

- Restrictions on `key` and `value`

  - <term>Atlas A2 training products/Atlas A2 inference products</term>: The data type can be FLOAT16, BFLOAT16, INT8, or INT4 (INT32).

- Restrictions on `pseShiftOptional`

  - If `Q_S` is not 1, `query` must be of type FLOAT16 or INT8 when `pseShiftOptional` is of type FLOAT16, and `query` must be of type BFLOAT16 when `pseShiftOptional` is of type BFLOAT16. The input shape must be (B, N, Q_S, KV_S) or (1, N, Q_S, KV_S), where `Q_S` is the S dimension in the shape of `query`, and `KV_S` is the S dimension in the shapes of `key` and `value`. For scenarios where `KV_S` of `pseShiftOptional` is not 32-byte aligned, it is recommended that the value be padded to 32 bytes to improve performance. There is no requirement on the padding value of the extra part.
  - If `Q_S` is 1, `query` must be of type FLOAT16 when `pseShiftOptional` is of type FLOAT16, and `query` must be of type BFLOAT16 when `pseShiftOptional` is of type BFLOAT16. The input shape must be (B, N, 1, KV_S) or (1, N, 1, KV_S), where `KV_S` is the S dimension in the shapes of `key` and `value`. For scenarios where `KV_S` of `pseShiftOptional` is not 32-byte aligned, it is recommended that the value be padded to 32 bytes to improve performance. There is no requirement on the padding value of the extra part.

- `attenMaskOptional`: masks the QK product to define the attention visibility between tokens.

  - If `Q_S` is not 1, the recommended input shapes are (Q_S, KV_S), (B, Q_S, KV_S), (1, Q_S, KV_S), (B, 1, Q_S, KV_S), and (1, 1, Q_S, KV_S).
  - If `Q_S` is 1, the recommended input shapes are (B, KV_S), (B, 1, KV_S), and (B, 1, 1, KV_S).
  - `Q_S` refers to the S dimension in the shape of `query`, while `KV_S` refers to the S dimension in the shapes of `key` and `value`. If `Q_S` or `KV_S` is not 16- or 32-byte aligned, it can be rounded up to the nearest aligned value.

- When the data type of `attenMask` is INT8 or UINT8, the value in the tensor must be 0 or 1.

- `actualSeqLengthsOptional`: aclIntArray on the host, indicating the valid sequence lengths of `query` in different batches. The data type can be INT64. If you do not specify `seqlen`, `nullptr` is passed, indicating that the length is the same as that of `S` in the shape of `query`. Restriction: The valid sequence length of each batch in the input parameter must be less than or equal to that of the corresponding batch in `query`. This parameter is invalid when `Q_S` is `1`. If the input length of `seqlen` is 1, all batches use the same `seqlen`. If the input length is greater than or equal to the batch size, the first *N* elements (where *N* equals the batch size) of `seqlen` are used. Other lengths are not supported.

- `actualSeqLengthsKvOptional`: aclIntArray on the host. It can be set to `nullptr`, indicating the valid sequence lengths of `key`/`value` in different batches. The data type can be INT64. If you do not specify `seqlen`, `nullptr` is passed, indicating that the length is the same as that of `S` in the shape of `key`/`value`. Restriction: The valid sequence length of each batch in the input parameter must be less than or equal to that of the corresponding batch in `key`/`value`. If the input length of `seqlenKv` is 1, all batches use the same `seqlenKv`. If the input length is greater than or equal to the batch size, the first *N* elements (where *N* equals the batch size) of `seqlenKv` are used. Other lengths are not supported.

- Constraints on the number of input parameters and input and output data formats related to INT8 quantization:

  - If both the input and output are of the INT8 type, the input parameters `deqScale1`, `quantScale1`, `deqScale2`, and `quantScale2` must exist at the same time. `quantOffset2` is optional and defaults to 0 if not passed.
  - If the input is of the INT8 type and the output is of the FLOAT16 type, the input parameters `deqScale1`, `quantScale1`, and `deqScale2` must exist at the same time. If the input parameter `quantOffset2` or `quantScale2` exists (not `nullptr`), an error is reported and returned.
  - When the input is of the FLOAT16 or BFLOAT16 type and the output is of the INT8 type, the input parameter `quantScale2` must exist, and `quantOffset2` is optional (defaults to 0 if not passed). If the input parameter `deqScale1`, `quantScale1`, or `deqScale2` exists (not `nullptr`), an error is reported and returned.
  - The input parameters `quantScale2` and `quantOffset2` support both the per-tensor and per-channel formats and the FLOAT32 and BFLOAT16 data types. If `quantOffset2` is passed, ensure that its type and shape are consistent with those of `quantScale2`. When the input is of the BFLOAT16 type, both FLOAT32 and BFLOAT16 are supported. Otherwise, only FLOAT32 is supported. In per-channel format, when the output layout is BSH, the product of all dimensions of `quantScale2` must be equal to **H**. For other layouts, the product must be equal to **N** × **D**. (When the output layout is BSH, it is recommended that the shape of `quantScale2` be set to [1,1,H] or [H]. When the output layout is BNSD, it is recommended that the shape of `quantScale2` be set to [1,N,1,D] or [N,D]. When the output layout is BSND, it is recommended that the shape of `quantScale2` be set to [1,1,N,D] or [N,D].)

- Constraints on the fake-quantization parameters `antiquantScale` and `antiquantOffset`:

  - Only the fake-quantization scenario where `kv_dtype` is INT8 is supported.
  - Per-channel mode: The shapes of the two parameters can be \(2, N, 1, D\), \(2, N, D\), \(2, H\), where **N** is `numKeyValueHeads`. The data type is the same as that of `query`, and `antiquantMode` is set to `0`.
  - Per-tensor mode: The shapes of the two parameters are (2), the data type is the same as that of `query`, and `antiquantMode` is set to `0`.
  - Per-token mode: The shapes of the two parameters are \(2, B, S\), the data type is fixed at FLOAT32, and `antiquantMode` is set to `1`.
  - In asymmetric quantization mode, both `antiquantScale` and `antiquantOffset` must be present.
  - In symmetric quantization mode, `antiquantOffset` can be `nullptr`. If `antiquantOffset` is `nullptr`, symmetric quantization is performed. Otherwise, asymmetric quantization is performed.
  - <term>Atlas A2 training products/Atlas A2 inference products</term>: When `Q_S` is 1, the data type can be FLOAT16, BFLOAT16, or FLOAT32. If `Q_S` is greater than or equal to 2, only FLOAT16 is supported.

- `keyAntiquantOffsetOptional`: aclTensor on the device. The data type can be FLOAT16, BFLOAT16, or FLOAT32. The data format can be ND. When the KV fake-quantization parameters are separated, this parameter indicates the dequantization offset of `key`. If this function is used, the data type and shape must be the same as those of `keyAntiquantScaleOptional`. If this function is not used, pass `nullptr`.

  - <term>Atlas A2 training products/Atlas A2 inference products</term>: When `Q_S` is greater than or equal to 2, only the per-token and per-channel modes are supported. When `Q_S` is 1, only the per-tensor, per-channel, and per-token modes are supported. The per-tensor mode can be combined with the per-head mode, the per-token mode can be combined with the per-head mode, the per-token mode can be combined with the page attention mode to manage scale/offset, and the per-token mode can be combined with the per-head mode and the page attention mode to manage scale/offset.

- `valueAntiquantScaleOptional`: aclTensor on the device. The data format can be ND. When the KV fake-quantization parameters are separated, this parameter indicates the dequantization factor of `value`. When `Q_S` is greater than or equal to 2, only the per-token and per-channel modes are supported. If this function is not used, pass `nullptr`.

  - <term>Atlas A2 training products/Atlas A2 inference products</term>: The data type can be FLOAT16, BFLOAT16, or FLOAT32. The per-tensor, per-channel, and per-token modes are supported. The per-tensor mode can be combined with the per-head mode, the per-token mode can be combined with the per-head mode, the per-token mode can be combined with the page attention mode to manage scale/offset, and the per-token mode can be combined with the per-head mode and the page attention mode to manage scale/offset.

- `actualSharedPrefixLenOptional`: aclIntArray on the host. `nullptr` can be passed, indicating the valid sequence length of `keySharedPrefix`/`valueSharedPrefix`. The data type can be INT64. If `seqlen` is not specified, `nullptr` can be passed, indicating that the length is the same as that of `S` in `keySharedPrefix`/`valueSharedPrefix`. Restriction: The valid sequence length in this input parameter must be less than or equal to that in `keySharedPrefix`/`valueSharedPrefix`.

- `inputLayout`: data layout of the input `query`, `key`, and `value`. Currently, BSH, BSND, BNSD, and BNSD_BSND are supported. (If the input layout is BNSD, the output layout is BSND, and only `Q_S` greater than 1 is supported.) If no specific layout is required, BSH is recommended.

  - Note: The data layout of `query`, `key`, and `value` can be interpreted from multiple dimensions. **B (Batch)** indicates the batch size of input samples, **S (Seq-Length)** indicates the sequence length of input samples, **H (Head-Size)** indicates the size of the hidden layer, **N (Head-Num)** indicates the number of heads, and **D (Head-Dim)** indicates the minimum unit size of the hidden layer (**D** = **H**/**N**).

- Restrictions on `numKeyValueHeads`: `numHeads` must be exactly divided by `numKeyValueHeads`, and the ratio of `numHeads` to `numKeyValueHeads` cannot be greater than 64. In the BSND, BNSD, and BNSD_BSND scenarios, the value must be the same as the N-axis value of `key`/`value` in the shape. Otherwise, an exception occurs.

- Restrictions on `sparseMode`

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
                <td>defaultMask mode</td>
                <td>
                    <ul style="margin: 0; padding-left: 20px;">
                        <li>If attenmask is not passed, the mask operation is not performed, and preTokens and nextTokens are ignored (internally set to INT_MAX).</li>
                        <li>If attenmask is passed, a complete attenmask matrix (S1 × S2) needs to be passed, indicating that the portion between preTokens and nextTokens needs to be calculated.</li>
                    </ul>
                </td>
            </tr>
            <tr>
                <td>1</td>
                <td>allMask mode</td>
                <td>A complete attenmask matrix (S1 × S2) must be passed.</td>
            </tr>
            <tr>
                <td>2</td>
                <td>leftUpCausal mode</td>
                <td>An optimized attenmask matrix (2048 × 2048) must be passed.</td>
            </tr>
            <tr>
                <td>3</td>
                <td>rightDownCausal mode</td>
                <td>This corresponds to a lower-triangular matrix partitioned by the top-right vertex. In this case, an optimized attenmask matrix (2048 × 2048) needs to be passed.</td>
            </tr>
            <tr>
                <td>4</td>
                <td>band mode</td>
                <td>An optimized attenmask matrix (2048 × 2048) must be passed.</td>
            </tr>
            <tr>
                <td>5</td>
                <td>prefix mode</td>
                <td>This mode is not supported currently. If no specific value is required, 0 is recommended.</td>
            </tr>
            <tr>
                <td>6</td>
                <td>global mode</td>
                <td>This mode is not supported currently. If no specific value is required, 0 is recommended.</td>
            </tr>
            <tr>
                <td>7</td>
                <td>dilated mode</td>
                <td>This mode is not supported currently. If no specific value is required, 0 is recommended.</td>
            </tr>
            <tr>
                <td>8</td>
                <td>block_local mode</td>
                <td>This mode is not supported currently. If no specific value is required, 0 is recommended.</td>
            </tr>
            <tr>
                <td colspan="3" style="text-align: left; ">
                    <strong>Note:</strong> When Q_S is 1 and no rope input is provided, sparseMode is invalid.
                </td>
            </tr>
          </tbody>
      </table>
  </div>

- Restrictions on `innerPrecise`

  - There are four modes (0, 1, 2, and 3) in total, represented by 2-bit combinations. Bit 0 indicates whether to use the high-precision or high-performance mode, and bit 1 indicates whether to perform invalid row correction.

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

  - Note: The high-precision and high-performance modes are applicable to both BFLOAT16 and INT8. Invalid row correction takes effect for FLOAT16, BFLOAT16, and INT8. The values `0` and `1` are reserved. If the masks involved in the computation are all 1s, the precision may be affected. In this case, you can set this parameter to `2` or `3` to enable invalid row correction to improve the precision. However, this configuration deteriorates the performance. If the operator can determine that invalid rows exist, the invalid row correction is automatically enabled, such as in scenarios where `sparseMode` is set to `3` and Sq is greater than Skv.

- Restrictions on `keyAntiquantMode`

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
                    <strong>Note:</strong> This parameter must be the same as valueAntiquantMode except when keyAntiquantMode is 0 and valueAntiquantMode is 1.
                </td>
            </tr>
          </tbody>
      </table>
  </div>

  - <term>Atlas A2 training products/Atlas A2 inference products</term>: When `Q_S` is 1, passing a value other than 0, 1, 2, 3, 4, or 5 will result in an execution error. If `Q_S` is greater than or equal to 2, only the values 0 and 1 are supported. Other values will result in an execution error.

- Restrictions on `valueAntiquantMode`

  - Except for the scenario where `keyAntiquantMode` is 0 and `valueAntiquantMode` is 1, the value must be the same as that of `keyAntiquantMode`.
  - <term>Atlas A2 training products/Atlas A2 inference products</term>: When `Q_S` is 1, passing a value other than 0, 1, 2, 3, 4, or 5 will result in an execution error. If `Q_S` is greater than or equal to 2, only the values 0 and 1 are supported. Other values will result in an execution error.

- Restrictions on `softmaxLse`

  - In the ring attention algorithm, the product of query and key is first processed to obtain softmax_max. This max value is subtracted from the product before calculating the exponential, which is then summed to yield softmax_sum. Finally, the log of softmax_sum is added back to softmax_max to obtain the final result.
  - When `softmaxLseFlag` is `True`, the shape must be [B, N, Q_S, 1] in general. When `inputLayout` is `TND` or `NTD_TND`, the shape must be [T, N, 1].
  - When `softmaxLseFlag` is `False`, if the `softmaxLse` tensor is not `nullptr`, the tensor data is returned directly. If `softmaxLse` is `nullptr`, a tensor of shape {1} filled with zeros is returned.

- **When Q_S is greater than 1**

  - Restrictions on `query`, `key`, and `value`:

    - The B axis must be less than or equal to 65536.

    - <term>Atlas A2 training products/Atlas A2 inference products</term>: If the input type is INT8 and the D axis is not 32-byte aligned, the maximum value of the B axis is 128. If the input type is FLOAT16 or BFLOAT16 and the D axis is not 16-byte aligned, the maximum value of the B axis is also 128.

    - The N axis can be less than or equal to 256, and the D axis can be less than or equal to 512. If `inputLayout` is BSH or BSND, it is recommended that N × D be less than 65535.

    - The S axis must be less than or equal to 20971520 (20M). In some long sequence scenarios, if the computation load is too large, the PFA operator execution may time out (an AI Core error is reported, and `errorStr` is `timeout or trap error`). In this case, S axis splitting is recommended. Note: The computation load is affected by parameters such as **B**, **S**, **N**, and **D**. Larger values indicate larger computation loads. The following lists some typical scenarios with long sequences (that is, the product of **B**, **S**, **N**, and **D** is large).

      <div style="overflow-x: auto;">
      <table style="undefined;table-layout: fixed; width: 900px"><colgroup>
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

    - Restrictions on the D axis: For the <term>Atlas A2 training products/Atlas A2 inference products</term>, if `query`, `key`, `value`, or `attentionOut` is INT8, the D axis must be 32-element aligned. If `query`, `key`, `value`, or `attentionOut` is INT4, the D axis must be 64-element aligned. If they are all FLOAT16 or BFLOAT16, the D axis must be 16-element aligned.

  - Currently, `sparseMode` can only be set to 0, 1, 2, 3, or 4. An error will be reported if it is set to other values.

    - When `sparseMode` is set to 0, if `attenMask` is a null pointer or is passed in the left padding scenario, the input parameters `preTokens` and `nextTokens` are ignored.
    - When `sparseMode` is set to 2, 3, or 4, the shape of `attenMask` must be (S, S), (1, S, S), or (1, 1, S, S), where S must be fixed at 2048. Additionally, the input`attenMask` must be a lower triangular matrix. If `attenMask` is not passed or the passed shape is incorrect, an error will be reported.
    - When `sparseMode` is set to 1, 2, or 3, the input parameters `preTokens` and `nextTokens` are ignored, and their values are assigned based on related rules.
  - In the synthesis parameter scenario of KV cache dequantization, only when `query` is of the FLOAT16 type, `key` and `value` of the INT8 type can be dequantized to FLOAT16. If the product of the data ranges of the input `key` and `value` and the data range of the input `antiquantScale` must be within the range of (–1, 1), the high-performance mode can ensure precision. Otherwise, the high-precision mode needs to be enabled to ensure precision.
  - Page attention scenario

    - The prerequisite for enabling page attention is that `blockTable` exists and is valid, and `key` and `value` are arranged in a continuous memory based on the indexes in `blockTable`. In this scenario, `inputLayout` of `key` and `value` is invalid. `blockTable` is filled with block IDs. Currently, the validity of block IDs is not verified. You need to ensure the validity of block IDs.
      - <term>Atlas A2 training products/Atlas A2 inference products</term>: The data types of `key` and `value` can be FLOAT16 or BFLOAT16.
    - `blockSize` is a user-defined parameter. Its value affects the page attention performance. When page attention is enabled, the value of `blockSize` must be a multiple of 128, ranging from 128 to 512. Generally, page attention can improve the throughput but deteriorate the performance.
    - In the page attention scenario, if the input KV cache layout is BnBsH (blocknum, blocksize, H) and the product of `KV_N` multiplied by `D` exceeds 65535, an error will be reported due to hardware instruction restrictions. This problem can be solved by enabling GQA (decreasing `KV_N`) or adjusting the KV cache layout to BnNBsD (blocknum, KV_N, blocksize, D). When `inputLayout` of `query` is BNSD or TND, the KV cache layout can beBnBsH or BnNBsD. When `inputLayout` of `query` is BSH or BSND, the KV cache layout can only be BnBsH. The value of `blocknum` cannot be less than the sum of blocks in each batch calculated based on `actualSeqLengthsKv` and `blockSize`. The shapes of `key` and `value` must be the same.
    - Page attention fake-quantization scenario
      - <term>Atlas A2 training products/Atlas A2 inference products</term>: The data type of `query` can be FLOAT16 or BFLOAT16, and the data types of `key` and `value` can be INT8.
    - Page attention full-quantization scenario
      - <term>Atlas A2 training products/Atlas A2 inference products</term>: The data type of `query` cannot be INT8.
    - Page attention does not support the tensor list or left padding.
    - In the page attention scenario, `actualSeqLengthsKv` must be passed.
    - In the page attention scenario, `blockTable` must be two-dimensional. The length of the first dimension must be equal to **B**, and the length of the second dimension must be greater than or equal to `maxBlockNumPerSeq` (the maximum number of blocks corresponding to `actualSeqLengthsKv` in different batches).
    - When page attention is enabled, the input `KV_S` must be greater than or equal to `maxBlockNumPerSeq` × `blockSize` in the following scenarios:
      - When `attenMask` is passed, the mask shape is (B, 1, Q_S, KV_S).
      - When `pseShift` is passed, the `pseShift` shape is (B, N, Q_S, KV_S).
  - Left padding for `query`

    - The transfer start point of `query` is calculated as follows: `Q_S` – `queryPaddingSize` – `actualSeqLengths`. The transfer end point of `query` is calculated as follows: `Q_S` – `queryPaddingSize`. The transfer start point of `query` cannot be less than 0, while the end point cannot be greater than `Q_S`. Otherwise, the result will not meet the expectation.
    - If `kvPaddingSize` is less than 0, it will be set to 0.
    - It must be enabled together with `actualSeqLengths`. Otherwise, the default scenario is right padding for `query`.
    - It does not support page attention and cannot be enabled together with `blockTable`.
    - It is not supported when Q is BF16/FP16 and KV is INT4.
  - Left padding for `kv`:

    - The transfer start point of `key` and `value` is calculated as follows: `KV_S` – `kvPaddingSize` – `actualSeqLengthsKv`. The transfer end point of `key` and `value` is calculated as follows: `KV_S` – `kvPaddingSize`. The transfer start point of `key` and `value` cannot be less than 0, while the end point cannot be greater than `KV_S`. Otherwise, the result will not meet the expectation.
    - If `kvPaddingSize` is less than 0, it will be set to 0.
    - It must be enabled together with `actualSeqLengthsKv`. Otherwise, the default scenario is right padding for `kv`.
    - It does not support page attention and cannot be enabled together with `blockTable`.
    - It is not supported when Q is BF16/FP16 and KV is INT4.
  - When the output is of type INT8 and `quantScale2` and `quantOffset2` are per-channel, left padding, ring attention, or non-32-byte alignment of D is not supported.
  - When the output is of type INT8, `sparse` cannot be `band` and `preTokens` or `nextTokens` cannot be negative.
  - Restrictions on `pseShift`

    - This function is supported when the data type of `query` is FLOAT16, BFLOAT16, or INT8.
    - When the data type of `query` is FLOAT16 and `pseShift` exists, the high-precision mode is forcibly used. The corresponding restrictions are the same as those of the high-precision mode.
    - `Q_S` must be greater than or equal to `S` of `query`, and `KV_S` must be greater than or equal to `S` of `key`. In the prefix scenario, `KV_S` must be greater than or equal to the sum of `actualSharedPrefixLen` and `S` of `key`.
  - When the output of type INT8, if the input parameter`quantOffset2` is a non-null pointer and a non-null tensor, and `sparseMode`, `preTokens`, and `nextTokens` meet the following conditions, certain rows of the matrix will not be involved in computation, resulting in a computation result error. In this scenario, the computation will be intercepted. (Solution: To prevent interception, perform post-quantization outside the FIA interface.)

    - When `sparseMode` is 0 and `attenMask` is a non-null pointer, interception occurs if `actualSeqLengths` – `actualSeqLengthsKV` – `actualSharedPrefixLen` – `preTokens` > 0 or `nextTokens` < 0 for any batch.
    - When `sparseMode` is 1 or 2, interception does not occur.
    - When `sparseMode` is 3, interception occurs if `actualSeqLengthsKV` + `actualSharedPrefixLen` – `actualSeqLengths` < 0 for any batch.
    - When `sparseMode` is 4, interception occurs if `preTokens` < 0 or `nextTokens` + `actualSeqLengthsKV` + `actualSharedPrefixLen` – `actualSeqLengths` < 0 for any batch.
  - Restrictions on prefix parameters

    - Both `keySharedPrefix` and `valueSharedPrefix` must be either null or non-null.
    - If neither `keySharedPrefix` nor `valueSharedPrefix` is null, the dimensions and data types of `keySharedPrefix`, `valueSharedPrefix`, `key`, and `value` must be the same.
    - If neither `keySharedPrefix` nor `valueSharedPrefix` is null, the first dimension (batch) of the shape of `keySharedPrefix` must be 1. When the layout is BNSD or BSND, the N and D axes must be the same as those of `key`. When the layout is BSH, the H axis must be the same as that of `key`. The same rules apply to `valueSharedPrefix`. `S` of `keySharedPrefix` and `valueSharedPrefix` must be the same.
    - When `actualSharedPrefixLen` exists, its shape must be [1], and its value cannot be greater than `S` of `keySharedPrefix` and `valueSharedPrefix`.
    - The sum of `S` of the public prefix and `S` of `key` or `value` must meet the original restriction on `S` of `key` or `value`.
    - The prefix does not support page attention, left padding, or tensor list.
    - In the prefix scenario, when `sparse` is 0 or 1 and `attenMask` is passed, `S2` must be greater than or equal to the sum of `actualSharedPrefixLen` and `S` of `key`.
    - In the prefix scenario, the input `qkv` cannot be all INT8.
  - KV fake-quantization parameter separation

    - <term>Atlas A2 training products/Atlas A2 inference products</term>
      - `keyAntiquantMode` and `valueAntiquantMode` must be the same.
      - Both `keyAntiquantScale` and `valueAntiquantScale` must be either null or non-null. Both `keyAntiquantOffset` and `valueAntiquantOffset` must be either null or non-null.
      - If neither `keyAntiquantScale` nor `valueAntiquantScale` is null, their shapes must be the same. If neither `keyAntiquantOffset` nor `valueAntiquantOffset` is null, their shapes must be the same.
      - Only the per-token and per-channel modes are supported. In per-token mode, the shapes of the two parameters must be both \(B, S\), and the data type is fixed at FLOAT32. In per-channel mode, the shapes of the two parameters must be (N, D\), (N, 1, D\), or (H\), and the data type is fixed at BF16.
      - When both fake-quantization parameters and KV separation quantization parameters are passed, the KV separation quantization parameters take effect.
      - When `keyAntiquantScale` and `valueAntiquantScale` are non-null, `S` of `query` must be less than or equal to 16.
      - When `keyAntiquantScale` and `valueAntiquantScale` are non-null, the data type of `query` must be BFLOAT16, the data type of `key` and `value` must be INT8, and the data type of the output must be BFLOAT16.
      - When `keyAntiquantScale` and `valueAntiquantScale` are non-null, the tensor list, left padding, and page attention functions are not supported.

- **When Q_S is equal to 1**

  - Restrictions on `query`, `key`, and `value`:
    - The B axis can be less than or equal to 65536, the N axis can be less than or equal to 256, and the D axis can be less than or equal to 512.
    - The input types of `query`, `key`, and `value` cannot be all INT8.
    - In INT4 (INT32) fake-quantization scenarios, the aclnn single-operator call supports KV inputs in either INT4 format or INT4-packed INT32 format. Using `dynamicQuant` to generate INT4 data is recommended, which stores eight INT4 elements within one INT32.
    - In INT4 (INT32) fake-quantization scenarios, if KV INT4 values are packed into INT32 inputs, the N, D, or H dimensions of KV must be 1/8 of their actual values (the same applies to prefix).
    - Restrictions on the D axis for `key` and `value` in specific data types
      - <term>Atlas A2 training products/Atlas A2 inference products</term>: When the input type of `key` and `value` is INT4 (INT32), the D axis must be 64-element aligned (or 8-element aligned for INT32).
  - Page attention scenario
    - The prerequisite for enabling page attention is that `blockTable` exists and is valid, and `key` and `value` are arranged in a continuous memory based on the indexes in `blockTable`. In this scenario, `inputLayout` of `key` and `value` is invalid.
    - `blockSize` is a user-defined parameter. Its value affects the page attention performance. When page attention is enabled, `blockSize` must be set to a non-zero value, and the maximum value of `blockSize` cannot exceed 512. Generally, page attention can improve the throughput but deteriorate the performance.
    - In the page attention scenario, when `inputLayout` of `query` is BNSD or TND, the KV cache layout can be BnBsH (blocknum, blocksize, H) or BnNBsD (blocknum, KV_N, blocksize, D). When `inputLayout` of `query` is BSH or BSND, the KV cache layout can only be BnBsH. The value of `blocknum` cannot be less than the sum of blocks in each batch calculated based on `actualSeqLengthsKv` and `blockSize`. The shapes of `key` and `value` must be the same.
    - In the page attention scenario, the performance is generally better when the KV cache layout is BnNBsD than when it is BnBsH. Therefore, BnNBsD is recommended.
    - In the page attention scenario, if the input KV cache layout is BnBsH and numKvHeads × headDim exceeds 64 KB, an error will be reported due to hardware instruction constraints. This problem can be solved by enabling GQA (decreasing `numKvHeads`) or adjusting the KV cache layout to BnNBsD.
    - Page attention does not support the tensor list, left padding, or the scenario where Q is BF16/FP16 and KV is INT4 (INT32).
    - In the page attention scenario, `actualSeqLengthsKv` must be passed.
    - In the page attention scenario, `blockTable` must be two-dimensional. The length of the first dimension must be equal to **B**, and the length of the second dimension must be greater than or equal to `maxBlockNumPerSeq` (the maximum number of blocks corresponding to `actualSeqLengthsKv` in each batch).
    - When page attention is enabled, the input `S` must be greater than or equal to `maxBlockNumPerSeq` × `blockSize` in the following scenarios:
      - When the attention mask is enabled, the mask shape is \(B, 1, 1, S\).
      - When `pseShift` is enabled, the `pseShift` shape is \(B, N, 1, S\).
      - When the fake-quantization per-token mode is enabled, the shape of the input parameters `antiquantScale` and `antiquantOffset` is both \(2, B, S\).
    - Left padding for `kv`:
      - <term>Atlas A2 training products/Atlas A2 inference products</term>: Scenarios where Q is BF16/FP16 and KV is INT4 (INT32) are not supported.
      - The transfer start point of `kvCache` is calculated as follows: `KV_S` – `kvPaddingSize` – `actualSeqLengths` The transfer end point of `kvCache` is calculated as follows: `KV_S` – `kvPaddingSize` If the transfer start point or end point is less than 0, the returned data is all 0s.
      - If `kvPaddingSize` is less than 0, it will be set to 0.
      - It must be enabled together with `actualSeqLengths`. Otherwise, the default scenario is right padding for `kv`.
      - Page attention and tensor list are not supported. Otherwise, the default scenario is right padding for `kv`.
      - When it is enabled together with `attenMask`, ensure that the meaning of `attenMask` is correct, that is, invalid data can be correctly masked. Otherwise, precision issues may occur.
    - Restrictions on `pseShift`
      - The data types of `pseShift` and `query` must be consistent.
    - KV fake-quantization parameter separation
      - Except when `keyAntiquantMode` is 0 and `valueAntiquantMode` is 1, the values of `keyAntiquantMode` and `valueAntiquantMode` must be the same.
      - Both `keyAntiquantScale` and `valueAntiquantScale` must be either null or non-null. Both `keyAntiquantOffset` and `valueAntiquantOffset` must be either null or non-null.
      - If neither `keyAntiquantScale` nor `valueAntiquantScale` is null, their shapes must be the same, except when `keyAntiquantMode` is 0 and `valueAntiquantMode` is 1. If neither `keyAntiquantOffset` nor `valueAntiquantOffset` is null, their shapes must be the same, except when `keyAntiquantMode` is 0 and `valueAntiquantMode` is 1.
      - <term>Atlas A2 training products/Atlas A2 inference products</term>: The following eight modes are supported: per-channel, per-tensor, per-token, per-tensor + per-head, per-token + per-head, per-token + page attention to manage scale/offset, per-token + per-head + page attention to manage scale/offset, and `key` in per-channel + `value` in per-token. In the following description, `N` indicates `numKeyValueHeads`.
        - Per-channel mode: The shapes of the two parameters can be \(1, N, 1, D\), \(1, N, D\), or \(1, H\). Their data types are the same as that of `query`. This mode is supported when the data types of `key` and `value` are INT8 or INT4 (INT32).
        - Per-tensor mode: The shapes of the two parameters are both \(1\), and their data types are the same as that of `query`. This mode is supported when the data types of `key` and `value` are INT8.
        - Per-token mode: The shapes of the two parameters are both \(1, B, S\), and their data type is fixed at FLOAT32. This mode is supported when the data types of `key` and `value` are INT8 or INT4 (INT32).
        - Per-tensor + per-head mode: The shapes of the two parameters are both \(N\), and their data types are the same as that of `query`. This mode is supported when the data types of `key` and `value` are INT8.
        - Per-channel for `key` + per-token for `value` mode: For `key` in per-channel, the shapes of the two parameters can be \(1, N, 1, D\), \(1, N, D\), or \(1, H\), and their data types are the same as that of `query`. For `value` in per-token, the shapes of the two parameters are both \(1, B, S\), and their data types are fixed at FLOAT32. This mode is supported when the data types of `key` and `value` are INT8 or INT4 (INT32). When the data types of `key` and `value` are INT8, only the data types of `query` and `attentionOut` can be FLOAT16.
      - Per-token + per-head mode: The shapes of the two parameters are both (B, N, S), and their data types are fixed at FLOAT32. This mode is supported when the data types of `key` and `value` are INT8 or INT4 (INT32).
      - Per-token + page attention to manage scale/offset: The shapes of the two parameters are both (blocknum, blocksize), and their data types are fixed at FLOAT32. This mode is supported when the data types of `key` and `value` are INT8.
      - Per-token + per-head + page attention to manage scale/offset: The shapes of the two parameters are both (blocknum, N, blocksize), and their data types are fixed at FLOAT32. This mode is supported when the data types of `key` and `value` are INT8.
      - When both fake-quantization parameters and KV separation quantization parameters are passed, the KV separation quantization parameters take effect.
      - In the INT4 (INT32) fake-quantization scenario where only KV fake-quantization parameter separation is supported, the following modes are supported:
        - Per-channel mode
        - Per-token mode
        - Per-token + per-head mode
        - Per-channel for `key` + per-token for `value` mode
      - Post-quantization is not supported in INT4 (INT32) fake-quantization scenarios.
    - Restrictions on prefix parameters
      - Both `keySharedPrefix` and `valueSharedPrefix` must be either null or non-null.
      - If neither `keySharedPrefix` nor `valueSharedPrefix` is null, the dimensions and data types of `keySharedPrefix`, `valueSharedPrefix`, `key`, and `value` must be the same.
      - If neither `keySharedPrefix` nor `valueSharedPrefix` is null, the first dimension (batch) of the shape of `keySharedPrefix` must be 1. When the layout is BNSD or BSND, the N and D axes must be the same as those of `key`. When the layout is BSH, the H axis must be the same as that of `key`. The same rules apply to `valueSharedPrefix`. `S` of `keySharedPrefix` and `valueSharedPrefix` must be the same.
      - When `actualSharedPrefixLen` exists, its shape must be [1], and its value cannot be greater than `S` of `keySharedPrefix` and `valueSharedPrefix`.
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
#include "aclnnop/aclnn_fused_infer_attention_score_v2.h"

using namespace std;

#define CHECK_RET(cond, return_expr)                                                                                   \
    do {                                                                                                               \
        if (!(cond)) {                                                                                                 \
            return_expr;                                                                                               \
        }                                                                                                              \
    } while (0)

#define LOG_PRINT(message, ...)                                                                                        \
    do {                                                                                                               \
        printf(message, ##__VA_ARGS__);                                                                                \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
  int64_t shapeSize = 1;
  for (auto i : shape) {
      shapeSize *= i;
  }
  return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream)
{
  // (Boilerplate) Initialize AscendCL.
  auto ret = aclInit(nullptr);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
  ret = aclrtSetDevice(deviceId);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
  ret = aclrtCreateStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
  return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void** deviceAddr,
                  aclDataType dataType, aclTensor** tensor)
{
  auto size = GetShapeSize(shape) * aclDataTypeSize(dataType);
  // Call aclrtMalloc to allocate memory on the device.
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  // Call aclrtMemcpy to copy data from the host to the device.
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // Compute the strides of the contiguous tensor.
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
      strides[i] = shape[i + 1] * strides[i + 1];
  }

  // Call aclCreateTensor to create an aclTensor.
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

int main()
{
  // 1. (Boilerplate) Initialize the device and stream. For details, see the AscendCL external API list.
  // Set the device ID in use.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the input and output based on the API.
  int32_t batchSize = 1;
  int32_t numHeads = 2;
  int32_t numKeyValueHeads = 2;
  int32_t sequenceLengthQ = 1;
  int32_t sequenceLengthKV = 512;
  int32_t headDims = 128;
  std::vector<int64_t> queryShape = {batchSize, numHeads, sequenceLengthQ, headDims};           // BNSD
  std::vector<int64_t> keyShape = {batchSize, numKeyValueHeads, sequenceLengthKV, headDims};    // BNSD
  std::vector<int64_t> valueShape = {batchSize, numKeyValueHeads, sequenceLengthKV, headDims};  // BNSD
  std::vector<int64_t> attenMaskShape = {batchSize, 1, sequenceLengthQ, sequenceLengthKV};      // B 1 S1 S2
  std::vector<int64_t> outShape = {batchSize, numHeads, sequenceLengthQ, headDims};             // BNSD
  void* queryDeviceAddr = nullptr;
  void* keyDeviceAddr = nullptr;
  void* valueDeviceAddr = nullptr;
  void* attenMaskDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* queryTensor = nullptr;
  aclTensor* keyTensor = nullptr;
  aclTensor* valueTensor = nullptr;
  aclTensor* attenMaskTensor = nullptr;
  aclTensor* outTensor = nullptr;
  int64_t queryShapeSize = GetShapeSize(queryShape);          // BNSD
  int64_t keyShapeSize = GetShapeSize(keyShape);              // BNSD
  int64_t valueShapeSize = GetShapeSize(valueShape);          // BNSD
  int64_t attenMaskShapeSize = GetShapeSize(attenMaskShape);  // B 1 S1 S2
  int64_t outShapeSize = GetShapeSize(outShape);              // BNSD
  std::vector<op::fp16_t> queryHostData(queryShapeSize, 1.0);
  std::vector<op::fp16_t> keyHostData(keyShapeSize, 1.0);
  std::vector<op::fp16_t> valueHostData(valueShapeSize, 1.0);
  std::vector<int8_t> attenMaskHostData(attenMaskShapeSize, 0);
  std::vector<op::fp16_t> outHostData(outShapeSize, 1.0);

  // Create a query aclTensor.
  ret = CreateAclTensor(queryHostData, queryShape, &queryDeviceAddr, aclDataType::ACL_FLOAT16, &queryTensor);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a key aclTensor.
  ret = CreateAclTensor(keyHostData, keyShape, &keyDeviceAddr, aclDataType::ACL_FLOAT16, &keyTensor);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  int kvTensorNum = 1;
  aclTensor* tensorsOfKey[kvTensorNum];
  tensorsOfKey[0] = keyTensor;
  auto tensorKeyList = aclCreateTensorList(tensorsOfKey, kvTensorNum);
  // Create a value aclTensor.
  ret = CreateAclTensor(valueHostData, valueShape, &valueDeviceAddr, aclDataType::ACL_FLOAT16, &valueTensor);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  aclTensor* tensorsOfValue[kvTensorNum];
  tensorsOfValue[0] = valueTensor;
  auto tensorValueList = aclCreateTensorList(tensorsOfValue, kvTensorNum);
  // Create attenMask aclTensor.
  ret = CreateAclTensor(attenMaskHostData, attenMaskShape, &attenMaskDeviceAddr, aclDataType::ACL_INT8, &attenMaskTensor);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT16, &outTensor);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  std::vector<int64_t> actualSeqlenVector = {sequenceLengthKV};
  auto actualSeqLengths = aclCreateIntArray(actualSeqlenVector.data(), actualSeqlenVector.size());

  double scaleValue = 1 / sqrt(headDims); // 1 / sqrt(d)
  int64_t preTokens = 65535;
  int64_t nextTokens = 65535;
  string sLayerOut = "BNSD";
  char layerOut[sLayerOut.length()];
  strcpy(layerOut, sLayerOut.c_str());
  int64_t sparseMode = 0;
  int64_t innerPrecise = 0;
  int blockSize = 0;
  int antiquantMode = 0;
  bool softmaxLseFlag = false;
  int keyAntiquantMode = 0;
  int valueAntiquantMode = 0;

  // 3. Call the CANN operator library API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API.
  ret = aclnnFusedInferAttentionScoreV2GetWorkspaceSize(
      queryTensor, tensorKeyList, tensorValueList, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
      nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
      nullptr, nullptr, numHeads, scaleValue, preTokens, nextTokens, layerOut, numKeyValueHeads, sparseMode,
      innerPrecise, blockSize, antiquantMode, softmaxLseFlag, keyAntiquantMode, valueAntiquantMode, outTensor,
      nullptr, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFusedInferAttentionScoreV2GetWorkspaceSize failed. ERROR: %d\n", ret);
            return ret);
  // Allocate device memory based on the computed workspaceSize.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
      ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API.
  ret = aclnnFusedInferAttentionScoreV2(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFusedInferAttentionScoreV2 failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate) Wait until the task execution is complete.
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
