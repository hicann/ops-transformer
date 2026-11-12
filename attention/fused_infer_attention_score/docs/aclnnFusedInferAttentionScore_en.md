# aclnnFusedInferAttentionScore

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-21T09:44:03.428Z -->

**NOTE: This API will be deprecated in subsequent versions. Use the latest API [aclnnFusedInferAttentionScoreV5](./aclnnFusedInferAttentionScoreV5_en.md) instead.**

## Applicable Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Not supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 products</term>: Supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 products</term>: Supported
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2 inference products</term>: Not supported
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas inference products</term>: Not supported
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas training products</term>: Not supported
<!-- end id6 -->

## Function

- Description: A FlashAttention operator suitable for both decode and prefill scenarios. It supports the prefill computation scenario (PromptFlashAttention) and the decode computation scenario (IncreFlashAttention).

  **NOTE**

  KV Cache specific to the decode scenario: KV Cache is a common technique for optimizing inference performance in large models. During sampling, the Transformer model performs inference using the given prompt/context as the initial input (which can be processed in parallel), and then generates additional tokens one by one to further refine the generated sequence (reflecting the autoregressive nature of the model). During the sampling process, the Transformer performs self-attention operations, which requires extracting key-value (KV) vectors for each item in the current sequence (whether prompt/context or generated tokens). These vectors are stored in a matrix, commonly referred to as the KV cache.
- Formula:

  Self-attention builds an attention model using the relationships within the input samples themselves. The principle is as follows: assume there is an input sample sequence $x$ of length $n$, where each element of $x$ is a $d$-dimensional vector. Each $d$-dimensional vector can be regarded as a token embedding. After passing such a sequence through three weight matrices, three matrices of dimension $n*d$ are obtained.

  The calculation formula of self-attention is generally defined as follows, where $Q$, $K$, and $V$ are important property elements of the input sample, obtained through spatial transformation of the input sample and can be unified into a feature space. "Attention" in the formula and operator name is an abbreviation of "self-attention".

  $$
  Attention(Q,K,V)=Score(Q,K)V
  $$

  In this operator, the Score function uses the Softmax function, and the self-attention calculation formula is:

  $$
  Attention(Q,K,V)=Softmax(\frac{QK^T}{\sqrt{d}})V
  $$

  Where: the product of $Q$ and $K^T$ represents the attention of the input $x$. To prevent this value from becoming too large, it is usually divided by the square root of $d$ for scaling, and each row is normalized by softmax. After multiplying with $V$, an $n*d$ matrix is obtained.

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). You must first call the "aclnnFusedInferAttentionScoreGetWorkspaceSize" API to obtain the input parameters and calculate the required workspace size according to the computation process, and then call the "aclnnFusedInferAttentionScore" API to perform the computation.

```cpp
 aclnnStatus aclnnFusedInferAttentionScoreGetWorkspaceSize(
     const aclTensor     *query,
     const aclTensorList *key,
     const aclTensorList *value,
     const aclTensor     *pseShift,
     const aclTensor     *attenMask,
     const aclIntArray   *actualSeqLengths,
     const aclIntArray   *actualSeqLengthsKv,
     const aclTensor     *deqScale1,
     const aclTensor     *quantScale1,
     const aclTensor     *deqScale2,
     const aclTensor     *quantScale2,
     const aclTensor     *quantOffset2,
     const aclTensor     *antiquantScale,
     const aclTensor     *antiquantOffset,
     const aclTensor     *blockTable,
     const aclTensor     *queryPaddingSize,
     const aclTensor     *kvPaddingSize,
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
     const aclTensor     *attentionOut,
     const aclTensor     *softmaxLse,
     uint64_t            *workspaceSize,
     aclOpExecutor       **executor)
```

```cpp
aclnnStatus aclnnFusedInferAttentionScore(
      void                *workspace,
      uint64_t             workspaceSize,
      aclOpExecutor       *executor,
      const aclrtStream    stream)
```

## aclnnFusedInferAttentionScoreGetWorkspaceSize

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
        <th>Dimension (shape)</th>
        <th>Non-contiguous Tensor</th>
      </tr></thead>
    <tbody>
      <tr>
        <td>query</td>
        <td>Input</td>
        <td>Input Q in the Formula.</td>
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
        <td>FLOAT16, BFLOAT16, INT8</td>
        <td>ND</td>
        <td>3-4</td>
        <td>×</td>
      </tr>
      <tr>
        <td>value</td>
        <td>Input</td>
        <td>Input V in the formula.</td>
        <td>-</td>
        <td>FLOAT16, BFLOAT16, INT8</td>
        <td>ND</td>
        <td>3-4</td>
        <td>×</td>
      </tr>
      <tr>
        <td>pseShift</td>
        <td>Input</td>
        <td>Position encoding.</td>
        <td><ul><li>Empty tensors are not supported.</li>
        <li>Pass in nullptr when this feature is not used.</li>
            <li>For comprehensive constraints, see <a href="#constraints">Constraints</a>.</li></ul></td>
        <td>FLOAT16, BFLOAT16</td>
        <td>ND</td>
        <td>4</td>
        <td>×</td>
      </tr>
      <tr>
        <td>attenMask</td>
        <td>Input</td>
        <td>Mask matrix</td>
        <td><ul><li>Empty tensors are not supported.</li>
        <li>Pass in nullptr if this function is not used.</li>
            <li>For comprehensive constraints, see <a href="#constraints">Constraints</a>.</li></ul></td>
        <td>BOOL, INT8, UINT8</td>
        <td>ND</td>
        <td>2-4</td>
        <td>×</td>
      </tr>
      <tr>
        <td>actualSeqLengths</td>
        <td>Input</td>
        <td>Valid sequence lengths of queries in different batches.</td>
        <td><ul><li>If the sequence length is not specified, pass in nullptr.</li>
            <li>For comprehensive constraints, see <a href="#constraints">Constraints</a>.</li></ul></td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>actualSeqLengthsKv</td>
        <td>Input</td>
        <td>Effective sequence length of key/value in different batches.</td>
        <td><ul><li>If the sequence length is not specified, nullptr can be passed in, indicating that the length is the same as the S dimension of the key/value shape.</li>
            <li>Restrictions as Follows: The effective Sequence Length of each batch in this parameter should not be greater than the Sequence Length of the corresponding batch in key/value. If the length of seqlenKv passed in is 1, each batch uses the same seqlenKv; if the passed-in length is greater than or equal to the number of batches, the first Batch values in seqlenKv are used. Other lengths are not supported.</li>
            <li>For comprehensive constraints, see <a href="#constraints">Constraints</a>.</li></ul></td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>deqScale1</td>
        <td>Input</td>
        <td>Dequantization factor after BMM1.</td>
          <td><ul><li>Empty tensor not supported.</li>
          <li>per-tensor is supported.</li>
              <li>Pass in nullptr if this function is not used.</li>
              <li>For comprehensive constraints, see <a href="#constraints">Constraints</a>.</li></ul></td>
        <td>UINT64, FLOAT32</td>
        <td>ND</td>
        <td>1</td>
        <td>-</td>
      </tr>
      <tr>
        <td>quantScale1</td>
        <td>Input</td>
        <td>Quantization factor before BMM2.</td>
        <td><ul><li>Empty tensor Not Supported.</li>
        <li>per-tensor is supported. </li>
            <li>nullptr can be passed in when this function is not used.</li>
            <li>For comprehensive constraints, see <a href="#constraints">Constraints</a>.</li></ul></td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>1</td>
        <td>-</td>
      </tr>
      <tr>
        <td>deqScale2</td>
        <td>Input</td>
        <td>Dequantization factor after BMM2.</td>
        <td><ul><li>Empty tensors are not supported.</li>
        <li>per-tensor is supported. </li>
           <li>Pass in nullptr when this function is not used.</li>
            <li>For comprehensive constraints, see <a href="#constraints">Constraints</a>.</li></ul></td>
        <td>UINT64, FLOAT32</td>
        <td>ND</td>
        <td>1</td>
        <td>-</td>
      </tr>
      <tr>
        <td>quantScale2</td>
        <td>Input</td>
        <td>Quantization factor of the output.</td>
        <td><ul><li>Empty tensors are not supported.</li>
        <li>per-tensor and per-channel are supported. </li>
            <li>Pass in nullptr if this function is not used.</li>
             <li>For comprehensive constraints, see <a href="#constraints">Constraints</a>.</li></ul></td>
        <td>FLOAT32, BFLOAT16</td>
        <td>ND</td>
        <td>1-4</td>
        <td>-</td>
      </tr>
      <tr>
        <td>quantOffset2</td>
        <td>Input</td>
        <td>Quantization offset of the output.</td>
        <td><ul><li>Empty tensors not supported.</li>
        <li>per-tensor and per-channel are supported. </li>
            <li>nullptr can be passed in when this function is not used.</li>
             <li>For comprehensive constraints, see <a href="#constraints">Constraints</a>.</li></ul></td>
        <td>FLOAT32, BFLOAT16</td>
        <td>ND</td>
        <td>1-4</td>
        <td>-</td>
      </tr>
      <tr>
        <td>antiquantScale</td>
        <td>Input</td>
        <td>Pseudo-quantization factor.</td>
        <td><ul><li>Empty tensors not supported.</li>
        <li>per-tensor, per-channel, and per-token are supported. </li>
            <li>nullptr can be passed in when this function is not used.</li>
            <li>For comprehensive constraints, see <a href="#constraints">Constraints</a>.</li>
            <li>Recommended to use the KV pseudo-quantization parameter separation mode.</li></ul></td>
        <td>FLOAT16, BFLOAT16, FLOAT32</td>
        <td>ND</td>
        <td>1-4</td>
        <td>-</td>
      </tr>
        <tr>
      <td>antiquantOffset</td>
        <td>Input</td>
        <td>Pseudo-quantization offset.</td>
        <td><ul><li>Empty tensors are not supported.</li>
        <li>per-tensor, per-channel, and per-token are supported. </li>
            <li>When used, the shape must keep consistent with that of antiquantScale.</li>
            <li>Pass in nullptr if this function is not used.</li>
            <li>For comprehensive constraints, see <a href="#constraints">Constraints</a>.</li>
            <li>It is recommended to use the KV pseudo-quantization parameter separation mode.</li></ul></td>
        <td>FLOAT16, BFLOAT16, FLOAT32</td>
        <td>ND</td>
        <td>1-4</td>
        <td>-</td>
      </tr>
    <tr>
       <td>blockTable</td>
        <td>Input</td>
        <td>Block mapping table used by KV storage in PageAttention.</td>
        <td><ul><li>Empty tensor Not Supported.</li>
        <li>Pass in nullptr when this feature is not used.</li></ul></td>
        <td>INT32</td>
        <td>ND</td>
        <td>2</td>
        <td>-</td>
      </tr>
      <tr>
       <td>queryPaddingSize</td>
        <td>Input</td>
        <td>Indicates whether the data of each batch in Query is right-aligned, and the number of right-aligned elements.</td>
        <td><ul><li>Empty tensors are not supported.</li>
        <li>This parameter is valid only when Q_S is greater than 1. In other use cases, this parameter does not take effect.</li>
            <li>Pass in nullptr if this function is not used.</li></ul></td>
        <td>INT64</td>
        <td>ND</td>
        <td>1</td>
        <td>-</td>
      </tr>
      <tr>
       <td>kvPaddingSize</td>
        <td>Input</td>
        <td>Indicates whether the data of each batch in key/value is right-aligned and the number of right-aligned elements.</td>
        <td><ul><li>Empty tensors are not supported.</li>
        <li>If this feature is not used, pass in nullptr.</li></ul></td>
        <td>INT64</td>
        <td>ND</td>
        <td>1</td>
        <td>-</td>
      </tr>
      <tr>
        <td>numHeads</td>
        <td>Input</td>
        <td>Number of query heads.</td>
        <td>In the BNSD use case, this value must be the same as the N-axis shape value of query in shape. Otherwise, execution errors occur.</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>scaleValue</td>
        <td>Input</td>
        <td>In the Formula, the reciprocal of the square root of d.</td>
        <td><ul><li>The data type must satisfy the data type deduction rule with the data type of query. </li>
            <li>Recommended to pass in 1.0 if the user does not specifically specify a value. </li></ul></td>
        <td>DOUBLE</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>preTokens</td>
        <td>Input</td>
        <td>Used for sparse computation, indicating that attention needs to be computed with the preceding several Tokens.</td>
          <td><ul><li>Recommended to pass in 2147483647 when not specifically specified.</li>
              <li>This parameter is invalid when Q_S is 1.</li></ul></td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>nextTokens</td>
        <td>Input</td>
        <td>Indicates that attention needs to be computed in association with the next few tokens.</td>
        <td><ul><li>Recommended to Pass In 2147483647 if not specified.</li>
            <li>This parameter is invalid when Q_S is 1.</li></ul></td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>inputLayout</td>
        <td>Input</td>
        <td>Identifies the data layout format of input query, key, and value.</td>
        <td><ul><li>"BSH" is recommended to pass in if not specified.</li></ul>
            <ul><li>For comprehensive constraints, see <a href="#constraints">Constraints</a></li></ul></td>
        <td>STRING</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>numKeyValueHeads</td>
        <td>Input</td>
        <td>Number of heads in key and value.</td>
        <td><ul><li>0 is recommended to pass in if not specified, indicating that key/value and query have the same number of heads.</li></ul>
            <ul><li>For comprehensive constraints, see <a href="#constraints">Constraints</a></li></ul></td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>sparseMode</td>
        <td>Input</td>
        <td>Sparse mode.</td>
        <td>For comprehensive constraints, see <a href="#constraints">Constraints</a>.</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>innerPrecise</td>
        <td>Input</td>
        <td>Indicates the selection of high precision or high performance.</td>
        <td>For comprehensive constraints, see <a href="#constraints">Constraints</a>.</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>blockSize</td>
        <td>Input</td>
        <td>Maximum number of tokens in each block of KV storage in PageAttention.</td>
        <td>See <a href="#constraints">Constraints</a> for comprehensive constraints.</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>antiquantMode</td>
        <td>Input</td>
        <td>Pseudo-quantization mode.</td>
          <td><ul><li>If 0 is passed in, per-channel is used (per-channel includes per-tensor).</li>
              <li>If 1 is passed in, per-token is used.</li>
              <li>Recommended to pass in 0 if not specifically specified.</li>
              <li>When Q_S is 1, passing in a value other than 0 or 1 causes an execution exception. When Q_S is greater than or equal to 2, this parameter is invalid.</li>
              <li>Recommended to use the KV pseudo-quantization parameter separation mode.</li></ul></td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>softmaxLseFlag</td>
        <td>Input</td>
        <td>Whether to output softmax_lse.</td>
          <td><ul><li>Supports splitting along the S axis (output is added).</li>
              <li>Recommended to pass in false if not specifically specified by the user.</li></ul></td>
        <td>BOOL</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>attentionOut</td>
        <td>Output</td>
        <td>The output in the formula.</td>
        <td>The D dimension of this parameter must be kept consistent with the D of value, and the other dimensions must be kept consistent with the shape of query.</td>
        <td>FLOAT16, BFLOAT16, INT8</td>
        <td>ND</td>
        <td>3-4</td>
        <td>-</td>
      </tr>
      <tr>
        <td>softmaxLse</td>
        <td>Output</td>
        <td>Result of the ring attention algorithm on query multiplied by key.</td>
        <td>For comprehensive constraints, see <a href="#constraints">Constraints</a>.</td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>4</td>
        <td>-</td>
      </tr>
      <tr>
        <td>workspaceSize</td>
        <td>Output</td>
        <td>Returns the workspace size that the user needs to allocate on the Device.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>executor</td>
        <td>Output</td>
        <td>Returns the op executor, which contains the operator computation flow.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
    </tbody></table>
    </div>

- **Return Value**

  Returns an aclnnStatus status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
  <col style="width: 280px">
  <col style="width: 119px">
  <col style="width: 751px">
  </colgroup>
  <thead>
    <tr>
      <th>Return Value</th>
      <th>Error Code</th>
      <th>Description</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>The passed-in query, key, value, or attentionOut is a null pointer.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_PARAM_INVALID</td>
      <td>161002</td>
      <td>The data types and data formats of query, key, value, pseShift, attenMask, and attentionOut are not within the supported range.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_RUNTIME_ERROR</td>
      <td>361001</td>
      <td>An exception occurs when the API calls the npu runtime interface for memory.</td>
    </tr>
  </tbody>
  </table>

## aclnnFusedInferAttentionScore

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
  <col style="width: 168px">
  <col style="width: 128px">
  <col style="width: 854px">
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
      <td>Address of the workspace memory allocated on the device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace memory allocated on the device, obtained by calling the first-phase API aclnnFusedInferAttentionScoreGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Executor of the operator, containing the computation process of the operator.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream for executing tasks.</td>
    </tr>
  </tbody>
  </table>

- **Return Value**

  Returns an aclnnStatus status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnPromptFlashAttention defaults to a deterministic implementation.
- When using this API with PyTorch, ensure that the versions of CANN-related packages and PyTorch-related packages are compatible.

- Handling of null input parameters: The operator internally checks whether the query parameter is null. If it is null, the operator returns directly. If the query parameter is a non-null tensor and the key and value parameters are null tensors (i.e., S2 is 0), attentionOut is filled with all zeros. When attentionOut is a null tensor, the AscendCLNN framework handles it. For other input parameters marked as "nullable" in the parameter descriptions above, no processing is performed when they are null pointers.

- The shapes of the corresponding tensors in key and value must be exactly the same. In non-contiguous use cases, the batch in the tensorlists of key and value can only be 1, the number of tensors in the tensorlists must be equal to B of query, and N and D must be the same. Due to tensorlist restrictions, B cannot be greater than 256 in non-contiguous use cases.

- pseShift usage restrictions:

  - If Q_S is not 1, when pseShift is of the FLOAT16 type, query must be of the FLOAT16 or INT8 type; when pseShift is of the BFLOAT16 type, query must be of the BFLOAT16 type. The input shape must be (B,N,Q_S,KV_S) or (1,N,Q_S,KV_S), where Q_S is S in the shape of query, and KV_S is S in the shape of key and value. If KV_S of pseShift is not 32-byte aligned, it is recommended to pad it to 32 bytes to improve performance. There is no requirement on the padding values of the extra part.
  - Q_S is 1. When pseShift is of the FLOAT16 type, the query must be of the FLOAT16 type; when pseShift is of the BFLOAT16 type, the query must be of the BFLOAT16 type. The input shape must be (B,N,1,KV_S) or (1,N,1,KV_S), where KV_S is the S in the shapes of key and value. For use cases where the KV_S of pseShift is not 32-aligned, padding to 32 bytes is recommended to improve performance. The padding values of the extra part are not restricted.

- attenMask: Masks the QK result to indicate whether to compute the correlation between tokens.

  - When Q_S is not 1, the recommended input shapes are (Q_S,KV_S); (B,Q_S,KV_S); (1,Q_S,KV_S); (B,1,Q_S,KV_S); (1,1,Q_S,KV_S).
  - When Q_S is 1, the recommended shape input is (B,KV_S); (B,1,KV_S); (B,1,1,KV_S).
  - Q_S is the S in the shape of query, and KV_S is the S in the shape of key and value. However, if Q_S or KV_S is not aligned to 16 or 32, it can be rounded up to the aligned S.

- When the data type of attenMask is INT8 or UINT8, the values in its tensor must be 0 or 1.

- actualSeqLengths: An aclIntArray on the host side, representing the valid sequence length of query in different batches. The data type supports INT64. If seqlen does not need to be specified, nullptr can be passed, indicating that the length is the same as S in the shape of query. Constraint: The valid sequence length of each batch in this input must not be greater than the sequence length of query in the corresponding batch. This parameter has no effect when Q_S is 1. If the passed length of seqlen is 1, each batch uses the same seqlen. If the passed length is greater than or equal to the number of batches, the first batch count of seqlen values are used. Other lengths are not supported.

- actualSeqLengthsKv: An aclIntArray on the host side. You can pass nullptr to indicate the effective sequence length of key/value in different batches. The data type supports INT64. If seqlen is not specified, you can pass nullptr, indicating that the length is the same as the S length of the key/value shape. Restriction: The effective sequence length of each batch in this input parameter must not exceed the sequence length of the corresponding batch in key/value. When the input length of seqlenKv is 1, each batch uses the same seqlenKv. When the input length is greater than or equal to the number of batches, the first batch count of seqlenKv is used. Other lengths are not supported.

- Comprehensive constraints on the number of int8 quantization-related input parameters and the input/output data formats:

  - When both input and output are INT8: The input parameters deqScale1, quantScale1, deqScale2, and quantScale2 must all be present. quantOffset2 is optional and defaults to 0 if not passed.
  - Use case where the input is INT8 and the output is FLOAT16: The input parameters deqScale1, quantScale1, and deqScale2 must be passed in at the same time. If the input parameter quantOffset2 or quantScale2 exists (that is, is not nullptr), an error is reported and returned.
  - Use case where the input is FLOAT16 or BFLOAT16 and the output is INT8: The input parameter quantScale2 must exist. quantOffset2 is optional and defaults to 0 if not passed in. If the input parameter deqScale1, quantScale1, or deqScale2 exists (that is, is not nullptr), an error is reported and returned.
  - The input parameters quantScale2 and quantOffset2 support two formats (per-tensor/per-channel) and two data types (FLOAT32/BFLOAT16). If quantOffset2 is passed in, its type and shape information must be consistent with those of quantScale2. When the input is BFLOAT16, both FLOAT32 and BFLOAT16 are supported; otherwise, only FLOAT32 is supported. In per-channel format, when the output layout is BSH, the product of all dimensions of quantScale2 must be equal to H; for other layouts, the product must be equal to N*D. (When the output layout is BSH, it is recommended to pass in quantScale2 of shape [1,1,H] or [H]; when the output is BNSD, it is recommended to pass in [1,N,1,D] or [N,D]; when the output is BSND, it is recommended to pass in [1,1,N,D] or [N,D]).

- Constraints on the pseudo-quantization parameters antiquantScale and antiquantOffset:

  - Per-channel mode: The shape of the two parameters can be \(2, N, 1, D\), \(2, N, D\), or \(2, H\), where N is numKeyValueHeads. The parameter data type is the same as the query data type, and antiquantMode is set to 0.
  - per-tensor mode: The shapes of the two parameters are both (2), the data type is the same as the query data type, and antiquantMode is set to 0.
  - per-token mode: The shapes of the two parameters are both \(2, B, S\), the data type is fixed to FLOAT32, and antiquantMode is set to 1.
  - In asymmetric quantization mode, both antiquantScale and antiquantOffset parameters must be present.
  - In symmetric quantization mode, antiquantOffset can be null (i.e., nullptr). When the antiquantOffset parameter is null, symmetric quantization is performed; otherwise, asymmetric quantization is performed.
  - When Q_S is greater than or equal to 2, only FLOAT16 and FLOAT32 are supported (FLOAT32 is supported only in PageAttention scenarios).
  - It is recommended to use the KV pseudo-quantization parameter separation mode.

- inputLayout: Specifies the data layout format of the input query, key, and value. Currently supports BSH, BSND, BNSD, and BNSD_BSND (when the input is BNSD, the output format is BSND, supported only when Q_S is greater than 1). If not explicitly specified by the user, it is recommended to pass "BSH".

  - Description: The data layout format of query, key, and value can be interpreted from multiple dimensions, where B (Batch) indicates the batch size of input samples, S (Seq-Length) indicates the sequence length of input samples, H (Hidden-Size) indicates the hidden layer size, N (Head-Num) indicates the number of heads, and D (Head-Dim) indicates the smallest unit dimension of the hidden layer, satisfying D = H/N.

- numKeyValueHeads usage restrictions: numHeads must be divisible by numKeyValueHeads. In BSND, BNSD, and BNSD_BSND scenarios, numKeyValueHeads must also be equal to the N-axis shape value of key/value in the shape; otherwise, an exception occurs.

- The usage restrictions of sparseMode are as follows:

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
                <td>                    <ul style="margin: 0; padding-left: 20px;">
                        <li>If attenmask is not passed in, the mask operation is not performed, and preTokens and nextTokens are ignored (set to INT_MAX internally);</li>
                        <li>If attenmask is passed in, the complete attenmask matrix (S1 * S2) needs to be passed in, indicating that the part between preTokens and nextTokens needs to be calculated.</li>
                    </ul>
                </td>
            </tr>
            <tr>
                <td>1</td>
                <td>allMask mode</td>
                <td>Must pass in the complete attenmask matrix (S1 * S2).</td>
            </tr>
            <tr>
                <td>2</td>
                <td>leftUpCausal mode</td>
                <td>Need to pass in the optimized attenmask matrix (2048*2048).</td>
            </tr>
            <tr>
                <td>3</td>
                <td>rightDownCausal mode</td>
                <td>Corresponds to the lower triangular use case divided by the right vertex. Need to pass in the optimized attenmask matrix (2048*2048).</td>
            </tr>
            <tr>
                <td>4</td>
                <td>band mode</td>
                <td>Need to pass in the optimized attenmask matrix (2048*2048).</td>
            </tr>
            <tr>
                <td>5</td>
                <td>prefix mode</td>
                <td>This mode is not supported yet. If the user does not specifically specify it, it is recommended to pass in 0.</td>
            </tr>
            <tr>
                <td>6</td>
                <td>global mode</td>
                <td>This mode is not supported yet. It is recommended to pass in 0 if the user does not specifically specify it.</td>
            </tr>
            <tr>
                <td>7</td>
                <td>dilated mode</td>
                <td>This mode is not supported yet. It is recommended to pass in 0 if the user does not specifically specify it.</td>
            </tr>
            <tr>
                <td>8</td>
                <td>block_local mode</td>
                <td>This mode is not supported yet. It is recommended to pass in 0 if the user does not specifically specify it.</td>
            </tr>
            <tr>
                <td colspan="3" style="text-align: left; ">                    <strong>Special constraints:</strong> This parameter is invalid when Q_S is 1.
                </td>
            </tr>
          </tbody>
      </table>
  </div>

- The usage restrictions of innerPrecise are as follows:

  - There are four modes: 0, 1, 2, and 3. They occupy two bits in total. Bit 0 indicates the selection between high precision and high performance, and bit 1 indicates whether row invalidation correction is performed.

    <table style="undefined;table-layout: fixed; width: 600px"><colgroup>
    <col style="width: 200px">
    <col style="width: 200px">
      <col style="width: 200px">
    </colgroup><thead>
    <tr>
    <th>innerPrecise</th>
    <th>Mode</th>
    <th>Invalid row correction</th>
    </tr></thead>
    <tbody>
    <tr>
    <td>0</td>
    <td>High precision</td>
    <td>×</td>
    </tr>
    <tr>
    <td>1</td>
    <td>High performance</td>
    <td>×</td>
    </tr>
    <tr>
    <td>2</td>
    <td>High precision</td>
    <td>√</td>
    </tr>
    <tr>
    <td>3</td>
    <td>High performance</td>
    <td>√</td>
    </tr>
    </tbody>
    </table>

  - Note: BFLOAT16 and INT8 do not distinguish between high precision and high performance. Row-invalid correction takes effect on FLOAT16, BFLOAT16, and INT8. Currently, 0 and 1 are reserved configuration values. If the "mask part involved in computation" contains an entire row of all 1s, precision loss may occur. In this case, you can set this parameter to 2 or 3 to enable the row-invalid function to improve precision. However, this configuration degrades performance. If the operator can determine that a row-invalid use case exists, row-invalid computation is automatically enabled, for example, when sparse_mode is 3 and Sq > Skv.

- softmaxLse usage constraints are as follows:

  - In the ring attention algorithm, the max is first taken over the result of query multiplied by key to obtain softmax_max. softmax_max is then subtracted from the result of query multiplied by key, the exp is taken, and the sum is computed to obtain softmax_sum. Finally, the log is taken over softmax_sum, and softmax_max is added to obtain the result.
  - When softmaxLseFlag is True, the shape must be [B, N, Q_S, 1]. Data with inf indicates invalid data.
  - When softmaxLseFlag is False, if the softmaxLse tensor passed in is not empty, the tensor data is directly returned. If nullptr is passed for softmaxLse, a tensor of shape {1} with all zeros is returned.

- **When Q_S is greater than 1**:

  - The function usage restrictions for query, key, and value inputs are as follows:

    - The B axis must be less than or equal to 65536. If the input type is INT8 and the D axis is not 32-byte aligned, the maximum supported value of the B axis is 128. If the input type is FLOAT16 or BFLOAT16 and the D axis is not 16-byte aligned, the B axis is also only supported up to 128.

    - The N axis must be less than or equal to 256, and the D axis must be less than or equal to 512. When inputLayout is BSH or BSND, N*D must be less than 65535.

    - S must be less than or equal to 20971520 (20M). In some long sequence scenarios, if the computation load is too large, the PFA operator may time out (aicore error type, with errorStr: timeout or trap error). In such scenarios, S-axis splitting is recommended. Note: The computation load is affected by B, S, N, D, etc. Larger values result in greater computation load. Typical long sequence scenarios that may cause timeouts (i.e., where the product of B, S, N, and D is large) include but are not limited to:

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

    - When query, key, value, or attentionOut contains the INT8 type, the D axis must be 32-aligned. When all types are FLOAT16 or BFLOAT16, the D axis must be 16-aligned.

  - The sparseMode parameter currently only supports values of 0, 1, 2, 3, and 4. An error is reported when other values are used.

    - When sparseMode = 0, if attenMask is a null pointer or attenMask is passed in a left-padding scenario, the input parameters preTokens and nextTokens are ignored.
    - When sparseMode = 2, 3, or 4, the shape of attenMask must be S,S or 1,S,S or 1,1,S,S, where the value of S must be fixed to 2048, and the user must ensure that the passed attenMask is lower triangular. If attenMask is nullptr or the passed shape is incorrect, an error is reported.
    - In scenarios where sparseMode = 1, 2, or 3, the input parameters preTokens and nextTokens are ignored and assigned values according to relevant rules.

  - The composite parameter scenario for kvCache dequantization supports only the case where query is FLOAT16, dequantizing key and value of INT8 type to FLOAT16. The product range of the datarange of the input parameter key/value and the datarange of the input parameter antiQuantScale falls within (-1, 1). The high-performance mode can guarantee precision; otherwise, the high-precision mode must be enabled to guarantee precision.

  - Page attention scenario:

    - The prerequisite for enabling page attention is that blockTable exists and is valid, and key and value are arranged in contiguous memory based on the indices in blockTable. In this use case, the inputLayout parameters of key and value are invalid. blockTable is filled with blockid. The validity of blockid is not verified currently and must be ensured by users.
    - blockSize is a user-defined parameter whose value affects the performance of page attention. When page attention is enabled, blockSize ranges from 128 to 512 and must be a multiple of 128. Generally, page attention improves throughput but degrades performance.
    - In the page attention use case, when the input kv cache layout format is BnBsH (blocknum, blocksize, H) and KV_N * D exceeds 65535, an error is intercepted due to hardware instruction constraints. The issue can be resolved by enabling GQA (reducing KV_N) or changing the kv cache layout format to BnNBsD (blocknum, KV_N, blocksize, D). When the inputLayout of query is BNSD, the kv cache layout supports two formats: BnBsH and BnNBsD. When the inputLayout of query is BSH or BSND, the kv cache layout supports only BnBsH. blocknum cannot be less than the total number of blocks per batch calculated based on actualSeqLengthsKv and blockSize. In addition, the shapes of key and value must be consistent
    - Page attention does not support pseudo-quantization scenarios, tensorlist scenarios, or left padding scenarios.
    - In the page attention scenario, actualSeqLengthsKv must be passed.
    - In page attention scenarios, blockTable must be two-dimensional. The first dimension length must equal B, and the second dimension length cannot be less than maxBlockNumPerSeq (maxBlockNumPerSeq is the number of blocks corresponding to the maximum actualSeqLengthsKv across different batches).
    - In page attention scenarios, int8 is not supported for query.
    - When page attention is enabled, the input KV_S in the following use cases must be greater than or equal to maxBlockNumPerSeq * blockSize
      - When attenMask is passed in, for example, the mask shape is (B, 1, Q_S, KV_S)
      - When pseShift is passed in, for example, the pseShift shape is (B, N, Q_S, KV_S)

  - Query left-padding scenario:

    - The formula for the query copy start position is: Q_S - queryPaddingSize - actualSeqLengths. The formula for the query copy end position is: Q_S - queryPaddingSize. The query copy start position must not be less than 0, and the end position must not be greater than Q_S. Otherwise, the result will not meet expectations.
    - When kvPaddingSize is less than 0, it is set to 0.
    - This use case must be enabled together with the actualSeqLengths parameter. Otherwise, the query right-padding use case is used by default.
    - This use case does not support PageAttention and cannot be enabled together with the blockTable parameter.

  - kv left-padding scenario:

    - The copy start point of key and value is calculated as: KV_S - kvPaddingSize - actualSeqLengthsKv. The copy end point of key and value is calculated as: KV_S - kvPaddingSize. The copy start point of key and value must not be less than 0, and the end point must not be greater than KV_S. Otherwise, the result will not meet expectations.
    - If kvPaddingSize is less than 0, it is set to 0.
    - It must be enabled together with the actualSeqLengthsKv parameter. Otherwise, the kv right-padding use case is used by default.
    - PageAttention is not supported, and it cannot be enabled together with the blockTable parameter.

  - When the output is int8 and quantScale2 and quantOffset2 are per-channel, left padding, Ring Attention, or use cases where D is not 32-byte aligned are not supported yet.

  - When the output is INT8, the scenario where sparse is band and preTokens/nextTokens are negative is not supported.

  - pseShift function usage restrictions as follows:

    - The pseShift function is supported when the query data type is FLOAT16, BFLOAT16, or INT8.
    - When the query data type is FLOAT16 and pseShift is present, the high-precision mode is forcibly used, and the corresponding restrictions are inherited from those of the high-precision mode.
    - Q_S must be greater than or equal to the S length of query, and KV_S must be greater than or equal to the S length of key.

  - When the output is INT8, if quantOffset2 is passed in as a non-null pointer with a non-empty tensor value, and sparseMode, preTokens, and nextTokens meet the following conditions, some rows of the matrix do not participate in the computation, resulting in errors in the computation results. In this case, the call is intercepted (solution: to avoid interception, perform post-quantization outside the FIA API instead of enabling it inside the FIA API):

  - sparseMode = 0: if attenMask is a non-null pointer, and actualSeqLengths - actualSeqLengthsKV - preTokens > 0 or nextTokens < 0 for each batch, the interception condition is met.
  - sparseMode = 1 or 2: the interception condition is never met.
  - When sparseMode is 3, the interception condition is met if actualSeqLengthsKV - actualSeqLengths < 0 for each batch.
  - sparseMode = 4: if preTokens < 0 or nextTokens + actualSeqLengthsKV - actualSeqLengths < 0 for each batch, the interception condition is met.

- **When Q_S equals 1**:

  - The function usage restrictions for query, key, and value inputs are as follows:
    - The B-axis must be less than or equal to 65536, the N-axis must be less than or equal to 256, and the D-axis must be less than or equal to 512.
    - The scenario where query, key, and value input types are all INT8 is not supported yet.
  - Page attention scenario:
    - The prerequisites for enabling page attention are that blockTable exists and is valid, and key and value are arranged in a contiguous memory space according to the indexes in blockTable. The dtype of key and value can be FLOAT16/BFLOAT16/INT8. In this use case, the inputLayout parameter of key and value is invalid. blockTable is populated with blockid. Currently, the validity of blockid is not checked and must be ensured by users.
    - blockSize is a user-defined parameter, and its value affects the performance of page attention. When page attention is enabled, blockSize must be a non-zero value and cannot exceed 512. Generally, page attention improves throughput but causes performance degradation.
    - In the page attention scenario, when the inputLayout of query is BNSD, the KV cache layout supports two formats: (blocknum, blocksize, H) and (blocknum, KV_N, blocksize, D). When the inputLayout of query is BSH or BSND, the KV cache layout supports only one format: (blocknum, blocksize, H). blocknum must not be less than the sum of the number of blocks for each batch calculated based on actualSeqLengthsKv and blockSize. In addition, the shapes of key and value must be consistent.
    - In the page attention scenario, the performance of the KV cache layout (blocknum, KV_N, blocksize, D) is generally better than that of the KV cache layout (blocknum, blocksize, H). It is recommended to prioritize the (blocknum, KV_N, blocksize, D) format.
    - In page attention use cases, when the input kv cache layout format is (blocknum, blocksize, H) and KV_N * D exceeds 64k, the call is intercepted and an error is reported due to hardware instruction constraints. To resolve this issue, enable GQA (to reduce KV_N) or change the kv cache layout format to (blocknum, KV_N, blocksize, D).
    - Page attention does not support the tensorlist scenario or the left-padding scenario.
    - In the page attention scenario, actualSeqLengthsKv must be passed.
    - In the page attention scenario, blockTable must be two-dimensional, with the first dimension length equal to B and the second dimension length not less than maxBlockNumPerSeq (maxBlockNumPerSeq is the number of blocks corresponding to the maximum actualSeqLengthsKv in each batch).
    - When page attention is enabled, KV_S in the input must be greater than or equal to maxBlockNumPerSeq * blockSize in the following use cases:
      - When attenMask is passed in, if the mask shape is (B, 1, Q_S, KV_S)
      - When pseShift is passed in, if the pseShift shape is (B, N, Q_S, KV_S)
  - KV left padding scenarios:
    - The formula for the copy start point of kvCache is: KV_S – kvPaddingSize – actualSeqLengths. The formula for the copy end point of kvCache is: KV_S – kvPaddingSize. If the copy start point or copy end point of kvCache is less than 0, the returned data result is all zeros.
    - If kvPaddingSize is less than 0, it is set to 0.
    - Must be enabled together with the actualSeqLengths parameter. Otherwise, the kv right padding use case is used by default.
    - When enabled together with the attenMask parameter, ensure that attenMask is correctly defined to properly mask invalid data. Otherwise, precision issues may be introduced.
  - pseShift function usage restrictions as follows:
    - The data type of pseShift must be consistent with that of query.

## Example

The following sample code is for reference only. For details about how to compile and run the code, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```cpp
#include <iostream>
#include <vector>
#include <math.h>
#include <cstring>
#include "acl/acl.h"
#include "aclnn/opdev/fp16_t.h"
#include "aclnnop/aclnn_fused_infer_attention_score.h"

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
    // Boilerplate: AscendCL initialization.
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
    // Call aclrtMemcpy to copy data on the host to the device memory.
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // Calculate the strides of a contiguous tensor.
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
    // 1. (Boilerplate) Initialize the device/stream. For details, see the AscendCL public API list.
    // Fill in deviceId based on your actual device.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. Construct the input and output, which need to be customized according to the API.
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

    // Create the query aclTensor.
    ret = CreateAclTensor(queryHostData, queryShape, &queryDeviceAddr, aclDataType::ACL_FLOAT16, &queryTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the key aclTensor.
    ret = CreateAclTensor(keyHostData, keyShape, &keyDeviceAddr, aclDataType::ACL_FLOAT16, &keyTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    int kvTensorNum = 1;
    aclTensor* tensorsOfKey[kvTensorNum];
    tensorsOfKey[0] = keyTensor;
    auto tensorKeyList = aclCreateTensorList(tensorsOfKey, kvTensorNum);
    // Create the value aclTensor.
    ret = CreateAclTensor(valueHostData, valueShape, &valueDeviceAddr, aclDataType::ACL_FLOAT16, &valueTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    aclTensor* tensorsOfValue[kvTensorNum];
    tensorsOfValue[0] = valueTensor;
    auto tensorValueList = aclCreateTensorList(tensorsOfValue, kvTensorNum);
    // Create the attenMask aclTensor.
    ret = CreateAclTensor(attenMaskHostData, attenMaskShape, &attenMaskDeviceAddr, aclDataType::ACL_INT8, &attenMaskTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the out aclTensor.
    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT16, &outTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    std::vector<int64_t> actualSeqlenVector = {sequenceLengthKV};
    auto actualSeqLengths = aclCreateIntArray(actualSeqlenVector.data(), actualSeqlenVector.size());

    double scaleValue = 1 / sqrt(headDims); // 1 / sqrt(d)
    int64_t preTokens = 2147483647;
    int64_t nextTokens = 2147483647;
    string sLayerOut = "BNSD";
    char layerOut[sLayerOut.length()+1];
    strcpy(layerOut, sLayerOut.c_str());
    int64_t sparseMode = 0;
    int64_t innerPrecise = 0;
    int blockSize = 0;
    int antiquantMode = 0;
    bool softmaxLseFlag = false;

    // 3. Call the CANN operator library API.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // Call the first-phase API.
    ret = aclnnFusedInferAttentionScoreGetWorkspaceSize(
        queryTensor, tensorKeyList, tensorValueList, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, numHeads, scaleValue, preTokens, nextTokens,
        layerOut, numKeyValueHeads, sparseMode, innerPrecise, blockSize, antiquantMode, softmaxLseFlag, outTensor,
        nullptr, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFusedInferAttentionScoreGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);
    // Allocate device memory based on the workspaceSize calculated by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // Call the second-phase API.
    ret = aclnnFusedInferAttentionScore(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFusedInferAttentionScore failed. ERROR: %d\n", ret); return ret);

    // 4. (boilerplate) Synchronize and wait for the task execution to complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. Obtain the value of the output and copy the result on the device side to the host side. Modify it based on the specific API definition.
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
