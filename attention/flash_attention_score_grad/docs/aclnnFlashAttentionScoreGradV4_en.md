# aclnnFlashAttentionScoreGradV4

## Supported Products

|Product     | Supported|
|:----------------------------|:-----------:|
|Ascend 950PR/Ascend 950DT|      √     |
|<term>Atlas A3 training products/Atlas A3 inference products</term>|     x      |
|<term>Atlas A2 training products/Atlas A2 inference products</term>|     x      |
|<term>Atlas 200I/500 A2 inference products</term>|      ×     |
|<term>Atlas inference products</term>|      ×     |
|<term>Atlas training products</term>|      ×     |

## Function

- Description: Computes the attention backpropagation output in training scenarios, which is the backpropagation of [FlashAttentionScoreV4](../../flash_attention_score/docs/aclnnFlashAttentionScoreV4_en.md). The query, key, and value parameters of this API support multiple sequences with the same or different lengths.
  - This API combines the [FlashAttentionScoreGradV2](./aclnnFlashAttentionScoreGradV2.md) and [FlashAttentionUnpaddingScoreGradV2](./aclnnFlashAttentionUnpaddingScoreGradV2.md) APIs and adjusts the Dropout function.
    - Ascend 950PR/Ascend 950DT: When keepProb is less than 1.0, if there is no external DropoutMask, the new parameter is used to generate the DropoutMask. If there is an external DropoutMask, the external DropoutMask is used.
- Formulas:

  - When pseType is set to 1, the calculation formula is the same as that of [FlashAttentionScoreGrad](./aclnnFlashAttentionScoreGrad.md).
  - When pseType is set to other values, the formula is as follows:

  $$
  Y=Dropout(Softmax(Mask(\frac{QK^T}{\sqrt{d}}+pse),atten\_mask),keep\_prob)V
  $$

    For convenience, the formula can be represented using variables $S$ and $P$:

  $$
  S=Mask(\frac{QK^T}{\sqrt{d}}+pse),atten\_mask
  $$

  $$
  P=Dropout(Softmax(S),keep\_prob)
  $$

  $$
  Y=PV
  $$

    Then the backward propagation formula for attention is as follows:

  $$
  dV=P^TdY
  $$

  $$
  dQ=\frac{((dS)*K)}{\sqrt{d}}
  $$

  $$
  dK=\frac{((dS)^T*Q)}{\sqrt{d}}
  $$

    **NOTE**
    The data formats of **query**, **keyIn**, and **value** can be interpreted from multiple dimensions. T (Total S Length) indicates the total length of S corresponding to all batches, B (Batch) indicates the batch size of the input sample, S (Seq-Length) indicates the length of the input sample sequence, H (Head-Size) indicates the size of the hidden layer, N (Head-Num) indicates the number of heads, and d (Head-Dim) indicates the minimum unit size of the hidden layer. d = H/N.

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnFlashAttentionScoreGradV4GetWorkspaceSize` is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, `aclnnFlashAttentionScoreGradV4` is called to perform computation.

```c++
aclnnStatus aclnnFlashAttentionScoreGradV4GetWorkspaceSize(
  const aclTensor   *query,
  const aclTensor   *keyIn, 
  const aclTensor   *value, 
  const aclTensor   *dy, 
  const aclTensor   *pseShiftOptional, 
  const aclTensor   *dropMaskOptional, 
  const aclTensor   *paddingMaskOptional, 
  const aclTensor   *attenMaskOptional, 
  const aclTensor   *softmaxMaxOptional, 
  const aclTensor   *softmaxSumOptional, 
  const aclTensor   *softmaxInOptional, 
  const aclTensor   *attentionInOptional, 
  const aclTensor   *sinkInOptional, 
  const aclTensor   *queryRopeOptional, 
  const aclTensor   *keyRopeOptional, 
  const aclTensor   *dScaleQOptional, 
  const aclTensor   *dScaleKOptional, 
  const aclTensor   *dScaleVOptional, 
  const aclTensor   *dScaleDyOptional, 
  const aclTensor   *dScaleOOptional, 
  const aclIntArray *prefixOptional, 
  const aclIntArray *actualSeqQLenOptional, 
  const aclIntArray *actualSeqKvLenOptional, 
  const aclIntArray *qStartIdxOptional, 
  const aclIntArray *kvStartIdxOptional, 
  double             scaleValue,
  double             keepProb,
  int64_t            preTokens,
  int64_t            nextTokens,
  int64_t            headNum,
  char              *inputLayout,
  char              *softmaxInLayout,
  int64_t            innerPrecise,
  int64_t            sparseMode,
  int64_t            pseType,
  int64_t            seed,
  int64_t            offset,
  int64_t            outDtype,
  aclTensor         *dqOut, 
  aclTensor         *dkOut, 
  aclTensor         *dvOut, 
  aclTensor         *dqRopeOut, 
  aclTensor         *dkRopeOut, 
  aclTensor         *dpseOut, 
  aclTensor         *dsinkOut,
  uint64_t          *workspaceSize, 
  aclOpExecutor    **executor)`
```

```c++
aclnnStatus aclnnFlashAttentionScoreGradV4(
  void             *workspace, 
  uint64_t          workspaceSize, 
  aclOpExecutor    *executor, 
  aclrtStream       stream)
```

## aclnnFlashAttentionScoreGradV4GetWorkspaceSize

- **Parameters:**
  <table style="undefined;table-layout: fixed; width: 1565px">
  <colgroup>
    <col style="width: 146px">
    <col style="width: 135px">
    <col style="width: 326px">
    <col style="width: 246px">
    <col style="width: 275px">
    <col style="width: 101px">
    <col style="width: 190px">
    <col style="width: 146px">
  </colgroup>
  <thead>
    <tr>
      <th>Name</th>
      <th>Input/Output</th>
      <th>Description</th>
      <th>Usage</th>
      <th>Data Type</th>
      <th>Data Format</th>
      <th>Dimension (Shape)</th>
      <th>Non-contiguous Tensor</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>query</td>
      <td>Input</td>
      <td>Q in the formula.</td>
      <td>The data type must be the same as that of keyIn or value.</td>
      <td>FLOAT8_E5M2, FLOAT8_E4M3FN, FLOAT16, BFLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>0, 3, 4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>keyIn</td>
      <td>Input</td>
      <td>K in the formula.</td>
      <td>The data type must be the same as that of query or value.</td>
      <td>FLOAT8_E5M2, FLOAT8_E4M3FN, FLOAT16, BFLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>0, 3, 4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>keyRopeOptional</td>
      <td>Input</td>
      <td>Rope part of the input K in the formula.</td>
      <td>The data type must be the same as that of keyIn.</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>0, 3, 4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>value</td>
      <td>Input</td>
      <td>V in the formula.</td>
      <td>The data type must be the same as that of query or keyIn.</td>
      <td>FLOAT8_E5M2, FLOAT8_E4M3FN, FLOAT16, BFLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>0, 3, 4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>dy</td>
      <td>Input</td>
      <td>dY in the formula.</td>
      <td>-</td>
      <td>FLOAT8_E5M2, FLOAT8_E4M3FN, FLOAT16, BFLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>0, 3, 4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>pseShiftOptional</td>
      <td>Optional input</td>
      <td>pse in the formula, indicating the position encoding.</td>
      <td>[B,N,S,S], [B,N,1,S], [1,N,S,S], [B,N,H,S], and [1,N,H,S] are supported.</td>
      <td>FLOAT16, BFLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>0, 4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>dropMaskOptional</td>
      <td>Optional input</td>
      <td>Dropout in the formula.</td>
      <td>-</td>
      <td>UINT8</td>
      <td>ND</td>
      <td>0, 1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>paddingMaskOptional</td>
      <td>Optional input</td>
      <td>This parameter is reserved and not used currently.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>qStartIdxOptional</td>
      <td>Optional input</td>
      <td>Start index of the current block Q sequence in the global sequence in the outer-cutting scenario.</td>
      <td>-</td>
      <td>INT64</td>
      <td>ND</td>
      <td>0, 1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>kvStartIdxOptional</td>
      <td>Optional input</td>
      <td>Start index of the current block KV sequence in the global sequence in the outer-cutting scenario.</td>
      <td>-</td>
      <td>INT64</td>
      <td>ND</td>
      <td>0, 1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>attenMaskOptional</td>
      <td>Optional input</td>
      <td><code>atten_mask</code> in the formula.</td>
      <td>
        <ul>
          <li>The value 1 indicates that the bit is not involved in the calculation, and the value 0 indicates that the bit is involved in the calculation.</li>
          <li>[B,N,S,S], [B,1,S,S], [1,1,S,S], and [S,S] are supported.</li>
        </ul>
      </td>
      <td>BOOL or UINT8</td>
      <td>ND</td>
      <td>0, 2, 4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>softmaxMaxOptional</td>
      <td>Optional input</td>
      <td>Intermediate output of the forward attention calculation.</td>
      <td>shape=[B,N,Sq,8],[N,T,8].</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>0, 4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>softmaxSumOptional</td>
      <td>Optional input</td>
      <td>Intermediate output of the forward attention calculation.</td>
      <td>shape=[B,N,Sq,8],[N,T,8].</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>0, 4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>softmaxInOptional</td>
      <td>Optional input</td>
      <td>Intermediate output of the forward attention calculation. This parameter is reserved and not used currently.</td>
      <td>shape=[B,N,Sq,8].</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>0, 4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>attentionInOptional</td>
      <td>Optional input</td>
      <td>Final output of the forward attention calculation.</td>
      <td>The data type and shape must be the same as those of query.</td>
      <td>FLOAT8_E5M2, FLOAT8_E4M3FN, FLOAT16, BFLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>0, 3, 4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>sinkInOptional</td>
      <td>Input</td>
      <td>Reserved parameter, which is not used currently.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dScaleQOptional</td>
      <td>Optional input</td>
      <td>Dequantization parameter of the query input.</td>
      <td>Support [B,N2,G,Sq/128,1].</td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>0, 1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>dScaleKOptional</td>
      <td>Optional input</td>
      <td>Dequantization parameter of the key input.</td>
      <td>Support [B,N2,1,Skv/128,1].</td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>0, 1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>dScaleVOptional</td>
      <td>Optional input</td>
      <td>Dequantization parameter of the value input.</td>
      <td>Support [B,N2,1,Skv/128,1].</td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>0, 1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>dScaleDyOptional</td>
      <td>Optional input</td>
      <td>Dequantization parameter of the dy input.</td>
      <td>Support [B,N2,G,Sq/128,1].</td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>0, 1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>dScaleOOptional</td>
      <td>Optional input</td>
      <td>Dequantization parameter of the attentionOptional input.</td>
      <td>Support [B,N2,G,Sq/128,1].</td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>0, 1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>prefixOptional</td>
      <td>Optional input</td>
      <td>N of each batch in the prefix sparse computation scenario.</td>
      <td>-</td>
      <td>INT64</td>
      <td>ND</td>
      <td>0, 1</td>
      <td>-</td>
    </tr>
    <tr>
      <td>actualSeqQLenOptional</td>
      <td>Input</td>
      <td>Query sequence length of each batch.</td>
      <td>-</td>
      <td>INT64</td>
      <td>ND</td>
      <td>0, 1</td>
      <td>-</td>
    </tr>
    <tr>
      <td>actualSeqKvLenOptional</td>
      <td>Input</td>
      <td>KV sequence length of each batch.</td>
      <td>-</td>
      <td>INT64</td>
      <td>ND</td>
      <td>0, 1</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scaleValue</td>
      <td>Input</td>
      <td>scale in the formula, indicating the scale factor.</td>
      <td>-</td>
      <td>DOUBLE</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>keepProb</td>
      <td>Input</td>
      <td>Ratio of 1s in dropMask.</td>
      <td>-</td>
      <td>DOUBLE</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>preTokens</td>
      <td>Input</td>
      <td>Left boundary of the sliding window for sparse computation.</td>
      <td>-</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>nextTokens</td>
      <td>Input</td>
      <td>Right boundary of the sliding window for sparse computation.</td>
      <td>-</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>headNum</td>
      <td>Input</td>
      <td>Number of heads on a single device, that is, the length of the N axis of query.</td>
      <td>-</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>inputLayout</td>
      <td>Input</td>
      <td>Data layout of query, key and value.</td>
      <td>BSH, SBH, BSND, and BNSD are supported.</td>
      <td>String</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>softmaxInLayout</td>
      <td>Input</td>
      <td>Reserved parameter. Not used currently.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>innerPrecise</td>
      <td>Input</td>
      <td>This parameter is reserved and not used currently.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>sparseMode</td>
      <td>Input</td>
      <td>Sparse mode.</td>
      <td>The value ranges from 0 to 8.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>pseType</td>
      <td>Input</td>
      <td>Integer on the host.</td>
      <td>The value ranges from 0 to 3.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>seed</td>
      <td>Input</td>
      <td>When keepProbOptional is less than 1.0, DropoutMask is generated based on seedOptional and offsetOptional.</td>
      <td>-</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>offset</td>
      <td>Input</td>
      <td>Int type on the host.</td>
      <td>-</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>outDtype</td>
      <td>Input</td>
      <td>If the value is 0, the output such as dqOut is FLOAT16. If the value is 1, the output is BFLOAT16.</td>
      <td>-</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dqOut</td>
      <td>Output</td>
      <td>dQ in the formula, indicating the gradient of query.</td>
      <td>-</td>
      <td>FLOAT16, BFLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>0, 3, 4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>dqRopeOut</td>
      <td>Output</td>
      <td>Gradient of queryRope.</td>
      <td>-</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>0, 3, 4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>dkOut</td>
      <td>Output</td>
      <td>dK in the formula, indicating the gradient of keyIn.</td>
      <td>-</td>
      <td>FLOAT16, BFLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>0, 3, 4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>dkRopeOut</td>
      <td>Output</td>
      <td>Gradient of keyInRope in the formula, dkRope.</td>
      <td>-</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>0, 3, 4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>dvOut</td>
      <td>Output</td>
      <td>dV in the formula, indicating the gradient of value.</td>
      <td>-</td>
      <td>FLOAT16, BFLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>0, 3, 4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>dpseOut</td>
      <td>Output</td>
      <td>d(pse) gradient.</td>
      <td>Reserved.</td>
      <td>FLOAT16, BFLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>0, 4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>dsinkOut</td>
      <td>Output</td>
      <td>Reserved parameter, which is not used currently.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
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
      <td>Operator executor, containing the computation process.</td>
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

  The first-phase API implements input parameter validation. The following error codes may be returned.

  <table style="undefined;table-layout: fixed; width: 1166px"><colgroup>
  <col style="width: 267px">
  <col style="width: 124px">
  <col style="width: 775px">
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
      <td>The required input, output, or attribute is passed as a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="2">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="2">161002</td>
      <td>The data type of query, keyIn, value, dy, pseShiftOptional, dropMaskOptional, paddingMaskOptional, attenMaskOptional, softmaxMaxOptional, softmaxSumOptional, softmaxInOptional, attentionInOptional, dqOut, dkOut, or dvOut is not supported.</td>
    </tr>
  </tbody>
  </table>

## aclnnFlashAttentionScoreGradV4

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1166px"><colgroup>
  <col style="width: 173px">
  <col style="width: 133px">
  <col style="width: 860px"> 
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
      <td>Memory address of the workspace to be allocated on the device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Workspace size allocated on the device, which is obtained by the first API aclnnFlashAttentionScoreGradV4GetWorkspaceSize.</td>
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

## Constraints<a name="1"></a>

- Deterministic computation:
  - `aclnnFlashAttentionScoreGradV4` defaults to non-deterministic implementation. You can call `aclrtCtxSetSysParamOpt` to enable deterministic computation.
- The restrictions on the inputs query, key, value, and dy are as follows:
  - **B**: The batch sizes must be equal.
  - `inputLayout` must be consistent.
  - D: The head dimension (D) of the query and key must be the same, and that of the value and dy must be the same. In addition, the D of the query and key must be greater than or equal to that of the value and dy.
- **N** of the input `query` or `dy` can be different from **N** of the `key` or `value`, but they must be proportional. That is, Nq/Nkv must be a non-zero integer. The value of Nq ranges from 1 to 256.
- The following uses the `inputLayout` TND as an example to describe the constraints on the shape:

    - **B**: The value ranges from 1 to 2K. When `prefixOptional` is passed, **B** supports a maximum of 1K.
    - **N**: The value ranges from 1 to 256.
    - **S**: The value ranges from 1 to 1M.
    - **D**: The value ranges from 1 to 768.
    - **KeepProb**: The value range is (0, 1].
- In some scenarios, if the computation load is too large, the operator execution may time out (an AI Core error is reported, and `errorStr` is `timeout or trap error`). In this case, you are advised to perform axis splitting. Note: The computation load is affected by parameters such as **B**, **S**, **N**, and **D**. Larger values indicate larger computation loads.
- The **prefixOptional** sparse computing supports only compression scenarios. sparseModeOptional = 6. When Sq > Skv, the value range of **N** of **prefix** is \[0, Skv\]. When Sq ≤ Skv, the value range of **N** of **prefix** is \[Skv – Sq, Skv\]. When sparseModeOptional is set to 5 or prefixOptional is not passed, full computation is performed. When sparseModeOptional is set to 6, prefixOptional must be passed.
- When sparseMode is set to 7, the optional input pseShiftOptional is not supported.
- When sparseMode is set to 8 and the length of q and kv in each sequence is the same, the optional input pseShiftOptional is supported. The global PSE is generated. The q direction can be used for external splitting. The q and kv of each sequence must have the same length before external splitting. Then, **actualSeqQLenOptional[0] - actualSeqKvLenOptional[0] + qStartIdxOptional - kvStartIdxOptional == 0** (experimental function).
- The `actualSeqQLenOptional` input supports the S length of 0 in a batch. In this case, the `pseShiftOptional` input is not supported.
- The restrictions on the softmaxMax and softmaxSum parameters are as follows: The input format is fixed to [B, N, S, 8], except for the TND input format, which is [N, T, 8]. Note that T = B x S.
- The value of `headNum` must be the same as the value of **N** in `query`.
- Ascend 950PR/Ascend 950DT:

    - seedOptional and offsetOptional take effect only when keepProbOptional is less than 1.0. Otherwise, they do not take effect.
    - When keepProbOptional is less than 1.0, if dropMaskOptional is not nullptr, the input dropMask is used. Otherwise, the dropMask generated by seed and offset is used.
- In TND format, some batches at the end are not involved in the calculation. In this case, you can pass zeros of the corresponding number to the end of actual_seq_q_len and actual_seq_kv_len. Assume that the actual S length is [2, 3, 4, 5, 6]. If the last two batches are not involved in the calculation, the input actual_seq_q_len is [2, 3, 4, 0, 0]. In this case, if prefixOptional needs to be passed, the same number of zeros also needs to be passed to the end, for example, [1, 1, 1, 0, 0].

## Examples

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```C++
#include <iostream>
#include <vector>
#include <cstdint>
#include <cmath>
#include <random>
#include "acl/acl.h"
#include "aclnnop/aclnn_flash_attention_score_grad.h"

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

void PrintOutResult(std::vector<int64_t> &shape, void** deviceAddr) {
  auto size = GetShapeSize(shape);
  std::vector<float> resultData(size, 0);
  auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]),
                         *deviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("mean result[%ld] is: %f\n", i, resultData[i]);
  }
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
  auto size = GetShapeSize(shape) * sizeof(T);
  // Call aclrtMalloc to allocate memory on the device.
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  // Call aclrtMemcpy to copy the data on the host to the memory on the device. 
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

int main() {
  // 1. (Boilerplate) Initialize the device and stream. For details, see the list of external AscendCL APIs.
  // Set the device ID in use.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the inputs and outputs based on the API definition.
  int64_t B = 1;
  int64_t N1 = 1;
  int64_t N2 = 1;
  int64_t S1 = 128;
  int64_t S2 = 128;
  int64_t D = 128;

  int64_t H1 = N1 * D;
  int64_t H2 = N2 * D;

  int64_t q_size = S1 * B * H1;
  int64_t kv_size = S2 * B * H2;
  int64_t atten_mask_size = S1 * S2;
  int64_t softmax_size = B * N1 * S1 * 8;

  std::vector<int64_t> qShape = {S1, B, H1};
  std::vector<int64_t> kShape = {S2, B, H2};
  std::vector<int64_t> vShape = {S2, B, H2};
  std::vector<int64_t> dxShape = {S1, B, H1};
  std::vector<int64_t> attenmaskShape = {S1, S2};
  std::vector<int64_t> softmaxMaxShape = {B, N1, S1, 8};
  std::vector<int64_t> softmaxSumShape = {B, N1, S1, 8};
  std::vector<int64_t> attentionInShape = {S1, B, H1};
  std::vector<int64_t> dqShape = {S1, B, H1};
  std::vector<int64_t> dkShape = {S2, B, H2};
  std::vector<int64_t> dvShape = {S2, B, H2};

  void* qDeviceAddr = nullptr;
  void* kDeviceAddr = nullptr;
  void* vDeviceAddr = nullptr;
  void* dxDeviceAddr = nullptr;
  void* attenmaskDeviceAddr = nullptr;
  void* softmaxMaxDeviceAddr = nullptr;
  void* softmaxSumDeviceAddr = nullptr;
  void* attentionInDeviceAddr = nullptr;
  void* dqDeviceAddr = nullptr;
  void* dkDeviceAddr = nullptr;
  void* dvDeviceAddr = nullptr;

  aclTensor* q = nullptr;
  aclTensor* k = nullptr;
  aclTensor* v = nullptr;
  aclTensor* dx = nullptr;
  aclTensor* pse = nullptr;
  aclTensor* dropMask = nullptr;
  aclTensor* padding = nullptr;
  aclTensor* attenmask = nullptr;
  aclTensor* queryRope = nullptr;
  aclTensor* keyRope = nullptr;
  aclTensor* dScaleQ = nullptr;
  aclTensor* dScaleK = nullptr;
  aclTensor* dScaleV = nullptr;
  aclTensor* dScaleDy = nullptr;
  aclTensor* dScaleO = nullptr;
  aclTensor* softmaxMax = nullptr;
  aclTensor* softmaxSum = nullptr;
  aclTensor* softmaxIn = nullptr;
  aclTensor* attentionIn = nullptr;
  aclTensor* dq = nullptr;
  aclTensor* dk = nullptr;
  aclTensor* dv = nullptr;
  aclTensor* dpse = nullptr;
  aclTensor* dqRope = nullptr;
  aclTensor* dkRope = nullptr;

  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<float> dist(0.0f, 1.0f); // Normal distribution with mean 0 and standard deviation 1

  std::vector<float> qHostData(q_size);
  for (auto& val : qHostData) {
      val = dist(gen);
  }

  std::vector<float> kHostData(kv_size);
  for (auto& val : kHostData) {
      val = dist(gen);
  }

  std::vector<float> vHostData(kv_size);
  for (auto& val : vHostData) {
      val = dist(gen);
  }

  std::vector<float> dxHostData(q_size);
  for (auto& val : dxHostData) {
      val = dist(gen);
  }

  std::vector<uint8_t> attenmaskHostData(atten_mask_size, 0);
  std::vector<float> softmaxMaxHostData(softmax_size, 3.0);
  std::vector<float> softmaxSumHostData(softmax_size, 3.0);
  std::vector<float> attentionInHostData(q_size, 255.0);
  std::vector<float> dqHostData(q_size, 2.0);
  std::vector<float> dkHostData(kv_size, 2.0);
  std::vector<float> dvHostData(kv_size, 3.0);

  ret = CreateAclTensor(qHostData, qShape, &qDeviceAddr, aclDataType::ACL_FLOAT, &q);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(kHostData, kShape, &kDeviceAddr, aclDataType::ACL_FLOAT, &k);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(vHostData, vShape, &vDeviceAddr, aclDataType::ACL_FLOAT, &v);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(dxHostData, dxShape, &dxDeviceAddr, aclDataType::ACL_FLOAT, &dx);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(attenmaskHostData, attenmaskShape, &attenmaskDeviceAddr, aclDataType::ACL_UINT8, &attenmask);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(softmaxMaxHostData, softmaxMaxShape, &softmaxMaxDeviceAddr, aclDataType::ACL_FLOAT, &softmaxMax);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(softmaxSumHostData, softmaxSumShape, &softmaxSumDeviceAddr, aclDataType::ACL_FLOAT, &softmaxSum);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(attentionInHostData, attentionInShape, &attentionInDeviceAddr, aclDataType::ACL_FLOAT, &attentionIn);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(dqHostData, dqShape, &dqDeviceAddr, aclDataType::ACL_FLOAT, &dq);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(dkHostData, dkShape, &dkDeviceAddr, aclDataType::ACL_FLOAT, &dk);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(dvHostData, dvShape, &dvDeviceAddr, aclDataType::ACL_FLOAT, &dv);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  double scaleValue = 1.0/sqrt(128);
  double keepProb = 1.0;
  int64_t preTokens = 65536;
  int64_t nextTokens = 65536;
  int64_t headNum = 1;
  int64_t innerPrecise = 0;
  int64_t sparseMode = 0;
  int64_t pseType = 1;
  int64_t outDtype = 1;
  int64_t seed = 0;
  int64_t offset = 0;
  char inputLayOut[5] = {'S', 'B', 'H', 0};

  // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  // Call the first-phase API of aclnnFlashAttentionScoreGradV4.
  ret = aclnnFlashAttentionScoreGradV4GetWorkspaceSize(q, k, v, dx, pse, dropMask, padding,
            attenmask, softmaxMax, softmaxSum, softmaxIn, attentionIn, nullptr, queryRope, keyRope, dScaleQ, dScaleK, dScaleV, 
            dScaleDy, dScaleO, nullptr, nullptr, nullptr, nullptr, nullptr, scaleValue, keepProb,
            preTokens, nextTokens, headNum, inputLayOut, nullptr, innerPrecise, sparseMode,outDtype, pseType, seed, offset,
            dq,dk,dv,dqRope,dkRope,dpse, nullptr, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFlashAttentionScoreGradV4GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  // Allocate device memory based on workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }

  // Call the second-phase API of aclnnFlashAttentionScoreGradV4.
  ret = aclnnFlashAttentionScoreGradV4(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFlashAttentionScoreGradV4 failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device memory to the host. Modify the code based on the API definition.
  PrintOutResult(dqShape, &dqDeviceAddr);
  PrintOutResult(dkShape, &dkDeviceAddr);
  PrintOutResult(dvShape, &dvDeviceAddr);

  // 6. Release aclTensor and aclScalar. Modify the code based on the API definition.
  aclDestroyTensor(q);
  aclDestroyTensor(k);
  aclDestroyTensor(v);
  aclDestroyTensor(dx);
  aclDestroyTensor(attenmask);
  aclDestroyTensor(softmaxMax);
  aclDestroyTensor(softmaxSum);
  aclDestroyTensor(attentionIn);
  aclDestroyTensor(dq);
  aclDestroyTensor(dk);

  // 7. Free device resources.
  aclrtFree(qDeviceAddr);
  aclrtFree(kDeviceAddr);
  aclrtFree(vDeviceAddr);
  aclrtFree(dxDeviceAddr);
  aclrtFree(attenmaskDeviceAddr);
  aclrtFree(softmaxMaxDeviceAddr);
  aclrtFree(softmaxSumDeviceAddr);
  aclrtFree(attentionInDeviceAddr);
  aclrtFree(dqDeviceAddr);
  aclrtFree(dkDeviceAddr);
  aclrtFree(dvDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}
```
