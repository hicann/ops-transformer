# aclnnFlashAttentionScoreV4

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-21T08:53:02.133Z -->

## Applicable Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 products</term>: Supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 products</term>: Not supported
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

- Description: In training use cases, this API implements self-attention computation using the FlashAttention algorithm. To align with competitor products and adapt to the gptoss model, this API supports the sink feature. **The query, key, and value parameters of this API support multiple sequences of equal or unequal lengths**
  - **Compared with the [FlashAttentionScoreV3](./aclnnFlashAttentionScoreV3_en.md) API, the functional differences of this API are as follows:**:
    - Dropout function adjustment: When keepProb is less than 1.0, if no external DropoutMask is passed in, the newly added parameters seed and offset are used to generate the DropoutMask; if an external DropoutMask is passed in, the external DropoutMask is used.
    Added the optional input sinkOptional, which supports the FLOAT32 data type.
  - **Compared with the [FlashAttentionVarLenScoreV5](./aclnnFlashAttentionVarLenScoreV5_en.md) API, the functional differences are as follows**:
    - Dropout function adjustment: When keepProb is less than 1.0, if no external DropoutMask is passed in, the newly added parameters seed and offset are used to generate the DropoutMask; if an external DropoutMask is passed in, the external DropoutMask is used.

- Formula:

   The forward calculation formula of attention is as follows:

   - When psetype = 1, the formula is the same as that of [FlashAttentionScore](./aclnnFlashAttentionScore_en.md).

   - When psetype is set to other values, the formula is as follows:

     $$
     attention\_out=Dropout(Softmax(Mask(scale*(query*key^T) + pse),atten\_mask),keep\_prob)*value
     $$
  After **sink** is added, the calculation logic is as follows. The main modifications are in the softmax_max and softmax_sum calculation logic.

  $$
  S = Q * K^{T}
  $$

  $$
  m = max(sink, max(S))
  $$

  $$
  Attention = \frac{e^{S - m} * V}{\sum e^{S-m} + e^{sink - m}}
  $$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). First, call the "aclnnFlashAttentionScoreV4GetWorkspaceSize" API to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, call the "aclnnFlashAttentionScoreV4" API to perform the computation.

```c++
aclnnStatus aclnnFlashAttentionScoreV4GetWorkspaceSize(
  const aclTensor   *query,
  const aclTensor   *key,
  const aclTensor   *value,
  const aclTensor   *realShiftOptional,
  const aclTensor   *dropMaskOptional,
  const aclTensor   *paddingMaskOptional,
  const aclTensor   *attenMaskOptional,
  const aclTensor   *queryRopeOptional,
  const aclTensor   *keyRopeOptional,
  const aclTensor   *dScaleQOptional,
  const aclTensor   *dScaleKOptional,
  const aclTensor   *dScaleVOptional,
  const aclTensor   *sinkOptional,
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
  int64_t            innerPrecise,
  int64_t            sparseMode,
  int64_t            outDtype,
  int64_t            pseType,
  char              *softmaxOutLayout,
  int64_t            seed,
  int64_t            offset,
  const aclTensor   *softmaxMaxOut,
  const aclTensor   *softmaxSumOut,
  const aclTensor   *softmaxOutOut,
  const aclTensor   *attentionOutOut,
  uint64_t          *workspaceSize,
  aclOpExecutor    **executor)
```

```c++
aclnnStatus aclnnFlashAttentionScoreV4(
  void             *workspace,
  uint64_t          workspaceSize,
  aclOpExecutor    *executor,
  const aclrtStream stream)
```

## aclnnFlashAttentionScoreV4GetWorkspaceSize

- **Parameters:**
  <table style="undefined;table-layout: fixed; width: 1573px"><colgroup>
    <col style="width: 213px">
    <col style="width: 121px">
    <col style="width: 253px">
    <col style="width: 262px">
    <col style="width: 295px">
    <col style="width: 115px">
    <col style="width: 169px">
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
        <td>The query in the formula.</td>
        <td>The data type is the same as that of key/value.</td>
        <td>FLOAT16, BFLOAT16, FLOAT32</td>
        <td>ND</td>
        <td>0, 3, 4</td>
        <td>√</td>
      </tr>
      <tr>
        <td>key</td>
        <td>Input</td>
        <td>The key in the formula.</td>
        <td>The data type must be the same as that of query/value.</td>
        <td>FLOAT16, BFLOAT16, FLOAT32</td>
        <td>ND</td>
        <td>0, 3, 4</td>
        <td>√</td>
      </tr>
      <tr>
        <td>value</td>
        <td>Input</td>
        <td>The value in the formula.</td>
        <td>The data type must be the same as that of query/key.</td>
        <td>FLOAT16, BFLOAT16, FLOAT32</td>
        <td>ND</td>
        <td>0, 3, 4</td>
        <td>√</td>
      </tr>
      <tr>
        <td>realShiftOptional</td>
        <td>Optional input</td>
        <td>pse in the formula.</td>
        <td>The data type must be the same as that of attentionOut.</td>
        <td>FLOAT16, BFLOAT16, FLOAT32</td>
        <td>ND</td>
        <td>[B,N,Sq,Skv], [B,N,1,Skv], [1,N,Sq,Skv], [B,N,1024,Skv], [1,N,1024,Skv], [B,N], [N]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>paddingMaskOptional</td>
        <td>Input</td>
        <td>Not used currently.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>dropMaskOptional</td>
        <td>Input</td>
        <td>Dropout in the formula.</td>
        <td>-</td>
        <td>UINT8</td>
        <td>ND</td>
        <td>0, 1</td>
        <td>√</td>
      </tr>
      <tr>
        <td>attenMaskOptional</td>
        <td>Input</td>
        <td>atten_mask in the formula.</td>
        <td>A value of 1 indicates the position does not participate in computation. A value of 0 indicates the position participates in computation.</td>
        <td>BOOL, UINT8</td>
        <td>ND</td>
        <td>[B,N,S,S], [B,1,S,S], [1,1,S,S], [S,S]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>queryRopeOptional</td>
        <td>Input</td>
        <td>An aclTensor on the device.</td>
        <td>The data type must be the same as that of query.</td>
        <td>FLOAT16, BFLOAT16</td>
        <td>ND</td>
        <td>0, 3, 4</td>
        <td>√</td>
      </tr>
      <tr>
        <td>keyRopeOptional</td>
        <td>Input</td>
        <td>The aclTensor on the device.</td>
        <td>The data type is the same as that of key.</td>
        <td>FLOAT16, BFLOAT16</td>
        <td>ND</td>
        <td>0, 3, 4</td>
        <td>√</td>
      </tr>
      <tr>
        <td>dScaleQOptional</td>
        <td>Input</td>
        <td>Quantization parameters of query.</td>
        <td>Input shape: [B,N1,Ceil(Sq/128),1].</td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>0, 3, 4</td>
        <td>√</td>
      </tr>
      <tr>
        <td>dScaleKOptional</td>
        <td>Input</td>
        <td>Quantization parameters of key.</td>
        <td>Input shape: [B,N2,Ceil(Skv/256),1].</td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>0, 3, 4</td>
        <td>√</td>
      </tr>
      <tr>
        <td>dScaleVOptional</td>
        <td>Input</td>
        <td>Quantization parameters of value.</td>
        <td>Input shape is [B,N2,Ceil(Skv/512),1].</td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>0, 3, 4</td>
        <td>√</td>
      </tr>
      <tr>
        <td>sinkOptional</td>
        <td>Input</td>
        <td>sink in the formula. Provides the function of resolving the issue where attention computation concentrates on the first part of TOKENs.</td>
        <td>If the input is not a null pointer, only a 1D tensor is supported.</td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>[N1]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>prefixOptional</td>
        <td>Input</td>
        <td>Represents the N value of each Batch in the prefix sparse computation use case.</td>
        <td>-</td>
        <td>INT64</td>
        <td>ND</td>
        <td>0, 1</td>
        <td>-</td>
      </tr>
      <tr>
        <td>actualSeqQLenOptional</td>
        <td>Input</td>
        <td>Describes the sequence length of query for each batch.</td>
        <td>-</td>
        <td>INT64</td>
        <td>ND</td>
        <td>0, 1</td>
        <td>-</td>
      </tr>
      <tr>
        <td>actualSeqKvLenOptional</td>
        <td>Input</td>
        <td>Describes the sequence length of the key/value for each Batch.</td>
        <td>-</td>
        <td>INT64</td>
        <td>ND</td>
        <td>0, 1</td>
        <td>-</td>
      </tr>
      <tr>
        <td>qStartIdxOptional</td>
        <td>Input</td>
        <td>Represents the global start index of the current chunk's query sequence in the pre-split use case.</td>
        <td>-</td>
        <td>INT64</td>
        <td>ND</td>
        <td>0, 1</td>
        <td>-</td>
      </tr>
      <tr>
        <td>kvStartIdxOptional</td>
        <td>Input</td>
        <td>Indicates the start index of the current chunk's key and value sequences in the global sequence in the external chunking use case.</td>
        <td>-</td>
        <td>INT64</td>
        <td>ND</td>
        <td>0, 1</td>
        <td>-</td>
      </tr>
      <tr>
        <td>scaleValue</td>
        <td>Input</td>
        <td>scale in the formula, representing the scaling coefficient.</td>
        <td>-</td>
        <td>DOUBLE</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>keepProb</td>
        <td>Input</td>
        <td>Represents the ratio of 1s in dropMaskOptional.</td>
        <td>Value range: (0, 1].</td>
        <td>DOUBLE</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>preTokens</td>
        <td>Input</td>
        <td>Indicates the left boundary of the sliding window for sparse computation.</td>
        <td>-</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>nextTokens</td>
        <td>Input</td>
        <td>Indicates the right boundary of the sliding window for sparse computation.</td>
        <td>-</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>headNum</td>
        <td>Input</td>
        <td>Number of heads per device, that is, the length of the N axis of the input query.</td>
        <td>-</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>inputLayout</td>
        <td>Input</td>
        <td>Data layout format of the input query, key, and value.</td>
        <td>Supports BSH, SBH, BSND, BNSD, and TND.</td>
        <td>String</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>innerPrecise</td>
        <td>Input</td>
        <td>int64_t on the host.</td>
        <td>Enables invalid row handling when set to 2. Do not set it unless necessary.</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>sparseMode</td>
        <td>Input</td>
        <td>Specifies the sparse mode.</td>
        <td>Supported values are 0, 1, 2, 3, 4, 5, and 6.</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>outDtype</td>
        <td>Input</td>
        <td>int64_t on the host.</td>
        <td>Takes effect only in quantization use cases. 0 indicates that the attenout output is in fp16 format, and 1 indicates that the output is in bf16 format.</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>softmaxOutLayout</td>
        <td>Input</td>
        <td>Controls the softmax output in TND use cases.</td>
        <td>If "same_as_input" is passed, the softmax output layout is the same as the input, in TN8 layout. If an empty string is passed, the logic remains unchanged, and the softmax output layout is BNS8.</td>
        <td>String</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>pseType</td>
        <td>Input</td>
        <td>int64_t on the Host side.</td>
        <td>Controls the computation order of mul and add. The value can be 0, 1, 2, or 3.</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>seed</td>
        <td>Input</td>
        <td>int64_t on the Host, the seed for generating dropmask.</td>
        <td>If internal dropmask generation is enabled, this parameter must be passed in. Otherwise, the default value seed=0 is used for internal dropmask generation.</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>offset</td>
        <td>Input</td>
        <td>Offset for generating dropmask.</td>
        <td>If internal dropmask generation is enabled, this parameter is required. Otherwise, the default value offset=0 is used.</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>softmaxMaxOut</td>
        <td>Output</td>
        <td>Max intermediate result of the Softmax computation, used for backward computation.</td>
        <td>The output shape is determined by softmaxOutLayout.</td>
        <td>FLOAT</td>
        <td>ND</td>
        <td>0, 3, 4</td>
        <td>√</td>
      </tr>
      <tr>
        <td>softmaxSumOut</td>
        <td>Output</td>
        <td>Sum intermediate result of Softmax computation, used for backward computation.</td>
        <td>The output shape is determined by softmaxOutLayout.</td>
        <td>FLOAT</td>
        <td>ND</td>
        <td>0, 3, 4</td>
        <td>√</td>
      </tr>
      <tr>
        <td>attentionOutOut</td>
        <td>Output</td>
        <td>Final output of the formula.</td>
        <td>The data type and shape are the same as those of query.</td>
        <td>FLOAT16, BFLOAT16, FLOAT32</td>
        <td>ND</td>
        <td>0, 3, 4</td>
        <td>√</td>
      </tr>
      <tr>
        <td>workspaceSize</td>
        <td>Output</td>
        <td>Returns the workspace size to be allocated on the Device.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>executor</td>
        <td>Output</td>
        <td>Returns the op executor containing the operator computation flow.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
    </tbody>
  </table>

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table style="undefined;table-layout: fixed;width: 1202px"><colgroup>
  <col style="width: 262px">
  <col style="width: 121px">
  <col style="width: 819px">
  </colgroup>
  <thead>
    <tr>
      <th>Return Values</th>
      <th>Error Codes</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>The parameter is a required input, output, or attribute, and is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="2">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="2">161002</td>
      <td>The data types of query, key, value, realShiftOptional, dropMaskOptional, paddingMaskOptional, attenMaskOptional, sinkOptional, softmaxMaxOut, softmaxSumOut, softmaxOutOut, and attentionOutOut are out of the supported range.</td>
    </tr>
  </tbody>
  </table>

## aclnnFlashAttentionScoreV4

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1154px"><colgroup>
  <col style="width: 153px">
  <col style="width: 121px">
  <col style="width: 880px">
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
      <td>Workspace memory address allocated on the Device side.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace allocated on the Device side, obtained by the first-phase API aclnnFlashAttentionScoreV4GetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>op executor, which contains the operator computation flow.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Specifies the stream for task execution.</td>
    </tr>
  </tbody>
  </table>

- **Return value**

    Returns an aclnnStatus status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnFlashAttentionScoreV4 defaults to a deterministic implementation.
- The input query, key, and value
  - B: The batchsize must be equal.
  - D: The head dimensions must satisfy (qD == kD && kD >= vD).
  - The inputLayout must be consistent.
- The input queryRopeOptional and query have the same input shape except in the D dimension. Other shape parameters should be identical.
- The input keyRopeOptional and key have the same input shape except in the D dimension. Other shape parameters should be identical.
- MLA concat use case: only dSize=128 and dSizeRope=64 are supported
- Constraints on data shapes, taking TND, BSND, and BNSD of inputLayout as examples (H = N * D for BSH and SBH), where:
    - T (B * S): The value range is 1 to 1M. In TND format, the maximum supported length of actualSeqQLenOptional is 20000.
    - B: The value range is 1 to 2M. When prefixOptional is present, B supports a maximum of 2K.
    - N: ranges from 1 to 256.
    - S: ranges from 1 to 1M.
    - D: ranges from 1 to 768.
- Empty tensor use case:
    - If any of B, N1 (headNum), or S1 (sequence length of query) is 0, the operator performs no computation and returns success directly, with workspaceSize being 0.
    - If any of N2 (number of heads of key/value), S2 (sequence length of key/value), or D (Head-Dim) is 0, query/key/value are empty tensors (shapeSize is 0), but the output shapes of softmaxMaxOut and softmaxSumOut may be non-empty (the shapes do not depend on the D dimension). In this case, the operator performs empty input processing and initializes the non-empty outputs.
    - In the preceding empty tensor use cases, the operator returns normally without errors.
- The data layout format of query, key, and value supports interpretation from multiple dimensions, where B (Batch) indicates the input sample batch size, S (Seq-Length) indicates the input sample sequence length, H (Head-Size) indicates the hidden layer size, N (Head-Num) indicates the number of heads, D (Head-Dim) indicates the minimum unit dimension of the hidden layer, and D = H/N is satisfied.
- innerPrecise: Currently, 0 and 1 are reserved configuration values, and 2 enables invalid row computation to avoid precision loss caused by entire-row masks during computation. However, this configuration degrades performance. If the operator can detect invalid rows, it automatically enables invalid row computation, for example, when sparseMode is 3 and Sq > Skv.
- Meaning of each pseType value

    | pseType     | Meaning                       |      Note |
    | ----------- | --------------------------------- | ----------|
    | 0           | pse passed in when calling the operator: mul first, then add. | - |
    | 1           | pse passed in when calling the operator: add first, then mul. | Consistent with the implementation of [FlashAttentionScore](./aclnnFlashAttentionScore_en.md). |
    | 2           | pse automatically generated by the operator: mul first, then add. | - |
    | 3           | pse automatically generated by the operator: mul first, then add, then sqrt. | - |

- When pseType is 2 or 3, only equal-length Sq and Skv are supported.
- If sinkOptional is not None, input tensors such as query and key support only float16 and bfloat16.
- The constraints on sparseMode are as follows:
  - When all attenMaskOptional shapes are smaller than 2048 and identical, it is recommended to use the default mode to reduce memory usage.
  - When sparseMode is set to 1, 2, 3, or 5, the user-configured preTokens and nextTokens do not take effect.
  - When configured as 0 or 4, ensure that attenMaskOptional is consistent with the ranges of preTokens and nextTokens.
  - If not explicitly specified, it is recommended to pass 0.
  - For details about different sparse modes, see [sparseMode Introduction](../../../docs/en/context/sparse_mode_introduction.md).
- In some scenarios, if the computation workload is too large, the operator execution may time out (aicore error with errorStr: timeout or trap error). In this case, axis splitting is recommended. Note: The computation workload here is affected by parameters such as B, S, N, and D. Larger values result in a heavier workload.
- In the band scenario, preTokens and nextTokens must have an intersection.
- For the prefixOptional sparse computation scenario, the scenarios include sparseMode=5 and sparseMode=6 when sequences have equal lengths, and sparseMode=6 when sequences have unequal lengths. In these two scenarios, when Sq > Skv, the N value range of prefix is [0, Skv]; when Sq <= Skv, the N value range of prefix is [Skv-Sq, Skv]. When sparseModeOptional=5 and prefix N > Skv, or when prefixOptional is not passed, full computation is performed. sparseModeOptional=6 requires that prefixOptional must be passed.
- realShiftOptional: When Sq is greater than 1024 and BNHS or 1NHS format is configured, Sq and Skv must be of equal length.
- The actualSeqQLenOptional input supports an S length of 0 on a certain batch. In this case, the optional input realShiftOptional is not supported.
- In TND use cases, the attenMaskOptional input does not support padding. That is, no row in attenMaskOptional can be all 1s.
- An S length of 0 on a certain batch in actualSeqQLenOptional is supported. If S is 0, the pse input is not supported.
  Assume that the actual S length is [2, 2, 0, 2, 2], then the passed actualSeqQLenOptional is [2, 4, 4, 6, 8].

<!-- npu="950" id7 -->
- <term>Ascend 950PR&950DT products</term>:
    - seed and offset take effect only when keepProb is less than 1.0. Otherwise, they do not take effect.
    - When keepProb is less than 1.0, if dropMaskOptional is not nullptr, the input dropMask is used. Otherwise, the dropMask generated from seed and offset is used.
<!-- end id7 -->
- In TND format, trailing batches that do not participate in computation are supported. In this case, pass the corresponding number of zeros at the end of actual_seq_q_len and actual_seq_kv_len. Assume the actual S length is [2, 3, 4, 5, 6]. If you want the last two batches to not participate in computation, pass actual_seq_q_len as [2, 3, 4, 0, 0]. In this case, if prefixOptional needs to be passed, its trailing part must also contain the same number of zeros, for example, [1, 1, 1, 0, 0].
- softmaxOutLayout supports the following input values: an empty string or "same_as_input".
- realShiftOptional: If Sq is greater than 1024, Sq and Skv of each batch have the same length, and sparseMode is 0, 2, or 3 with a lower triangular mask, alibi position encoding compression can be enabled. In this case, only the last 1024 rows of the original PSE need to be input for memory optimization, that is, alibi_compress = ori_pse[:, :, -1024:, :]. Details are as follows:
  - When the parameter differs for each batch, the shape is BNHSkv (H = 1024).
  - When the parameter is the same for each batch, the shape is 1NHSkv (H = 1024).
  - If pseType is 2 or 3, the data type must be FLOAT32, and the supported shapes are [B,N] or [N].
  - If this parameter is not enabled, pass nullptr to realShiftOptional and 1 to pseType.

## Example

The sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```C++
#include <iostream>
#include <vector>
#include <cmath>
#include "acl/acl.h"
#include "aclnn/opdev/fp16_t.h"
#include "aclnnop/aclnn_flash_attention_score.h"

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

void PrintOutResultFp16(std::vector<int64_t> &shape, void** deviceAddr) {
  auto size = GetShapeSize(shape);
  std::vector<aclFloat16> resultData(size, 0);
  auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]),
                         *deviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("mean result[%ld] is: %f\n", i, aclFloat16ToFloat(resultData[i]));
  }
}

int Init(int32_t deviceId, aclrtContext* context, aclrtStream* stream) {
  // Boilerplate: Initialize AscendCL.
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
                    aclDataType dataType, aclTensor** tensor) {
  auto size = GetShapeSize(shape) * sizeof(T);
  // Call aclrtMalloc to allocate device memory.
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  // Call aclrtMemcpy to copy host data to device memory.
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
  // Calculate the strides of the contiguous tensor.
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
  // 1. (Boilerplate) Initialize device/context/stream. For details, see the AscendCL external API list.
  // Set deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtContext context;
  aclrtStream stream;
  auto ret = Init(deviceId, &context, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the input and output. Customize the construction according to the API definition.
  int64_t B = 1;
  int64_t N1 = 1;
  int64_t N2 = 1;
  int64_t S1 = 256;
  int64_t S2 = 256;
  int64_t D = 128;
  int64_t H1 = N1 * D;
  int64_t H2 = N2 * D;

  int64_t q_size = S1 * B * H1;
  int64_t kv_size = S2 * B * H2;
  int64_t atten_mask_size = S1 * S2;
  int64_t softmax_size = B * N1 * S1 * 8;
  int64_t sink_size = N1;

  std::vector<int64_t> qShape = {S1, B, H1};
  std::vector<int64_t> kShape = {S2, B, H2};
  std::vector<int64_t> vShape = {S2, B, H2};
  std::vector<int64_t> attenmaskShape = {S1, S2};
  std::vector<int64_t> sinkShape = {N1};

  std::vector<int64_t> attentionOutShape = {S1, B, H1};
  std::vector<int64_t> softmaxMaxShape = {B, N1, S1, 8};
  std::vector<int64_t> softmaxSumShape = {B, N1, S1, 8};

  void* qDeviceAddr = nullptr;
  void* kDeviceAddr = nullptr;
  void* vDeviceAddr = nullptr;
  void* attenmaskDeviceAddr = nullptr;
  void* sinkDeviceAddr = nullptr;
  void* attentionOutDeviceAddr = nullptr;
  void* softmaxMaxDeviceAddr = nullptr;
  void* softmaxSumDeviceAddr = nullptr;

  aclTensor* q = nullptr;
  aclTensor* k = nullptr;
  aclTensor* v = nullptr;
  aclTensor* pse = nullptr;
  aclTensor* dropMask = nullptr;
  aclTensor* padding = nullptr;
  aclTensor* attenmask = nullptr;
  aclTensor* queryRope = nullptr;
  aclTensor* keyRope = nullptr;
  aclTensor* dScaleQ = nullptr;
  aclTensor* dScaleK = nullptr;
  aclTensor* dScaleV = nullptr;
  aclTensor* sink = nullptr;
  aclTensor* attentionOut = nullptr;
  aclTensor* softmaxMax = nullptr;
  aclTensor* softmaxSum = nullptr;
  aclTensor* softmaxOut = nullptr;

  std::vector<op::fp16_t> qHostData(q_size, 1.0);
  std::vector<op::fp16_t> kHostData(kv_size, 1.0);
  std::vector<op::fp16_t> vHostData(kv_size, 1.0);
  std::vector<uint8_t> attenmaskHostData(atten_mask_size, 0);
  std::vector<float> sinkHostData(sink_size, 3.0);
  std::vector<op::fp16_t> attentionOutHostData(q_size, 255.0);
  std::vector<float> softmaxMaxHostData(softmax_size, 3.0);
  std::vector<float> softmaxSumHostData(softmax_size, 3.0);

  ret = CreateAclTensor(qHostData, qShape, &qDeviceAddr, aclDataType::ACL_FLOAT16, &q);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(kHostData, kShape, &kDeviceAddr, aclDataType::ACL_FLOAT16, &k);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(vHostData, vShape, &vDeviceAddr, aclDataType::ACL_FLOAT16, &v);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(attenmaskHostData, attenmaskShape, &attenmaskDeviceAddr, aclDataType::ACL_UINT8, &attenmask);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(sinkHostData, sinkShape, &sinkDeviceAddr, aclDataType::ACL_FLOAT, &sink);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(attentionOutHostData, attentionOutShape, &attentionOutDeviceAddr, aclDataType::ACL_FLOAT16, &attentionOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(softmaxMaxHostData, softmaxMaxShape, &softmaxMaxDeviceAddr, aclDataType::ACL_FLOAT, &softmaxMax);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(softmaxSumHostData, softmaxSumShape, &softmaxSumDeviceAddr, aclDataType::ACL_FLOAT, &softmaxSum);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  std::vector<int64_t> prefixOp = {0};
  std::vector<int64_t> qStartIdxOp = {0};
  std::vector<int64_t> kvStartIdxOp = {0};
  std::vector<int64_t> actualSeqQLenOp = {128};
  std::vector<int64_t> actualSeqKVLenOp = {128};

  aclIntArray *prefix = aclCreateIntArray(prefixOp.data(), 1);
  aclIntArray *qStartIdx = aclCreateIntArray(qStartIdxOp.data(), 1);
  aclIntArray *kvStartIdx = aclCreateIntArray(kvStartIdxOp.data(), 1);
  aclIntArray* actualSeqQLen = aclCreateIntArray(actualSeqQLenOp.data(), 1);
  aclIntArray* actualSeqKVLen = aclCreateIntArray(actualSeqKVLenOp.data(), 1);

  double scaleValue = 0.088388;
  double keepProb = 1;
  int64_t preTokens = 65536;
  int64_t nextTokens = 65536;
  int64_t headNum = 1;
  int64_t innerPrecise = 0;
  int64_t sparseMode = 0;
  int64_t outDtype = 0;
  int64_t pseType = 1;
  int64_t seed = 0;
  int64_t offset = 0;
  char layOut[5] = {'S', 'B', 'H', 0};
  char *softmaxOutLayout = "same_as_input";

  // 3. Call the CANN operator library API. Replace the API name with the actual one.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  // Call the first-phase API of aclnnFlashAttentionScoreV4.
    ret = aclnnFlashAttentionScoreV4GetWorkspaceSize(
            q, k, v, pse, dropMask, padding, attenmask, queryRope, keyRope, dScaleQ, dScaleK, dScaleV, sink, prefix,
            actualSeqQLen, actualSeqKVLen, qStartIdx, kvStartIdx, scaleValue, keepProb, preTokens, nextTokens,
            headNum, layOut, innerPrecise, sparseMode, outDtype, pseType, softmaxOutLayout, seed, offset, softmaxMax, softmaxSum,
            softmaxOut, attentionOut, &workspaceSize, &executor);

  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFlashAttentionScoreV4GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  // Allocate device memory based on the workspaceSize calculated by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }

  // Call the second-phase API of aclnnFlashAttentionScoreV4.
  ret = aclnnFlashAttentionScoreV4(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFlashAttentionScoreV4 failed. ERROR: %d\n", ret); return ret);

  // 4. (boilerplate) Synchronously wait for the task execution to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value. Copy the result from the device to the host. Modify according to the actual API definition.
  PrintOutResultFp16(attentionOutShape, &attentionOutDeviceAddr);
  PrintOutResult(softmaxMaxShape, &softmaxMaxDeviceAddr);
  PrintOutResult(softmaxSumShape, &softmaxSumDeviceAddr);

  // 6. Release aclTensor and aclScalar. Modify according to the actual API definition.
  aclDestroyTensor(q);
  aclDestroyTensor(k);
  aclDestroyTensor(v);
  aclDestroyTensor(attenmask);
  aclDestroyTensor(sink);
  aclDestroyTensor(attentionOut);
  aclDestroyTensor(softmaxMax);
  aclDestroyTensor(softmaxSum);

  // 7. Release device resources.
  aclrtFree(qDeviceAddr);
  aclrtFree(kDeviceAddr);
  aclrtFree(vDeviceAddr);
  aclrtFree(attenmaskDeviceAddr);
  aclrtFree(sinkDeviceAddr);
  aclrtFree(attentionOutDeviceAddr);
  aclrtFree(softmaxMaxDeviceAddr);
  aclrtFree(softmaxSumDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtDestroyContext(context);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}
```
