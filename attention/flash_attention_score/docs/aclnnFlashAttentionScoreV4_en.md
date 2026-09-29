# aclnnFlashAttentionScoreV4

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

- Uses the FlashAttention algorithm to perform self-attention computation in training scenarios. The query, key, and value parameters of this API support multiple sequences with the same length or different lengths.
  - Compared with the [FlashAttentionScoreV3](./aclnnFlashAttentionScoreV2.md) API, this API has the following differences:
    - Adjusting the Dropout function: When keepProb is less than 1.0, if no external DropoutMask is passed, the new parameters seed and offset are used to generate the DropoutMask. If an external DropoutMask is passed, the external DropoutMask is used.
  - Compared with the [FlashAttentionVarLenScoreV5](./aclnnFlashAttentionVarLenScoreV5.md) API, this API has the following differences:
    - Adjusting the Dropout function: When keepProb is less than 1.0, if no external DropoutMask is passed, the new parameters seed and offset are used to generate the DropoutMask. If an external DropoutMask is passed, the external DropoutMask is used.

- Formulas:

  The forward propagation formula for attention is as follows:

  - When psetype is set to 1, the calculation formula is the same as that of [FlashAttentionScore](./aclnnFlashAttentionScore.md).

  - When `psetype` is set to other values, the formula is as follows:

    $$
    attention\_out=Dropout(Softmax(Mask(scale*(query*key^T) + pse),atten\_mask),keep\_prob)*value
    $$

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnFlashAttentionScoreV4GetWorkspaceSize` is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, `aclnnFlashAttentionScoreV4` is called to perform computation.

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
        <th>Name</th>
        <th>Input/Output</th>
        <th>Description</th>
        <th>Usage</th>
        <th>Data Type</th>
        <th>Data Format</th>
        <th>Dimension (Shape)</th>
        <th>Non-contiguous Tensor</th>
      </tr></thead>
    <tbody>
      <tr>
        <td>query</td>
        <td>Input</td>
        <td><code>query</code> in the formulas.</td>
        <td>The data type must be the same as that of key and value.</td>
        <td>FLOAT16, BFLOAT16, FLOAT32</td>
        <td>ND</td>
        <td>0, 3, 4</td>
        <td>√</td>
      </tr>
      <tr>
        <td>key</td>
        <td>Input</td>
        <td><code>key</code> in the formulas.</td>
        <td>The data type must be the same as that of query and value.</td>
        <td>FLOAT16, BFLOAT16, FLOAT32</td>
        <td>ND</td>
        <td>0, 3, 4</td>
        <td>√</td>
      </tr>
      <tr>
        <td>value</td>
        <td>Input</td>
        <td><code>value</code> in the formula.</td>
        <td>The data type must be the same as that of query and key.</td>
        <td>FLOAT16, BFLOAT16, FLOAT32</td>
        <td>ND</td>
        <td>0, 3, 4</td>
        <td>√</td>
      </tr>
      <tr>
        <td>realShiftOptional</td>
        <td>Optional input</td>
        <td>pse in the formula.</td>
        <td>The data type is the same as that of attentionOut.</td>
        <td>FLOAT16, BFLOAT16, FLOAT32</td>
        <td>ND</td>
        <td>[B,N,S,S], [B,N,1,Skv], [1,N,S,S]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>paddingMaskOptional</td>
        <td>Input</td>
        <td>Reserved.</td>
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
        <td><code>atten_mask</code> in the formula.</td>
        <td>A value of 1 indicates that the position does not participate in the calculation, while a value of 0 indicates that it does.</td>
        <td>BOOL or UINT8</td>
        <td>ND</td>
        <td>[B,N,S,S], [B,1,S,S], [1,1,S,S], [S,S]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>queryRopeOptional</td>
        <td>Input</td>
        <td>aclTensor on the device.</td>
        <td>The data type must be the same as that of query.</td>
        <td>FLOAT16, BFLOAT16</td>
        <td>ND</td>
        <td>0, 3, 4</td>
        <td>√</td>
      </tr>
      <tr>
        <td>keyRopeOptional</td>
        <td>Input</td>
        <td>aclTensor on the device.</td>
        <td>The data type is the same as that of the key.</td>
        <td>FLOAT16, BFLOAT16</td>
        <td>ND</td>
        <td>0, 3, 4</td>
        <td>√</td>
      </tr>
      <tr>
        <td>dScaleQOptional</td>
        <td>Input</td>
        <td>Quantization parameter of the query.</td>
        <td>The input shape is [B, N1, Ceil(Sq/128), 1].</td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>0, 3, 4</td>
        <td>√</td>
      </tr>
      <tr>
        <td>dScaleKOptional</td>
        <td>Input</td>
        <td>Quantization parameter of the key.</td>
        <td>The input shape is [B, N2, Ceil(Skv/256), 1].</td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>0, 3, 4</td>
        <td>√</td>
      </tr>
      <tr>
        <td>dScaleVOptional</td>
        <td>Input</td>
        <td>Quantization parameter of the value.</td>
        <td>The input shape is [B, N2, Ceil(Skv/256), 1].</td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>0, 3, 4</td>
        <td>√</td>
      </tr>
      <tr>
        <td>sinkOptional</td>
        <td>Input</td>
        <td>Reserved parameter, which is not used currently.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>prefixOptional</td>
        <td>Input</td>
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
        <td>Sequence length of query corresponding to each batch.</td>
        <td>-</td>
        <td>INT64</td>
        <td>ND</td>
        <td>0, 1</td>
        <td>-</td>
      </tr>
      <tr>
        <td>actualSeqKvLenOptional</td>
        <td>Input</td>
        <td>Sequence length of key/value corresponding to each batch.</td>
        <td>-</td>
        <td>INT64</td>
        <td>ND</td>
        <td>0, 1</td>
        <td>-</td>
      </tr>
      <tr>
        <td>qStartIdxOptional</td>
        <td>Input</td>
        <td>Global start index of the query sequence for the current chunk in an outer splitting scenario.</td>
        <td>-</td>
        <td>INT64</td>
        <td>ND</td>
        <td>0, 1</td>
        <td>-</td>
      </tr>
      <tr>
        <td>kvStartIdxOptional</td>
        <td>Input</td>
        <td>Global start index of the query sequence for the current chunk in an outer splitting scenario.</td>
        <td>-</td>
        <td>INT64</td>
        <td>ND</td>
        <td>0, 1</td>
        <td>-</td>
      </tr>
      <tr>
        <td>scaleValue</td>
        <td>Input</td>
        <td><code>scale</code> in the formula, indicating the scaling coefficient.</td>
        <td>-</td>
        <td>DOUBLE</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>keepProb</td>
        <td>Input</td>
        <td>Proportion of 1s in dropMaskOptional.</td>
        <td>The value range is (0, 1].</td>
        <td>DOUBLE</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>preTokens</td>
        <td>Input</td>
        <td>Left boundary of the sliding window, used for sparse computation.</td>
        <td>-</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>nextTokens</td>
        <td>Input</td>
        <td>Right boundary of the sliding window, used for sparse computation.</td>
        <td>-</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>headNum</td>
        <td>Input</td>
        <td>Number of heads on a single device, that is, the length of the N axis of the input query.</td>
        <td>-</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>inputLayout</td>
        <td>Input</td>
        <td>Layout of the input <code>query</code>, <code>key</code>, and <code>value</code>.</td>
        <td>BSH, SBH, BSND, BNSD, and TND are supported.</td>
        <td>String</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>innerPrecise</td>
        <td>Input</td>
        <td>int64_t on the host.</td>
        <td>Setting this parameter to 2 enables the processing of invalid rows. You are advised not to set this parameter unless necessary.</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>sparseMode</td>
        <td>Input</td>
        <td>Sparse mode.</td>
        <td>The value can be 0, 1, 2, 3, 4, 5, or 6.</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>outDtype</td>
        <td>Input</td>
        <td>int64_t on the host.</td>
        <td>This parameter is valid only in quantization scenarios. The value 0 indicates that the output of attenout is in fp16 format, and the value 1 indicates that the output is in bf16 format.</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>softmaxOutLayout</td>
        <td>Input</td>
        <td>Reserved parameter, which is not used currently.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>pseType</td>
        <td>Input</td>
        <td>int64_t on the host.</td>
        <td>Calculation sequence of multiplication and addition. The value can be 0, 1, 2, or 3.</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>seed</td>
        <td>Input</td>
        <td>int64_t on the host, which is the seed for generating dropmask.</td>
        <td>-</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>offset</td>
        <td>Input</td>
        <td>Offset for generating the dropmask.</td>
        <td>-</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>softmaxMaxOut</td>
        <td>Output</td>
        <td>Intermediate result of the Max operation in softmax, used for backward computation.</td>
        <td>The output shape is [B, N, Sq, 8].</td>
        <td>FLOAT</td>
        <td>ND</td>
        <td>0, 4</td>
        <td>√</td>
      </tr>
      <tr>
        <td>softmaxSumOut</td>
        <td>Output</td>
        <td>Intermediate result of the Sum operation in softmax, used for backward computation.</td>
        <td>The output shape is [B, N, Sq, 8].</td>
        <td>FLOAT</td>
        <td>ND</td>
        <td>0, 4</td>
        <td>√</td>
      </tr>
      <tr>
        <td>attentionOutOut</td>
        <td>Output</td>
        <td>Final output of the formula.</td>
        <td>The data type and shape must be the same as those of query.</td>
        <td>FLOAT16, BFLOAT16, FLOAT32</td>
        <td>ND</td>
        <td>0, 3, 4</td>
        <td>√</td>
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
    </tbody>
  </table>

- **Returns**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API implements input parameter validation. The following error codes may be returned.

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
      <td>The required input, output, or attribute is passed as a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="2">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="2">161002</td>
      <td>The data type of query, key, value, realShiftOptional, dropMaskOptional, paddingMaskOptional, attenMaskOptional, softmaxMaxOut, softmaxSumOut, softmaxOutOut, or attentionOutOut is not supported.</td>
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
      <td>Size of the workspace to be allocated on the device, which is obtained by calling the first-phase API aclnnFlashAttentionScoreV2GetWorkspaceSize.</td>
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
  - The default deterministic implementation of aclnnFlashAttentionScoreV4.
- When this API is used together with PyTorch, ensure that the CANN package versions match the PyTorch package versions.
- Input query, key, and value
  - **B**: The batch sizes must be equal.
  - **D**: Head-Dim must satisfy (qD == kD && kD >= vD).
  - `inputLayout` must be consistent.
- The input shape of queryRopeOptional is the same as that of query in all dimensions except the D dimension.
- The input shape of keyRopeOptional is the same as that of key in all dimensions except the D dimension.
- The restrictions on the data shape are as follows, using TND, BSND, and BNSD of inputLayout as examples (H = N x D in BSH and SBH):
    - T(B x S): The value ranges from 1 to 1M. In TND format, the maximum length supported by actualSeqQLenOptional is 20,000.
    - `B`: The value ranges from 1 to 2M. When `prefixOptional` is passed, **B** supports a maximum of 2K.
    - **N**: The value ranges from 1 to 256.
    - **S**: The value ranges from 1 to 1M.
    - **D**: The value ranges from 1 to 768.
- The data layout of **query**, **key**, and **value** can be interpreted from multiple dimensions. To be specific, **B (Batch)** indicates the size of an input sample batch, **S (Seq-Length)** indicates the length of the input sample sequence, **H (Head-Size)** indicates the size of the hidden layer, **N (Head-Num)** indicates the number of heads, and **D (Head-Dim)** indicates the minimum unit size of the hidden layer (D = H/N).
- `innerPrecise`: 0 and 1 are reserved, and 2 indicates that invalid row calculation is enabled. This function is used to prevent precision loss caused by the mask of the entire row during calculation. However, this configuration deteriorates the performance. If the operator can determine that invalid rows exist, the invalid row computation is automatically enabled, such as in scenarios where `sparseMode` is set to 3 and Sq is greater than Skv.
- Meanings of `pseType` values:

    | pseType     | Meaning                             |      Remarks  |
    | ----------- | --------------------------------- | ----------|
    | 0           | A value is externally passed to `pse`, and multiplication is required before addition.             | - |
    | 1           | A value is externally passed to `pse`, and addition is required before multiplication.             | The implementation is the same as that of [FlashAttentionScore](./aclnnFlashAttentionScore.md).|
    | 2           | A value is internally passed to `pse`, and multiplication is required before addition.             | - |
    | 3           | A value is internally passed to `pse`, and multiplication and addition are required before square root operation.        | - |

- When `pseType` is set to 2 or 3, Sq and Skv must be of the same length.
- The constraints on sparseMode are as follows:
  - When the shapes of all `attenMaskOptional` are less than 2048 and are the same, the default mode is recommended to reduce memory usage.
  - When the value is set to 1, 2, 3, or 5, the user-configured `preTokens` and `nextTokens` do not take effect.
  - When the value is set to 0 or 4, ensure that the ranges of `attenMaskOptional`, `preTokens`, and `nextTokens` are consistent.
  - If no specific value is required, 0 is recommended.
  - For details about the sparse modes, see [Sparse Mode Description](../../../docs/en/context/sparse_mode_introduction.md).
- In some scenarios, if the computation load is too large, the operator execution may time out (an AI Core error is reported, and `errorStr` is `timeout or trap error`). In this case, you are advised to perform axis splitting. Note: The computation load is affected by parameters such as **B**, **S**, **N**, and **D**. Larger values indicate larger computation loads.
- In the band scenario, the values of `preTokens` and `nextTokens` must overlap.
- In the prefixOptional sparse computing scenario, sparseMode is set to 5 or 6 when the sequence lengths are the same, and sparseMode is set to 6 when the sequence lengths are different. In the two scenarios, when Sq > Skv, the value range of N for prefix is [0, Skv]. When Sq <= Skv, the value range of N for prefix is [Skv-Sq, Skv]. If sparseModeOptional is set to 5 and prefix N > Skv or prefixOptional is not passed, full computation is performed. If sparseModeOptional is set to 6, prefixOptional must be passed.
- If Sq of `realShiftOptional` is greater than 1024, if BNHS and 1NHS are configured, Sq and Skv must have the same length.
- The `actualSeqQLenOptional` input supports the S length of 0 in a batch. In this case, the `realShiftOptional` input is not supported.
- The `attenMaskOptional` input does not support padding. That is, `attenMaskOptional` cannot contain a row of all 1s.
- The **S** length of a batch in `actualSeqQLenOptional` can be 0. If the **S** length is 0, the `pse` input is not supported.
  If the actual S length is \[2,2,0,2,2\], the value of `actualSeqQLenOptional` is \[2,4,4,6,8\].
- Ascend 950PR/Ascend 950DT:
    - seed and offset take effect only when keepProb is less than 1.0. Otherwise, they do not take effect.
    - When keepProb is less than 1.0, if dropMaskOptional is not nullptr, the input dropMask is used. Otherwise, the dropMask generated by seed and offset is used.
- In TND format, some batches at the end do not participate in computation. In this case, you can pass 0s to the end of actual_seq_q_len and actual_seq_kv_len. For example, if the actual S length is [2, 3, 4, 5, 6] and the last two batches are not required to participate in computation, the input actual_seq_q_len is [2, 3, 4, 0, 0]. In this case, if prefixOptional needs to be passed, the same number of 0s must be passed to the end of prefixOptional, for example, [1, 1, 1, 0, 0].

## Examples

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```C++
#include <iostream>
#include <vector>
#include <cmath>
#include "acl/acl.h"
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
  // 1. (Fixed writing) Initialize the device, context, and stream. For details, see the list of external AscendCL APIs.
  // Set the device ID in use.
  int32_t deviceId = 0;
  aclrtContext context;
  aclrtStream stream;
  auto ret = Init(deviceId, &context, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the inputs and outputs based on the API definition.
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

  std::vector<int64_t> qShape = {S1, B, H1};
  std::vector<int64_t> kShape = {S2, B, H2};
  std::vector<int64_t> vShape = {S2, B, H2};
  std::vector<int64_t> attenmaskShape = {S1, S2};

  std::vector<int64_t> attentionOutShape = {S1, B, H1};
  std::vector<int64_t> softmaxMaxShape = {B, N1, S1, 8};
  std::vector<int64_t> softmaxSumShape = {B, N1, S1, 8};

  void* qDeviceAddr = nullptr;
  void* kDeviceAddr = nullptr;
  void* vDeviceAddr = nullptr;
  void* attenmaskDeviceAddr = nullptr;
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
 
  std::vector<float> qHostData(q_size, 1.0);
  std::vector<float> kHostData(kv_size, 1.0);
  std::vector<float> vHostData(kv_size, 1.0);
  std::vector<uint8_t> attenmaskHostData(atten_mask_size, 0);
  std::vector<float> attentionOutHostData(q_size, 255);
  std::vector<float> softmaxMaxHostData(softmax_size, 3.0);
  std::vector<float> softmaxSumHostData(softmax_size, 3.0);

  ret = CreateAclTensor(qHostData, qShape, &qDeviceAddr, aclDataType::ACL_FLOAT, &q);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(kHostData, kShape, &kDeviceAddr, aclDataType::ACL_FLOAT, &k);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(vHostData, vShape, &vDeviceAddr, aclDataType::ACL_FLOAT, &v);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(attenmaskHostData, attenmaskShape, &attenmaskDeviceAddr, aclDataType::ACL_UINT8, &attenmask);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(attentionOutHostData, attentionOutShape, &attentionOutDeviceAddr, aclDataType::ACL_FLOAT, &attentionOut);
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
  char *softmaxLayout = nullptr;

  // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  // Call the first-phase API of aclnnFlashAttentionScoreV4.
    ret = aclnnFlashAttentionScoreV4GetWorkspaceSize(
            q, k, v, pse, dropMask, padding, attenmask, queryRope, keyRope, dScaleQ, dScaleK, dScaleV, sink, prefix,
            actualSeqQLen, actualSeqKVLen, qStartIdx, kvStartIdx, scaleValue, keepProb, preTokens, nextTokens,
            headNum, layOut, innerPrecise, sparseMode, outDtype, pseType, softmaxLayout, seed, offset, softmaxMax, softmaxSum,
            softmaxOut, attentionOut, &workspaceSize, &executor);

  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFlashAttentionScoreV4GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  // Allocate device memory based on workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }

  // Call the second-phase API of aclnnFlashAttentionScoreV4.
  ret = aclnnFlashAttentionScoreV4(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFlashAttentionScoreV4 failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device memory to the host. Modify the code based on the API definition.
  PrintOutResult(attentionOutShape, &attentionOutDeviceAddr);
  PrintOutResult(softmaxMaxShape, &softmaxMaxDeviceAddr);
  PrintOutResult(softmaxSumShape, &softmaxSumDeviceAddr);

  // 6. Release aclTensor and aclScalar. Modify the code based on the API definition.
  aclDestroyTensor(q);
  aclDestroyTensor(k);
  aclDestroyTensor(v);
  aclDestroyTensor(attenmask);
  aclDestroyTensor(attentionOut);
  aclDestroyTensor(softmaxMax);
  aclDestroyTensor(softmaxSum);

  // 7. Free device resources.
  aclrtFree(qDeviceAddr);
  aclrtFree(kDeviceAddr);
  aclrtFree(vDeviceAddr);
  aclrtFree(attenmaskDeviceAddr);
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
