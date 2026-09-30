# aclnnFlashAttentionUnpaddingScoreGradV5

## Supported Products

|Product     | Supported|
|:----------------------------|:-----------:|
|<term>Atlas A3 training products/Atlas A3 inference products</term>|      √     |
|<term>Atlas A2 training products</term>|      √     |
|<term>Atlas A2 inference products</term>|      ×     |
|<term>Atlas 200I/500 A2 inference products</term>|      ×     |
|<term>Atlas inference products</term>|      ×     |
|<term>Atlas training products</term>|      ×     |

## Description

- **API function**: Uses the FlashAttention algorithm to perform self-attention computation in training scenarios. `sinkInOptional` is added as an optional parameter.

- Formula:

  The forward propagation formula for attention is as follows:

  $$
  =Dropout(Softmax(Mask(\frac{QK^T+pse}{\sqrt{d}}),atten\_mask),keep\_prob)V
  $$

  For convenience, the formula can be represented using variables $S$ and $P$:

  $$
  =Mask(\frac{QK^T+pse}{\sqrt{d}}),atten\_mask
  $$

  $$
  =Dropout(Softmax(S),keep\_prob)
  $$

  $$
  =PV
  $$

  Then the backward propagation formula for attention is as follows:

  $$
  V=P^TdY
  $$

  $$
  Q=\frac{((dS)*K)}{\sqrt{d}}
  $$

  $$
  K=\frac{((dS)^T*Q)}{\sqrt{d}}
  $$

The following calculation logic applies after sink is added, primarily modifying the calculation parts related to softmax_max and softmax_sum:

$$
S = Q @ K^{T}
$$

$$
m = max(sink, max(S))
$$

$$
Attention = \frac{e^{S - m} @ V}{\sum e^{S-m} + S^{sink - m}}
$$

$$
dSink = reduce(-P * dP * SimpleSoftmax(sink, x\_max, x\_sum))
$$

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md)calls. First, `aclnnFlashAttentionUnpaddingScoreGradV5GetWorkspaceSize` is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, `aclnnFlashAttentionUnpaddingScoreGradV5` is called to perform computation.

```c++
aclnnStatus aclnnFlashAttentionUnpaddingScoreGradV5GetWorkspaceSize(
    const aclTensor *query, 
    const aclTensor *queryRope, 
    const aclTensor *keyIn, 
    const aclTensor *keyInRope, 
    const aclTensor *value, 
    const aclTensor *dy, 
    const aclTensor *pseShiftOptional, 
    const aclTensor *dropMaskOptional, 
    const aclTensor *paddingMaskOptional, 
    const aclTensor *attenMaskOptional, 
    const aclTensor *softmaxMaxOptional, 
    const aclTensor *softmaxSumOptional, 
    const aclTensor *softmaxInOptional, 
    const aclTensor *attentionInOptional, 
    const aclTensor *sinkInOptional, 
    const aclIntArray *prefixOptional, 
    const aclIntArray *actualSeqQLenOptional, 
    const aclIntArray *actualSeqKvLenOptional, 
    const aclIntArray *qStartIdxOptional, 
    const aclIntArray *kvStartIdxOptional, 
    double scaleValue, 
    double keepProb, 
    int64_t preTokens, 
    int64_t nextTokens, 
    int64_t headNum, 
    char *inputLayout, 
    int64_t innerPrecise, 
    int64_t sparseMode, 
    int64_t pseType, 
    char *softmaxInLayout, 
    const aclTensor *dqOut, 
    const aclTensor *dqRopeOut, 
    const aclTensor *dkOut, 
    const aclTensor *dkRopeOut, 
    const aclTensor *dvOut, 
    const aclTensor *dpseOut, 
    const aclTensor *dsinkOut, 
    uint64_t *workspaceSize, 
    aclOpExecutor **executor);
```

```c++
aclnnStatus aclnnFlashAttentionUnpaddingScoreGradV5(
  void             *workspace,
  uint64_t          workspaceSize,
  aclOpExecutor    *executor,
  const aclrtStream stream)
```

## aclnnFlashAttentionUnpaddingScoreGradV5GetWorkspaceSize

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1529px"><colgroup>
    <col style="width: 198px">
    <col style="width: 120px">
    <col style="width: 289px">
    <col style="width: 302px">
    <col style="width: 238px">
    <col style="width: 106px">
    <col style="width: 130px">
    <col style="width: 146px">
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
        <td>Q in the formula.</td>
        <td>The data type must be the same as that of keyIn or value.</td>
        <td>FLOAT16, BFLOAT16</td>
        <td>ND</td>
        <td>[TND]</td>
        <td>√</td>
      </tr>
    <tr>
        <td>queryRope</td>
        <td>Input</td>
        <td>Rotary positional encoding (RoPE) part of Q.</td>
        <td>The data type must be the same as that of query.</td>
        <td>BFLOAT16</td>
        <td>ND</td>
        <td>[TND]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>keyIn</td>
        <td>Input</td>
        <td>K in the formula.</td>
        <td>The data type must be the same as that of query or value.</td>
        <td>FLOAT16, BFLOAT16</td>
        <td>ND</td>
        <td>[TND]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>keyInRope</td>
        <td>Input</td>
        <td>RoPE part of K.</td>
        <td>The data type must be the same as that of keyIn.</td>
        <td>BFLOAT16</td>
        <td>ND</td>
        <td>[TND]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>value</td>
        <td>Input</td>
        <td>V in the formula.</td>
        <td>The data type must be the same as that of query or keyIn.</td>
        <td>FLOAT16, BFLOAT16</td>
        <td>ND</td>
        <td>[TND]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>dy</td>
        <td>Input</td>
        <td>dY in the formula.</td>
        <td>-</td>
        <td>FLOAT16, BFLOAT16</td>
        <td>ND</td>
        <td>[TND]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>pseShiftOptional</td>
        <td>Optional input</td>
        <td>pse in the formula.</td>
        <td>The data type must match query. Use this parameter with pseType.</td>
        <td>FLOAT16, BFLOAT16</td>
        <td>ND</td>
        <td>[B,N,1024,Skv], [1,N,1024,Skv], [B,N], [N]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>dropMaskOptional</td>
        <td>Optional input</td>
        <td>Dropout mask.</td>
        <td>This parameter is incompatible with RoPE. Pass a null pointer if you do not intend to use it.</td>
        <td>UINT8</td>
        <td>ND</td>
        <td>0, 1</td>
        <td>√</td>
      </tr>
      <tr>
        <td>paddingMaskOptional</td>
        <td>Input</td>
        <td>Reserved parameter.</td>
        <td>A null pointer must be passed to this parameter when the API is called.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>attenMaskOptional</td>
        <td>Input</td>
        <td>atten_mask in the formula.</td>
        <td>A value of 1 indicates that the position does not participate in the calculation, while a value of 0 indicates that it does.</td>
        <td>BOOL, UINT8</td>
        <td>ND</td>
        <td>[B,N,Sq,Skv], [B,1,Sq,Skv], [1,1,Sq,Skv], [Sq,Skv] </td>
        <td>√</td>
      </tr>
      <tr>
        <td>softmaxMaxOptional</td>
        <td>Optional input</td>
        <td>Intermediate output of the Softmax forward propagation.</td>
        <td>-</td>
        <td>FLOAT</td>
        <td>ND</td>
        <td>[TN8], [NT8]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>softmaxSumOptional</td>
        <td>Optional input</td>
        <td>Intermediate output of the Softmax forward propagation.</td>
        <td>-</td>
        <td>FLOAT</td>
        <td>ND</td>
        <td>[TN8], [NT8]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>softmaxInOptional</td>
        <td>Input</td>
        <td>Intermediate output of the Softmax forward propagation.</td>
        <td>Reserved.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>attentionInOptional</td>
        <td>Input</td>
        <td>Forward attention output.</td>
        <td>The data type and shape must be the same as those of query.</td>
        <td>FLOAT16, BFLOAT16, FLOAT32</td>
        <td>ND</td>
        <td>[TND]</td>
        <td>√</td>
      </tr>
    <tr>
        <td>sinkInOptional</td>
        <td>Optional input</td>
        <td>sink in the formula.</td>
        <td>The length is headNum.</td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>[headNum]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>prefixOptional</td>
        <td>Optional input</td>
        <td>N of each batch in the prefix sparse computation scenario.</td>
        <td>If this parameter is not used, a null pointer can be passed.</td>
        <td>INT64</td>
        <td>ND</td>
        <td>0, 1</td>
        <td>-</td>
      </tr>
      <tr>
        <td>actualSeqQLenOptional</td>
        <td>Input</td>
        <td>Actual query sequence length.</td>
        <td>-</td>
        <td>INT64</td>
        <td>ND</td>
        <td>1</td>
        <td>-</td>
      </tr>
      <tr>
        <td>actualSeqKvLenOptional</td>
        <td>Input</td>
        <td>Actual key/value sequence length.</td>
        <td>-</td>
        <td>INT64</td>
        <td>ND</td>
        <td>1</td>
        <td>-</td>
      </tr>
      <tr>
        <td>scaleValue</td>
        <td>Input</td>
        <td>Scale factor.</td>
        <td>Generally, set this parameter to D^-0.5.</td>
        <td>DOUBLE</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>keepProb</td>
        <td>Input</td>
        <td>Ratio of 1s in dropMaskOptional.</td>
        <td>Generally, set this parameter to 1.0.</td>
        <td>DOUBLE</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>preTokens</td>
        <td>Input</td>
        <td>Left boundary of the sliding window for sparse computation.</td>
        <td>If no specific value is required, 2147483647 is recommended.</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>nextTokens</td>
        <td>Input</td>
        <td>Right boundary of the sliding window for sparse computation.</td>
        <td>If no specific value is required, 2147483647 is recommended.</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>headNum</td>
        <td>Input</td>
        <td>Number of heads on a single rank, that is, the length of the N axis of query.</td>
        <td>See the constraint description.</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>inputLayout</td>
        <td>Input</td>
        <td>Data layout of input Q/K/V.</td>
        <td>TND is supported.</td>
        <td>String</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>innerPrecise</td>
        <td>Input</td>
        <td>Internal calculation precision control.</td>
        <td>Reserved.</td>
        <td>INT32</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>sparseMode</td>
        <td>Input</td>
        <td>Sparse mode.</td>
        <td>The value ranges from 0 to 8, excluding 5. When RoPE inputs are provided, 6 is not supported.</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>pseType</td>
        <td>Input</td>
        <td>pse type.</td>
        <td>The value ranges from 0 to 3.</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>softmaxInLayout</td>
        <td>Input</td>
        <td>Controls the actual data layout of softmaxMax and softmaxSum.</td>
        <td>Pass "same_as_input" for TND and "" for NTD.</td>
        <td>String</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>dqOut</td>
        <td>Output</td>
        <td>dQ in the formula, indicating the gradient of query.</td>
        <td>-</td>
        <td>FLOAT16, BFLOAT16</td>
        <td>ND</td>
        <td>[TND]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>dkOut</td>
        <td>Output</td>
        <td>dK in the formula, indicating the gradient of key.</td>
        <td>-</td>
        <td>FLOAT16, BFLOAT16</td>
        <td>ND</td>
        <td>[TND]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>dvOut</td>
        <td>Output</td>
        <td>dV in the formula, indicating the gradient of value.</td>
        <td>-</td>
        <td>FLOAT16, BFLOAT16</td>
        <td>ND</td>
        <td>[TND]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>dpseOut</td>
        <td>Output</td>
        <td>d(pse) gradient.</td>
        <td>Reserved.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>dsinkOut</td>
        <td>Output</td>
        <td>dSink in the formula, indicating the d(sinkInOptional) gradient.</td>
        <td>-</td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>[headNum]</td>
        <td>√</td>
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

  aclnnStatus: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API implements input parameter verification. The following errors may be thrown.

  <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
  <col style="width: 286px">
  <col style="width: 118px">
  <col style="width: 746px">
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
      <td>The data type of query, keyIn, value, dy, pseShiftOptional, dropMaskOptional, paddingMaskOptional, attenMaskOptional, softmaxMaxOptional, softmaxSumOptional, softmaxInOptional, attentionInOptional, sinkInOptional, dqOut, dkOut, dvOut, dsinkOut, or softmaxInLayout is not supported.</td>
    </tr>
    <tr>
      <td>The data format of query, keyIn, value, dy, pseShiftOptional, dropMaskOptional, paddingMaskOptional, attenMaskOptional, softmaxMaxOptional, softmaxSumOptional, softmaxInOptional, attentionInOptional, sinkInOptional, dqOut, dkOut, dvOut, dsinkOut, or softmaxInLayout is not supported.</td>
    </tr>
  </tbody>
  </table>

## aclnnFlashAttentionUnpaddingScoreGradV5

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
      <td>Size of the workspace to be allocated on the device, obtained by calling the first-phase API aclnnFlashAttentionUnpaddingScoreGradV5GetWorkspaceSize.</td>
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

  aclnnStatus status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computing:
  - `aclnnFlashAttentionUnpaddingScoreGradV5` defaults to a non-deterministic implementation. You can call `aclrtCtxSetSysParamOpt` to enable deterministic computing.
- When this API is used together with PyTorch, ensure that the CANN package versions match the PyTorch package versions.
- **B** (batch size) of the input `query`, `queryRope`, `key`, `keyRope`, `value`, and `dy` must be the same.
- The `inputLayout` of the input `query`, `key`, `value`, and `dy` must be the same.
- `inputLayout` of the input `query`, `queryRope`, `key`, `keyRope`, `value`, and `dy` must be TND.
- The input data types of `query`, `key`, `value`, and `pseShiftOptional` must be the same.
- The shapes of the input `key` and `value` must be the same. If the **D** values of `query`, `key`, and `value` are the same, the shapes of `query` and `dy` must be the same. Note: In the current version, the **D** values of `query`, `key`, and `value` must be the same.
- **N** of the input `query` can be different from **N** of the `key` or `value`, but they must be proportional. That is, Nq/Nkv must be a non-zero integer and the value of Nq ranges from 1 to 256.
- The following uses the `inputLayout` TND as an example to describe the constraints on the shape:

    - **T(B*S)**: The value ranges from 1 to 1M.
    - **B**: The value ranges from 1 to 2K. When **prefixOptional** is carried, B supports a maximum of 1K.
    - **N**: The value ranges from 1 to 256.
    - **S**: The value ranges from 1 to 1M.
    - **D**: The value ranges from 1 to 768.
    - **KeepProb**: The value range is (0, 1].
- The data format of `query`, `key`, and `value` can only be TND. **T** indicates the data closely arranged on the B and S axes (SeqLenQ and SeqLenKV of each batch). **B (Batch)** indicates the batch size of the input sample, and **S (Seq-Length)** indicates the length of the input sample sequence. **H (Head-Size)** indicates the size of the hidden layer, **N (Head-Num)** indicates the number of heads. **D (Head-Dim)** indicates the minimum unit size of the hidden layer (**D** = **H**/**N**).
- `pseShiftOptional`: If Sq is greater than 1024, Sq and Skv of each batch are of equal length, and it is a lower triangular mask scenario with `sparseMode` being 0, 2, or 3, ALiBi positional encoding compression can be enabled. In this case, only the last 1024 rows of the original PSE need to be input for memory optimization, that is, `alibi_compress = ori_pse[:, :, -1024:, :]`. Specifically:
  - If the parameters of each batch are different, the shape is BNHSkv (H=1024).
  - When each batch is the same, the shape is 1NHSkv (H=1024).
  - If `pseType` is 2 or 3, the data type must be FLOAT32, and the supported shapes are [B,N] and [N].
  - If this parameter is not enabled, pass a null pointer to `pseShiftOptional` and 1 to `pseType`.
- Meanings of `pseType` values:

  | pseType | Description| Remarks|
  | ----------- | --------------------------------- | ----------|
  | 0 | A value is externally passed to `pse`, and multiplication is required before addition.| - |
  | 1 | A value is externally passed to `pse`, and addition is required before multiplication.| The implementation is the same as that of [FlashAttentionScoreGrad](./aclnnFlashAttentionScoreGrad_en.md).|
  | 2 | A value is internally passed to `pse`, and multiplication is required before addition.| - |
  | 3 | A value is internally passed to `pse`, and multiplication and addition are required before square root operation.| - |

- The constraints for `sparseMode` are as follows:
  - If the shape values of all `attenMaskOptional` are the same and less than 2048, you are advised to use the default mode to reduce memory usage.
  - When the value is set to 1, 2, 3, or 5, the user-configured `preTokens` and `nextTokens` do not take effect.
  - When the value is set to 0 or 4, ensure that the ranges of `attenMaskOptional`, `preTokens`, and `nextTokens` are consistent.
  - If no specific value is required, you are advised to set it to 0.
  - For details about the sparse modes, see [Sparse Mode Description](../../../docs/en/context/sparse_mode_introduction.md).
  - When the value is set to 7, `realShiftOptional` is not supported.
  - When the value is set to 8, `realShiftOptional` is supported when the q and kv of each sequence have the same length. PSE generation is performed globally. Outer splitting in the q direction is supported. q and kv of each sequence must have the same length before outer splitting, and `actualSeqQLenOptional` is passed after outer splitting.
- For details about different data formats, see [Data Format](../../../docs/en/context/data_format.md).
- In some scenarios, if the computation load is too large, the operator execution may time out (AI Core error, errorStr: timeout or trap error). In this case, you are advised to perform axis splitting. Note: The computation load is affected by parameters such as **B**, **S**, **N**, and **D**. Larger values indicate larger computation loads.
- The `prefixOptional` sparse computing supports only compression scenarios (`sparseMode = 6`). When Sq > Skv, the value range of **N** of `prefix` is \[0, Skv\]. When Sq ≤ Skv, the value range of **N** of `prefix` is \[Skv – Sq, Skv\].
  `[0]` - `actualSeqKvLenOptional[0]` + `qStartIdxOptional` - `kvStartIdxOptional` == 0 (experimental feature)
- The `actualSeqQLenOptional` input supports the S length of 0 in a batch. In this case, the `pseShiftOptional` input is not supported.
- Constraints on the `softmaxMax` and `softmaxSum` parameters: The input format is fixed at \[B, N, S, 8\], except TND format, which is \[T, N, 8\]. Note: T = B x S.
- The value of `headNum` must be the same as the value of **N** in `query`.
- When the data layout of `softmaxSum` and `softmaxMax` is TND, `softmaxInLayout` must be set to `same_as_input`.
- `sinkInOptional` has one dimension. The length must be the same as that of `headnum` of `query`.

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```c++
#include <iostream>
#include <vector>
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

int Init(int32_t deviceId, aclrtContext* context, aclrtStream* stream) {
  // (Fixed writing) Initialize resources.
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
  // 1. (Fixed writing) Initialize the device and stream. For details, see the ACL API manual.
  // Set the device ID in use.
  int32_t deviceId = 0;
  aclrtContext context;
  aclrtStream stream;
  auto ret = Init(deviceId, &context, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the inputs and outputs based on the API definition.
  std::vector<int64_t> qShape = {256, 1, 128};
  std::vector<int64_t> kShape = {256, 1, 128};
  std::vector<int64_t> vShape = {256, 1, 128};
  std::vector<int64_t> dxShape = {256, 1, 128};
  std::vector<int64_t> attenmaskShape = {256, 256};
  std::vector<int64_t> softmaxMaxShape = {256, 1, 8};
  std::vector<int64_t> softmaxSumShape = {256, 1, 8};
  std::vector<int64_t> attentionInShape = {256, 1, 128};
  std::vector<int64_t> sinkInOptionalShape = {1};
  std::vector<int64_t> dqShape = {256, 1, 128};
  std::vector<int64_t> dkShape = {256, 1, 128};
  std::vector<int64_t> dvShape = {256, 1, 128};
  std::vector<int64_t> dsinkShape = {1};  

  void* qDeviceAddr = nullptr;
  void* kDeviceAddr = nullptr;
  void* vDeviceAddr = nullptr;
  void* dxDeviceAddr = nullptr;
  void* attenmaskDeviceAddr = nullptr;
  void* softmaxMaxDeviceAddr = nullptr;
  void* softmaxSumDeviceAddr = nullptr;
  void* attentionInDeviceAddr = nullptr;
  void* sinkInOptionalDeviceAddr = nullptr;

  void* dqDeviceAddr = nullptr;
  void* dkDeviceAddr = nullptr;
  void* dvDeviceAddr = nullptr;
  void* dsinkDeviceAddr = nullptr;

  aclTensor* q = nullptr;
  aclTensor* k = nullptr;
  aclTensor* v = nullptr;
  aclTensor* dx = nullptr;
  aclTensor* pse = nullptr;
  aclTensor* dropMask = nullptr;
  aclTensor* padding = nullptr;
  aclTensor* attenmask = nullptr;
  aclTensor* softmaxMax = nullptr;
  aclTensor* softmaxSum = nullptr;
  aclTensor* softmaxIn = nullptr;
  aclTensor* attentionIn = nullptr;
  aclTensor* sinkInOptional = nullptr;
  aclTensor* dq = nullptr;
  aclTensor* dk = nullptr;
  aclTensor* dv = nullptr;
  aclTensor* dpse = nullptr;
  aclTensor* dsink = nullptr;

  std::vector<float> qHostData(32768, 1);
  std::vector<float> kHostData(32768, 1);
  std::vector<float> vHostData(32768, 1);
  std::vector<float> dxHostData(32768, 1);
  std::vector<uint8_t> attenmaskHostData(65536, 0);
  std::vector<float> softmaxMaxHostData(2048, 3.0);
  std::vector<float> softmaxSumHostData(2048, 3.0);
  std::vector<float> attentionInHostData(32768, 1);
  std::vector<float> sinkInOptionalHostData(1, 0);
  std::vector<float> dqHostData(32768, 0);
  std::vector<float> dkHostData(32768, 0);
  std::vector<float> dvHostData(32768, 0);
  std::vector<float> dsinkHostData(1, 0);
  ret = CreateAclTensor(qHostData, qShape, &qDeviceAddr, aclDataType::ACL_FLOAT16, &q);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(kHostData, kShape, &kDeviceAddr, aclDataType::ACL_FLOAT16, &k);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(vHostData, vShape, &vDeviceAddr, aclDataType::ACL_FLOAT16, &v);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(dxHostData, dxShape, &dxDeviceAddr, aclDataType::ACL_FLOAT16, &dx);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(attenmaskHostData, attenmaskShape, &attenmaskDeviceAddr, aclDataType::ACL_UINT8, &attenmask);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(softmaxMaxHostData, softmaxMaxShape, &softmaxMaxDeviceAddr, aclDataType::ACL_FLOAT, &softmaxMax);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(softmaxSumHostData, softmaxSumShape, &softmaxSumDeviceAddr, aclDataType::ACL_FLOAT, &softmaxSum);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(attentionInHostData, attentionInShape, &attentionInDeviceAddr, aclDataType::ACL_FLOAT16, &attentionIn);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(sinkInOptionalHostData, sinkInOptionalShape, &sinkInOptionalDeviceAddr, aclDataType::ACL_FLOAT, &sinkInOptional);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(dqHostData, dqShape, &dqDeviceAddr, aclDataType::ACL_FLOAT16, &dq);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(dkHostData, dkShape, &dkDeviceAddr, aclDataType::ACL_FLOAT16, &dk);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(dvHostData, dvShape, &dvDeviceAddr, aclDataType::ACL_FLOAT16, &dv);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(dsinkHostData, dsinkShape, &dsinkDeviceAddr, aclDataType::ACL_FLOAT, &dsink);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  std::vector<int64_t> prefixOp = {0};
  aclIntArray* prefix = aclCreateIntArray(prefixOp.data(), 1);
  std::vector<int64_t>  acSeqQLenOp = {256};
  std::vector<int64_t>  acSeqKvLenOp = {256};
  aclIntArray* acSeqQLen = aclCreateIntArray(acSeqQLenOp.data(), acSeqQLenOp.size());
  aclIntArray* acSeqKvLen = aclCreateIntArray(acSeqKvLenOp.data(), acSeqKvLenOp.size());
  std::vector<int64_t> qStartIdxOp = {0};
  std::vector<int64_t> kvStartIdxOp = {0};
  aclIntArray *qStartIdx = aclCreateIntArray(qStartIdxOp.data(), 1);
  aclIntArray *kvStartIdx = aclCreateIntArray(kvStartIdxOp.data(), 1);
  double scaleValue = 0.088388;
  double keepProb = 1;
  int64_t preTokens = 65536;
  int64_t nextTokens = 65536;
  int64_t headNum = 1;
  int64_t innerPrecise = 0;
  int64_t sparseMode = 0;
  int64_t pseType = 1;
  char softmaxInLayoutArr[] = "same_as_input";
  char layOut[5] = {'T', 'N', 'D', 0};

  // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  // Call the first-phase API of aclnnFlashAttentionUnpaddingScoreGradV5.
  ret = aclnnFlashAttentionUnpaddingScoreGradV5GetWorkspaceSize(q, nullptr, k, nullptr, v, dx, pse, dropMask, padding,
              attenmask, softmaxMax, softmaxSum, softmaxIn, attentionIn, sinkInOptional, prefix, acSeqQLen, acSeqKvLen,
              qStartIdx, kvStartIdx, scaleValue, keepProb, preTokens, nextTokens, headNum, layOut, innerPrecise, sparseMode,
              pseType, softmaxInLayoutArr, dq, nullptr, dk, nullptr, dv, dpse, dsink, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFlashAttentionUnpaddingScoreGradV5GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  // Allocate device memory based on the computed workspaceSize.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }

  // Call the second-phase API of aclnnFlashAttentionUnpaddingScoreGradV5.
  ret = aclnnFlashAttentionUnpaddingScoreGradV5(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFlashAttentionUnpaddingScoreGradV5 failed. ERROR: %d\n", ret); return ret);

  // 4. (Fixed writing) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device to the host. Modify the code based on the API definition.
  PrintOutResult(dqShape, &dqDeviceAddr);
  PrintOutResult(dkShape, &dkDeviceAddr);
  PrintOutResult(dvShape, &dvDeviceAddr);
  PrintOutResult(dsinkShape, &dsinkDeviceAddr);

  // 6. Release aclTensor and aclScalar. Modify the configuration based on the API definition.
  aclDestroyTensor(q);
  aclDestroyTensor(k);
  aclDestroyTensor(v);
  aclDestroyTensor(dx);
  aclDestroyTensor(attenmask);
  aclDestroyTensor(softmaxMax);
  aclDestroyTensor(softmaxSum);
  aclDestroyTensor(attentionIn);
  aclDestroyTensor(sinkInOptional);  
  aclDestroyTensor(dq);
  aclDestroyTensor(dk);
  aclDestroyTensor(dv);
  aclDestroyTensor(dsink);
  // 7. Release device resources.
  aclrtFree(qDeviceAddr);
  aclrtFree(kDeviceAddr);
  aclrtFree(vDeviceAddr);
  aclrtFree(dxDeviceAddr);
  aclrtFree(attenmaskDeviceAddr);
  aclrtFree(softmaxMaxDeviceAddr);
  aclrtFree(softmaxSumDeviceAddr);
  aclrtFree(attentionInDeviceAddr);
  aclrtFree(sinkInOptionalDeviceAddr);
  aclrtFree(dqDeviceAddr);
  aclrtFree(dkDeviceAddr);
  aclrtFree(dvDeviceAddr);
  aclrtFree(dsinkDeviceAddr);
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
