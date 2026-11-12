Disclaimer: This document is licensed under the [Creative Commons License version 4.0](https://creativecommons.org/licenses/by/4.0/legalcode). Any reproduction, citation, modification, or other use must comply with this license.

# aclnnNormRopeConcat

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T15:51:02.811Z -->

## Applicable Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Supported
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

- Description: Performs normalization (Norm), rotary position embedding (Rope), and feature concatenation (Concat) on query, key, and Value in the attention mechanism of a (multimodal) transformer:

    - Normalization (Norm) currently supports layer normalization (LayerNorm), layer normalization with affine transformation parameters (AFFINE LayerNorm), root mean square normalization (RmsNorm), and root mean square normalization with affine transformation parameters (AFFINE RmsNorm).
    - Rotary position encoding (Rope) supports Interleave and Half types.
    - Feature concatenation (Concat) supports concatenation along the sequence dimension, with order distinction.

- Formula (using Query (video) and EncoderQuery (text) as an example, with `concatOrder=0`):

    $$
    \begin{aligned}
    hiddenState_q &= \text{Norm}(query, normQueryWeight, normQueryBias, eps) \\
    hiddenState_{eq} &= \text{Norm}(encoderQuery, normAddedQueryWeight, normAddedQueryBias, eps) \\
    concatenatedHiddenState &= \text{Concat}(hiddenState_q, hiddenState_{eq}) \\
    transposedHiddenState &= \text{Transpose}(concatenatedHiddenState, (0, 2, 1, 3)) \\
    hiddenState &= \text{RoPE}(transposedHiddenState, ropeSin, ropeCos)
    \end{aligned}
    $$

    When `ropeType` is `NONE`, RoPE does not transform the input. Otherwise, RoPE applies only to the first `seqRope` positions of the concatenated sequence. The remaining positions are unchanged.

- Description:
    1. The input `query` has shape `(B, S_q, N, D)`, `encoderQuery` has shape `(B, S_eq, N, D)`, and the output `hiddenState` has shape `(B, N, S_q + S_eq, D)`, where
    `B` is the batch size, `S_q` and `S_eq` are the sequence lengths of `query` and `encoderQuery`, respectively, `N` is the number of attention heads, and `D` is the head dimension.
    2. Norm has five modes (`normType`): `NONE(0), LAYER_NORM(1), LAYER_NORM_AFFINE(2), RMS_NORM(3), RMS_NORM_AFFINE(4)`, where:
        When `normType = NONE`:
        $$
        hiddenState_q = query
        $$

        When `normType = LAYER_NORM`:

        $$
        \begin{aligned}
        queryMean_{b,s,n} &= \frac{1}{D}\sum_{i=0}^{D-1}query_{b,s,n,i} \\
        queryVar_{b,s,n} &= \frac{1}{D}\sum_{i=0}^{D-1}(query_{b,s,n,i}-queryMean_{b,s,n})^2 \\
        queryRstd_{b,s,n} &= \frac{1}{\sqrt{queryVar_{b,s,n}+\epsilon}} \\
        hiddenState_q &= (query-queryMean)*queryRstd
        \end{aligned}
        $$

        When `normType = LAYER_NORM_AFFINE`, on top of the above:

        $$
        hiddenState_q = normQueryWeight*hiddenState_q + normQueryBias
        $$

        When `normType = RMS_NORM`:

        $$
        \begin{aligned}
        queryMs &= \frac{1}{D}\sum_{i=0}^{D-1}(query_{b,s,n,i})^2 \\
        queryRms &= \frac{1}{\sqrt{queryMs+\epsilon}} \\
        hiddenState_q &= query * queryRms
        \end{aligned}
        $$

        When `normType = RMS_NORM_AFFINE`, based on the above:

        $$
        hiddenState_q = normQueryWeight*hiddenState_q
        $$

    3. Concat refers to concatenation along the sequence dimension. The concatenation has an order distinction (`concatOrder`). When `concatOrder=0`, $hiddenState_q$ precedes $hiddenState_{eq}$; when `concatOrder=1`, $hiddenState_q$ follows $hiddenState_{eq}$.
    4. RoPE has three modes (`ropeType`): `NONE(0)`, `INTERLEAVE(1)`, `HALF(2)`. When `ropeType=NONE`, the output is directly passed through without transformation. For other cases, refer to the following:

        ```python
          def image_rotary_emb(hidden_states, rope_sin, rope_cos, mode=1):
              out = torch.empty_like(hidden_states)
              if mode == 1: # interleave
                  x = hidden_states.view(*hidden_states.shape[:-1], -1, 2)
                  x1, x2 = x[..., 0], x[..., 1]
                  rotated_x = torch.stack([-x2, x1],dim=-1).flatten(3)
                  out = hidden_states.float() * rope_cos + rotated_x.float()*rope_sin
                  return out.type_as(hidden_states)
              else: # half
                  x1, x2 = hidden_states.reshape(*hidden_states.shape[:-1], 2, -1).unbind(-2)
                  rotated_x = torch.cat([-x2, x1],dim=-1)
                  out = hidden_states.float() * rope_cos + rotated_x.float()*rope_sin
                  return out.type_as(hidden_states)
        ```

    5. The shape of the RoPE input `ropeSin` is `(seqRope, D)`, where

      $$
      seqRope \leq \min(seqQuery+seqEncoderQuery, seqKey+seqEncoderKey)
      $$

    6. In training scenarios, `queryMean, queryRstd, encoderQueryMean, encoderQueryRstd` are output for subsequent backward propagation.

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call "aclnnNormRopeConcatGetWorkspaceSize" first to obtain the input parameter and calculate the required workspace size based on the computation process, and then call "aclnnNormRopeConcat" to execute the computation.

```cpp
aclnnStatus aclnnNormRopeConcatGetWorkspaceSize(
  const aclTensor *query,
  const aclTensor *key,
  const aclTensor *value,
  const aclTensor *encoderQuery,
  const aclTensor *encoderKey,
  const aclTensor *encoderValue,
  const aclTensor *normQueryWeight,
  const aclTensor *normQueryBias,
  const aclTensor *normKeyWeight,
  const aclTensor *normKeyBias,
  const aclTensor *normAddedQueryWeight,
  const aclTensor *normAddedQueryBias,
  const aclTensor *normAddedKeyWeight,
  const aclTensor *normAddedKeyBias,
  const aclTensor *ropeSin,
  const aclTensor *ropeCos,
  int64_t         normType,
  int64_t         normAddedType,
  int64_t         ropeType,
  int64_t         concatOrder,
  double          eps,
  bool            isTraining,
  const aclTensor *queryOutput,
  const aclTensor *keyOutput,
  const aclTensor *valueOutput,
  const aclTensor *normQueryMean,
  const aclTensor *normQueryRstd,
  const aclTensor *normKeyMean,
  const aclTensor *normKeyRstd,
  const aclTensor *normAddedQueryMean,
  const aclTensor *normAddedQueryRstd,
  const aclTensor *normAddedKeyMean,
  const aclTensor *normAddedKeyRstd,
  uint64_t        *workspaceSize,
  aclOpExecutor   **executor)
```

```cpp
aclnnStatus aclnnNormRopeConcat(
  void            *workspace,
  uint64_t        workspaceSize,
  aclOpExecutor   *executor,
  aclrtStream     stream)
```

## aclnnNormRopeConcatGetWorkspaceSize

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1452px"><colgroup>
    <col style="width: 174px">
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
        <td>Indicates the Query in the attention mechanism.</td>
        <td>The data type is consistent with key and value.</td>
        <td>FLOAT16, BFLOAT16, FLOAT</td>
        <td>ND</td>
        <td>[BSND]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>key</td>
        <td>Input</td>
        <td>Indicates the Key in the attention mechanism.</td>
        <td>The data type is consistent with query and value.</td>
        <td>FLOAT16, BFLOAT16, FLOAT</td>
        <td>ND</td>
        <td>[BSND]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>value</td>
        <td>Input</td>
        <td>Indicates the Value in the attention mechanism.</td>
        <td>The data type is consistent with query and key.</td>
        <td>FLOAT16, BFLOAT16, FLOAT</td>
        <td>ND</td>
        <td>[BSND]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>encoderQuery</td>
        <td>Input</td>
        <td>Indicates the Query in the attention mechanism, from EncoderHiddenState.</td>
        <td>The data type must be consistent with that of key and value, or a null pointer.</td>
        <td>FLOAT16, BFLOAT16, FLOAT</td>
        <td>ND</td>
        <td>[BSND]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>encoderKey</td>
        <td>Input</td>
        <td>Indicates the Key in the attention mechanism, from EncoderHiddenState.</td>
        <td>The data type is consistent with query and value, or a null pointer.</td>
        <td>FLOAT16, BFLOAT16, FLOAT</td>
        <td>ND</td>
        <td>[BSND]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>encoderValue</td>
        <td>Input</td>
        <td>Indicates the Value in the attention mechanism, from EncoderHiddenState.</td>
        <td>The data type is consistent with query and key, or is a null pointer.</td>
        <td>FLOAT16, BFLOAT16, FLOAT</td>
        <td>ND</td>
        <td>[BSND]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>normQueryWeight</td>
        <td>Input</td>
        <td>Indicates the affine transformation parameters of LayerNorm, applied to Query.</td>
        <td>Optional, needs to be provided when normType=2.</td>
        <td>FLOAT16, BFLOAT16, FLOAT</td>
        <td>ND</td>
        <td>[D]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>normQueryBias</td>
        <td>Input</td>
        <td>Indicates the affine transformation parameters of LayerNorm, applied to Query.</td>
        <td>Optional, needs to be provided when normType=2 or 4.</td>
        <td>FLOAT16, BFLOAT16, FLOAT</td>
        <td>ND</td>
        <td>[D]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>normKeyWeight</td>
        <td>Input</td>
        <td>Indicates the affine transformation parameter of LayerNorm, applied to Key.</td>
        <td>Optional, needs to be provided when normType=2 or 4.</td>
        <td>FLOAT16, BFLOAT16, FLOAT</td>
        <td>ND</td>
        <td>[D]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>normKeyBias</td>
        <td>Input</td>
        <td>Indicates the affine transformation parameter of LayerNorm, applied to Key.</td>
        <td>Optional, needs to be provided when normType=2.</td>
        <td>FLOAT16, BFLOAT16, FLOAT</td>
        <td>ND</td>
        <td>[D]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>normAddedQueryWeight</td>
        <td>Input</td>
        <td>Indicates the affine transformation parameter of LayerNorm, applied to encoderQuery.</td>
        <td>Optional. Need to provide it when normAddedType=2 or 4.</td>
        <td>FLOAT16, BFLOAT16, FLOAT</td>
        <td>ND</td>
        <td>[D]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>normAddedQueryBias</td>
        <td>Input</td>
        <td>Indicates the affine transformation parameter of LayerNorm, applied to encoderQuery.</td>
        <td>Optional, needs to be provided when normAddedType=2.</td>
        <td>FLOAT16, BFLOAT16, FLOAT</td>
        <td>ND</td>
        <td>[D]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>normAddedKeyWeight</td>
        <td>Input</td>
        <td>Indicates the affine transformation parameter of LayerNorm, applied to encoderKey.</td>
        <td>Optional, needs to be provided when normAddedType=2 or 4.</td>
        <td>FLOAT16, BFLOAT16, FLOAT</td>
        <td>ND</td>
        <td>[D]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>normAddedKeyBias</td>
        <td>Input</td>
        <td>Indicates the affine transformation parameters of LayerNorm, applied to encoderKey.</td>
        <td>Optional, needs to be provided when normAddedType=2.</td>
        <td>FLOAT16, BFLOAT16, FLOAT</td>
        <td>ND</td>
        <td>[D]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>ropeSin</td>
        <td>Input</td>
        <td>Indicates the sine encoding of RoPE.</td>
        <td>Need to provide when ropeType!=0.</td>
        <td>FLOAT16, BFLOAT16, FLOAT</td>
        <td>ND</td>
        <td>[SD]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>ropeCos</td>
        <td>Input</td>
        <td>Cosine encoding of RoPE.</td>
        <td>Need to provide when ropeType!=0.</td>
        <td>FLOAT16, BFLOAT16, FLOAT</td>
        <td>ND</td>
        <td>[SD]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>normType</td>
        <td>attribute</td>
        <td>Indicates the normalization type applied to q and k. 0: no normalization, 1: LayerNorm, 2: LayerNormAffine, 3: RmsNorm, 4: RmsNormAffine.</td>
        <td>None.</td>
        <td>int64</td>
        <td>scalar</td>
        <td>scalar</td>
        <td></td>
      </tr>
      <tr>
        <td>normAddedType</td>
        <td>attribute</td>
        <td>Indicates the normalization type applied to encoderQuery and encoderKey. 0: no normalization, 1: LayerNorm, 2: LayerNormAffine, 3: RmsNorm, 4: RmsNormAffine.</td>
        <td>None.</td>
        <td>int64</td>
        <td>scalar</td>
        <td>Scalar</td>
        <td></td>
      </tr>
      <tr>
        <td>ropeType</td>
        <td>attribute</td>
        <td>Indicates the RoPE mode, of the int64 type. 0: no RoPE, 1: Interleave, 2: Half.</td>
        <td>None.</td>
        <td>int64</td>
        <td>Scalar</td>
        <td>Scalar</td>
        <td></td>
      </tr>
      <tr>
        <td>concatOrder</td>
        <td>attribute</td>
        <td>Indicates the concatenation order, of the int64 type. 0: query first, 1: query last.</td>
        <td>None.</td>
        <td>int64</td>
        <td>scalar</td>
        <td>scalar</td>
        <td></td>
      </tr>
      <tr>
        <td>eps</td>
        <td>attribute</td>
        <td>Indicates the epsilon value in normalization.</td>
        <td>None.</td>
        <td>float32</td>
        <td>scalar</td>
        <td>scalar</td>
        <td></td>
      </tr>
      <tr>
        <td>isTraining</td>
        <td>attribute</td>
        <td>Indicates whether it is the training phase, determining whether to output values used for backward.</td>
        <td>None.</td>
        <td>bool</td>
        <td>scalar</td>
        <td>scalar</td>
        <td></td>
      </tr>
      <tr>
        <td>queryOutput</td>
        <td>Output</td>
        <td>Indicates the query output in the attention mechanism.</td>
        <td>Data type is consistent with query.</td>
        <td>FLOAT16, BFLOAT16, FLOAT</td>
        <td>ND</td>
        <td>[BNSD]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>keyOutput</td>
        <td>Output</td>
        <td>Indicates the key output in the attention mechanism.</td>
        <td>Data type must be consistent with key.</td>
        <td>FLOAT16, BFLOAT16, FLOAT</td>
        <td>ND</td>
        <td>[BNSD]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>valueOutput</td>
        <td>Output</td>
        <td>Indicates the value output in the attention mechanism.</td>
        <td>Data type must be consistent with value.</td>
        <td>FLOAT16, BFLOAT16, FLOAT</td>
        <td>ND</td>
        <td>[BNSD]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>normQueryMean</td>
        <td>Output</td>
        <td>Query mean output in LayerNorm, for backward.</td>
        <td>Valid when normType!=0 and isTraining=true.</td>
        <td>FLOAT</td>
        <td>ND</td>
        <td>[BS]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>normQueryRstd</td>
        <td>Output</td>
        <td>Standard deviation output of the query in LayerNorm, for backward.</td>
        <td>Valid when normType!=0 and isTraining=true.</td>
        <td>FLOAT</td>
        <td>ND</td>
        <td>[BS]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>normKeyMean</td>
        <td>Output</td>
        <td>Mean output of the key in LayerNorm, for backward.</td>
        <td>Valid when normType!=0 and isTraining=true.</td>
        <td>FLOAT</td>
        <td>ND</td>
        <td>[BS]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>normKeyRstd</td>
        <td>Output</td>
        <td>Standard deviation output of the key in LayerNorm, for backward.</td>
        <td>Valid when normType!=0 and isTraining=true.</td>
        <td>FLOAT</td>
        <td>ND</td>
        <td>[BS]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>normAddedQueryMean</td>
        <td>Output</td>
        <td>Mean output of encoderQuery in LayerNorm, for backward.</td>
        <td>Valid when normAddedType!=0 and isTraining=true.</td>
        <td>FLOAT</td>
        <td>ND</td>
        <td>[BS]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>normAddedQueryRstd</td>
        <td>Output</td>
        <td>Standard deviation output of encoderQuery in LayerNorm, for backward.</td>
        <td>Valid when normAddedType!=0 and isTraining=true.</td>
        <td>FLOAT</td>
        <td>ND</td>
        <td>[BS]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>normAddedKeyMean</td>
        <td>Output</td>
        <td>The mean output of encoderKey in LayerNorm, for backward.</td>
        <td>Valid when normAddedType!=0 and isTraining=true.</td>
        <td>FLOAT</td>
        <td>ND</td>
        <td>[BS]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>normAddedKeyRstd</td>
        <td>Output</td>
        <td>Standard deviation output of encoderKey in LayerNorm, for backward.</td>
        <td>Valid when normAddedType!=0 and isTraining=true.</td>
        <td>FLOAT</td>
        <td>ND</td>
        <td>[BS]</td>
        <td>√</td>
      </tr>
    </tbody></table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:
  <table style="undefined;table-layout: fixed;width: 1149px"><colgroup>
  <col style="width: 281px">
  <col style="width: 119px">
  <col style="width: 749px">
  </colgroup>
  <thead>
    <tr>
      <th>Return Value</th>
      <th>Error Code</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>The input query, key, or value is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="2">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="2">161002</td>
      <td>The data type of an input or output is not supported.</td>
    </tr>
    <tr>
      <td>The input shape is not supported.</td>
    </tr>
    <tr>
      <td rowspan="2">ACLNN_ERR_INNER_TILING_ERROR</td>
      <td rowspan="2">561002</td>
      <td>The NormType or NormAddedType of the input parameter is not in the supported range.</td>
    </tr>
    <tr>
      <td>The shapes of the input tensors do not meet the constraints.</td>
    </tr>
     <tr>
      <td>ACLNN_ERR_INNER_TILING_ERROR</td>
      <td>361001</td>
      <td>The ropeType or concatOrder of the input parameter takes a value beyond the constraints.</td>
    </tr>
  </tbody>
  </table>

## aclnnNormRopeConcat

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1149px">
  <colgroup>
    <col style="width: 281px">
    <col style="width: 119px">
    <col style="width: 749px">
  </colgroup>
  <tr>
    <th align="center">Parameter</th>
    <th align="center">Input/Output</th>
    <th align="center">Description</th>
  </tr>
  <tbody>
    <tr>
      <td>workspace</td>
      <td>Input</td>
      <td>Address of the workspace memory allocated on the Device side.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace allocated on the Device side, obtained by calling the first-phase API aclnnNormRopeConcatGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Op executor, which contains the operator computation flow.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream for task execution.</td>
    </tr>
  </tbody>
  </table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnNormRopeConcat defaults to a deterministic implementation.
- The data types of query, key, value, encoderQuery, encoderKey, and encoderValue must be consistent.
- The headDim length is in the range [1, 1024] and must be an even number.
- The seqRope length ranges from 1 to Min(seqQuery+seqEncoderQuery, seqKey+seqEncoderKey).

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

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
  #include "aclnnop/aclnn_norm_rope_concat.h"

  #define SUCCESS 0
  #define FAILED 1

  #define INFO_LOG(fmt, args...) fprintf(stdout, "[INFO]  " fmt "\n", ##args)
  #define WARN_LOG(fmt, args...) fprintf(stdout, "[WARN]  " fmt "\n", ##args)
  #define ERROR_LOG(fmt, args...) fprintf(stderr, "[ERROR]  " fmt "\n", ##args)

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

  int64_t GetShapeSize(const std::vector<int64_t> &shape)
  {
      int64_t shapeSize = 1;
      for (auto i : shape) {
          shapeSize *= i;
      }
      return shapeSize;
  }

  int Init(int32_t deviceId, aclrtContext *context, aclrtStream *stream)
  {
      // Boilerplate for ACL initialization.
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
  int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                      aclDataType dataType, aclTensor **result)
  {
      auto size = GetShapeSize(shape) * sizeof(T);
      // Call aclrtMalloc to allocate device-side memory
      auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
      // Call aclrtMemcpy to copy host-side data to device-side memory
      ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

      // Calculate strides of the contiguous result
      std::vector<int64_t> strides(shape.size(), 1);
      for (int64_t i = shape.size() - 2; i >= 0; i--) {
          strides[i] = shape[i + 1] * strides[i + 1];
      }

      // Call aclCreateTensor to create an aclTensor
      *result = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                                shape.data(), shape.size(), *deviceAddr);
      return 0;
  }

  int main()
  {
      // 1. (boilerplate) Initialize device/context/stream. For details, see the acl external API list
      // Set deviceId based on the actual device
      int32_t deviceId = 0;
      aclrtContext context;
      aclrtStream stream;
      auto ret = Init(deviceId, &context, &stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

      // 2. Construct inputs and outputs, customized based on the API
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
      float eps = 1e-5f;
      uint32_t normType = 2;
      uint32_t normAddedType = 2;
      uint32_t ropeType = 1;
      uint32_t concatOrder = 0;

      std::vector<int64_t> queryShape = {batchSize, querySeq, headNum, headDim};
      std::vector<int64_t> keyShape = {batchSize, keySeq, headNum, headDim};
      std::vector<int64_t> valueShape = {batchSize, keySeq, headNum, headDim};
      std::vector<int64_t> encoderQueryShape = {batchSize, encoderQuerySeq, headNum, headDim};
      std::vector<int64_t> encoderKeyShape = {batchSize, encoderKeySeq, headNum, headDim};
      std::vector<int64_t> encoderValueShape = {batchSize, encoderKeySeq, headNum, headDim};

      std::vector<int64_t> normQueryWeightShape = {headDim};
      std::vector<int64_t> normQueryBiasShape = {headDim};
      std::vector<int64_t> normKeyWeightShape = {headDim};
      std::vector<int64_t> normKeyBiasShape = {headDim};
      std::vector<int64_t> normAddedQueryWeightShape = {headDim};
      std::vector<int64_t> normAddedQueryBiasShape = {headDim};
      std::vector<int64_t> normAddedKeyWeightShape = {headDim};
      std::vector<int64_t> normAddedKeyBiasShape = {headDim};

      std::vector<int64_t> ropeSinShape = {ropeSeq, headDim};
      std::vector<int64_t> ropeCosShape = {ropeSeq, headDim};

      std::vector<int64_t> queryOutputShape = {batchSize, headNum, querySeq + encoderQuerySeq, headDim};
      std::vector<int64_t> keyOutputShape = {batchSize, headNum, keySeq + encoderKeySeq, headDim};
      std::vector<int64_t> valueOutputShape = {batchSize, headNum, valueSeq + encoderValueSeq, headDim};

      std::vector<int64_t> normQueryMeanShape = {batchSize, querySeq, headNum, 1};
      std::vector<int64_t> normQueryRstdShape = {batchSize, querySeq, headNum, 1};
      std::vector<int64_t> normKeyMeanShape = {batchSize, keySeq, headNum, 1};
      std::vector<int64_t> normKeyRstdShape = {batchSize, keySeq, headNum, 1};
      std::vector<int64_t> normAddedQueryMeanShape = {batchSize, encoderQuerySeq, headNum, 1};
      std::vector<int64_t> normAddedQueryRstdShape = {batchSize, encoderQuerySeq, headNum, 1};
      std::vector<int64_t> normAddedKeyMeanShape = {batchSize, encoderKeySeq, headNum, 1};
      std::vector<int64_t> normAddedKeyRstdShape = {batchSize, encoderKeySeq, headNum, 1};

      void *queryDeviceAddr = nullptr;
      void *keyDeviceAddr = nullptr;
      void *valueDeviceAddr = nullptr;
      void *encoderQueryDeviceAddr = nullptr;
      void *encoderKeyDeviceAddr = nullptr;
      void *encoderValueDeviceAddr = nullptr;
      // layernorm
      void *normQueryWeightDeviceAddr = nullptr;
      void *normQueryBiasDeviceAddr = nullptr;
      void *normKeyWeightDeviceAddr = nullptr;
      void *normKeyBiasDeviceAddr = nullptr;
      void *normAddedQueryWeightDeviceAddr = nullptr;
      void *normAddedQueryBiasDeviceAddr = nullptr;
      void *normAddedKeyWeightDeviceAddr = nullptr;
      void *normAddedKeyBiasDeviceAddr = nullptr;
      // rope
      void *ropeSinDeviceAddr = nullptr;
      void *ropeCosDeviceAddr = nullptr;

      void *queryOutputDeviceAddr = nullptr;
      void *keyOutputDeviceAddr = nullptr;
      void *valueOutputDeviceAddr = nullptr;

      void *normQueryMeanDeviceAddr = nullptr;
      void *normQueryRstdDeviceAddr = nullptr;
      void *normKeyMeanDeviceAddr = nullptr;
      void *normKeyRstdDeviceAddr = nullptr;
      void *normAddedQueryMeanDeviceAddr = nullptr;
      void *normAddedQueryRstdDeviceAddr = nullptr;
      void *normAddedKeyMeanDeviceAddr = nullptr;
      void *normAddedKeyRstdDeviceAddr = nullptr;

      aclTensor *query = nullptr;
      aclTensor *key = nullptr;
      aclTensor *value = nullptr;
      aclTensor *encoderQuery = nullptr;
      aclTensor *encoderKey = nullptr;
      aclTensor *encoderValue = nullptr;
      aclTensor *normQueryWeight = nullptr;
      aclTensor *normQueryBias = nullptr;
      aclTensor *normKeyWeight = nullptr;
      aclTensor *normKeyBias = nullptr;
      aclTensor *normAddedQueryWeight = nullptr;
      aclTensor *normAddedQueryBias = nullptr;
      aclTensor *normAddedKeyWeight = nullptr;
      aclTensor *normAddedKeyBias = nullptr;

      aclTensor *ropeSin = nullptr;
      aclTensor *ropeCos = nullptr;

      aclTensor *normQueryMean = nullptr;
      aclTensor *normQueryRstd = nullptr;
      aclTensor *normKeyMean = nullptr;
      aclTensor *normKeyRstd = nullptr;
      aclTensor *normAddedQueryMean = nullptr;
      aclTensor *normAddedQueryRstd = nullptr;
      aclTensor *normAddedKeyMean = nullptr;
      aclTensor *normAddedKeyRstd = nullptr;

      aclTensor *queryOutput = nullptr;
      aclTensor *keyOutput = nullptr;
      aclTensor *valueOutput = nullptr;

      std::vector<float> queryOutputHostData(batchSize * headNum * (querySeq + encoderQuerySeq) * headDim, 0.0);
      std::vector<float> keyOutputHostData(batchSize * headNum * (keySeq + encoderKeySeq) * headDim, 0.0);
      std::vector<float> valueOutputHostData(batchSize * headNum * (valueSeq + encoderValueSeq) * headDim, 0.0);

      std::vector<float> encoderQueryHostData(batchSize * headNum * encoderQuerySeq * headDim, 4.0);
      std::vector<float> encoderKeyHostData(batchSize * headNum * encoderKeySeq * headDim, 5.0);
      std::vector<float> encoderValueHostData(batchSize * headNum * encoderValueSeq * headDim, 6.0);

      std::vector<float> normQueryWeightHostData(headDim, 1.0);
      std::vector<float> normQueryBiasHostData(headDim, 2.0);
      std::vector<float> normKeyWeightHostData(headDim, 3.0);
      std::vector<float> normKeyBiasHostData(headDim, 4.0);
      std::vector<float> normAddedQueryWeightHostData(headDim, 5.0);
      std::vector<float> normAddedQueryBiasHostData(headDim, 6.0);
      std::vector<float> normAddedKeyWeightHostData(headDim, 7.0);
      std::vector<float> normAddedKeyBiasHostData(headDim, 8.0);
      std::vector<float> ropeSinHostData(ropeSeq * headDim, 9.0);
      std::vector<float> ropeCosHostData(ropeSeq * headDim, 10.0);

      std::vector<float> queryHostData(batchSize * headNum * querySeq * headDim, 0.0);
      std::vector<float> keyHostData(batchSize * headNum * keySeq * headDim, 0.0);
      std::vector<float> valueHostData(batchSize * headNum * keySeq * headDim, 0.0);
      std::vector<float> normQueryMeanHostData(batchSize * headNum * querySeq * 1, 0.0);
      std::vector<float> normQueryRstdHostData(batchSize * headNum * querySeq * 1, 0.0);
      std::vector<float> normKeyMeanHostData(batchSize * headNum * keySeq * 1, 0.0);
      std::vector<float> normKeyRstdHostData(batchSize * headNum * keySeq * 1, 0.0);
      std::vector<float> normAddedQueryMeanHostData(batchSize * headNum * encoderQuerySeq * 1, 0.0);
      std::vector<float> normAddedQueryRstdHostData(batchSize * headNum * encoderQuerySeq * 1, 0.0);
      std::vector<float> normAddedKeyMeanHostData(batchSize * headNum * encoderKeySeq * 1, 0.0);
      std::vector<float> normAddedKeyRstdHostData(batchSize * headNum * encoderKeySeq * 1, 0.0);

      CHECK_RET(ret == ACL_SUCCESS, return ret);
      ret = CreateAclTensor(queryHostData, queryShape, &queryDeviceAddr, aclDataType::ACL_FLOAT, &query);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      ret = CreateAclTensor(keyHostData, keyShape, &keyDeviceAddr, aclDataType::ACL_FLOAT, &key);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      ret = CreateAclTensor(valueHostData, valueShape, &valueDeviceAddr, aclDataType::ACL_FLOAT, &value);
      CHECK_RET(ret == ACL_SUCCESS, return ret);

      ret = CreateAclTensor(encoderQueryHostData, encoderQueryShape, &encoderQueryDeviceAddr, aclDataType::ACL_FLOAT,
                            &encoderQuery);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      ret = CreateAclTensor(encoderKeyHostData, encoderKeyShape, &encoderKeyDeviceAddr, aclDataType::ACL_FLOAT,
                            &encoderKey);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      ret = CreateAclTensor(encoderValueHostData, encoderValueShape, &encoderValueDeviceAddr, aclDataType::ACL_FLOAT,
                            &encoderValue);
      CHECK_RET(ret == ACL_SUCCESS, return ret);

      ret = CreateAclTensor(normQueryWeightHostData, normQueryWeightShape, &normQueryWeightDeviceAddr,
                            aclDataType::ACL_FLOAT, &normQueryWeight);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      ret = CreateAclTensor(normQueryBiasHostData, normQueryBiasShape, &normQueryBiasDeviceAddr, aclDataType::ACL_FLOAT,
                            &normQueryBias);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      ret = CreateAclTensor(normKeyWeightHostData, normKeyWeightShape, &normKeyWeightDeviceAddr, aclDataType::ACL_FLOAT,
                            &normKeyWeight);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      ret = CreateAclTensor(normKeyBiasHostData, normKeyBiasShape, &normKeyBiasDeviceAddr, aclDataType::ACL_FLOAT,
                            &normKeyBias);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      ret = CreateAclTensor(normAddedQueryWeightHostData, normAddedQueryWeightShape, &normAddedQueryWeightDeviceAddr,
                            aclDataType::ACL_FLOAT, &normAddedQueryWeight);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      ret = CreateAclTensor(normAddedQueryBiasHostData, normAddedQueryBiasShape, &normAddedQueryBiasDeviceAddr,
                            aclDataType::ACL_FLOAT, &normAddedQueryBias);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      ret = CreateAclTensor(normAddedKeyWeightHostData, normAddedKeyWeightShape, &normAddedKeyWeightDeviceAddr,
                            aclDataType::ACL_FLOAT, &normAddedKeyWeight);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      ret = CreateAclTensor(normAddedKeyBiasHostData, normAddedKeyBiasShape, &normAddedKeyBiasDeviceAddr,
                            aclDataType::ACL_FLOAT, &normAddedKeyBias);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      ret = CreateAclTensor(ropeSinHostData, ropeSinShape, &ropeSinDeviceAddr, aclDataType::ACL_FLOAT, &ropeSin);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      ret = CreateAclTensor(ropeCosHostData, ropeCosShape, &ropeCosDeviceAddr, aclDataType::ACL_FLOAT, &ropeCos);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      ret = CreateAclTensor(queryOutputHostData, queryOutputShape, &queryOutputDeviceAddr, aclDataType::ACL_FLOAT,
                            &queryOutput);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      ret = CreateAclTensor(keyOutputHostData, keyOutputShape, &keyOutputDeviceAddr, aclDataType::ACL_FLOAT, &keyOutput);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      ret = CreateAclTensor(valueOutputHostData, valueOutputShape, &valueOutputDeviceAddr, aclDataType::ACL_FLOAT,
                            &valueOutput);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      ret = CreateAclTensor(normQueryMeanHostData, normQueryMeanShape, &normQueryMeanDeviceAddr, aclDataType::ACL_FLOAT,
                            &normQueryMean);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      ret = CreateAclTensor(normQueryRstdHostData, normQueryRstdShape, &normQueryRstdDeviceAddr, aclDataType::ACL_FLOAT,
                            &normQueryRstd);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      ret = CreateAclTensor(normKeyMeanHostData, normKeyMeanShape, &normKeyMeanDeviceAddr, aclDataType::ACL_FLOAT,
                            &normKeyMean);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      ret = CreateAclTensor(normKeyRstdHostData, normKeyRstdShape, &normKeyRstdDeviceAddr, aclDataType::ACL_FLOAT,
                            &normKeyRstd);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      ret = CreateAclTensor(normAddedQueryMeanHostData, normAddedQueryMeanShape, &normAddedQueryMeanDeviceAddr,
                            aclDataType::ACL_FLOAT, &normAddedQueryMean);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      ret = CreateAclTensor(normAddedQueryRstdHostData, normAddedQueryRstdShape, &normAddedQueryRstdDeviceAddr,
                            aclDataType::ACL_FLOAT, &normAddedQueryRstd);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      ret = CreateAclTensor(normAddedKeyMeanHostData, normAddedKeyMeanShape, &normAddedKeyMeanDeviceAddr,
                            aclDataType::ACL_FLOAT, &normAddedKeyMean);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      ret = CreateAclTensor(normAddedKeyRstdHostData, normAddedKeyRstdShape, &normAddedKeyRstdDeviceAddr,
                            aclDataType::ACL_FLOAT, &normAddedKeyRstd);
      CHECK_RET(ret == ACL_SUCCESS, return ret);

      // 3. Call the CANN operator library API and replace it with the actual API name
      uint64_t workspaceSize = 0;
      aclOpExecutor *executor;
      bool isTraining = false;
      // Call the first-phase API of aclnnNormRopeConcat
      ret = aclnnNormRopeConcatGetWorkspaceSize(
          query, key, value, encoderQuery, encoderKey, encoderValue, normQueryWeight, normQueryBias, normKeyWeight,
          normKeyBias, normAddedQueryWeight, normAddedQueryBias, normAddedKeyWeight, normAddedKeyBias, ropeSin, ropeCos,
          normType, normAddedType, ropeType, concatOrder, eps, isTraining, queryOutput, keyOutput, valueOutput,
          normQueryMean, normQueryRstd, normKeyMean, normKeyRstd, normAddedQueryMean, normAddedQueryRstd,
          normAddedKeyMean, normAddedKeyRstd, &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNormRopeConcatGetWorkspaceSize failed. ERROR: %d\n", ret);
                return ret);
      // Allocate device memory based on the workspaceSize calculated by the first-phase API
      void *workspaceAddr = nullptr;
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
      }
      // Call the second-phase API of aclnnNormRopeConcat
      ret = aclnnNormRopeConcat(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNormRopeConcat failed. ERROR: %d\n", ret); return ret);

      // 4. (boilerplate) Synchronously wait for the task execution to finish
      ret = aclrtSynchronizeStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

      // 5. Obtain the output value and copy the result from device memory to host memory. Modify it according to the specific API definition
      auto size = GetShapeSize(queryOutputShape);
      std::vector<float> queryOutputData(size, 0);
      ret = aclrtMemcpy(queryOutputData.data(), queryOutputData.size() * sizeof(queryOutputData[0]),
                        queryOutputDeviceAddr, size * sizeof(queryOutputData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy queryOutput result from device to host failed. ERROR: %d\n", ret);
                return ret);
      for (int64_t i = 0; i < size; i++) {
          LOG_PRINT("queryOutput result[%ld] is: %f\n", i, queryOutputData[i]);
      }
      size = GetShapeSize(keyOutputShape);
      std::vector<float> keyOutputData(size, 0);
      ret = aclrtMemcpy(keyOutputData.data(), keyOutputData.size() * sizeof(keyOutputData[0]), keyOutputDeviceAddr,
                        size * sizeof(keyOutputData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy keyOutput result from device to host failed. ERROR: %d\n", ret);
                return ret);
      for (int64_t i = 0; i < size; i++) {
          LOG_PRINT("keyOutput result[%ld] is: %f\n", i, keyOutputData[i]);
      }
      size = GetShapeSize(valueOutputShape);
      std::vector<float> valueOutputData(size, 0);
      ret = aclrtMemcpy(valueOutputData.data(), valueOutputData.size() * sizeof(valueOutputData[0]),
                        valueOutputDeviceAddr, size * sizeof(valueOutputData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy valueOutput result from device to host failed. ERROR: %d\n", ret);
                return ret);
      for (int64_t i = 0; i < size; i++) {
          LOG_PRINT("valueOutput result[%ld] is: %f\n", i, valueOutputData[i]);
      }
      size = GetShapeSize(normQueryMeanShape);
      std::vector<float> normQueryMeanData(size, 0);
      ret = aclrtMemcpy(normQueryMeanData.data(), normQueryMeanData.size() * sizeof(normQueryMeanData[0]),
                        normQueryMeanDeviceAddr, size * sizeof(normQueryMeanData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy normQueryMean result from device to host failed. ERROR: %d\n", ret);
                return ret);
      for (int64_t i = 0; i < size; i++) {
          LOG_PRINT("normQueryMean result[%ld] is: %f\n", i, normQueryMeanData[i]);
      }
      size = GetShapeSize(normQueryRstdShape);
      std::vector<float> normQueryRstdData(size, 0);
      ret = aclrtMemcpy(normQueryRstdData.data(), normQueryRstdData.size() * sizeof(normQueryRstdData[0]),
                        normQueryRstdDeviceAddr, size * sizeof(normQueryRstdData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy normQueryRstd result from device to host failed. ERROR: %d\n", ret);
                return ret);
      for (int64_t i = 0; i < size; i++) {
          LOG_PRINT("normQueryRstd result[%ld] is: %f\n", i, normQueryRstdData[i]);
      }
      size = GetShapeSize(normKeyMeanShape);
      std::vector<float> normKeyMeanData(size, 0);
      ret = aclrtMemcpy(normKeyMeanData.data(), normKeyMeanData.size() * sizeof(normKeyMeanData[0]),
                        normKeyMeanDeviceAddr, size * sizeof(normKeyMeanData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy normKeyMean result from device to host failed. ERROR: %d\n", ret);
                return ret);
      for (int64_t i = 0; i < size; i++) {
          LOG_PRINT("normKeyMean result[%ld] is: %f\n", i, normKeyMeanData[i]);
      }
      size = GetShapeSize(normKeyRstdShape);
      std::vector<float> normKeyRstdData(size, 0);
      ret = aclrtMemcpy(normKeyRstdData.data(), normKeyRstdData.size() * sizeof(normKeyRstdData[0]),
                        normKeyRstdDeviceAddr, size * sizeof(normKeyRstdData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy normKeyRstd result from device to host failed. ERROR: %d\n", ret);
                return ret);
      for (int64_t i = 0; i < size; i++) {
          LOG_PRINT("normKeyRstd result[%ld] is: %f\n", i, normKeyRstdData[i]);
      }
      size = GetShapeSize(normAddedQueryMeanShape);
      std::vector<float> normAddedQueryMeanData(size, 0);
      ret =
          aclrtMemcpy(normAddedQueryMeanData.data(), normAddedQueryMeanData.size() * sizeof(normAddedQueryMeanData[0]),
                      normAddedQueryMeanDeviceAddr, size * sizeof(normAddedQueryMeanData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS,
                LOG_PRINT("copy normAddedQueryMean result from device to host failed. ERROR: %d\n", ret);
                return ret);
      for (int64_t i = 0; i < size; i++) {
          LOG_PRINT("normAddedQueryMean result[%ld] is: %f\n", i, normAddedQueryMeanData[i]);
      }
      size = GetShapeSize(normAddedQueryRstdShape);
      std::vector<float> normAddedQueryRstdData(size, 0);
      ret =
          aclrtMemcpy(normAddedQueryRstdData.data(), normAddedQueryRstdData.size() * sizeof(normAddedQueryRstdData[0]),
                      normAddedQueryRstdDeviceAddr, size * sizeof(normAddedQueryRstdData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS,
                LOG_PRINT("copy normAddedQueryRstd result from device to host failed. ERROR: %d\n", ret);
                return ret);
      for (int64_t i = 0; i < size; i++) {
          LOG_PRINT("normAddedQueryRstd result[%ld] is: %f\n", i, normAddedQueryRstdData[i]);
      }
      size = GetShapeSize(normAddedKeyMeanShape);
      std::vector<float> normAddedKeyMeanData(size, 0);
      ret = aclrtMemcpy(normAddedKeyMeanData.data(), normAddedKeyMeanData.size() * sizeof(normAddedKeyMeanData[0]),
                        normAddedKeyMeanDeviceAddr, size * sizeof(normAddedKeyMeanData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS,
                LOG_PRINT("copy normAddedKeyMean result from device to host failed. ERROR: %d\n", ret);
                return ret);
      for (int64_t i = 0; i < size; i++) {
          LOG_PRINT("normAddedKeyMean result[%ld] is: %f\n", i, normAddedKeyMeanData[i]);
      }
      size = GetShapeSize(normAddedKeyRstdShape);
      std::vector<float> normAddedKeyRstdData(size, 0);
      ret = aclrtMemcpy(normAddedKeyRstdData.data(), normAddedKeyRstdData.size() * sizeof(normAddedKeyRstdData[0]),
                        normAddedKeyRstdDeviceAddr, size * sizeof(normAddedKeyRstdData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS,
                LOG_PRINT("copy normAddedKeyRstd result from device to host failed. ERROR: %d\n", ret);
                return ret);
      for (int64_t i = 0; i < size; i++) {
          LOG_PRINT("normAddedKeyRstd result[%ld] is: %f\n", i, normAddedKeyRstdData[i]);
      }


      // 6. Release aclTensor and aclScalar. Modify it according to the specific API definition
      aclDestroyTensor(query);
      aclDestroyTensor(key);
      aclDestroyTensor(value);
      aclDestroyTensor(encoderQuery);
      aclDestroyTensor(encoderKey);
      aclDestroyTensor(encoderValue);
      aclDestroyTensor(normQueryWeight);
      aclDestroyTensor(normQueryBias);
      aclDestroyTensor(normKeyWeight);
      aclDestroyTensor(normKeyBias);
      aclDestroyTensor(normAddedQueryWeight);
      aclDestroyTensor(normAddedQueryBias);
      aclDestroyTensor(normAddedKeyWeight);
      aclDestroyTensor(normAddedKeyBias);
      aclDestroyTensor(ropeSin);
      aclDestroyTensor(ropeCos);
      aclDestroyTensor(queryOutput);
      aclDestroyTensor(keyOutput);
      aclDestroyTensor(valueOutput);
      aclDestroyTensor(normQueryMean);
      aclDestroyTensor(normQueryRstd);
      aclDestroyTensor(normKeyMean);
      aclDestroyTensor(normKeyRstd);
      aclDestroyTensor(normAddedQueryMean);
      aclDestroyTensor(normAddedQueryRstd);
      aclDestroyTensor(normAddedKeyMean);
      aclDestroyTensor(normAddedKeyRstd);

      // 7. Release device resources. Modify it according to the specific API definition
      aclrtFree(queryDeviceAddr);
      aclrtFree(keyDeviceAddr);
      aclrtFree(valueDeviceAddr);
      aclrtFree(encoderQueryDeviceAddr);
      aclrtFree(encoderKeyDeviceAddr);
      aclrtFree(encoderValueDeviceAddr);
      aclrtFree(normQueryWeightDeviceAddr);
      aclrtFree(normQueryBiasDeviceAddr);
      aclrtFree(normKeyWeightDeviceAddr);
      aclrtFree(normKeyBiasDeviceAddr);
      aclrtFree(normAddedQueryWeightDeviceAddr);
      aclrtFree(normAddedQueryBiasDeviceAddr);
      aclrtFree(normAddedKeyWeightDeviceAddr);
      aclrtFree(normAddedKeyBiasDeviceAddr);
      aclrtFree(ropeSinDeviceAddr);
      aclrtFree(ropeCosDeviceAddr);
      aclrtFree(queryOutputDeviceAddr);
      aclrtFree(keyOutputDeviceAddr);
      aclrtFree(valueOutputDeviceAddr);
      aclrtFree(normQueryMeanDeviceAddr);
      aclrtFree(normQueryRstdDeviceAddr);
      aclrtFree(normKeyMeanDeviceAddr);
      aclrtFree(normKeyRstdDeviceAddr);
      aclrtFree(normAddedQueryMeanDeviceAddr);
      aclrtFree(normAddedQueryRstdDeviceAddr);
      aclrtFree(normAddedKeyMeanDeviceAddr);
      aclrtFree(normAddedKeyRstdDeviceAddr);

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
