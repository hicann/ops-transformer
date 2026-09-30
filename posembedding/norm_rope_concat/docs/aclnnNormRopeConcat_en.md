# aclnnNormRopeConcat

## Supported Products

|Product     | Supported|
|:----------------------------|:-----------:|
|<term>Atlas A3 training products/Atlas A3 inference products</term>|      √     |
|<term>Atlas A2 training products/Atlas A2 inference products</term>|      √     |
|<term>Atlas 200I/500 A2 inference products</term>|      ×     |
|<term>Atlas inference products</term>|      ×     |
|<term>Atlas training products</term>|      ×     |

## Function

- Description: Implements normalization (Norm), Rotary Position Embedding (RoPE), and feature concatenation (Concat) for `query`, `key`, and `value` in the (multi-modal) transformer attention mechanism.

    - Currently, Norm supports layer normalization (LayerNorm) and layer normalization with affine transformation parameters (AFFINE LayerNorm).
    - RoPE supports the Interleave and Half types.
    - Concat can be performed in the sequence dimension with support for specifying different concatenation orders.

- The computation formulas (using `query` (video) and `encoderQuery` (text) as examples) are as follows:

      $$
      hiddenState_q = \text{LayerNorm}(query, normQueryWeight, normQueryBias, eps) \\
      hiddenState_{eq} = \text{LayerNorm}(encoderQuery, normEncoderQueryWeight, normEncoderQueryBias, eps) \\
      concatedHiddenState = \text{Concat}(hiddenState_q, hiddenState_{eq}) \\
      transposedHiddenState = \text{Transpose}(concatedHiddenState, (0, 2, 1, 3)) \\
      hiddenState = \text{RoPE}(concatedHiddenState, ropeSin, ropeCos)
      $$

- Note:
    1. The input and output layouts are as follows: The shape of the input `query` is `(B, S, N, D)`, and the shape of the output `hiddenState` is `(B, N, S, D)`.
    `B` indicates `batch`, `S` indicates `sequenceLen`, `N` indicates `headNum`, and `D` indicates `headDim`.
    2. LayerNorm has three modes (specified by `normType`): `NONE(0)`, `LAYER_NORM(1)`, and `LAYER_NORM_AFFINE(2)`.
        When `normType = NONE`:

        $$
        hiddenState_q = query
        $$

        When `normType = LAYER_NORM`:

        $$
        queryMean_{b,s,n} = \frac{1}{D}\sum_{i=0}^{D}query_{b,s,n} \\
        queryVar_{b,s,n} = \frac{1}{D}\sum_{i=0}^{D}(query-queryMean_{b,s,n})^2 \\
        queryRstd_{b,s,n}=  \frac{1}{\sqrt{queryVar_{b,s,n}+\epsilon}} \\
        hiddenState_q = (query-queryMean)*queryRstd
        $$

        When `normType =LAYER_NORM_AFFINE` (based on the preceding formulas):

        $$
        hiddenState_q = normQueryWeight*hiddenState_q + normQueryBias
        $$

    3. Concat is performed in the sequence dimension. The concatenation order is specified by `concatOrder`. When `concatOrder=0`, $hiddenState_q$ is before $hiddenState_{eq}$. When `concatOrder=1`, $hiddenState_q$ is after $hiddenState_{eq}$.
    4. RoPE has three modes (specified by `ropeType`):`NONE(0)`, `INTERLEAVE(1)`, and `HALF(2)`. When `ropeType=NONE`, the output is directly generated without transformation. In other cases, see the following:
        
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

    5. The shape of the input `ropeSin` for RoPE is `(seqRope, D)`, where

       $$seqRope ≤ min(seqQuery+seqEncoderQuery, seqKey+seqEncoderKey)$$

    6. In the training scenario, `queryMean`, `queryRstd`, `encoderQueryMean`, and `encoderQueryRstd` are output for subsequent backward propagation.

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnNormRopeConcatGetWorkspaceSize` is called to obtain the input parameters and compute the required workspace size based on the process. Then, `aclnnNormRopeConcat` is called to perform computation.

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
  uint64_t *workspaceSize,
  aclOpExecutor **executor)
```

```cpp
aclnnStatus aclnnNormRopeConcat(
  void *workspace,
  uint64_t workspaceSize,
  aclOpExecutor *executor,
  aclrtStream stream)
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
        <td>Query in the attention mechanism.</td>
        <td>The data type must be the same as that of <code>key</code> and <code>value</code>.</td>
        <td>FLOAT16, BFLOAT16, FLOAT</td>
        <td>ND</td>
        <td>[BSND]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>key</td>
        <td>Input</td>
        <td>Key in the attention mechanism.</td>
        <td>The data type must be the same as that of <code>query</code> and <code>value</code>.</td>
        <td>FLOAT16, BFLOAT16, FLOAT</td>
        <td>ND</td>
        <td>[BSND]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>value</td>
        <td>Input</td>
        <td>Value in the attention mechanism.</td>
        <td>The data type must be the same as that of <code>query</code> and <code>key</code>.</td>
        <td>FLOAT16, BFLOAT16, FLOAT</td>
        <td>ND</td>
        <td>[BSND]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>encoderQuery</td>
        <td>Input</td>
        <td>Query in the attention mechanism, which comes from EncoderHiddenState.</td>
        <td>Its data type must be the same as that of <code>key</code> and <code>value</code>, or it is a null pointer.</td>
        <td>FLOAT16, BFLOAT16, FLOAT</td>
        <td>ND</td>
        <td>[BSND]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>encoderKey</td>
        <td>Input</td>
        <td>Key in the attention mechanism, which comes from EncoderHiddenState.</td>
        <td>Its data type must be the same as that of <code>query</code> and <code>value</code>, or it is a null pointer.</td>
        <td>FLOAT16, BFLOAT16, FLOAT</td>
        <td>ND</td>
        <td>[BSND]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>encoderValue</td>
        <td>Input</td>
        <td>Value in the attention mechanism, which comes from EncoderHiddenState.</td>
        <td>Its data type must be the same as that of <code>query</code> and <code>key</code>, or it is a null pointer.</td>
        <td>FLOAT16, BFLOAT16, FLOAT</td>
        <td>ND</td>
        <td>[BSND]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>normQueryWeight</td>
        <td>Input</td>
        <td>Affine transformation parameter of LayerNorm, which is applied to <code>query</code>.</td>
        <td>Optional. This parameter is required when <code>normType</code> is set to <code>2</code>.</td>
        <td>FLOAT16, BFLOAT16, FLOAT</td>
        <td>ND</td>
        <td>[D]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>normQueryBias</td>
        <td>Input</td>
        <td>Affine transformation parameter of LayerNorm, which is applied to <code>query</code>.</td>
        <td>Optional. This parameter is required when <code>normType</code> is set to <code>2</code>.</td>
        <td>FLOAT16, BFLOAT16, FLOAT</td>
        <td>ND</td>
        <td>[D]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>normKeyWeight</td>
        <td>Input</td>
        <td>Affine transformation parameter of LayerNorm, which is applied to <code>key</code>.</td>
        <td>Optional. This parameter is required when <code>normType</code> is set to <code>2</code>.</td>
        <td>FLOAT16, BFLOAT16, FLOAT</td>
        <td>ND</td>
        <td>[D]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>normKeyBias</td>
        <td>Input</td>
        <td>Affine transformation parameter of LayerNorm, which is applied to <code>key</code>.</td>
        <td>Optional. This parameter is required when <code>normType</code> is set to <code>2</code>.</td>
        <td>FLOAT16, BFLOAT16, FLOAT</td>
        <td>ND</td>
        <td>[D]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>normAddedQueryWeight</td>
        <td>Input</td>
        <td>Affine transformation parameter of LayerNorm, which is applied to <code>encoderQuery</code>.</td>
        <td>Optional. This parameter is required when <code>normAddedType</code> is set to <code>2</code>.</td>
        <td>FLOAT16, BFLOAT16, FLOAT</td>
        <td>ND</td>
        <td>[D]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>normAddedQueryBias</td>
        <td>Input</td>
        <td>Affine transformation parameter of LayerNorm, which is applied to <code>encoderQuery</code>.</td>
        <td>Optional. This parameter is required when <code>normAddedType</code> is set to <code>2</code>.</td>
        <td>FLOAT16, BFLOAT16, FLOAT</td>
        <td>ND</td>
        <td>[D]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>normAddedKeyWeight</td>
        <td>Input</td>
        <td>Affine transformation parameter of LayerNorm, which is applied to <code>encoderKey</code>.</td>
        <td>Optional. This parameter is required when <code>normAddedType</code> is set to <code>2</code>.</td>
        <td>FLOAT16, BFLOAT16, FLOAT</td>
        <td>ND</td>
        <td>[D]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>normAddedKeyBias</td>
        <td>Input</td>
        <td>Affine transformation parameter of LayerNorm, which is applied to <code>encoderKey</code>.</td>
        <td>Optional. This parameter is required when <code>normAddedType</code> is set to <code>2</code>.</td>
        <td>FLOAT16, BFLOAT16, FLOAT</td>
        <td>ND</td>
        <td>[D]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>ropeSin</td>
        <td>Input</td>
        <td>Sine encoding of RoPE.</td>
        <td>This parameter is required when <code>ropeType</code> is not 0.</td>
        <td>FLOAT16, BFLOAT16, FLOAT</td>
        <td>ND</td>
        <td>[SD]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>ropeCos</td>
        <td>Input</td>
        <td>Cosine encoding of RoPE.</td>
        <td>This parameter is required when <code>ropeType</code> is not 0.</td>
        <td>FLOAT16, BFLOAT16, FLOAT</td>
        <td>ND</td>
        <td>[SD]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>normType</td>
        <td>Attribute</td>
        <td>Regularization type applied to <code>query</code> and <code>key</code>. <code>0</code>: no regularization; <code>1</code>: LayerNorm; <code>2</code>: LayerNormAffine.</td>
        <td>None.</td>
        <td>int64</td>
        <td>Scalar</td>
        <td>Scalar</td>
        <td></td>
      </tr>
      <tr>
        <td>normAddedType</td>
        <td>Attribute</td>
        <td>Regularization type applied to <code>encoderQuery</code> and <code>encoderKey</code>. <code>0</code>: no regularization; <code>1</code>: LayerNorm; <code>2</code>: LayerNormAffine.</td>
        <td>None.</td>
        <td>int64</td>
        <td>Scalar</td>
        <td>Scalar</td>
        <td></td>
      </tr>
      <tr>
        <td>ropeType</td>
        <td>Attribute</td>
        <td>RoPE mode, which is of the int64 type. <code>0</code>: no RoPE; <code>1</code>: Interleave; <code>2</code>: Half.</td>
        <td>None.</td>
        <td>int64</td>
        <td>Scalar</td>
        <td>Scalar</td>
        <td></td>
      </tr>
      <tr>
        <td>concatOrder</td>
        <td>Attribute</td>
        <td>Concatenation order, which is of the int64 type. <code>0</code>: <code>query</code> first; <code>1</code>: <code>query</code> last.</td>
        <td>None.</td>
        <td>int64</td>
        <td>Scalar</td>
        <td>Scalar</td>
        <td></td>
      </tr>
      <tr>
        <td>eps</td>
        <td>Attribute</td>
        <td>Epsilon value in regularization.</td>
        <td>None.</td>
        <td>float32</td>
        <td>Scalar</td>
        <td>Scalar</td>
        <td></td>
      </tr>
      <tr>
        <td>isTraining</td>
        <td>Attribute</td>
        <td>Whether the phase is the training phase, which determines whether to output the values used for backward propagation.</td>
        <td>None.</td>
        <td>bool</td>
        <td>Scalar</td>
        <td>Scalar</td>
        <td></td>
      </tr>
      <tr>
        <td>queryOutput</td>
        <td>Output</td>
        <td>Output query in the attention mechanism.</td>
        <td>The data type must be the same as that of <code>query</code>.</td>
        <td>FLOAT16, BFLOAT16, FLOAT</td>
        <td>ND</td>
        <td>[BNSD]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>keyOutput</td>
        <td>Output</td>
        <td>Output key in the attention mechanism.</td>
        <td>The data type must be the same as that of <code>key</code>.</td>
        <td>FLOAT16, BFLOAT16, FLOAT</td>
        <td>ND</td>
        <td>[BNSD]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>valueOutput</td>
        <td>Output</td>
        <td>Output value in the attention mechanism.</td>
        <td>The data type must be the same as that of <code>value</code>.</td>
        <td>FLOAT16, BFLOAT16, FLOAT</td>
        <td>ND</td>
        <td>[BNSD]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>normQueryMean</td>
        <td>Output</td>
        <td>Output mean value of <code>query</code> in LayerNorm, which is used for backward propagation.</td>
        <td>This parameter is valid only when <code>normType</code> is not 0 and <code>isTraining</code> is <code>true</code>.</td>
        <td>FLOAT</td>
        <td>ND</td>
        <td>[BS]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>normQueryRstd</td>
        <td>Output</td>
        <td>Output standard deviation of <code>query</code> in LayerNorm, which is used for backward propagation.</td>
        <td>This parameter is valid only when <code>normType</code> is not 0 and <code>isTraining</code> is <code>true</code>.</td>
        <td>FLOAT</td>
        <td>ND</td>
        <td>[BS]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>normKeyMean</td>
        <td>Output</td>
        <td>Output mean value of <code>key</code> in LayerNorm, which is used for backward propagation.</td>
        <td>This parameter is valid only when <code>normType</code> is not 0 and <code>isTraining</code> is <code>true</code>.</td>
        <td>FLOAT</td>
        <td>ND</td>
        <td>[BS]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>normKeyRstd</td>
        <td>Output</td>
        <td>Output standard deviation of <code>key</code> in LayerNorm, which is used for backward propagation.</td>
        <td>This parameter is valid only when <code>normType</code> is not 0 and <code>isTraining</code> is <code>true</code>.</td>
        <td>FLOAT</td>
        <td>ND</td>
        <td>[BS]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>normEncoderQueryMean</td>
        <td>Output</td>
        <td>Output mean value of <code>encoderQuery</code> in LayerNorm, which is used for backward propagation.</td>
        <td>This parameter is valid only when <code>normAddedType</code> is not 0 and <code>isTraining</code> is <code>true</code>.</td>
        <td>FLOAT</td>
        <td>ND</td>
        <td>[BS]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>normEncoderQueryRstd</td>
        <td>Output</td>
        <td>Output standard deviation of <code>encoderQuery</code> in LayerNorm, which is used for backward propagation.</td>
        <td>This parameter is valid only when <code>normAddedType</code> is not 0 and <code>isTraining</code> is <code>true</code>.</td>
        <td>FLOAT</td>
        <td>ND</td>
        <td>[BS]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>normEncoderKeyMean</td>
        <td>Output</td>
        <td>Output mean value of <code>encoderKey</code> in LayerNorm, which is used for backward propagation.</td>
        <td>This parameter is valid only when <code>normAddedType</code> is not 0 and <code>isTraining</code> is <code>true</code>.</td>
        <td>FLOAT</td>
        <td>ND</td>
        <td>[BS]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>normEncoderKeyRstd</td>
        <td>Output</td>
        <td>Output standard deviation of <code>encoderKey</code> in LayerNorm, which is used for backward propagation.</td>
        <td>This parameter is valid only when <code>normAddedType</code> is not 0 and <code>isTraining</code> is <code>true</code>.</td>
        <td>FLOAT</td>
        <td>ND</td>
        <td>[BS]</td>
        <td>√</td>
      </tr>
    </tbody></table>

- **Returns**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API implements input parameter verification. The following errors may be thrown.
  <table style="undefined;table-layout: fixed;width: 1149px"><colgroup>
  <col style="width: 281px">
  <col style="width: 119px">
  <col style="width: 749px">
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
      <td>The input <code>query</code>, <code>key</code>, or <code>value</code> is a null pointer.</td>
    </tr>
  </tbody>
  </table>

## aclnnNormRopeConcat

- **Parameters**

  <table style="undifined;table-layout: fixed; width: 1149px">
  <colgroup>
    <col style="width: 281px">
    <col style="width: 119px">
    <col style="width: 749px">
  </colgroup>
  <tr>
    <th align="center">Name</th>
    <th align="center">Input/Output</th>
    <th align="center">Description</th>
  </tr>
  <tbody>
    <tr>
      <td>workspace</td>
      <td>Input</td>
      <td>Address of the workspace to be allocated on the device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace to be allocated on the device, which is obtained by calling the first-phase API <code>aclnnNormRopeConcatGetWorkspaceSize</code>.</td>
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

- The data types of `query`, `key`, `value`, `encoderQuery`, `encoderKey`, and `encoderValue` must be the same.
- The value of `headDim` must be an even number in the range of [1, 1024].
- The value of `seqRope` must be in the range of [1, Min(`seqQuery` + `seqEncoderQuery`, `seqKey` + `seqEncoderKey`)].

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

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
      // (Boilerplate) Initialize AscendCL.
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
      // Call aclrtMalloc to allocate memory on the device.
      auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
      // Call aclrtMemcpy to copy host data to the device memory.
      ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

      // Compute the strides of the contiguous result.
      std::vector<int64_t> strides(shape.size(), 1);
      for (int64_t i = shape.size() - 2; i >= 0; i--) {
          strides[i] = shape[i + 1] * strides[i + 1];
      }

      // Call aclCreateTensor to create an aclTensor.
      *result = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                                shape.data(), shape.size(), *deviceAddr);
      return 0;
  }

  int main()
  {
      // 1. (Boilerplate) Initialize the device, context, and stream. For details, see the list of external AscendCL APIs.
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

      std::vector<int64_t> queryOutputShape = {batchSize, headNum, querySeq * 2, headDim};
      std::vector<int64_t> keyOutputShape = {batchSize, headNum, querySeq * 2, headDim};
      std::vector<int64_t> valueOutputShape = {batchSize, headNum, querySeq * 2, headDim};

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

      std::vector<float> queryOutputHostData(batchSize * headNum * querySeq + encoderQuerySeq * headDim, 0.0);
      std::vector<float> keyOutputHostData(batchSize * headNum * keySeq + encoderKeySeq * headDim, 0.0);
      std::vector<float> valueOutputHostData(batchSize * headNum * valueSeq + encoderValueSeq * headDim, 0.0);

      std::vector<float> encoderQueryHostData(batchSize * headNum * encoderQuerySeq * headDim, 4.0);
      std::vector<float> encoderKeyHostData(batchSize * headNum * encoderKeySeq * headDim, 5.0);
      std::vector<float> encoderValueHostData(batchSize * headNum * encoderKeySeq * headDim, 6.0);

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
      ret = CreateAclTensor(normQueryMeanHostData, normQueryMeanShape, &normQueryMeanDeviceAddr, aclDataType::ACL_FLOAT,
                            &normQueryMean);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      ret = CreateAclTensor(normQueryRstdHostData, normQueryRstdShape, &normQueryRstdDeviceAddr, aclDataType::ACL_FLOAT,
                            &normQueryRstd);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      ret = CreateAclTensor(normKeyWeightHostData, normKeyWeightShape, &normKeyWeightDeviceAddr, aclDataType::ACL_FLOAT,
                            &normKeyWeight);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      ret = CreateAclTensor(normKeyMeanHostData, normKeyMeanShape, &normKeyMeanDeviceAddr, aclDataType::ACL_FLOAT,
                            &normKeyMean);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      ret = CreateAclTensor(normKeyRstdHostData, normKeyRstdShape, &normKeyRstdDeviceAddr, aclDataType::ACL_FLOAT,
                            &normKeyRstd);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      ret = CreateAclTensor(normAddedQueryWeightHostData, normAddedQueryWeightShape, &normAddedQueryWeightDeviceAddr,
                            aclDataType::ACL_FLOAT, &normAddedQueryWeight);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      ret = CreateAclTensor(normAddedQueryMeanHostData, normAddedQueryMeanShape, &normAddedQueryMeanDeviceAddr,
                            aclDataType::ACL_FLOAT, &normAddedQueryMean);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      ret = CreateAclTensor(normAddedQueryRstdHostData, normAddedQueryRstdShape, &normAddedQueryRstdDeviceAddr,
                            aclDataType::ACL_FLOAT, &normAddedQueryRstd);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      ret = CreateAclTensor(normAddedKeyWeightHostData, normAddedKeyWeightShape, &normAddedKeyWeightDeviceAddr,
                            aclDataType::ACL_FLOAT, &normAddedKeyWeight);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      ret = CreateAclTensor(normAddedKeyMeanHostData, normAddedKeyMeanShape, &normAddedKeyMeanDeviceAddr,
                            aclDataType::ACL_FLOAT, &normAddedKeyMean);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      ret = CreateAclTensor(normAddedKeyRstdHostData, normAddedKeyRstdShape, &normAddedKeyRstdDeviceAddr,
                            aclDataType::ACL_FLOAT, &normAddedKeyRstd);
      CHECK_RET(ret == ACL_SUCCESS, return ret);

      // 3. Call the CANN operator library API. Change the API name to the actual one.
      uint64_t workspaceSize = 0;
      aclOpExecutor *executor;
      bool isTraing = false;
      // Call the first-phase API of aclnnGeGluBackward.
      ret = aclnnNormRopeConcatGetWorkspaceSize(
          query, key, value, encoderQuery, encoderKey, encoderValue, normQueryWeight, normQueryBias, normKeyWeight,
          normKeyBias, normAddedQueryWeight, normAddedQueryBias, normAddedKeyWeight, normAddedKeyBias, ropeSin, ropeCos,
          normType, normAddedType, ropeType, concatOrder, eps, isTraing, queryOutput, keyOutput, valueOutput,
          normQueryMean, normQueryRstd, normKeyMean, normKeyRstd, normAddedQueryMean, normAddedQueryRstd,
          normAddedKeyMean, normAddedKeyRstd, &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNormRopeConcatGetWorkspaceSize failed. ERROR: %d\n", ret);
                return ret);
      // Allocate device memory based on workspaceSize computed by the first-phase API.
      void *workspaceAddr = nullptr;
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
      }
      // Call the second-phase API of aclnnGeGluBackward.
      ret = aclnnNormRopeConcat(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNormRopeConcat failed. ERROR: %d\n", ret); return ret);

      // 4. (Boilerplate) Wait until the task execution is complete.
      ret = aclrtSynchronizeStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

      // 5. Obtain the output value and copy the result from the device memory to the host. Modify the code based on the API definition.
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


      // 6. Release aclTensors and aclScalars. Modify the code based on the API definition.
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

      // 7. Release device resources. Modify the code based on the API definition.
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
