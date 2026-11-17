# aclnnRecurrentKda

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-21T12:56:09.515Z -->

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

- Description: aclnnRecurrentKda is the recurrent implementation of the linear attention forward computation API based on the Gated Delta Rule, designed for short-sequence use cases such as decode / MTP (Multi-Token Prediction). The API advances the recurrent state $S_t$ token by token, supports variable-length inputs (with `cuSeqlensOptional`) and MTP slot reuse (with `ssmStateIndicesOptional` and `numAcceptedTokensOptional`), and optionally performs numerical stabilization in the kernel, such as interpreting `gate` as raw gate, applying L2 normalization to q/k, and applying sigmoid to beta.
- Formula:

  Assume the internal state matrix participates in the computation in the `[V_dim, K_dim]` layout. When `useQkL2normInKernel=true`, the operator first applies L2 normalization to each token's $q_t$ and $k_t$; then only $q_t$ is multiplied by the scaling factor `scale`, yielding $\bar q_t$, $\bar k_t$.

  When `useGateInKernel=false`, the input `gate` is already the step log gate, i.e. $\ell_t=gate_t$. When `useGateInKernel=true`, let $x_t=gate_t+dtBias$ (no bias is added when `dtBiasOptional` is not passed) and $a=\exp(A_{\log})$. Then:

  $$
  \ell_t =
  \begin{cases}
  lowerBound\cdot\sigma(a x_t), & safeGate=true \\
  -a\cdot\operatorname{softplus}(x_t), & safeGate=false
  \end{cases},
  \qquad
  d_t=\exp(\ell_t)
  $$

  The effective value of beta is:

  $$
  \bar\beta_t =
  \begin{cases}
  \beta_t, & useBetaSigmoidInKernel=false \\
  \sigma(\beta_t), & useBetaSigmoidInKernel=true,\ allowNegEigval=false \\
  2\sigma(\beta_t), & useBetaSigmoidInKernel=true,\ allowNegEigval=true
  \end{cases}
  $$

  The recurrent update of each token is:

  $$
  \begin{aligned}
  \bar S_t &= S_{t-1}\odot d_t \\
  \Delta_t &= \bar\beta_t\left(v_t-\bar S_t\bar k_t\right) \\
  S_t &= \bar S_t+\Delta_t\bar k_t^T \\
  o_t &= S_t\bar q_t
  \end{aligned}
  $$

  where $S_{t-1},S_t \in R^{V_{dim}\times K_{dim}}$, $q_t,k_t\in R^{K_{dim}}$, $v_t,o_t\in R^{V_{dim}}$. $d_t$ is broadcast along the K dimension of the state matrix. Multiple value heads can share the same query/key head, and $H_v$ must be divisible by $H_k$.

- Symbol Description

  | Symbol      | Meaning |
  | ----------- | ----------- |
  | B           | Batch Size, the batch size of input samples. |
  | T           | Token Capacity, B*T in BSND and the first input dimension in TND; the number of valid tokens can be less than T. |
  | H_k         | Number of Query/Key heads. |
  | H_v         | Number of Value/Gate heads (H_v must be divisible by H_k). |
  | K_dim       | Dimension of each Key head. |
  | V_dim       | Dimension of each Value head. |
  | seq_num     | Number of logical sequences (equals B for fixed length, and the length of `cuSeqlensOptional` minus 1 for variable length). |
  | scale       | Scaling coefficient, usually K_dim**-0.5. |
  | gate        | gate tensor. When `useGateInKernel=true`, it is the raw gate input, which is converted to the step log gate $\ell_t$ inside the operator; when `useGateInKernel=false`, it is the step log gate precomputed externally. |
  | $d_t$       | Recurrent decay coefficient, $d_t=\exp(\ell_t)$, broadcast along the K dimension of the state matrix. |
  | A_log       | Raw gate branch parameter. The operator uses $\exp(A_{\log})$ in the gate computation. |
  | dtBias      | Raw gate bias. |
  | beta        | Delta update coefficient. |
  | $S_t$       | Recurrent state matrix, stored as `[state_capacity, H_v, V_dim, K_dim]` in the `stateVFirst=True` layout, and as `[state_capacity, H_v, K_dim, V_dim]` in the `stateVFirst=False` layout. |
  | $\bar S_t$  | Decayed state, $\bar S_t=S_{t-1}\odot d_t$. |
  | $\Delta_t$  | Delta update amount, $\Delta_t=\bar\beta_t(v_t-\bar S_t\bar k_t)$. |

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). First, call the "aclnnRecurrentKdaGetWorkspaceSize" API to obtain the workspace size required for the computation and the executor that contains the operator computation process. Then, call the "aclnnRecurrentKda" API to perform the computation.

```Cpp
aclnnStatus aclnnRecurrentKdaGetWorkspaceSize(
    const aclTensor     *query,
    const aclTensor     *key,
    const aclTensor     *value,
    const aclTensor     *gate,
    const aclTensor     *beta,
    aclTensor           *initialStateRef,
    const aclTensor     *cuSeqlensOptional,
    const aclTensor     *ssmStateIndicesOptional,
    const aclTensor     *aLogOptional,
    const aclTensor     *dtBiasOptional,
    const aclTensor     *numAcceptedTokensOptional,
    const char          *layout,
    double               scale,
    bool                 outputFinalState,
    bool                 inplaceFinalState,
    bool                 useQkL2normInKernel,
    bool                 useGateInKernel,
    bool                 useBetaSigmoidInKernel,
    bool                 allowNegEigval,
    bool                 safeGate,
    double               lowerBound,
    bool                 stateVFirst,
    const aclTensor     *attnOut,
    const aclTensor     *finalState,
    uint64_t            *workspaceSize,
    aclOpExecutor      **executor)
```

```Cpp
aclnnStatus aclnnRecurrentKda(
    void             *workspace,
    uint64_t          workspaceSize,
    aclOpExecutor    *executor,
    aclrtStream       stream)
```

## aclnnRecurrentKdaGetWorkspaceSize

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1494px"><colgroup>
  <col style="width: 146px">
  <col style="width: 110px">
  <col style="width: 301px">
  <col style="width: 500px">
  <col style="width: 328px">
  <col style="width: 101px">
  <col style="width: 400px">
  <col style="width: 146px">
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
      <td>query (aclTensor)</td>
      <td>Input</td>
      <td>Query tensor.</td>
      <td>With layout BSND: [B,T,H_k,K_dim] or TND: [T,H_k,K_dim].</td>
      <td>BFLOAT16</td>
      <td>ND</td>
      <td>[B,T,H_k,K_dim] or [T,H_k,K_dim]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>key (aclTensor)</td>
      <td>Input</td>
      <td>Key tensor.</td>
      <td>Same shape and data type as query.</td>
      <td>BFLOAT16</td>
      <td>ND</td>
      <td>[B,T,H_k,K_dim] or [T,H_k,K_dim]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>value (aclTensor)</td>
      <td>Input</td>
      <td>Value tensor.</td>
      <td>Depending on layout: BSND: [B,T,H_v,V_dim] or TND: [T,H_v,V_dim].</td>
      <td>BFLOAT16</td>
      <td>ND</td>
      <td>[B,T,H_v,V_dim] or [T,H_v,V_dim]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gate (aclTensor)</td>
      <td>Input</td>
      <td>Precomputed step log gate or raw gate.</td>
      <td>Layout is BSND: [B,T,H_v,K_dim] or TND: [T,H_v,K_dim].</td>
      <td>FLOAT, BFLOAT16, FLOAT16</td>
      <td>ND</td>
      <td>[B,T,H_v,K_dim] or [T,H_v,K_dim]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>beta (aclTensor)</td>
      <td>Input</td>
      <td>Delta update coefficient.</td>
      <td>BSND: [B,T,H_v] or TND: [T,H_v]</td>
      <td>FLOAT, BFLOAT16, FLOAT16</td>
      <td>ND</td>
      <td>[B,T,H_v] or [T,H_v]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>initialStateRef (aclTensor)</td>
      <td>Input/Output</td>
      <td>Initial state or state pool.</td>
      <td>Shape: [state_capacity,H_v,V_dim,K_dim] when stateVFirst=true; Shape: [state_capacity,H_v,K_dim,V_dim] when stateVFirst=false. state_capacity must be equal to seq_num when ssmStateIndicesOptional is not passed.</td>
      <td>BFLOAT16, FLOAT</td>
      <td>ND</td>
      <td>[state_capacity,H_v,V_dim,K_dim] or [state_capacity,H_v,K_dim,V_dim]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>cuSeqlensOptional (aclTensor)</td>
      <td>Input</td>
      <td>Cumulative token offset of variable-length sequences.</td>
      <td>Optional; a device-side 1-D Tensor, Shape: [seq_num+1], with a length of at least 2.</td>
      <td>INT32, INT64</td>
      <td>ND</td>
      <td>[seq_num+1]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>ssmStateIndicesOptional (aclTensor)</td>
      <td>Input</td>
      <td>Index of the state slot corresponding to each token.</td>
      <td>Optional; supports a packed 1-D shape [>=T] or a speculative 2-D shape [seq_num,max_step], and numAcceptedTokensOptional can be omitted.</td>
      <td>INT32, INT64</td>
      <td>ND</td>
      <td>[>=T] or [seq_num,max_step]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>aLogOptional (aclTensor)</td>
      <td>Input</td>
      <td>A_log parameter.</td>
      <td>Required when useGateInKernel=True; Shape: [H_v].</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>[H_v]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>dtBiasOptional (aclTensor)</td>
      <td>Input</td>
      <td>Raw gate bias.</td>
      <td>Shape: [H_v*K_dim] or [H_v,K_dim].</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>[H_v*K_dim] or [H_v,K_dim]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>numAcceptedTokensOptional (aclTensor)</td>
      <td>Input</td>
      <td>Number of accepted tokens for each sequence.</td>
      <td>Optional; if passed, ssmStateIndicesOptional must also be passed; Shape: [seq_num], each value must be in [1,seq_len] of the corresponding sequence and selects the initial State slot.</td>
      <td>INT32, INT64</td>
      <td>ND</td>
      <td>[seq_num]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>layout (char)</td>
      <td>Input</td>
      <td>Data Layout format of query, key, and value.</td>
      <td>Enum values: BSND | TND.</td>
      <td>STRING</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scale (double)</td>
      <td>Input</td>
      <td>Scaling factor.</td>
      <td>Multiplied to query, usually K_dim**-0.5.</td>
      <td>DOUBLE</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>outputFinalState (bool)</td>
      <td>Input</td>
      <td>Whether to output a valid final state.</td>
      <td>When true, finalState returns a valid final state; whether it is updated in place is determined by inplaceFinalState.</td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>inplaceFinalState (bool)</td>
      <td>Input</td>
      <td>Whether the state is updated in place.</td>
      <td>If true, the final state is written back to initialStateRef; if false, initialStateRef is not updated.</td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>useQkL2normInKernel (bool)</td>
      <td>Input</td>
      <td>Whether to perform L2 normalization on q/k.</td>
      <td>-</td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>useGateInKernel (bool)</td>
      <td>Input</td>
      <td>Whether to interpret gate as a raw gate.</td>
      <td>If true, aLogOptional is required.</td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>useBetaSigmoidInKernel (bool)</td>
      <td>Input</td>
      <td>Whether to compute sigmoid(beta) in the kernel.</td>
      <td>-</td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>allowNegEigval (bool)</td>
      <td>Input</td>
      <td>Whether to multiply by 2 after beta sigmoid.</td>
      <td>-</td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>safeGate (bool)</td>
      <td>Input</td>
      <td>Switch for the safe branch of the raw gate.</td>
      <td>Used together with useGateInKernel.</td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>lowerBound (double)</td>
      <td>Input</td>
      <td>Lower bound of the safe gate.</td>
      <td>Effective when safeGate=True. Value range: [-5,0).</td>
      <td>DOUBLE</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>stateVFirst (bool)</td>
      <td>Input</td>
      <td>State layout flag.</td>
      <td>Supports true ([state_capacity,H_v,V_dim,K_dim]) / false ([state_capacity,H_v,K_dim,V_dim]); defaults to false.</td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>attnOut (aclTensor)</td>
      <td>Output</td>
      <td>Recurrent KDA output.</td>
      <td>Same shape as value.</td>
      <td>BFLOAT16</td>
      <td>ND</td>
      <td>[B,T,H_v,V_dim] or [T,H_v,V_dim]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>finalState (aclTensor)</td>
      <td>Output</td>
      <td>Final state.</td>
      <td>The shape and data type must be the same as those of initialStateRef; the content is valid when outputFinalState=true.</td>
      <td>BFLOAT16, FLOAT</td>
      <td>ND</td>
      <td>[state_capacity,H_v,V_dim,K_dim] or [state_capacity,H_v,K_dim,V_dim]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize (uint64_t*)</td>
      <td>Output</td>
      <td>Returns the workspace size to be allocated on the Device side.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor (aclOpExecutor)</td>
      <td>Output</td>
      <td>Returns the op executor, which contains the operator computation process.</td>
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

  <table style="undefined;table-layout: fixed;width: 1155px"><colgroup>
    <col style="width: 319px">
    <col style="width: 144px">
    <col style="width: 671px">
    </colgroup>
        <thead>
            <th>Return Value</th>
            <th>Error Code</th>
            <th>Description</th>
        </thead>
        <tbody>
            <tr>
                <td>ACLNN_ERR_PARAM_NULLPTR</td>
                <td>161001</td>
                <td>If a passed parameter is a required input, output, or required attribute and is a null pointer, 161001 is returned.</td>
            </tr>
            <tr>
                <td rowspan="20">ACLNN_ERR_PARAM_INVALID</td>
                <td rowspan="20">161002</td>
            </tr>
            <tr><td>query/key/value dtype is not {BFLOAT16}.</td></tr>
            <tr><td>gate/beta dtype is not in {FLOAT, BFLOAT16, FLOAT16}.</td></tr>
            <tr><td>initialState/finalState dtype is not in {BFLOAT16, FLOAT}, or the two are inconsistent.</td></tr>
            <tr><td>cuSeqlensOptional/ssmStateIndicesOptional/numAcceptedTokensOptional dtype is not in {INT32, INT64}.</td></tr>
            <tr><td>The dtype of aLogOptional/dtBiasOptional is not FLOAT.</td></tr>
            <tr><td>layout is not in the {BSND, TND} set or does not match rank.</td></tr>
            <tr><td>K_dim is not equal to 128; V_dim is not in {128, 256}.</td></tr>
            <tr><td>H_v is not divisible by H_k.</td></tr>
            <tr><td>The shapes of query/key are inconsistent, or the shape relationships of query/key/value/gate/beta do not meet the requirements of the corresponding layout.</td></tr>
            <tr><td>cuSeqlensOptional is not a one-dimensional tensor, or its length is less than 2.</td></tr>
            <tr><td>The shapes of initialState/finalState do not match stateVFirst; state_capacity is not equal to seq_num when ssmStateIndicesOptional is not passed.</td></tr>
            <tr><td>ssmStateIndicesOptional is neither a one-dimensional tensor whose length is greater than or equal to token capacity, nor a two-dimensional tensor of shape [seq_num,max_step].</td></tr>
            <tr><td>The length of numAcceptedTokensOptional is not equal to seq_num.</td></tr>
            <tr><td>aLogOptional is null when useGateInKernel=true.</td></tr>
            <tr><td>aLogOptional/dtBiasOptional is passed or safeGate=true is set when useGateInKernel=false.</td></tr>
            <tr><td>The shape of aLogOptional is not [H_v].</td></tr>
            <tr><td>The shape of dtBiasOptional is not [H_v*K_dim] or [H_v,K_dim].</td></tr>
            <tr><td>lowerBound is not in [-5,0) when safeGate=true.</td></tr>
            <tr><td>numAcceptedTokensOptional is not provided together with ssmStateIndicesOptional.</td></tr>
        </tbody>
    </table>

## aclnnRecurrentKda

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 953px"><colgroup>
  <col style="width: 173px">
  <col style="width: 112px">
  <col style="width: 668px">
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
      <td>Size of the workspace allocated on the Device side, obtained by the first-phase API aclnnRecurrentKdaGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>op executor, which contains the operator computation flow.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream that specifies the task to execute.</td>
    </tr>
  </tbody>
  </table>
- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation: aclnnRecurrentKda defaults to a deterministic implementation.
- Dense and variable-length sequences are processed recurrently in fixed-size token windows.
- The data types of query, key, and value must be the same, and only BFLOAT16 is supported; out has the same dtype as value.
- gate and beta support FLOAT, BFLOAT16, and FLOAT16 (internally converted to FLOAT for computation).
- initialState and finalState must be of the same data type, either BFLOAT16 or FLOAT; out has the same dtype as value. finalState must be provided, and its content is valid only when outputFinalState=true.
- cuSeqlensOptional, ssmStateIndicesOptional, and numAcceptedTokensOptional support INT32 or INT64, and are read by the kernel based on the actual dtype.
- aLogOptional and dtBiasOptional must be FLOAT.
- layout supports only BSND and TND (uppercase required).
- K_dim supports only 128, and V_dim supports only 128 or 256.
- H_v must be divisible by H_k.
- stateVFirst supports true and false (defaults to false): if true, the shapes of initialState and finalState must be `[state_capacity, H_v, V_dim, K_dim]`; if false, they must be `[state_capacity, H_v, K_dim, V_dim]`. If ssmStateIndicesOptional is not passed, state_capacity must be equal to the number of logical sequences; if it is passed, state_capacity can differ from seq_num, and valid slots are selected by indices.
- The first element of cuSeqlensOptional must be 0, and each offset must be non-decreasing and must not exceed the input token capacity. The last element can be less than the token capacity. In this case, the padding tail output is not guaranteed to be valid and the state is not updated.
- ssmStateIndicesOptional supports a 1D tensor whose length is greater than or equal to the token capacity, or a 2D tensor with the shape of [seq_num,max_step]; in the 2D form, max_step must cover the length of each actual sequence, and each index must be in [0,state_capacity). numAcceptedTokensOptional must be a 1D tensor with a length equal to seq_num, and each value must be in [1,seq_len] of the corresponding sequence; it depends on ssmStateIndicesOptional, but ssmStateIndicesOptional can be provided alone.
- aLogOptional is required when useGateInKernel=true; when useGateInKernel=false, an error is reported if aLogOptional or dtBiasOptional is passed in or safeGate=true is set.
- The shape of aLogOptional must be [H_v]; the shape of dtBiasOptional must be [H_v*K_dim] or [H_v,K_dim].
- When safeGate=true, lowerBound must be in the range [-5, 0).
- When inplaceFinalState=true, the final state is written back to initialState; when outputFinalState=true, finalState returns a valid final state. The two switches can be combined independently.
- query/key/value/gate/beta and the optional metadata tensors support non-contiguous formats, and the API internally calls Contiguous for processing. initialState/finalState directly support non-contiguous views, but the internal 2D state matrix must be dense: stride[3]=1 and stride[2]=shape[3]; the outer strides of slot/head must not cause overlapping address ranges.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include "securec.h"
#include "acl/acl.h"
#include "aclnnop/aclnn_recurrent_kda.h"

using namespace std;

namespace {

#define CHECK_RET(cond) ((cond) ? true :(false))

#define LOG_PRINT(message, ...)     \
  do {                              \
    (void)printf(message, ##__VA_ARGS__); \
  } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
  int64_t shapeSize = 1;
  for (auto i : shape) {
    shapeSize *= i;
  }
  return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream) {
  auto ret = aclInit(nullptr);
  if (!CHECK_RET(ret == ACL_SUCCESS)) {
    LOG_PRINT("aclInit failed. ERROR: %d\n", ret);
    return ret;
  }
  ret = aclrtSetDevice(deviceId);
  if (!CHECK_RET(ret == ACL_SUCCESS)) {
    LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret);
    return ret;
  }
  ret = aclrtCreateStream(stream);
  if (!CHECK_RET(ret == ACL_SUCCESS)) {
    LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret);
    return ret;
  }
  return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor) {
  auto size = GetShapeSize(shape) * sizeof(T);
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  if (!CHECK_RET(ret == ACL_SUCCESS)) {
    LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret);
    return ret;
  }

  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  if (!CHECK_RET(ret == ACL_SUCCESS)) {
    LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret);
    return ret;
  }

  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

struct TensorResources {
    void* queryDeviceAddr = nullptr;
    void* keyDeviceAddr = nullptr;
    void* valueDeviceAddr = nullptr;
    void* gateDeviceAddr = nullptr;
    void* betaDeviceAddr = nullptr;
    void* initialStateDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    void* finalStateDeviceAddr = nullptr;

    aclTensor* queryTensor = nullptr;
    aclTensor* keyTensor = nullptr;
    aclTensor* valueTensor = nullptr;
    aclTensor* gateTensor = nullptr;
    aclTensor* betaTensor = nullptr;
    aclTensor* initialStateTensor = nullptr;
    aclTensor* outTensor = nullptr;
    aclTensor* finalStateTensor = nullptr;
};

int InitializeTensors(TensorResources& resources) {
    // BSND layout: q/k [B,T,H_k,K], v [B,T,H_v,V], gate [B,T,H_v,K], beta [B,T,H_v]
    // state [seq_num,H_v,V,K]
    int64_t B = 1;
    int64_t T = 1;
    int64_t Hk = 2;
    int64_t Hv = 2;
    int64_t K = 128;
    int64_t V = 128;

    std::vector<int64_t> queryShape = {B, T, Hk, K};
    std::vector<int64_t> keyShape = {B, T, Hk, K};
    std::vector<int64_t> valueShape = {B, T, Hv, V};
    std::vector<int64_t> gateShape = {B, T, Hv, K};
    std::vector<int64_t> betaShape = {B, T, Hv};
    std::vector<int64_t> initialStateShape = {B, Hv, V, K};
    std::vector<int64_t> outShape = {B, T, Hv, V};
    std::vector<int64_t> finalStateShape = {B, Hv, V, K};

    int64_t querySize = GetShapeSize(queryShape);
    int64_t keySize = GetShapeSize(keyShape);
    int64_t valueSize = GetShapeSize(valueShape);
    int64_t gateSize = GetShapeSize(gateShape);
    int64_t betaSize = GetShapeSize(betaShape);
    int64_t initialStateSize = GetShapeSize(initialStateShape);
    int64_t outSize = GetShapeSize(outShape);
    int64_t finalStateSize = GetShapeSize(finalStateShape);

    std::vector<aclBfloat16> queryHostData(querySize, 0);
    std::vector<aclBfloat16> keyHostData(keySize, 0);
    std::vector<aclBfloat16> valueHostData(valueSize, 0);
    std::vector<float> gateHostData(gateSize, 0);
    std::vector<float> betaHostData(betaSize, 0);
    std::vector<float> initialStateHostData(initialStateSize, 0);
    std::vector<aclBfloat16> outHostData(outSize, 0);
    std::vector<float> finalStateHostData(finalStateSize, 0);

    // Create query aclTensor.
    int ret = CreateAclTensor(queryHostData, queryShape, &resources.queryDeviceAddr,
                             aclDataType::ACL_BF16, &resources.queryTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
      return ret;
    }

    // Create key aclTensor.
    ret = CreateAclTensor(keyHostData, keyShape, &resources.keyDeviceAddr,
                         aclDataType::ACL_BF16, &resources.keyTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
      return ret;
    }

    // Create value aclTensor.
    ret = CreateAclTensor(valueHostData, valueShape, &resources.valueDeviceAddr,
                         aclDataType::ACL_BF16, &resources.valueTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
      return ret;
    }

    // Create gate aclTensor.
    ret = CreateAclTensor(gateHostData, gateShape, &resources.gateDeviceAddr,
                         aclDataType::ACL_FLOAT, &resources.gateTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
      return ret;
    }

    // Create beta aclTensor.
    ret = CreateAclTensor(betaHostData, betaShape, &resources.betaDeviceAddr,
                         aclDataType::ACL_FLOAT, &resources.betaTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
      return ret;
    }

    // Create initialState aclTensor.
    ret = CreateAclTensor(initialStateHostData, initialStateShape, &resources.initialStateDeviceAddr,
                         aclDataType::ACL_FLOAT, &resources.initialStateTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
      return ret;
    }

    // Create out aclTensor.
    ret = CreateAclTensor(outHostData, outShape, &resources.outDeviceAddr,
                         aclDataType::ACL_BF16, &resources.outTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
      return ret;
    }

    // Create finalState aclTensor.
    ret = CreateAclTensor(finalStateHostData, finalStateShape, &resources.finalStateDeviceAddr,
                         aclDataType::ACL_FLOAT, &resources.finalStateTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
      return ret;
    }

    return ACL_SUCCESS;
}

int ExecuteRecurrentKda(TensorResources& resources, aclrtStream stream,
                        void** workspaceAddr, uint64_t* workspaceSize) {
    constexpr const char layoutStr[] = "BSND";
    double scale = 1.0 / sqrt(128.0);
    bool outputFinalState = true;
    bool inplaceFinalState = true;
    bool useQkL2normInKernel = false;
    bool useGateInKernel = false;
    bool useBetaSigmoidInKernel = false;
    bool allowNegEigval = false;
    bool safeGate = false;
    double lowerBound = -5.0;
    bool stateVFirst = true;
    aclOpExecutor* executor;

    int ret = aclnnRecurrentKdaGetWorkspaceSize(
        resources.queryTensor, resources.keyTensor, resources.valueTensor,
        resources.gateTensor, resources.betaTensor, resources.initialStateTensor,
        nullptr, nullptr, nullptr, nullptr, nullptr,
        layoutStr, scale, outputFinalState, inplaceFinalState, useQkL2normInKernel, useGateInKernel,
        useBetaSigmoidInKernel, allowNegEigval, safeGate, lowerBound, stateVFirst,
        resources.outTensor, resources.finalStateTensor, workspaceSize, &executor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclnnRecurrentKdaGetWorkspaceSize failed. ERROR: %d\n", ret);
        return ret;
    }

    if (*workspaceSize > 0ULL) {
        ret = aclrtMalloc(workspaceAddr, *workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (!CHECK_RET(ret == ACL_SUCCESS)) {
            LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
            return ret;
        }
    }

    ret = aclnnRecurrentKda(*workspaceAddr, *workspaceSize, executor, stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclnnRecurrentKda failed. ERROR: %d\n", ret);
        return ret;
    }

    return ACL_SUCCESS;
}

int PrintOutResult(std::vector<int64_t> &shape, void** deviceAddr) {
  auto size = GetShapeSize(shape);
  std::vector<aclBfloat16> resultData(size, 0);
  auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]),
                         *deviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret);
        return ret;
  }
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("mean result[%ld] is: %f\n", i, aclBfloat16ToFloat(resultData[i]));
  }
  return ACL_SUCCESS;
}

void CleanupResources(TensorResources& resources, void* workspaceAddr,
                     aclrtStream stream, int32_t deviceId) {
    if (resources.queryTensor) {
      aclDestroyTensor(resources.queryTensor);
    }
    if (resources.keyTensor) {
      aclDestroyTensor(resources.keyTensor);
    }
    if (resources.valueTensor) {
      aclDestroyTensor(resources.valueTensor);
    }
    if (resources.gateTensor) {
      aclDestroyTensor(resources.gateTensor);
    }
    if (resources.betaTensor) {
      aclDestroyTensor(resources.betaTensor);
    }
    if (resources.initialStateTensor) {
      aclDestroyTensor(resources.initialStateTensor);
    }
    if (resources.outTensor) {
      aclDestroyTensor(resources.outTensor);
    }
    if (resources.finalStateTensor) {
      aclDestroyTensor(resources.finalStateTensor);
    }

    if (resources.queryDeviceAddr) {
      aclrtFree(resources.queryDeviceAddr);
    }
    if (resources.keyDeviceAddr) {
      aclrtFree(resources.keyDeviceAddr);
    }
    if (resources.valueDeviceAddr) {
      aclrtFree(resources.valueDeviceAddr);
    }
    if (resources.gateDeviceAddr) {
      aclrtFree(resources.gateDeviceAddr);
    }
    if (resources.betaDeviceAddr) {
      aclrtFree(resources.betaDeviceAddr);
    }
    if (resources.initialStateDeviceAddr) {
      aclrtFree(resources.initialStateDeviceAddr);
    }
    if (resources.outDeviceAddr) {
      aclrtFree(resources.outDeviceAddr);
    }
    if (resources.finalStateDeviceAddr) {
      aclrtFree(resources.finalStateDeviceAddr);
    }

    if (workspaceAddr) {
      aclrtFree(workspaceAddr);
    }
    if (stream) {
      aclrtDestroyStream(stream);
    }

    aclrtResetDevice(deviceId);
    aclFinalize();
}

} // namespace

int main() {

    int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    TensorResources resources = {};
    void* workspaceAddr = nullptr;
    uint64_t workspaceSize = 0;
    std::vector<int64_t> outShape = {1, 1, 2, 128};
    int ret = ACL_SUCCESS;

    // 1. Initialize device and stream
    ret = Init(deviceId, &stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("Init acl failed. ERROR: %d\n", ret);
        return ret;
    }

    // 2. Initialize tensors
    ret = InitializeTensors(resources);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        CleanupResources(resources, workspaceAddr, stream, deviceId);
        return ret;
    }

    // 3. Execute the operation
    ret = ExecuteRecurrentKda(resources, stream, &workspaceAddr, &workspaceSize);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        CleanupResources(resources, workspaceAddr, stream, deviceId);
        return ret;
    }

    // 4. Synchronize stream
    ret = aclrtSynchronizeStream(stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
        CleanupResources(resources, workspaceAddr, stream, deviceId);
        return ret;
    }

    // 5. Process results
    printf("-----------out output -----------\n");
    PrintOutResult(outShape, &resources.outDeviceAddr);

    // 6. Cleanup resources
    CleanupResources(resources, workspaceAddr, stream, deviceId);
    return 0;
}
```
