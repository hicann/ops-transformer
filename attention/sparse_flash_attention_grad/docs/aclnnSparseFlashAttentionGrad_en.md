# aclnnSparseFlashAttentionGrad

## Supported Products

| Product   | Supported |
|:----------------------------|:-----------:|
|<term>Atlas A3 training products/Atlas A3 inference products</term>|      √     |
|<term>Atlas A2 training products/Atlas A2 inference products</term>|      √     |
|<term>Atlas 200I/500 A2 inference products</term>|      ×     |
|<term>Atlas inference products</term>|      ×     |
|<term>Atlas training products</term>|      ×     |

## Function

- **Description**: Rearrange the data of size `selectedBlockSize` based on `topkIndices` for key and value, then compute the backward output of attention in the training scenario.

- **Formula**: Based on the passed `topkIndice`, select `selectedBlockCount` pieces of data of size `selectedBlockSize` from `keyIn` and `value` for reordering. The formula is as follows:

  $$
   selectedKey\text{ }=\text{ }Gather \left( key,topkIndices \left[ i \left]  \left) ,\text{ }0\text{ } < =i < \text{ }selectBlockCount\right. \right. \right. \right.
  $$

  $$
   selectedValue\text{ }=\text{ }Gather \left( value,topkIndices \left[ i \left]  \left) ,\text{ }0\text{ } < =i < \text{ }selectBlockCount\right. \right. \right. \right.
  $$

<div style="padding-left:40px;">

  Phase 1: Compute $dP$ and $dV$ based on the matrix multiplication derivative rules:

</div>

  $$
   dP\mathop{{}}\nolimits_{{t,:}}=dO\mathop{{}}\nolimits_{{t,:}}\text{@}V\mathop{{}}\nolimits^{{T}}
  $$

  $$
   dV \left[ u \left] =P\mathop{{}}\nolimits_{{T}}^{{t,:}}\text{@}dO\mathop{{}}\nolimits_{{t,:}}\right. \right.
  $$

<div style="padding-left:40px;">

   Phase 2: Compute $dS$:

</div>

  $$
   d\mathop{{S}}\nolimits_{{t,:}}= \left[ P\mathop{{}}\nolimits_{{t,:}}@ \left( dP\mathop{{}}\nolimits_{{t,:}}-FlashSoftmaxGrad \left( dO,O \left)  \left)  \right] \right. \right. \right. \right.
  $$

<div style="padding-left:40px;">

   Phase 3: Compute $dQ$ and $dK$:

</div>

  $$
   d\mathop{{Q}}\nolimits_{{t,:}}=d\mathop{{S}}\nolimits_{{t,:}}@K \left[ u \left] \mathop{{}}\nolimits_{{:t,:}}/\sqrt{{d\mathop{{}}\nolimits_{{k,:}}}}\right. \right.
  $$

  $$
   dK \left[ u \left] \mathop{{}}\nolimits_{{:t,:}}=dS\mathop{{}}\nolimits_{{t,:t}}\mathop{{}}\nolimits^{{T}}\text{@}Q/\sqrt{{d\mathop{{}}\nolimits_{{t,:}}}}\right. \right. 
  $$

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md)calls. First call `aclnnSparseFlashAttentionGradGetWorkspaceSize` to obtain the required workspace size for computation and the executor that includes the operator's computation process. Then call `aclnnSparseFlashAttentionGrad` to perform the computation.

```c++
aclnnStatus aclnnSparseFlashAttentionGradGetWorkspaceSize(
    const aclTensor     *query, 
    const aclTensor     *key,
    const aclTensor     *value,
    const aclTensor     *sparseIndices,
    const aclTensor     *dOut,
    const aclTensor     *out,
    const aclTensor     *softmaxMax,
    const aclTensor     *softmaxSum,
    const aclTensor     *actualSeqLengthsQueryOptional,
    const aclTensor     *actualSeqLengthskvOptional,
    const aclTensor     *queryRopeOptional,
    const aclTensor     *keyRopeOptional,
    double               scaleValue,
    int64_t              sparseBlockSize,
    char                *layoutOptional,
    int64_t              sparseMode,
    int64_t              preTokens,
    int64_t              nextTokens,
    bool                 deterministic,
    const aclTensor     *dQueryOut,
    const aclTensor     *dKeyOut,
    const aclTensor     *dValueOut,
    const aclTensor     *dQueryRopeOutOptional,
    const aclTensor     *dKeyRopeOutOptional,
    uint64_t            *workspaceSize,
    aclOpExecutor      **executor)
```

```c++
aclnnStatus aclnnSparseFlashAttentionGrad(
    void             *workspace, 
    uint64_t          workspaceSize, 
    aclOpExecutor    *executor, 
    aclrtStream stream)
```

## aclnnSparseFlashAttentionGradGetWorkspaceSize

- **Parameters:**

    <table style="undefined;table-layout: fixed; width: 1550px">
        <colgroup>
            <col style="width: 220px">
            <col style="width: 120px">
            <col style="width: 200px">  
            <col style="width: 400px">  
            <col style="width: 212px">  
            <col style="width: 100px">
            <col style="width: 290px">
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
            <th>Dimension (shape)</th>
            <th>Non-contiguous Tensor</th>
        </tr></thead>
        <tbody>
        <tr>
            <td>query</td>
            <td>Input</td>
            <td>The input Q for the attention structure.</td>
            <td>
            The dimensions of the shapes for query, key, value, sparseIndices, dOut, out, softmaxMax, and softmaxSum remain consistent.
            </td>
            <td>BFLOAT16, FLOAT16</td>
            <td>ND</td>
            <td>(B,S1,N1,D), (T1,N1,D)<br>
            B: Supports generalization; S1: Supports generalization; N1: Supports 128, 64, 32, 16, 8, 4, 2, 1; D: 512; T2: B x S2
            </td>
            <td>√</td>
        </tr>
        <tr>
            <td>key</td>
            <td>Input</td>
            <td>Input K for the attention structure.</td>
            <td>-</td>
            <td>BFLOAT16, FLOAT16</td>
            <td>ND</td>
            <td>(B,S2,N2,D), (T2,N2,D)<br>
            B: Supports generalization; N2: 1; D: 512; T2: B × S2
            </td>
            <td>√</td>
        </tr>
        <tr>
            <td>value</td>
            <td>Input</td>
            <td>The input v for the attention structure.</td>
            <td>
            -
            </td>
            <td>BFLOAT16, FLOAT16</td>
            <td>ND</td>
            <td>(B,S2,N2,D), (T2,N2,D)<br>
            B: Supports generalization and remains consistent with the B in the query; S2: Supports generalization and remains consistent with the S2 in the value; N2: 1; D: 512; T2: B × S2
            </td>
            <td>√</td>
        </tr>
        <tr>
            <td>sparseIndices</td>
            <td>Input</td>
            <td>Attention indices with higher weights selected in sparse scenarios.</td>
            <td>
            -
            </td>
            <td>INT64</td>
            <td>ND</td>
            <td>(B,S1,N2,K), (T1,N2,K)<br>
            B: Supports generalization and is consistent with the B in the query; S2: Supports generalization and is consistent with the S2 in the value; N2: 1; K: 2048; T1: B × S1
            </td>
            <td>-</td>
        </tr>
        <tr>
            <td>dOut</td>
            <td>Input</td>
            <td>Gradient of the attention output matrix.</td>
            <td>
            -
            </td>
            <td>BFLOAT16, FLOAT16</td>
            <td>ND</td>
            <td>(B,S1,N1,D), (T1,N1,D)<br>
            B: Supports generalization and remains consistent with query B; S1: Supports generalization and remains consistent with query S1; D: 512; T1: B × S1
            </td>
            <td>√</td>
        </tr>
        <tr>
            <td>out</td>
            <td>Input</td>
            <td>Attention output matrix.</td>
            <td>
            -
            </td>
            <td>BFLOAT16, FLOAT16</td>
            <td>ND</td>
            <td>(B,S1,N1,D), (T1,N1,D)<br>
            B: Supports generalization and remains consistent with the B in the query; S1: Supports generalization and remains consistent with the S1 in the query; N1: Supports 128, 64, 32, 16, 8, 4, 2, 1; D: 512; T1: B × S1
            </td>
            <td>√</td>
        </tr>
        <tr>
            <td>softmaxMax</td>
            <td>Input</td>
            <td>Intermediate output of the attention forward computation.</td>
            <td>
            -
            </td>
            <td>FLOAT32</td>
            <td>ND</td>
            <td>(B,N2,S1,G), (N2,T1,G)<br>
            B: Supports generalization and remains consistent with the B in the query; N2: 1; S1: Supports generalization and remains consistent with the S1 in the query; G: N1/N2; T1: B × S1
            </td>
            <td>√</td>
        </tr>
        <tr>
            <td>softmaxSum</td>
            <td>Input</td>
            <td>Intermediate output of the attention forward computation.</td>
            <td>
            -
            </td>
            <td>FLOAT32</td>
            <td>ND</td>
            <td>(B,N2,S1,G), (N2,T1,G)<br>
            B: Supports generalization and is consistent with the B in the query; N2: 1; S1: Supports generalization and is consistent with the S1 in the query; G: N1/N2; T1: B × S1
            </td>
            <td>√</td>
        </tr>
  <tr>
            <td>actualSeqLengthsQueryOptional</td>
            <td>Input</td>
            <td>The number of valid tokens in the Query for each Batch.</td>
            <td>
            <ul>
                <li>Optional: This variable exists when the layout is TND.</li>
                <li>The length should be consistent with B.</li>
                <li>The cumulative sum remains consistent with T1.</li>
            </ul>
            </td>
            <td>INT32</td>
            <td>ND</td>
            <td>(B,)</td>
            <td>-</td>
        </tr>
        <tr>
            <td>actualSeqLengthskvOptional</td>
            <td>Input</td>
            <td>The number of valid tokens for Key and value in each Batch.</td>
            <td>
            <ul>
                <li>Optional: This variable exists when the layout is TND.</li>
                <li>The length should be consistent with B.</li>
                <li>The cumulative sum remains consistent with T2.</li>
            </ul>
            </td>
            <td>INT32</td>
            <td>ND</td>
            <td>(B,)</td>
            <td>-</td>
        </tr>
        <tr>
            <td>queryRopeOptional</td>
            <td>Input</td>
            <td>MLA rope part: Output of the query position encoding.</td>
            <td>
            -
            </td>
            <td>BFLOAT16, FLOAT16</td>
            <td>ND</td>
            <td>(B,S1,N1,Dr), (T1,N1,Dr)<br>
            B: Supports generalization and remains consistent with the B in the query; S1: Supports generalization and remains consistent with the S1 in the query; N1: Supports 128, 64, 32, 16, 8, 4, 2, 1; Dr: 64; T1: B × S1
            </td>
            <td>√</td>
        </tr>
        <tr>
            <td>keyRopeOptional</td>
            <td>Input</td>
            <td>MLA rope part: Output of Key position encoding.</td>
            <td>
            -
            </td>
            <td>BFLOAT16, FLOAT16</td>
            <td>ND</td>
            <td>(B,S2,N2,Dr), (T2,N2,Dr)<br>
            B: Supports generalization and is consistent with the B in the query; S2: Supports generalization and is consistent with the S2 in the value; N2: 1; Dr: 64; T2: B × S2</td>
            <td>√</td>
        </tr>
        <tr>
            <td>scaleValue</td>
            <td>Input</td>
            <td>Scaling factor.</td>
            <td>
            Suggested value: the reciprocal of the square root of d in the formula
            </td>
            <td>FLOAT32</td>
            <td>N/A</td>
            <td>-</td>
            <td>-</td>
        </tr>
        <tr>
            <td>sparseBlockSize</td>
            <td>Input</td>
            <td>The size of the selected block.</td>
            <td>
            Currently supports 1, 8, 16, 32, and 64.
            </td>
            <td>INT32</td>
            <td>N/A</td>
            <td>-</td>
            <td>-</td>
        </tr>
        <tr>
            <td>layout</td>
            <td>Input</td>
            <td>Layout format.</td>
            <td>
            Supports BSND, TND.
            </td>
            <td>STRING</td>
            <td>N/A</td>
            <td>-</td>
            <td>-</td>
        </tr>
        <tr>
            <td>sparseMode</td>
            <td>Input</td>
            <td>Sparse mode.</td>
        <td>
              <ul>
                <li>Represents the sparse mode. For detailed explanations of different sparse modes, please refer to <a href="##constraint-description">Constraint Description</a>.</li>
                <li>Only modes 0 and 3 are supported.</li>
              </ul>
        </td>
        <td>INT64</td>
        <td>N/A</td>
        <td>-</td>
        <td>-</td>
        </tr>
        <tr>
        <td>preTokens</td>
            <td>Input</td>
            <td>In the Attention operator, the starting position of the sliding window for the S matrix.</td>
            <td>
            <ul>
                <li>When sparse_mode=4, pre_tokens takes effect.</li>
                <li>Only supports the value: 2147483647</li>
            </ul>
            </td>
            <td>INT64</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
        </tr>
        <tr>
        <td>nextTokens</td>
            <td>Input</td>
            <td>In the Attention operator, the sliding window termination position for the S matrix.</td>
            <td>
            <ul>
                <li>When sparse_mode=4, next_tokens takes effect.</li>
                <li>Only supported value: 2147483647</li>
            </ul>
            </td>
            <td>INT64</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
        </tr>
        <tr>
        <td>deterministic</td>
            <td>Input</td>
            <td>Deterministic computation.</td>
            <td>
            Consistent with the global deterministic parameter use_deterministic_algorithms
            </td>
            <td>BOOL</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
        </tr>
        <tr>
            <td>dQuery</td>
            <td>Output</td>
            <td>Represents the gradient of the query.</td>
            <td>
            Keep the Shape dimension consistent with the input query
            </td>
            <td>BFLOAT16, FLOAT16</td>
            <td>ND</td>
            <td>(B,S1,N1,D), (T1,N1,D)<br>
            B: Supports generalization and remains consistent with the B in the query; S1: Supports generalization and remains consistent with the S1 in the query; N1: Supports 128, 64, 32, 16, 8, 4, 2, 1; D: 512; T1: B × S1
            </td>
            <td>√</td>
        </tr>
        <tr>
            <td>dKey</td>
            <td>Output</td>
            <td>Represents the gradient of the key.</td>
            <td>
            -
            </td>
            <td>BFLOAT16, FLOAT16</td>
            <td>ND</td>
            <td>(B,S2,N2,D), (T2,N2,D)<br>
            B: Supports generalization and remains consistent with the B in the query; S2: Supports generalization and remains consistent with the S2 in the value; N2: 1; D: 512; T2: B × S2
            </td>
            <td>√</td>
        </tr>
        <tr>  
            <td>dValue</td>
            <td>Output</td>
            <td>Represents the gradient of the value.</td>
            <td>
            Keep the Shape dimension consistent with the input value.
            </td>
            <td>BFLOAT16, FLOAT16</td>
            <td>ND</td>
            <td>(B,S2,N2,D), (T2,N2,D)<br>
            B: Supports generalization and remains consistent with the B in the query; S2: Supports generalization and remains consistent with the S2 in the value; N2: 1; D: 512; T2: B × S2</td>
            <td>√</td>
        </tr>
          <tr>
            <td>dQueryRopeOptional</td>
            <td>Output</td>
            <td>Represents the gradient of queryRope.</td>
            <td>
            <ul>
                <li>This variable will only be output when the input queryRope exists.</li>
                <li>Maintain the same shape dimensions as the input query.</li>
            </ul>
            </td>
            <td>BFLOAT16, FLOAT16</td>
            <td>ND</td>
            <td>(B,S1,N1,Dr), (T1,N1,Dr)<br>
            B: Supports generalization and remains consistent with the B in the query; S1: Supports generalization and remains consistent with the S1 in the query; N1: Supports 128, 64, 32, 16, 8, 4, 2, 1; Dr: 64; T1: B × S1
            </td>
            <td>√</td>
        </tr>
        <tr>
            <td>dKeyRopeOptional</td>
            <td>Output</td>
            <td>Represents the gradient of keyRope.</td>
            <td>
            When the input keyRope exists, this variable will be output.
            </td>
            <td>BFLOAT16, FLOAT16</td>
            <td>ND</td>
            <td>(B,S2,N2,Dr), (T2,N2,Dr)<br>
            B: Supports generalization and remains consistent with the B in the query; S2: Supports generalization and remains consistent with the S2 in the value; N2: 1; Dr: 64; T2: B × S2
            </td>
            <td>√</td>
        </tr>
        </tbody>
    </table>

- **Returns:**

  `aclnnStatus` status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API implements input parameter verification. The following errors may be thrown.

    <table style="undefined;table-layout: fixed;width: 1155px"><colgroup>
    <col style="width: 319px">
    <col style="width: 144px">
    <col style="width: 671px">
    </colgroup>
        <thead>
            <th>Return</th>
            <th>Error Code</th>
            <th>Description</th>
        </thead>
        <tbody>
            <tr>
                <td>ACLNN_ERR_PARAM_NULLPTR</td>
                <td>161001</td>
                <td>Required parameter or output is a null pointer.</td>
            </tr>
            <tr>
                <td>ACLNN_ERR_PARAM_INVALID</td>
                <td>161002</td>
                <td>The data types and formats of input variables, such as query, key, value, sparseIndices, are not within the supported range.</td>
            </tr>
            <tr>
                <td>ACLNN_ERR_RUNTIME_ERROR</td>
                <td>361001</td>
                <td>An exception of the npu runtime call.</td>
            </tr>
        </tbody>
    </table>

## aclnnSparseFlashAttentionGrad

- **Parameters:**

    <table style="undefined;table-layout: fixed; width: 1155px"><colgroup>
    <col style="width: 144px">
    <col style="width: 125px">
    <col style="width: 700px">
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
        <td>The memory address of the workspace allocated on the Device side.</td>
        </tr>
        <tr>
        <td>workspaceSize</td>
        <td>Input</td>
        <td>The size of the workspace allocated on the Device side, obtained from the first interface aclnnSparseFlashAttentionGradGetWorkspaceSize.</td>
        </tr>
        <tr>
        <td>executor</td>
        <td>Input</td>
        <td>Operator executor, which contains the calculation process of the operator.</td>
        </tr>
        <tr>
        <td>stream</td>
        <td>Input</td>
        <td>Specifies the Stream for task execution.</td>
        </tr>
    </tbody>
    </table>

- **Returns:**

  `aclnnStatus` status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - `aclnnSparseFlashAttentionGrad` defaults to a non-deterministic implementation. Enabling deterministic computation via `aclrtCtxSetSysParamOpt` is not supported.
- Common constraints
    - Handling scenarios where input parameters are empty:
        - If the query is an empty Tensor: return directly.

- Mask
    <table style="undefined;table-layout: fixed; width: 942px"><colgroup>
        <col style="width: 100px">
        <col style="width: 740px">
        <col style="width: 360px">
        </colgroup>
        <thead>
            <tr>
                <th>sparseMode</th>
                <th>Meaning</th>
                <th>Remarks</th>
            </tr>
        </thead>
        <tbody>
        <tr>
            <td>0</td>
            <td>No mask operation.</td>
            <td>Supported</td>
        </tr>
        <tr>
            <td>1</td>
            <td>allMask, the complete attenmask matrix must be passed in.</td>
            <td>Not Supported</td>
        </tr>
        <tr>
            <td>2</td>
            <td>Mask for leftUpCausal mode, requires passing the optimized attenmask matrix.</td>
            <td>Not Supported</td>
        </tr>
        <tr>
            <td>3</td>
            <td>The mask for the rightDownCausal mode, corresponding to the lower triangular scenario divided by the right vertex.</td>
            <td>Supported</td>
  </tr>
        <tr>
            <td>4</td>
            <td>The mask for band mode, requires passing the optimized attenmask matrix.</td>
            <td>Not Supported</td>
        </tr>
        <tr>
            <td>5</td>
            <td>prefix</td>
            <td>Not Supported</td>
        </tr>
        <tr>
            <td>6</td>
            <td>global</td>
            <td>Not Supported</td>
        </tr>
        <tr>
            <td>7</td>
            <td>dilated</td>
            <td>Not Supported</td>
        </tr>
        <tr>
            <td>8</td>
            <td>block_local</td>
            <td>Not Supported</td>
        </tr>
        </tbody>
    </table>
- Specification constraints
    <table style="undefined;table-layout: fixed; width: 942px"><colgroup>
        <col style="width: 100px">
        <col style="width: 300px">
        <col style="width: 360px">
        </colgroup>
        <thead>
            <tr>
                <th>Specification Item</th>
                <th>Specifications</th>
                <th>Specification</th>
            </tr>
        </thead>
        <tbody>
        <tr>
            <td>deterministic</td>
            <td>bool</td>
            <td>Supports deterministic computation</td>
        </tr>
        <tr>
            <td>B</td>
            <td>1~256</td>
            <td>-</td>
        </tr>
        <tr>
            <td>S1, S2</td>
            <td>1~128K</td>
            <td>S1, S2 support unequal length</td>
        </tr>
        <tr>
            <td>N1</td>
            <td>1, 2, 4, 8, 16, 32, 64, 128</td>
            <td>SparseFA is for MQA.</td>
        </tr>
        <tr>
            <td>N2</td>
            <td>1</td>
            <td>SparseFA is for MQA, Nidx2=1.</td>
        </tr>
        <tr>
            <td>D</td>
            <td>512</td>
            <td>-</td>
        </tr>
        <tr>
            <td>Drope</td>
            <td>64</td>
            <td>-</td>
        </tr>
        <tr>
            <td>layout</td>
            <td>BSND/TND</td>
            <td>-</td>
        </tr>
        </tbody>
    </table>

## Examples

The following uses the <term>Atlas A2 training products/Atlas A2 inference products</term> as an example. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```c++
#include <iostream>
#include <vector>
#include <numeric>
#include "acl/acl.h"
#include "aclnnop/aclnn_sparse_flash_attention_grad.h"

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
  std::vector<short> resultData(size, 0);
  auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]),
                         *deviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("mean result[%ld] is: %e\n", i, resultData[i]);
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
  // Call aclrtMalloc to allocate memory on the device side.
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  // Call aclrtMemcpy to copy data from the host side to the device side memory.
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // Calculate the strides of a contiguous tensor.
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  // Call the aclCreateTensor interface to create an aclTensor.
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

int main() {
  // 1. (Fixed writing) Initialize the device, context, and stream. For details, see the ACL API manual.
  // Set the deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtContext context;
  aclrtStream stream;
  auto ret = Init(deviceId, &context, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the input and output based on the API.
  std::vector<int64_t> qShape = {1, 16, 512};                // T1, N1, D
  std::vector<int64_t> kShape = {2048, 1, 512};              // T2, N2, D
  std::vector<int64_t> vShape = {2048, 1, 512};              // T2, N2, D
  std::vector<int64_t> sparseIndicesShape = {1, 1, 2048};    // T1, N2, K
  std::vector<int64_t> outShape = {1, 16, 512};             // T1, N1, D
  std::vector<int64_t> dOutShape = {1, 16, 512};            // T1, N1, D
  std::vector<int64_t> softmaxMaxShape = {1, 1, 16};        // N2, T1, G
  std::vector<int64_t> softmaxSumShape = {1, 1, 16};        // N2, T1, G
  std::vector<int64_t> actSeqQLenshape = {1};               // B
  std::vector<int64_t> actSeqKvLenshape = {1};              // B
  std::vector<int64_t> qRopeShape = {1, 16, 64};            // T1, N1, Drope
  std::vector<int64_t> kRopeShape = {2048, 1, 64};          // T2, N2, Drope

  void* qDeviceAddr = nullptr;
  void* kDeviceAddr = nullptr;
  void* vDeviceAddr = nullptr;
  void* sparseIndicesDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  void* dOutDeviceAddr = nullptr;
  void* softmaxMaxDeviceAddr = nullptr;
  void* softmaxSumDeviceAddr = nullptr;
  void* actSeqQLenDeviceAddr = nullptr;
  void* actSeqKvLenDeviceAddr = nullptr;
  void* qRopeDeviceAddr = nullptr;
  void* kRopeDeviceAddr = nullptr;
  void* dqDeviceAddr = nullptr;
  void* dkDeviceAddr = nullptr;
  void* dvDeviceAddr = nullptr;
  void* dqRopeDeviceAddr = nullptr;
  void* dkRopeDeviceAddr = nullptr;

  aclTensor* q = nullptr;
  aclTensor* k = nullptr;
  aclTensor* v = nullptr;
  aclTensor* sparseIndices = nullptr;
  aclTensor* out = nullptr;
  aclTensor* dOut = nullptr;
  aclTensor* softmaxMax = nullptr;
  aclTensor* softmaxSum = nullptr;
  aclTensor* actSeqQLen = nullptr;
  aclTensor* actSeqKvLen = nullptr;
  aclTensor* qRope = nullptr;
  aclTensor* kRope = nullptr;
  aclTensor* dq = nullptr;
  aclTensor* dk = nullptr;
  aclTensor* dv = nullptr;
  aclTensor* dqRope = nullptr;
  aclTensor* dkRope = nullptr;

  std::vector<short> qHostData(1 * 16 * 512, 1.0);
  std::vector<short> kHostData(2048 * 1 * 512, 1.0);
  std::vector<short> vHostData(2048 * 1 * 512, 1.0);
  std::vector<int32_t> sparseIndicesHostData(2048);
  std::iota(sparseIndicesHostData.begin(), sparseIndicesHostData.end(), 0);
  std::vector<short> outHostData(1 * 16 * 512, 1.0);
  std::vector<short> dOutHostData(1 * 16 * 512, 1.0);
  std::vector<float> softmaxMaxHostData(16, 3.0);
  std::vector<float> softmaxSumHostData(16, 3.0);
  std::vector<int64_t> actSeqQLenHostData(1, 1);
  std::vector<int64_t> actSeqKvLenHostData(1, 2048);
  std::vector<short> qRopeHostData(1 * 16 * 64, 1.0);
  std::vector<short> kRopeHostData(2048 * 1 * 64, 1.0);
  std::vector<short> dqHostData(1 * 16 * 512, 0);
  std::vector<short> dkHostData(2048 * 1 * 512, 0);
  std::vector<short> dvHostData(2048 * 1 * 512, 0);
  std::vector<short> dqRopeHostData(1 * 16 * 64, 0);
  std::vector<short> dkRopeHostData(2048 * 1 * 64, 0);

  ret = CreateAclTensor(qHostData, qShape, &qDeviceAddr, aclDataType::ACL_FLOAT16, &q);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(kHostData, kShape, &kDeviceAddr, aclDataType::ACL_FLOAT16, &k);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(vHostData, vShape, &vDeviceAddr, aclDataType::ACL_FLOAT16, &v);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(sparseIndicesHostData, sparseIndicesShape, &sparseIndicesDeviceAddr, aclDataType::ACL_INT32, &sparseIndices);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT16, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(dOutHostData, dOutShape, &dOutDeviceAddr, aclDataType::ACL_FLOAT16, &dOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(softmaxMaxHostData, softmaxMaxShape, &softmaxMaxDeviceAddr, aclDataType::ACL_FLOAT, &softmaxMax);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(softmaxSumHostData, softmaxSumShape, &softmaxSumDeviceAddr, aclDataType::ACL_FLOAT, &softmaxSum);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(actSeqQLenHostData, actSeqQLenshape, &actSeqQLenDeviceAddr, aclDataType::ACL_INT64, &actSeqQLen);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(actSeqKvLenHostData, actSeqKvLenshape, &actSeqKvLenDeviceAddr, aclDataType::ACL_INT64, &actSeqKvLen);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(qRopeHostData, qRopeShape, &dqDeviceAddr, aclDataType::ACL_FLOAT16, &qRope);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(kRopeHostData, kRopeShape, &dkDeviceAddr, aclDataType::ACL_FLOAT16, &kRope);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(dqHostData, qShape, &dqDeviceAddr, aclDataType::ACL_FLOAT16, &dq);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(dkHostData, kShape, &dkDeviceAddr, aclDataType::ACL_FLOAT16, &dk);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(dvHostData, vShape, &dvDeviceAddr, aclDataType::ACL_FLOAT16, &dv);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(dqRopeHostData, qRopeShape, &dqRopeDeviceAddr, aclDataType::ACL_FLOAT16, &dqRope);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(dkRopeHostData, kRopeShape, &dkRopeDeviceAddr, aclDataType::ACL_FLOAT16, &dkRope);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  
  double scaleValue = 0.088388;
  int64_t sparseBlockSize = 1;
  int64_t sparseMode = 0;
  int64_t preTokens = 2147483647;
  int64_t nextTokens = 2147483647;
  bool deterministic = false;
  char layout[5] = {'T', 'N', 'D', 0};
  
  // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  
  // Call the first-phase API of aclnnSparseFlashAttentionGrad.
  ret = aclnnSparseFlashAttentionGradGetWorkspaceSize(q, k, v, sparseIndices, dOut, out, softmaxMax, softmaxSum, actSeqQLen, actSeqKvLen,
            qRope, kRope, scaleValue, sparseBlockSize, layout, sparseMode, preTokens, nextTokens, deterministic, dq, dk, dv, dqRope, dkRope, 
            &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSparseFlashAttentionGradGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  
  // Allocate device memory based on workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  
  // Call the second-phase API of aclnnSparseFlashAttentionGrad.
  ret = aclnnSparseFlashAttentionGrad(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSparseFlashAttentionGrad failed. ERROR: %d\n", ret); return ret);
  
  // 4. (Fixed writing) Synchronize the stream and wait for task completion.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
  
  // 5. Obtain the output value and copy the result from the device to the host. Modify the code based on the API definition.
  PrintOutResult(qShape, &dqDeviceAddr);
  PrintOutResult(kShape, &dkDeviceAddr);
  PrintOutResult(vShape, &dvDeviceAddr);
  PrintOutResult(qRopeShape, &dqRopeDeviceAddr);
  PrintOutResult(kRopeShape, &dkRopeDeviceAddr);
  
  // 6. Release aclTensor and aclScalar. Modify the configuration based on the API definition.
  aclDestroyTensor(q);
  aclDestroyTensor(k);
  aclDestroyTensor(v);
  aclDestroyTensor(sparseIndices);
  aclDestroyTensor(out);
  aclDestroyTensor(dOut);
  aclDestroyTensor(softmaxMax);
  aclDestroyTensor(softmaxSum);
  aclDestroyTensor(actSeqQLen);
  aclDestroyTensor(actSeqKvLen);
  aclDestroyTensor(qRope);
  aclDestroyTensor(kRope);
  aclDestroyTensor(dq);
  aclDestroyTensor(dk);
  aclDestroyTensor(dv);
  aclDestroyTensor(dqRope);
  aclDestroyTensor(dkRope);
  
  // 7. Release device resources.
  aclrtFree(qDeviceAddr);
  aclrtFree(kDeviceAddr);
  aclrtFree(vDeviceAddr);
  aclrtFree(sparseIndicesDeviceAddr);
  aclrtFree(softmaxMaxDeviceAddr);
  aclrtFree(softmaxSumDeviceAddr);
  aclrtFree(outDeviceAddr);
  aclrtFree(dOutDeviceAddr);
  aclrtFree(qRopeDeviceAddr);
  aclrtFree(kRopeDeviceAddr);
  aclrtFree(dqDeviceAddr);
  aclrtFree(dkDeviceAddr);
  aclrtFree(dvDeviceAddr);
  aclrtFree(dqRopeDeviceAddr);
  aclrtFree(dkRopeDeviceAddr);
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
