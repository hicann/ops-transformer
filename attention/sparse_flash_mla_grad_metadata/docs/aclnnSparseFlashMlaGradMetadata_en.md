# aclnnSparseFlashMlaGradMetadata

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-21T13:25:25.330Z -->

## Applicable Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 products</term>: Not supported
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

- Description: This is an AI CPU operator interface and the preceding operator interface of the aclnnSparseFlashMlaGrad operator. It computes and outputs the load balancing result based on the input information of the aclnnSparseFlashMlaGrad operator interface. The output can be used as the input of the aclnnSparseFlashMlaGrad operator interface to reduce its execution time.

  **This operator is not recommended for standalone use. Use it together with the aclnnSparseFlashMlaGrad operator to form a complete workflow.**
    1. Accepts the input data shape information of the aclnnSparseFlashMlaGrad operator interface, including batchSize, qSeqlen, kSeqlen, and mask. Tiles the input and simulates the computation time to evenly distribute tiles across available cores, reducing the overall computation time of the aclnnSparseFlashMlaGrad operator and improving hardware utilization.
    2. After the allocation result is output, it is used as input for the aclnnSparseFlashMlaGrad operator.
    3. The allocation result contains the start point and end point of the basic block on each AIC core, and the FD task information of each AIV core. For details, see [Example](#example).

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call "aclnnSparseFlashMlaGradMetadataGetWorkspaceSize" first to obtain the workspace size, and then call "aclnnSparseFlashMlaGradMetadata" to perform computation.

``` cpp
aclnnStatus aclnnSparseFlashMlaGradMetadataGetWorkspaceSize(
    const aclTensor   *cuSeqlensQOptional,
    const aclTensor   *cuSeqlensOriKvOptional,
    const aclTensor   *cuSeqlensCmpKvOptional,
    const aclTensor   *sequsedQOptional,
    const aclTensor   *sequsedOriKvOptional,
    const aclTensor   *sequsedCmpKvOptional,
    const aclTensor   *cmpResidualKvOptional,
    const aclTensor   *oriTopkLengthOptional,
    const aclTensor   *cmpTopkLengthOptional,
    int64_t            numHeadsQ,
    int64_t            numHeadsKv,
    int64_t            headDim,
    int64_t            batchSize,
    int64_t            maxSeqlenQ,
    int64_t            maxSeqlenOriKv,
    int64_t            maxSeqlenCmpKv,
    int64_t            oriTopk,
    int64_t            cmpTopk,
    int64_t            cmpRatio,
    int64_t            oriMaskMode,
    int64_t            cmpMaskMode,
    int64_t            oriWinLeft,
    int64_t            oriWinRight,
    const char        *layoutQOptional,
    const char        *layoutKvOptional,
    bool               hasOriKv,
    bool               hasCmpKv,
    const aclTensor   *metadata,
    uint64_t          *workspaceSize,
    aclOpExecutor    **executor)
```

``` cpp
aclnnStatus aclnnSparseFlashMlaGradMetadata(
    void              *workspace,
    uint64_t           workspaceSize,
    aclOpExecutor     *executor,
    aclrtStream        stream)
```

## aclnnSparseFlashMlaGradMetadataGetWorkspaceSize

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1450px"><colgroup>
  <col style="width: 240px">
  <col style="width: 120px">
  <col style="width: 333px">
  <col style="width: 160px">
  <col style="width: 100px">
  <col style="width: 100px">
  <col style="width: 170px">
  <col style="width: 140px">
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
      <td>cuSeqlensQOptional (aclTensor*)</td>
      <td>Input</td>
      <td>Indicates the valid Sequence Length of q in different batches.</td>
      <td><ul><li>Supports empty tensors.</li><li>The shape is fixed to (B+1, ).</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>1D</td>
      <td>√</td>
    </tr>
    <tr>
      <td>cuSeqlensOriKvOptional (aclTensor*)</td>
      <td>Input</td>
      <td>Indicates the valid Sequence Length of oriKvOptional in different batches.</td>
      <td><ul><li>Supports empty tensors.</li><li>The shape is fixed to (B+1, ).</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>1D</td>
      <td>√</td>
    </tr>
    <tr>
      <td>cuSeqlensCmpKvOptional (aclTensor*)</td>
      <td>Input</td>
      <td>Indicates the valid Sequence Length of cmpKvOptional in different Batches.</td>
      <td><ul><li>Supports empty tensors.</li><li>The shape is fixed to (B+1, ).</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>1D</td>
      <td>√</td>
    </tr>
    <tr>
      <td>sequsedQOptional(aclTensor*)</td>
      <td>Input</td>
      <td>Indicates the Sequence Length of q that actually participate in the operation in different Batch.</td>
      <td><ul><li>Supports empty Tensor.</li><li>Shape is fixed to (B, ).</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>1D</td>
      <td>√</td>
    </tr>
    <tr>
      <td>sequsedOriKvOptional (aclTensor*)</td>
      <td>Input</td>
      <td>Indicates the Sequence Length of oriKvOptional that actually participate in the operation in different batches.</td>
      <td><ul><li>Supports empty tensors.</li><li>The shape is fixed to (B, ).</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>1D</td>
      <td>√</td>
    </tr>
    <tr>
      <td>sequsedCmpKvOptional (aclTensor*)</td>
      <td>Input</td>
      <td>Indicates the Sequence Length of cmpKvOptional that actually participates in the operation in different Batches.</td>
      <td><ul><li>Supports empty tensors.</li><li>Shape is fixed to (B, ).</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>1-D</td>
      <td>√</td>
    </tr>
    <tr>
      <td>cmpResidualKvOptional (aclTensor*)</td>
      <td>Input</td>
      <td>Indicates the remainder of the compressed Sequence Length of cmpKvOptional in different Batches. Used together with cmpRatio to implement the mask and load calculation for the cmpKvOptional part.</td>
      <td><ul><li>Supports empty tensor.</li><li>Must be passed in when cmpRatio is not 1 and mask is 3.</li><li>Shape is fixed to (B, ).</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>1D</td>
      <td>√</td>
    </tr>
    <tr>
      <td>oriTopkLengthOptional (aclTensor*)</td>
      <td>Input</td>
      <td>Indicates the number of key sparse tokens in oriKvOptional corresponding to different q tokens.</td>
      <td><ul><li>Supports empty tensor.</li><li>Shape is (B, S1, N2) or (T1, N2).</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>2D, 3D</td>
      <td>√</td>
    </tr>
    <tr>
      <td>cmpTopkLengthOptional (aclTensor*)</td>
      <td>Input</td>
      <td>Indicates the number of key sparse tokens in the cmpKvOptional part corresponding to different q tokens.</td>
      <td><ul><li>Supports empty tensors.</li><li>The shape is (B, S1, N2) or (T1, N2).</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>2D, 3D</td>
      <td>√</td>
    </tr>
    <tr>
      <td>numHeadsQ (int64_t)</td>
      <td>Input</td>
      <td>Indicates the number of heads of q.</td>
      <td>Currently supports [1, 128].</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>numHeadsKv (int64_t)</td>
      <td>Input</td>
      <td>Number of multi-heads corresponding to oriKvOptional and cmpKvOptional.</td>
      <td>Currently, only 1 is supported.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>headDim (int64_t)</td>
      <td>Input</td>
      <td>Dimension of the attention head.</td>
      <td>Currently only 512 is supported.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>batchSize (int64_t)</td>
      <td>Input</td>
      <td>Indicates the number of batches.</td>
      <td><ul><li>Supports non-negative numbers.</li><li>The recommended value is 0.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>maxSeqlenQ (int64_t)</td>
      <td>Input</td>
      <td>Indicates the maximum Sequence Length of q.</td>
      <td><ul><li>Supports non-negative values.</li><li>The recommended value is 0.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>maxSeqlenOriKv (int64_t)</td>
      <td>Input</td>
      <td>Indicates the maximum Sequence Length of oriKvOptional.</td>
      <td><ul><li>Supports non-negative values.</li><li>The recommended value is 0.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>maxSeqlenCmpKv (int64_t)</td>
      <td>Input</td>
      <td>Indicates the maximum Sequence Length of cmpKvOptional.</td>
      <td><ul><li>Supports non-negative numbers.</li><li>The recommended value is 0.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>oriTopk (int64_t)</td>
      <td>Input</td>
      <td>Indicates the number of key sparse tokens filtered from oriKvOptional. 0 indicates a non-sparse task.</td>
      <td><ul><li>Supports non-negative numbers.</li><li>The recommended value is 0.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cmpTopk (int64_t)</td>
      <td>Input</td>
      <td>Specifies the number of key sparse tokens selected from cmpKvOptional. 0 indicates a non-sparse use case.</td>
      <td><ul><li>Supports non-negative numbers.</li><li>The recommended value is 0.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cmpRatio (int64_t)</td>
      <td>Input</td>
      <td>Indicates the compression ratio of cmpKvOptional.</td>
      <td><ul><li>Currently supports [1, 128].</li><li>The recommended value is 1, indicating no compression.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>oriMaskMode (int64_t)</td>
      <td>Input</td>
      <td>Indicates the mask mode for the computation of q and oriKvOptional.</td>
      <td><ul><li>0 indicates No mask.</li><li>3 indicates the rightDownCausal mode.</li><li>4 indicates the sliding window mode, corresponding to the window use case divided by oriWinLeft and oriWinRight.</li><li>The recommended value is 0.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cmpMaskMode (int64_t)</td>
      <td>Input</td>
      <td>Indicates the mask mode for the computation of q and cmpKvOptional.</td>
      <td><ul><li>0: No mask.</li><li>3: rightDownCausal mode, corresponding to the lower triangle scenario with the right vertex as the division point.</li><li>The recommended value is 0.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>oriWinLeft (int64_t)</td>
      <td>Input</td>
      <td>Indicates the number of past tokens for q in the computation of q and oriKvOptional.</td>
      <td><ul><li>Value range ≥ -1, where -1 indicates infinity.</li><li>The recommended value is -1.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>oriWinRight (int64_t)</td>
      <td>Input</td>
      <td>Indicates the number of future tokens that q attends to in the computation of q and oriKvOptional.</td>
      <td><ul><li>Value range ≥ -1, where -1 indicates infinity.</li><li>The recommended value is -1.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layoutQOptional (char*)</td>
      <td>Input</td>
      <td>Indicates the layout format of q.</td>
      <td><ul><li>Supports BSND and TND.</li><li>The recommended value is BSND.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layoutKvOptional (char*)</td>
      <td>Input</td>
      <td>Indicates the layout of oriKvOptional and cmpKvOptional.</td>
      <td><ul><li>Supports BSND and TND.</li><li>The recommended value is BSND.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>hasOriKv (bool)</td>
      <td>Input</td>
      <td>Indicates whether oriKvOptional is included.</td>
      <td><ul><li>true: includes oriKvOptional.</li><li>false: does not include oriKvOptional.</li><li>The recommended value is true.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>hasCmpKv (bool)</td>
      <td>Input</td>
      <td>Indicates whether cmpKvOptional is included.</td>
      <td><ul><li>true: contains cmpKvOptional.</li><li>false: does not contain cmpKvOptional.</li><li>The recommended value is true.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>metadata (aclTensor*)</td>
      <td>Output</td>
      <td>Indicates the output of the load balancing result.</td>
      <td>The shape is fixed to (1024, ).</td>
      <td>INT32</td>
      <td>ND</td>
      <td>1-D</td>
      <td>×</td>
    </tr>
    <tr>
      <td>workspaceSize (uint64_t*)</td>
      <td>Output</td>
      <td>Returns the workspace size that needs to be applied on the side of the Device.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor (aclOpExecutor**)</td>
      <td>Output</td>
      <td>Returns the op executor, including the operator computation flow.</td>
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

    <table style="undefined;table-layout: fixed; width: 1134px"><colgroup>
    <col style="width: 319px">
    <col style="width: 144px">
    <col style="width: 671px">
    </colgroup>
    <thead>
      <tr>
        <th>Return Value</th>
        <th>Error Code</th>
        <th>Description</th>
      </tr></thead>
    <tbody>
      <tr>
        <td>ACLNN_ERR_INNER_CREATE_EXECUTOR</td>
        <td>561101</td>
        <td>Failed to create aclOpExecutor.</td>
      </tr>
      <tr>
        <td>ACLNN_ERR_INNER_NULLPTR</td>
        <td>561103</td>
        <td>The parameter workspaceSize or executor is a null pointer, or cuSeqlensQOptional, cuSeqlensOriKvOptional, cuSeqlensCmpKvOptional, sequsedQOptional, sequsedOriKvOptional, sequsedCmpKvOptional, cmpResidualKvOptional, oriTopkLengthOptional, or cmpTopkLengthOptional is a null pointer after Contiguous processing.</td>
      </tr>
      <tr>
        <td rowspan="1">ACLNN_ERR_PARAM_INVALID</td>
        <td rowspan="1">161002</td>
        <td>The specifications of parameters cuSeqlensQOptional, cuSeqlensOriKvOptional, cuSeqlensCmpKvOptional, sequsedQOptional, sequsedOriKvOptional, sequsedCmpKvOptional, cmpResidualKvOptional, oriTopkLengthOptional, cmpTopkLengthOptional, numHeadsQ, numHeadsKv, headDim, batchSize, maxSeqlenQ, maxSeqlenOriKv, maxSeqlenCmpKv, oriTopk, cmpTopk, cmpRatio, oriMaskMode, cmpMaskMode, oriWinLeft, oriWinRight, layoutQOptional, layoutKvOptional, hasOriKv, hasCmpKv are not in the supported range.</td>
      </tr>
    </tbody>
    </table>

## aclnnSparseFlashMlaGradMetadata

- **Parameters:**

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
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>workspace</td>
            <td>Input</td>
            <td>Workspace memory address applied on the side of the Device.</td>
        </tr>
        <tr>
            <td>workspaceSize</td>
            <td>Input</td>
            <td>Size of the workspace applied on the side of the Device, obtained through the first-phase API aclnnSparseFlashMlaGradMetadataGetWorkspaceSize.</td>
        </tr>
        <tr>
            <td>executor</td>
            <td>Input</td>
            <td>Executor of the op. It contains the computation flow of the operator.</td>
        </tr>
        <tr>
            <td>stream</td>
            <td>Input</td>
            <td>Specifies the stream for task execution.</td>
        </tr>
    </tbody>
    </table>

- **Return value**

    aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

  - aclnnSparseFlashMlaGradMetadata defaults to a deterministic implementation.
  - B (Batch) indicates the batch size of input samples. q, oriKvOptional, and cmpKvOptional are input parameters of the paired aclnnSparseFlashMlaGrad operator. S1 indicates the size of the S axis in the q shape when layoutQOptional=BSND. T1 indicates the size of the T axis in the q shape when layoutQOptional=TND. S2 indicates the size of the S axis in the oriKvOptional shape when layoutKvOptional=BSND. S3 indicates the size of the S axis in the cmpKvOptional shape when layoutKvOptional=BSND. N2 indicates the size of the N axis in the oriKvOptional and cmpKvOptional shapes.
  - The parameters cuSeqlensQOptional, cuSeqlensOriKvOptional, and cuSeqlensCmpKvOptional require their values to be the cumulative count of valid tokens in the current and preceding batches. The first element is fixed to 0, and the value of each subsequent element must be greater than or equal to that of the previous element.
  - Parameters sequsedQOptional, sequsedOriKvOptional, and sequsedCmpKvOptional must indicate the number of valid tokens in each Batch.
  - layoutQOptional and layoutKvOptional must be the same.
  - Parameter cmpResidualKvOptional must satisfy cmpResidualKvOptional[i] < cmpRatio.
  - numHeadsQ must be divisible by numHeadsKv.
  - If hasOriKv is true, oriKvOptional is considered sparse when oriTopk is greater than 0, and non-sparse when oriTopk is 0.
  - If hasCmpKv is true, cmpKvOptional is considered sparse when cmpTopk is greater than 0, and non-sparse when cmpTopk is 0.
  - If hasOriKv is true, oriTopk is not 0, and oriMaskMode is 0, oriTopkLengthOptional must be passed in. In this case, the valid seqlen of oriKvOptional corresponding to the current q token is the minimum value of the oriMaskMode rule and the elements in oriTopkLengthOptional. In other oriKvOptional sparse use cases, the valid seqlen is the minimum value of the oriMaskMode rule and oriTopk.
  - If hasCmpKv is true, cmpTopk is not 0, and cmpMaskMode is 0, cmpTopkLengthOptional must be passed in. In this case, the valid seqlen of cmpKvOptional corresponding to the current q token is the minimum value of the cmpMaskMode rule and the elements in cmpTopkLengthOptional. In other cmpKvOptional sparse use cases, the valid seqlen is the minimum value of the cmpMaskMode rule and cmpTopk.
  - layoutQOptional=BSND use case
    - maxSeqlenQ must be passed in with the value of S1.
  - layoutKvOptional=BSND case
    - If hasOriKv is true, maxSeqlenOriKv must be passed in with the value of S2.
    - If hasCmpKv is true, maxSeqlenCmpKv must be passed in with the value of S3.
  - layoutQOptional=TND case
    - cuSeqlensQOptional must be passed in.
  - layoutKvOptional=TND case
    - If hasOriKv is true, cuSeqlensOriKvOptional must be passed in.
    - If hasCmpKv is true, cuSeqlensCmpKvOptional must be passed in.
  - Batch value rule
    - If layoutQOptional is BSND, prefer to use the shape of sequsedQOptional to derive batch. If sequsedQOptional is not passed in, obtain the number of batches through batchSize.
    - If layoutQOptional is TND, prefer to use the shape of sequsedQOptional to derive batch. If sequsedQOptional is not passed in, derive batch through the shape of cuSeqlensQOptional.
  - q Seqlen value rule
    - If layoutQOptional is BSND, prefer to use the elements in sequsedQOptional to obtain seqlen. If sequsedQOptional is not passed in, obtain seqlen through maxSeqlenQ.
    - If layoutQOptional is TND, prefer to use the elements in sequsedQOptional to obtain seqlen. If sequsedQOptional is not passed in, obtain seqlen through the elements in cuSeqlensQOptional.
  - oriKvOptional Seqlen value rule
    - If layoutKvOptional is BSND, prefer to use the elements in sequsedOriKvOptional to obtain seqlen. If sequsedOriKvOptional is not passed in, obtain seqlen through maxSeqlenOriKv.
    - If layoutKvOptional is TND, prefer to use the elements in sequsedOriKvOptional to obtain seqlen. If sequsedOriKvOptional is not passed in, obtain seqlen through the elements in cuSeqlensOriKvOptional.
  - cmpKvOptional Seqlen value rule
    - If layoutKvOptional is BSND, prefer to use the elements in sequsedCmpKvOptional to obtain seqlen. If sequsedCmpKvOptional is not passed in, obtain seqlen through maxSeqlenCmpKv.
    - If layoutKvOptional is TND, prefer to use the elements in sequsedCmpKvOptional to obtain seqlen. If sequsedCmpKvOptional is not passed in, obtain seqlen through the elements in cuSeqlensCmpKvOptional.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

``` cpp
#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include <limits>
#include <functional>
#include <utility>
#include "acl/acl.h"
#include "aclnnop/aclnn_sparse_flash_mla_grad_metadata.h"

#define CHECK_LOG_RET(cond, ret_val, fmt, ...)      \
    do {                                            \
        if (!(cond)) {                              \
            printf(fmt "\n", ##__VA_ARGS__);        \
            return (ret_val);                       \
        }                                           \
    } while (0)

// Constants
constexpr uint32_t AIC_CORE_MAX_NUM = 36;
constexpr uint32_t AIV_CORE_MAX_NUM = 72;
constexpr uint32_t SMLAG_METADATA_TOTAL_SIZE = 1024;
using SMLAG_METADATA_T = int32_t;
constexpr uint32_t GRAD_METADATA_SIZE = 7;

// Grad Metadata Index Definitions
constexpr uint32_t TOTAL_NUM = 0;
constexpr uint32_t FORMER_CORE_PROCESS_NUM = 1;
constexpr uint32_t REMAIN_CORE_PROCESS_NUM = 2;
constexpr uint32_t REMAIN_CORE_NUM = 3;
constexpr uint32_t USED_CORE_NUM = 4;
constexpr uint32_t MAX_ORI_KV_SIZE = 5;
constexpr uint32_t MAX_CMP_KV_SIZE = 6;

struct SmlagMetadata {
    uint32_t gradMetadata[GRAD_METADATA_SIZE];
};

struct ScopeGuard
{
    explicit ScopeGuard(std::function<void()> onExitScope) : m_exitFunc(std::move(onExitScope)),
        m_isDismissed(false) {}
    // Do not copy.
    ScopeGuard(const ScopeGuard&) = delete;
    ScopeGuard& operator=(const ScopeGuard&) = delete;

    ~ScopeGuard()
    {
        if (!m_isDismissed) {
            m_exitFunc();
        }
    }

    void Dismiss()
    {
        m_isDismissed = true;
    }

    std::function<void()> m_exitFunc;
    bool m_isDismissed;
};

struct Tensor {
    void *hostAddr { nullptr };
    void *deviceAddr { nullptr };
    aclTensor *data { nullptr };
};

struct ArgScenario {
    bool hasCuSeq { false };
    bool hasSeqused { false };
};

struct ArgContext {
    // required input
    int64_t numHeadsQ { 0 };
    int64_t numHeadsKv { 0 };
    int64_t headDim { 0 };
    // optional input
    Tensor cuSeqlensQOptional {};
    Tensor cuSeqlensOriKvOptional {};
    Tensor cuSeqlensCmpKvOptional {};
    Tensor sequsedQOptional {};
    Tensor sequsedOriKvOptional {};
    Tensor sequsedCmpKvOptional {};
    Tensor cmpResidualKvOptional {};
    Tensor oriTopkLengthOptional {};
    Tensor cmpTopkLengthOptional {};
    int64_t batchSize { 0 };
    int64_t maxSeqlenQ { 0 };
    int64_t maxSeqlenOriKv { 0 };
    int64_t maxSeqlenCmpKv { 0 };
    int64_t oriTopk { 0 };
    int64_t cmpTopk { 0 };
    int64_t cmpRatio { 0 };
    int64_t oriMaskMode { 0 };
    int64_t cmpMaskMode { 0 };
    int64_t oriWinLeft { -1 };
    int64_t oriWinRight { -1 };
    char *layoutQOptional { nullptr };
    char *layoutKvOptional { nullptr };
    bool hasOriKv { true };
    bool hasCmpKv { true };
    // output
    Tensor metadata {};
};

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

aclnnStatus Init(int32_t deviceId, aclrtStream* stream)
{
    // Boilerplate, initialization.
    auto ret = aclInit(nullptr);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "aclInit failed. ERROR: %d", ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "aclrtSetDevice failed. ERROR: %d", ret);
    ret = aclrtCreateStream(stream);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "aclrtCreateStream failed. ERROR: %d", ret);
    return ACL_SUCCESS;
}

void Finalize(int32_t deviceId, aclrtStream stream)
{
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
}

aclnnStatus CreateTensor(aclDataType dataType, const std::vector<int64_t> &shape, Tensor &tensor)
{
    auto size = GetShapeSize(shape) * aclDataTypeSize(dataType);
    // Call aclrtMallocHost to allocate memory on the host side.
    auto ret = aclrtMallocHost(&(tensor.hostAddr), size);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "aclrtMallocHost failed. ERROR: %d", ret);
    memset(tensor.hostAddr, 0, size);
    // Call aclrtMalloc to allocate memory on the device side.
    ret = aclrtMalloc(&(tensor.deviceAddr), size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "aclrtMalloc failed. ERROR: %d", ret);
    // Call the aclCreateTensor API to create an aclTensor.
    tensor.data = aclCreateTensor(shape.data(), shape.size(), dataType, nullptr, 0, aclFormat::ACL_FORMAT_ND,
        shape.data(), shape.size(), tensor.deviceAddr);
    CHECK_LOG_RET(tensor.data != nullptr, ACL_ERROR_FAILURE, "aclCreateTensor failed");
    // Call aclrtMemcpy to copy host-side data to device-side memory.
    ret = aclrtMemcpy(tensor.deviceAddr, size, tensor.hostAddr, size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "aclrtMemcpy failed. ERROR: %d", ret);
    return ACL_SUCCESS;
}

void DestroyTensor(Tensor &tensor)
{
    if (tensor.data != nullptr) {
        aclDestroyTensor(tensor.data);
        tensor.data = nullptr;
    }
    if (tensor.deviceAddr != nullptr) {
        aclrtFree(tensor.deviceAddr);
        tensor.deviceAddr = nullptr;
    }
    if (tensor.hostAddr != nullptr) {
        aclrtFreeHost(tensor.hostAddr);
        tensor.hostAddr = nullptr;
    }
}

void DestroyArgs(ArgContext &context)
{
    DestroyTensor(context.metadata);
    DestroyTensor(context.cuSeqlensQOptional);
    DestroyTensor(context.cuSeqlensOriKvOptional);
    DestroyTensor(context.cuSeqlensCmpKvOptional);
    DestroyTensor(context.sequsedQOptional);
    DestroyTensor(context.sequsedOriKvOptional);
    DestroyTensor(context.sequsedCmpKvOptional);
    DestroyTensor(context.cmpResidualKvOptional);
    DestroyTensor(context.oriTopkLengthOptional);
    DestroyTensor(context.cmpTopkLengthOptional);

    if (context.layoutQOptional != nullptr) {
        free(context.layoutQOptional);
        context.layoutQOptional = nullptr;
    }
    if (context.layoutKvOptional != nullptr) {
        free(context.layoutKvOptional);
        context.layoutKvOptional = nullptr;
    }
}

aclnnStatus CreateArgs(const ArgScenario &scenario, ArgContext &context)
{
    ScopeGuard argsGuard([&] { DestroyArgs(context); });
    aclnnStatus ret;

    context.numHeadsQ = 64;
    context.numHeadsKv = 1;
    context.headDim = 512;
    ret = CreateTensor(aclDataType::ACL_INT32, { SMLAG_METADATA_TOTAL_SIZE }, context.metadata);     // 1024: Fix size
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "Create metadata failed. Error: %d", ret);
    context.oriTopk = 0;
    context.cmpTopk = 0;
    context.cmpRatio = 128;
    context.oriMaskMode = 4;
    context.cmpMaskMode = 3;
    context.oriWinLeft = 127;
    context.oriWinRight = 0;
    context.layoutQOptional = (char *)malloc(sizeof(char) * 16);
    context.layoutKvOptional = (char *)malloc(sizeof(char) * 16);
    CHECK_LOG_RET(context.layoutQOptional != nullptr, ACL_ERROR_FAILURE, "Create layoutQOptional failed");
    CHECK_LOG_RET(context.layoutKvOptional != nullptr, ACL_ERROR_FAILURE, "Create layoutKvOptional failed");
    strcpy(context.layoutQOptional, "BSND");                // BSND,TND
    strcpy(context.layoutKvOptional, "BSND");               // BSND,TND
    context.hasOriKv = true;
    context.hasCmpKv = true;

    context.batchSize = 4;
    context.maxSeqlenOriKv = 1024;
    context.maxSeqlenCmpKv = 1024;
    context.maxSeqlenQ = 1024;

    if (scenario.hasCuSeq) {
        // (B+1,), first element is always 0
        ret = CreateTensor(aclDataType::ACL_INT32, { context.batchSize + 1 }, context.cuSeqlensQOptional);
        CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "Create cuSeqlensQOptional failed. Error: %d", ret);
        ret = CreateTensor(aclDataType::ACL_INT32, { context.batchSize + 1 }, context.cuSeqlensOriKvOptional);
        CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "Create cuSeqlensOriKvOptional failed. Error: %d", ret);
        ret = CreateTensor(aclDataType::ACL_INT32, { context.batchSize + 1 }, context.cuSeqlensCmpKvOptional);
        CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "Create cuSeqlensCmpKvOptional failed. Error: %d", ret);
    }

    if (scenario.hasSeqused) {
        // (B,)
        ret = CreateTensor(aclDataType::ACL_INT32, { context.batchSize }, context.sequsedQOptional);
        CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "Create sequsedQOptional failed. Error: %d", ret);
        ret = CreateTensor(aclDataType::ACL_INT32, { context.batchSize }, context.sequsedOriKvOptional);
        CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "Create sequsedOriKvOptional failed. Error: %d", ret);
        ret = CreateTensor(aclDataType::ACL_INT32, { context.batchSize }, context.sequsedCmpKvOptional);
        CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "Create sequsedCmpKvOptional failed. Error: %d", ret);
    }

    if (context.hasCmpKv && context.cmpRatio != 1 && context.cmpMaskMode == 3) {
        // (B,)
        ret = CreateTensor(aclDataType::ACL_INT32, { context.batchSize }, context.cmpResidualKvOptional);
        CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "Create cmpResidualKvOptional failed. Error: %d", ret);
    }

    argsGuard.Dismiss();
    return ACL_SUCCESS;
}

int main() {
    // 1. (Boilerplate) Initialize device/stream. For details, see the external API list.
    // Set deviceId based on the actual device.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "Init acl failed. ERROR: %d", ret);
    ScopeGuard sysGuard([&] { Finalize(deviceId, stream); });

    // 2. Construct inputs and outputs based on the API definition.
    ArgScenario scenario {};
    scenario.hasCuSeq = false;
    scenario.hasSeqused = false;
    ArgContext context {};
    ret = CreateArgs(scenario, context);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "Create input arguments failed. ERROR: %d", ret);
    ScopeGuard argsGuard([&] { DestroyArgs(context); });

    // 3. Call the CANN operator library API. Replace it with the specific API.
    // Call the first-phase API of aclnnSparseFlashMlaGradMetadata.
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    void *workspaceAddr = nullptr;
    ret = aclnnSparseFlashMlaGradMetadataGetWorkspaceSize(
        context.cuSeqlensQOptional.data, context.cuSeqlensOriKvOptional.data, context.cuSeqlensCmpKvOptional.data,
        context.sequsedQOptional.data, context.sequsedOriKvOptional.data, context.sequsedCmpKvOptional.data,
        context.cmpResidualKvOptional.data, context.oriTopkLengthOptional.data, context.cmpTopkLengthOptional.data,
        context.numHeadsQ, context.numHeadsKv, context.headDim, context.batchSize, context.maxSeqlenQ,
        context.maxSeqlenOriKv, context.maxSeqlenCmpKv, context.oriTopk, context.cmpTopk, context.cmpRatio,
        context.oriMaskMode, context.cmpMaskMode, context.oriWinLeft, context.oriWinRight, context.layoutQOptional,
        context.layoutKvOptional, context.hasOriKv, context.hasCmpKv, context.metadata.data, &workspaceSize, &executor);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret,
        "aclnnSparseFlashMlaGradMetadataGetWorkspaceSize failed. ERROR: %d\n", ret);

    if (workspaceSize > static_cast<uint64_t>(0)) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "allocate workspace failed. ERROR: %d\n", ret);
    }
    ScopeGuard workspaceGuard([&] {
        if (workspaceAddr != nullptr) {
            aclrtFree(workspaceAddr);
            workspaceAddr = nullptr;
        }
    });

    // Call the second-phase API of aclnnSparseFlashMlaGradMetadata.
    ret = aclnnSparseFlashMlaGradMetadata(workspaceAddr, workspaceSize, executor, stream);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "aclnnSparseFlashMlaGradMetadata failed. ERROR: %d\n", ret);

    // 4. (Boilerplate) Synchronously wait for the task execution to complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "aclrtSynchronizeStream failed. ERROR: %d\n", ret);

    // 5. Print the output.
    SmlagMetadata result {};
    ret = aclrtMemcpy(&result, sizeof(result), context.metadata.deviceAddr, sizeof(result), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "aclrtMemcpy failed. ERROR: %d\n", ret);

    for (uint32_t i = 0; i < GRAD_METADATA_SIZE; ++i) {
        printf("AIC Core%u\n", i);
        printf("    Total Num                 : %u\n", result.gradMetadata[TOTAL_NUM]);
        printf("    Former Core Process Num   : %u\n", result.gradMetadata[FORMER_CORE_PROCESS_NUM]);
        printf("    Remain Core Process Num   : %u\n", result.gradMetadata[REMAIN_CORE_PROCESS_NUM]);
        printf("    Remain Core Num           : %u\n", result.gradMetadata[REMAIN_CORE_NUM]);
        printf("    Used Core Num             : %u\n", result.gradMetadata[USED_CORE_NUM]);
        printf("    Max Ori Kv Size           : %u\n", result.gradMetadata[MAX_ORI_KV_SIZE]);
        printf("    Max Cmp Kv Size           : %u\n", result.gradMetadata[MAX_CMP_KV_SIZE]);
    }

    return 0;
}
```
