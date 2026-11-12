# aclnnQuantSparseFlashMlaMetadata

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-21T12:50:14.983Z -->

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

- Description: `aclnnQuantSparseFlashMlaMetadata` is the preceding operator of `aclnnQuantSparseFlashMla`. It generates a load balancing task partition scheme for the subsequent Attention computation. This operator does not perform the actual Attention computation. Instead, it calculates in the AI CPU the start and end ranges of the Attention computation to be processed by each AI Core based on the input parameters. This maximizes the utilization of computing resources and avoids load imbalance across Cores.

  **This operator is not recommended for standalone use. Use it with the aclnnQuantSparseFlashMla operator to form a complete workflow.**
    1. Accepts the input data shape information of the aclnnQuantSparseFlashMla operator API, including batchSize, qSeqen, kvSeqlen, and mask. The input is divided into blocks, and the computation time is simulated to evenly distribute the blocks across the available cores. This reduces the overall computation time of the aclnnQuantSparseFlashMla operator and improves hardware utilization.
    2. After the allocation result is output, it is used as input for the aclnnQuantSparseFlashMla operator.
    3. The allocation result includes the start and end points of the basic blocks of each AIC core, and the FD task information of each AIV core. For details, see [Example](#example).

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call "aclnnQuantSparseFlashMlaMetadataGetWorkspaceSize" first to obtain the workspace size, and then call "aclnnQuantSparseFlashMlaMetadata" to perform the computation

``` cpp
aclnnStatus aclnnQuantSparseFlashMlaMetadataGetWorkspaceSize(
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
    int64_t            quantMode,
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
aclnnStatus aclnnQuantSparseFlashMlaMetadata(
    void              *workspace,
    uint64_t           workspaceSize,
    aclOpExecutor     *executor,
    aclrtStream        stream)
```

## aclnnQuantSparseFlashMlaMetadataGetWorkspaceSize

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
      <td>cuSeqlensQOptional(aclTensor*)</td>
      <td>Input</td>
      <td>Indicates the valid Sequence Length of Query in different batches.</td>
      <td><ul><li>Supports empty tensor</li><li>Required in scenarios where layoutQOptional is TND.</li><li>The first value is an additional value, fixed to 0.</li><li>The shape is fixed to (B+1, ).</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>1D</td>
      <td>√</td>
    </tr>
    <tr>
      <td>cuSeqlensOriKvOptional (aclTensor*)</td>
      <td>Input</td>
      <td>Indicates the valid Sequence Length of oriKvOptional in different batches.</td>
      <td><ul><li>Supports empty tensors.</li><li>Required in scenarios where layoutKvOptional is TND.</li><li>The first value is an additional value and is fixed to 0.</li><li>Shape fixed to (B+1, ).</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>1D</td>
      <td>√</td>
    </tr>
    <tr>
      <td>cuSeqlensCmpKvOptional (aclTensor*)</td>
      <td>Input</td>
      <td>Indicates the valid Sequence Length of cmpKvOptional in different Batches.</td>
      <td><ul><li>Supports empty tensors.</li><li>Required in scenarios where layoutKvOptional is TND.</li><li>The first value is an additional value and is fixed to 0.</li><li>Shape fixed to (B+1, ).</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>1-dimensional</td>
      <td>√</td>
    </tr>
    <tr>
      <td>sequsedQOptional (aclTensor*)</td>
      <td>Input</td>
      <td>Indicates different Batch Sequence Lengths of the Query actually involved in the operation.</td>
      <td><ul><li>Supports empty tensors.</li><li>The shape is fixed to (B, ).</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>1-dimensional</td>
      <td>√</td>
    </tr>
    <tr>
      <td>sequsedOriKvOptional (aclTensor*)</td>
      <td>Input</td>
      <td>Indicates the Sequence Length of oriKvOptional actually involved in the operation for different batches.</td>
      <td><ul><li>Supports empty tensors.</li><li>The shape is fixed to (B, ).</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>1D</td>
      <td>√</td>
    </tr>
    <tr>
      <td>sequsedCmpKvOptional (aclTensor*)</td>
      <td>Input</td>
      <td>Indicates the Sequence Length actually involved in the operation of cmpKvOptional in different Batches.</td>
      <td><ul><li>Supports empty tensor.</li><li>shape fixed to (B, ).</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>1-D</td>
      <td>√</td>
    </tr>
    <tr>
      <td>cmpResidualKvOptional (aclTensor*)</td>
      <td>Input</td>
      <td>Indicates the remainder of the compressed Sequence Length of cmpResidualKvOptional in different Batches, and works with cmpRatio to implement mask and load calculation for the cmpKvOptional part.</td>
      <td><ul><li>Supports empty tensor.</li><li>Required in use cases where cmpRatio is not 1.</li><li>Shape fixed to (B, ).</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>1D</td>
      <td>√</td>
    </tr>
    <tr>
      <td>oriTopkLengthOptional (aclTensor*)</td>
      <td>Input</td>
      <td>Used to identify the length of oriSparseIndicesOptional actually involved in the calculation.</td>
      <td>        <ul>
          <li>Supports empty tensor.</li>
          <li>Required in use cases where oriTopk!=0 and oriMaskMode=0.</li>
          <li>If layoutQ is BSND: (B, S1, N2)</li>
          <li>If layoutQ is TND: (T1, N2)</li>
        </ul>
      </td>
      <td>INT32</td>
      <td>ND</td>
      <td>2D, 3D</td>
      <td>√</td>
    </tr>
    <tr>
      <td>cmpTopkLengthOptional (aclTensor*)</td>
      <td>Input</td>
      <td>Used to identify the length of calculation actually involved in cmpSparseIndicesOptional.</td>
      <td>        <ul>
          <li>Supports empty tensors.</li>
          <li>Required in scenarios where cmpTopk!=0 and cmpMaskMode=0.</li>
          <li>When layoutQ is BSND: (B, S1, N2)</li>
          <li>When layoutQ is TND: (T1, N2)</li>
        </ul>
      </td>
      <td>INT32</td>
      <td>ND</td>
      <td>2D, 3D</td>
      <td>√</td>
    </tr>
    <tr>
      <td>numHeadsQ(int64_t)</td>
      <td>Input</td>
      <td>Indicates the number of Query heads.</td>
      <td>Value range: 1-128</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>numHeadsKv(int64_t)</td>
      <td>Input</td>
      <td>Number of heads corresponding to Key and Value.</td>
      <td>Currently only supports 1.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>headDim (int64_t)</td>
      <td>Input</td>
      <td>Dimension of the attention head.</td>
      <td>Currently only supports 512.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>quantMode (int64_t)</td>
      <td>Input</td>
      <td>Indicates the quantization mode.</td>
      <td><ul><li>1: Q, K, and V use per-tensor quantization, and the data type of Q, K, and V is HIFLOAT8.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>batchSize (int64_t)</td>
      <td>Input</td>
      <td>Indicates the number of batches.</td>
      <td><ul><li>Supports non-negative values.</li><li>Recommended value: 0.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>maxSeqlenQ(int64_t)</td>
      <td>Input</td>
      <td>Indicates the maximum Sequence Length of Query.</td>
      <td><ul><li>Supports non-negative numbers.</li><li>Recommended value is 0.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>maxSeqlenOriKv(int64_t)</td>
      <td>Input</td>
      <td>Indicates the maximum Sequence Length of oriKvOptional.</td>
      <td><ul><li>Supports non-negative numbers.</li><li>Recommended value is 0.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>maxSeqlenCmpKv (int64_t)</td>
      <td>Input</td>
      <td>Indicates the maximum Sequence Length of cmpKvOptional.</td>
      <td><ul><li>Supports non-negative numbers.</li><li>Recommended value is 0.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>oriTopk (int64_t)</td>
      <td>Input</td>
      <td>Indicates the number of key sparse tokens filtered from oriKvOptional. 0 indicates a non-sparse use case.</td>
      <td><ul><li>Currently only supports 0.</li><li>Recommended value is 0.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cmpTopk (int64_t)</td>
      <td>Input</td>
      <td>Indicates the number of key sparse tokens filtered from cmpKvOptional. 0 indicates a non-sparse use case.</td>
      <td><ul><li>Supports non-negative numbers.</li><li>Recommended value: 0.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cmpRatio (int64_t)</td>
      <td>Input</td>
      <td>Indicates the compression ratio of cmpKvOptional.</td>
      <td><ul><li>Currently supports 1 to 128.</li><li>Recommended value: 1, indicating no compression.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>oriMaskMode (int64_t)</td>
      <td>Input</td>
      <td>Indicates the mask mode of the computation between q and oriKvOptional.</td>
      <td><ul><li>Currently supports:<br>0: No Mask.<br>3: RightDownCausal mode.<br>4: Band mode.</li><li>Recommended value: 0.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cmpMaskMode (int64_t)</td>
      <td>Input</td>
      <td>Indicates the mask mode of the computation between q and cmpKvOptional.</td>
      <td><ul><li>Currently supports:<br>0: No Mask.<br>3: RightDownCausal mode.</li><li>Recommended value: 0.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>oriWinLeft (int64_t)</td>
      <td>Input</td>
      <td>Indicates the number of past tokens that q attends to in the calculation of q and oriKvOptional.</td>
      <td><ul><li>Value range: ≥-1. -1 indicates infinity.</li><li>Recommended value: -1.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>oriWinRight (int64_t)</td>
      <td>Input</td>
      <td>Indicates the number of future tokens computed by q in the calculation of q and oriKvOptional.</td>
      <td><ul><li>Value range: ≥-1, -1 indicates infinity.</li><li>Recommended value: -1</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layoutQOptional (char*)</td>
      <td>Input</td>
      <td>Indicates the layout of Query.</td>
      <td><ul><li>Supports BSND and TND.</li><li>Recommended value: BSND.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layoutKvOptional (char*)</td>
      <td>Input</td>
      <td>Indicates the layout format of Key.</td>
      <td><ul><li>Supports BSND, TND, and PA_BBND.</li><li>Recommended value: BSND.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>hasOriKv (bool)</td>
      <td>Input</td>
      <td>Used to identify whether oriKvOptional is present.</td>
      <td><ul><li>true: oriKvOptional is included.</li><li>false: oriKvOptional is not included.</li><li>Recommended value: true.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>hasCmpKv (bool)</td>
      <td>Input</td>
      <td>Used to identify whether cmpKvOptional is included.</td>
      <td><ul><li>true: cmpKvOptional is included.</li><li>false: cmpKvOptional is not included.</li><li>Recommended value: true.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>metadata (aclTensor*)</td>
      <td>Output</td>
      <td>Indicates the output of the load balancing result.</td>
      <td>-</td>
      <td>INT32</td>
      <td>ND</td>
      <td>1-D, shape fixed to (1024)</td>
      <td>×</td>
    </tr>
    <tr>
      <td>workspaceSize (uint64_t*)</td>
      <td>Output</td>
      <td>Returns the workspace size to be applied by the device side.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor (aclOpExecutor**)</td>
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

    <table style="undefined;table-layout: fixed; width: 1134px"><colgroup>
    <col style="width: 319px">
    <col style="width: 144px">
    <col style="width: 671px">
    </colgroup>
    <thead>
      <tr>
        <th>Return value</th>
        <th>Error code</th>
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
        <td>workspaceSize or executor is a null pointer, or cuSeqlensQOptional, cuSeqlensOriKvOptional, cuSeqlensCmpKvOptional, sequsedQOptional, sequsedOriKvOptional, sequsedCmpKvOptional, cmpResidualKvOptional, oriTopkLengthOptional, or cmpTopkLengthOptional is a null pointer after Contiguous conversion.</td>
      </tr>
      <tr>
        <td rowspan="1">ACLNN_ERR_PARAM_INVALID</td>
        <td rowspan="1">161002</td>
        <td>The specifications of parameters cuSeqlensQOptional, cuSeqlensOriKvOptional, cuSeqlensCmpKvOptional, sequsedQOptional, sequsedOriKvOptional, sequsedCmpKvOptional, cmpResidualKvOptional, oriTopkLengthOptional, cmpTopkLengthOptional, numHeadsQ, numHeadsKv, headDim, quantMode, batchSize, maxSeqlenQ, maxSeqlenOriKv, maxSeqlenCmpKv, oriTopk, cmpTopk, cmpRatio, oriMaskMode, cmpMaskMode, oriWinLeft, oriWinRight, layoutQOptional, layoutKvOptional, hasOriKv, and hasCmpKv are out of the supported range.</td>
      </tr>
    </tbody>
    </table>

## aclnnQuantSparseFlashMlaMetadata

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
            <td>Workspace memory address applied by the Device.</td>
        </tr>
        <tr>
            <td>workspaceSize</td>
            <td>Input</td>
            <td>Workspace size applied by the Device, obtained by the first-phase API aclnnQuantSparseFlashMlaMetadataGetWorkspaceSize.</td>
        </tr>
        <tr>
            <td>executor</td>
            <td>Input</td>
            <td>Executor of the op, including the computation flow of the operator.</td>
        </tr>
        <tr>
            <td>stream</td>
            <td>Input</td>
            <td>Specifies the stream for executing tasks.</td>
        </tr>
    </tbody>
    </table>

- **Return value**

    aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

  - aclnnQuantSparseFlashMlaMetadata defaults to a deterministic implementation.
  - B (Batch) indicates the batch size of input samples.
  - Batch value rules
    - Obtain the Batch information in sequsedQOptional first.
    - When sequsedQOptional is not passed, layoutQOptional is TND, and cuSeqlensQOptional is passed, obtain the Batch information in cuSeqlensQOptional.
    - In all other cases, use batchSize.
  - Query Sequence Length value rules
    - Obtain the Sequence Length information in sequsedQOptional first.
    - When sequsedQOptional is not passed, layoutQOptional is TND, and cuSeqlensQOptional is passed, obtain the Sequence Length information in cuSeqlensQOptional.
    - In all other cases, use maxSeqlenQ.
    - The rules for obtaining the Sequence Length of oriKvOptional and cmpKvOptional are the same as those for Query.
  - BSND use case
    - When the passed layoutQOptional is "BSND", the maxSeqlenQ parameter is mandatory when sequsedQOptional is not passed.
    - When the passed layoutKvOptional is "BSND", if hasOriKv is true, maxSeqlenOriKv is mandatory when sequsedOriKvOptional is not passed; if hasCmpKv is true, maxSeqlenCmpKv is mandatory when sequsedCmpKvOptional is not passed.
  - TND use case
    - When the passed layoutQOptional is "TND", cuSeqlensQOptional is mandatory.
    - When the passed layoutKvOptional is "TND", if hasOriKv is true, cuSeqlensOriKvOptional is mandatory; if hasCmpKv is true, cuSeqlensCmpKvOptional is mandatory.
  - TopkLength parameter
    - oriTopkLengthOptional and cmpTopkLengthOptional indicate the length actually involved in calculation of ori/cmp sparseIndices. The value cannot be greater than the size of the last dimension of sparseIndices. When sequsedQOptional is passed, the values of the valid part corresponding to topkLength must be greater than or equal to 0.
    - When oriMaskMode=0 and oriSparseIndicesOptional is not empty, oriTopkLengthOptional is mandatory.
    - When cmpMaskMode=0 and cmpSparseIndicesOptional is not empty, cmpTopkLengthOptional is mandatory.

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
#include "aclnnop/aclnn_quant_sparse_flash_mla_metadata.h"

#define CHECK_LOG_RET(cond, ret_val, fmt, ...)                                                                         \
    do {                                                                                                               \
        if (!(cond)) {                                                                                                 \
            printf(fmt "\n", ##__VA_ARGS__);                                                                           \
            return (ret_val);                                                                                          \
        }                                                                                                              \
    } while (0)

// Refer to quant_sparse_flash_mla_metadata.h
constexpr uint32_t AIC_CORE_MAX_NUM = 36;
constexpr uint32_t AIV_CORE_MAX_NUM = 72;
constexpr uint32_t MQSMLA_METADATA_TOTAL_SIZE = 1024;
constexpr uint32_t FA_METADATA_SIZE = 9;
constexpr uint32_t FD_METADATA_SIZE = 8;

// FA Metadata Index Definitions
constexpr uint32_t FA_CORE_ENABLE_INDEX = 0;
constexpr uint32_t FA_BN2_START_INDEX = 1;
constexpr uint32_t FA_M_START_INDEX = 2;
constexpr uint32_t FA_S2_START_INDEX = 3;
constexpr uint32_t FA_BN2_END_INDEX = 4;
constexpr uint32_t FA_M_END_INDEX = 5;
constexpr uint32_t FA_S2_END_INDEX = 6;
constexpr uint32_t FA_FIRST_FD_DATA_WORKSPACE_IDX_INDEX = 7;
constexpr uint32_t FA_S2_MAX_NUM = 8;

// FD Metadata Index Definitions
constexpr uint32_t FD_CORE_ENABLE_INDEX = 0;
constexpr uint32_t FD_BN2_IDX_INDEX = 1;
constexpr uint32_t FD_M_IDX_INDEX = 2;
constexpr uint32_t FD_WORKSPACE_IDX_INDEX = 3;
constexpr uint32_t FD_WORKSPACE_NUM_INDEX = 4;
constexpr uint32_t FD_M_START_INDEX = 5;
constexpr uint32_t FD_M_NUM_INDEX = 6;

struct MqsmlaMetadata {
    uint32_t faMetadata[AIC_CORE_MAX_NUM][FA_METADATA_SIZE];
    uint32_t fdMetadata[AIV_CORE_MAX_NUM][FD_METADATA_SIZE];
};

struct ScopeGuard {
    explicit ScopeGuard(std::function<void()> onExitScope) : m_exitFunc(std::move(onExitScope)), m_isDismissed(false)
    {
    }
    // Do not copy.
    ScopeGuard(const ScopeGuard &) = delete;
    ScopeGuard &operator=(const ScopeGuard &) = delete;

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
    void *hostAddr{nullptr};
    void *deviceAddr{nullptr};
    aclTensor *data{nullptr};
};

struct ArgScenario {
    bool hasCuSeq{false};
    bool hasSeqused{false};
};

struct ArgContext {
    // required input
    int64_t numHeadsQ{0};
    int64_t numHeadsKv{0};
    int64_t headDim{0};
    int64_t quantMode{1};
    // optional input
    Tensor cuSeqlensQOptional{};
    Tensor cuSeqlensOriKvOptional{};
    Tensor cuSeqlensCmpKvOptional{};
    Tensor sequsedQOptional{};
    Tensor sequsedOriKvOptional{};
    Tensor sequsedCmpKvOptional{};
    Tensor cmpResidualKvOptional{};
    Tensor oriTopkLengthOptional{};
    Tensor cmpTopkLengthOptional{};
    int64_t batchSize{0};
    int64_t maxSeqlenQ{0};
    int64_t maxSeqlenOriKv{0};
    int64_t maxSeqlenCmpKv{0};
    int64_t oriTopk{0};
    int64_t cmpTopk{0};
    int64_t cmpRatio{0};
    int64_t oriMaskMode{0};
    int64_t cmpMaskMode{0};
    int64_t oriWinLeft{-1};
    int64_t oriWinRight{-1};
    char *layoutQOptional{nullptr};
    char *layoutKvOptional{nullptr};
    bool hasOriKv{true};
    bool hasCmpKv{true};
    // output
    Tensor metadata{};
};

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

aclnnStatus Init(int32_t deviceId, aclrtStream *stream)
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
    // Call aclrtMallocHost to allocate host-side memory.
    auto ret = aclrtMallocHost(&(tensor.hostAddr), size);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "aclrtMallocHost failed. ERROR: %d", ret);
    memset(tensor.hostAddr, 0, size);
    // Call aclrtMalloc to allocate device-side memory.
    ret = aclrtMalloc(&(tensor.deviceAddr), size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "aclrtMalloc failed. ERROR: %d", ret);
    // Call aclCreateTensor to create an aclTensor.
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
    context.quantMode = 1;
    ret = CreateTensor(aclDataType::ACL_INT32, {MQSMLA_METADATA_TOTAL_SIZE}, context.metadata); // 1024: Fix size
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
    strcpy(context.layoutQOptional, "BSND");  // BSND,TND
    strcpy(context.layoutKvOptional, "BSND"); // BSND,TND,PA_BBND
    context.hasOriKv = true;
    context.hasCmpKv = true;

    context.batchSize = 4;
    context.maxSeqlenOriKv = 1024;
    context.maxSeqlenCmpKv = 1024;
    context.maxSeqlenQ = 1024;

    if (scenario.hasCuSeq) {
        // (B+1,), first element is always 0
        ret = CreateTensor(aclDataType::ACL_INT32, {context.batchSize + 1}, context.cuSeqlensQOptional);
        CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "Create cuSeqlensQOptional failed. Error: %d", ret);
        ret = CreateTensor(aclDataType::ACL_INT32, {context.batchSize + 1}, context.cuSeqlensOriKvOptional);
        CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "Create cuSeqlensOriKvOptional failed. Error: %d", ret);
        ret = CreateTensor(aclDataType::ACL_INT32, {context.batchSize + 1}, context.cuSeqlensCmpKvOptional);
        CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "Create cuSeqlensCmpKvOptional failed. Error: %d", ret);
    }

    if (scenario.hasSeqused) {
        // (B,)
        ret = CreateTensor(aclDataType::ACL_INT32, {context.batchSize}, context.sequsedQOptional);
        CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "Create sequsedQOptional failed. Error: %d", ret);
        ret = CreateTensor(aclDataType::ACL_INT32, {context.batchSize}, context.sequsedOriKvOptional);
        CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "Create sequsedOriKvOptional failed. Error: %d", ret);
        ret = CreateTensor(aclDataType::ACL_INT32, {context.batchSize}, context.sequsedCmpKvOptional);
        CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "Create sequsedCmpKvOptional failed. Error: %d", ret);
    }

    if (context.hasCmpKv && context.cmpRatio != 1 && context.cmpMaskMode == 3) {
        // (B,)
        ret = CreateTensor(aclDataType::ACL_INT32, {context.batchSize}, context.cmpResidualKvOptional);
        CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "Create cmpResidualKvOptional failed. Error: %d", ret);
    }

    argsGuard.Dismiss();
    return ACL_SUCCESS;
}

int main()
{
    // 1. (Boilerplate) Device/stream initialization. For details, see the external API list.
    // Set deviceId based on the actual device in use.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "Init acl failed. ERROR: %d", ret);
    ScopeGuard sysGuard([&] { Finalize(deviceId, stream); });

    // 2. Construct the input and output according to the API definition.
    ArgScenario scenario{};
    scenario.hasCuSeq = false;
    scenario.hasSeqused = false;
    ArgContext context{};
    ret = CreateArgs(scenario, context);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "Create input arguments failed. ERROR: %d", ret);
    ScopeGuard argsGuard([&] { DestroyArgs(context); });

    // 3. Call the CANN operator library API. Replace it with the specific API.
    // Call the first-phase API of aclnnQuantSparseFlashMlaMetadata.
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    void *workspaceAddr = nullptr;
    ret = aclnnQuantSparseFlashMlaMetadataGetWorkspaceSize(
        context.cuSeqlensQOptional.data, context.cuSeqlensOriKvOptional.data, context.cuSeqlensCmpKvOptional.data,
        context.sequsedQOptional.data, context.sequsedOriKvOptional.data, context.sequsedCmpKvOptional.data,
        context.cmpResidualKvOptional.data, context.oriTopkLengthOptional.data, context.cmpTopkLengthOptional.data,
        context.numHeadsQ, context.numHeadsKv, context.headDim, context.quantMode, context.batchSize,
        context.maxSeqlenQ, context.maxSeqlenOriKv, context.maxSeqlenCmpKv, context.oriTopk, context.cmpTopk,
        context.cmpRatio, context.oriMaskMode, context.cmpMaskMode, context.oriWinLeft, context.oriWinRight,
        context.layoutQOptional, context.layoutKvOptional, context.hasOriKv, context.hasCmpKv, context.metadata.data,
        &workspaceSize, &executor);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "aclnnQuantSparseFlashMlaMetadataGetWorkspaceSize failed. ERROR: %d\n", ret);

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

    // Call the second-phase API of aclnnQuantSparseFlashMlaMetadata.
    ret = aclnnQuantSparseFlashMlaMetadata(workspaceAddr, workspaceSize, executor, stream);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "aclnnQuantSparseFlashMlaMetadata failed. ERROR: %d\n", ret);

    // 4. (boilerplate) Synchronize and wait for the task execution to complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "aclrtSynchronizeStream failed. ERROR: %d\n", ret);

    // 5. Print output.
    MqsmlaMetadata result{};
    ret = aclrtMemcpy(&result, sizeof(result), context.metadata.deviceAddr, sizeof(result), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "aclrtMemcpy failed. ERROR: %d\n", ret);

    for (uint32_t i = 0; i < AIC_CORE_MAX_NUM; ++i) {
        printf("AIC Core%u\n", i);
        printf("    Core Enable : %u\n", result.faMetadata[i][FA_CORE_ENABLE_INDEX]);
        printf("    Start BN2   : %u\n", result.faMetadata[i][FA_BN2_START_INDEX]);
        printf("    Start M     : %u\n", result.faMetadata[i][FA_M_START_INDEX]);
        printf("    Start S2    : %u\n", result.faMetadata[i][FA_S2_START_INDEX]);
        printf("    End BN2     : %u\n", result.faMetadata[i][FA_BN2_END_INDEX]);
        printf("    End M       : %u\n", result.faMetadata[i][FA_M_END_INDEX]);
        printf("    End S2      : %u\n", result.faMetadata[i][FA_S2_END_INDEX]);
        printf("    First Workspace Index : %u\n", result.faMetadata[i][FA_FIRST_FD_DATA_WORKSPACE_IDX_INDEX]);
        printf("    Max S2 Block Num : %u\n", result.faMetadata[i][FA_S2_MAX_NUM]);
    }
    for (uint32_t i = 0; i < AIV_CORE_MAX_NUM; ++i) {
        printf("AIV Core%u\n", i);
        printf("    Core Enable             : %u\n", result.fdMetadata[i][FD_CORE_ENABLE_INDEX]);
        printf("    FD Task BN2 Idx         : %u\n", result.fdMetadata[i][FD_BN2_IDX_INDEX]);
        printf("    FD Task M Idx           : %u\n", result.fdMetadata[i][FD_M_IDX_INDEX]);
        printf("    FD Task S2 Idx          : %u\n", result.fdMetadata[i][FD_WORKSPACE_IDX_INDEX]);
        printf("    FD Task Workspace Num   : %u\n", result.fdMetadata[i][FD_WORKSPACE_NUM_INDEX]);
        printf("    FD Subtask M Start      : %u\n", result.fdMetadata[i][FD_M_START_INDEX]);
        printf("    FD Subtask M Num        : %u\n", result.fdMetadata[i][FD_M_NUM_INDEX]);
    }

    return 0;
}
```
