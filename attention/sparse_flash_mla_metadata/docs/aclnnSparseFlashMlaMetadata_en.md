# aclnnSparseFlashMlaMetadata

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-21T13:28:11.227Z -->

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

- Description: `aclnnSparseFlashMlaMetadata` is the preceding operator of the `aclnnSparseFlashMla` operator and generates a load balancing task partitioning scheme for subsequent Attention computation. This operator does not perform actual Attention computation. Instead, it uses the input parameters to compute in AI CPU the start and end ranges of Attention computation to be processed by each AI Core, maximizing the utilization of computing resources and avoiding load imbalance across Cores.

  **This operator is not recommended for standalone use. It is recommended to use it together with the aclnnSparseFlashMla operator to form a complete workflow.**
- Use case abbreviations: SWA (Sliding Window Attention), CSA (Compressed Sparse Attention), HCA (Heavily Compressed Attention).
- Formula:

  This operator is an AICPU-scheduled operator and does not involve numerical computation. The core workflow is: parse the Q/KV sequence lengths of each batch → calculate the valid S2 range of each S1G block based on the mask mode → perform load balancing core partitioning based on the cost model → output the core partitioning metadata.

  The output metadata tensor has a shape of (1024,) and a data type of INT32. Its internal structure is as follows:

  - FA metadata area (AIC_CORE_NUM × 9 INT32 values), containing the FA-stage task information of each AICore:

    | Index | Meaning |
    | :--- | :--- |
    | 0 | core_enable, Whether the core is enabled. |
    | 1 | bn2_start, BN2 start index. |
    | 2 | m_start, M (S1G) start index. |
    | 3 | s2_start, S2 start index. |
    | 4 | bn2_end, BN2 end index. |
    | 5 | m_end, M end index. |
    | 6 | s2_end, S2 end index. |
    | 7 | first_fd_data_workspace_idx, workspace offset of the first FD reduction data. |
    | 8 | max_s2_block_num, maximum number of s2 blocks allocated to a single core. |

  - FD metadata area (AIV_CORE_NUM × 8 INT32 values), containing the FD reduction task information of each AIVCore:

    | Index | Meaning |
    | :--- | :--- |
    | 0 | core_enable, Whether the core is enabled. |
    | 1 | bn2_idx, BN2 index of the reduction task. |
    | 2 | m_idx, M index of the reduction task. |
    | 3 | workspace_idx, storage location of the reduction data in the workspace. |
    | 4 | workspace_num, number of S2 splits across cores. |
    | 5 | m_start, M axis start. |
    | 6 | m_num, number of rows in the M axis. |

- Symbol Description

  | Symbol                | Meaning                                                      |
  | ------------------- | --------------------------------------------------------- |
  | B                   | Batch Size                                                |
  | N1/N2               | Number of Query/KV heads.                                              |
  | D                   | Dimension of each attention head.                                        |
  | G                   | GQA grouping ratio, G=N1/N2.                                        |
  | S1/S2               | Query/KV sequence length.                                          |
  | S1G                 | Block index along the S1×G direction.                                        |
  | mBaseSize           | Base block size of the M axis, equal to G.                                      |
  | s2BaseSize          | Base block size of the S2 axis, fixed to 512.                                  |

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). You must first call the `aclnnSparseFlashMlaMetadataGetWorkspaceSize` API to obtain the workspace size required for the computation and the executor that contains the operator computation process, and then call the `aclnnSparseFlashMlaMetadata` API to perform the actual computation.

```c++
aclnnStatus aclnnSparseFlashMlaMetadataGetWorkspaceSize(
    const aclTensor *cuSeqlensQOptional,
    const aclTensor *cuSeqlensOriKvOptional,
    const aclTensor *cuSeqlensCmpKvOptional,
    const aclTensor *sequsedQOptional,
    const aclTensor *sequsedOriKvOptional,
    const aclTensor *sequsedCmpKvOptional,
    const aclTensor *cmpResidualKvOptional,
    const aclTensor *oriTopkLengthOptional,
    const aclTensor *cmpTopkLengthOptional,
    int64_t          numHeadsQ,
    int64_t          numHeadsKv,
    int64_t          headDim,
    int64_t          batchSize,
    int64_t          maxSeqlenQ,
    int64_t          maxSeqlenOriKv,
    int64_t          maxSeqlenCmpKv,
    int64_t          oriTopk,
    int64_t          cmpTopk,
    int64_t          cmpRatio,
    int64_t          oriMaskMode,
    int64_t          cmpMaskMode,
    int64_t          oriWinLeft,
    int64_t          oriWinRight,
    const char      *layoutQOptional,
    const char      *layoutKvOptional,
    bool             hasOriKv,
    bool             hasCmpKv,
    const aclTensor *metaData,
    uint64_t        *workspaceSize,
    aclOpExecutor  **executor)
```

```c++
aclnnStatus aclnnSparseFlashMlaMetadata(
    void          *workspace,
    uint64_t       workspaceSize,
    aclOpExecutor *executor,
    aclrtStream    stream)
```

## aclnnSparseFlashMlaMetadataGetWorkspaceSize

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1500px"><colgroup>
    <col style="width: 220px">
    <col style="width: 100px">
    <col style="width: 300px">
    <col style="width: 300px">
    <col style="width: 120px">
    <col style="width: 100px">
    <col style="width: 160px">
    <col style="width: 100px">
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
      <td>Represents the number of valid tokens of q in each batch (prefix sum form).</td>
      <td><ul><li>Supports empty Tensor.</li><li>Shape fixed to (B+1, ).</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>1D</td>
      <td>√</td>
    </tr>
    <tr>
      <td>cuSeqlensOriKvOptional (aclTensor*)</td>
      <td>Input</td>
      <td>Represents the number of valid tokens of oriKv in each batch (prefix sum form).</td>
      <td><ul><li>Supports empty tensors.</li><li>Shape is fixed to (B+1, ).</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>1D</td>
      <td>√</td>
    </tr>
    <tr>
      <td>cuSeqlensCmpKvOptional(aclTensor*)</td>
      <td>Input</td>
      <td>Represents the number of valid tokens of cmpKv in different batches (prefix sum form).</td>
      <td><ul><li>Supports empty tensors.</li><li>Shape is fixed to (B+1, ).</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>1D</td>
      <td>√</td>
    </tr>
    <tr>
      <td>sequsedQOptional (aclTensor*)</td>
      <td>Input</td>
      <td>Indicates the number of q tokens actually participating in computation in each batch.</td>
      <td><ul><li>Supports empty tensors.</li><li>The shape is fixed to (B, ).</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>(B,)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>sequsedOriKvOptional (aclTensor*)</td>
      <td>Input</td>
      <td>Represents the number of tokens of oriKv actually participating in computation in different batches.</td>
      <td><ul><li>Supports empty tensor.</li><li>shape is fixed to (B, ).</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>1-D</td>
      <td>√</td>
    </tr>
    <tr>
      <td>sequsedCmpKvOptional (aclTensor*)</td>
      <td>Input</td>
      <td>Indicates the number of tokens of cmpKv actually participating in the computation in each batch.</td>
      <td><ul><li>Supports empty tensors.</li><li>The shape is fixed to (B, ).</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>1D</td>
      <td>√</td>
    </tr>
    <tr>
      <td>cmpResidualKvOptional (aclTensor*)</td>
      <td>Input</td>
      <td>Compressed KV remainder, used for recovery of the KV length before compression used by the cmp-side mask based on cmp_len * cmpRatio + residual.</td>
      <td>Mandatory in CSA, HCA, and use cases where cmpRatio is not 1 and cmpMaskMode is 3. Usable when layoutKvOptional is BSND, TND, or PA_BBND.</td>
      <td>INT32</td>
      <td>ND</td>
      <td>1D</td>
      <td>√</td>
    </tr>
    <tr>
      <td>oriTopkLengthOptional (aclTensor*)</td>
      <td>Input</td>
      <td>Represents the number of key sparse tokens in the oriKvOptional part corresponding to different q tokens.</td>
      <td><ul><li>Must be passed in for the SWA sparse ori_kv use case. An empty tensor is supported in other scenarios.</li><li>The shape is (B, S1, N2) or (T1, N2).</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>2D, 3D</td>
      <td>√</td>
    </tr>
    <tr>
      <td>cmpTopkLengthOptional (aclTensor*)</td>
      <td>Input</td>
      <td>Represents the number of key sparse tokens in cmpKvOptional corresponding to different q tokens.</td>
      <td><ul><li>Supports empty tensor.</li><li>The shape is (B, S1, N2) or (T1, N2).</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>2D, 3D</td>
      <td>√</td>
    </tr>
    <tr>
      <td>numHeadsQ (int64_t)</td>
      <td>Input</td>
      <td>Number of query heads.</td>
      <td>Supports [1, 128].</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>numHeadsKv (int64_t)</td>
      <td>Input</td>
      <td>Number of KV heads.</td>
      <td>Only 1 is supported.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>headDim (int64_t)</td>
      <td>Input</td>
      <td>Dimension of each attention head.</td>
      <td>Only 512 is supported.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>batchSize (int64_t)</td>
      <td>Input</td>
      <td>Batch size of the input samples.</td>
      <td>No manual specification is required when layoutQOptional is TND; the recommended value is 0.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>maxSeqlenQ (int64_t)</td>
      <td>Input</td>
      <td>Maximum number of valid q tokens in all batches.</td>
      <td>If 0 is passed in, the value is obtained through interface derivation. The recommended value is 0.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>maxSeqlenOriKv (int64_t)</td>
      <td>Input</td>
      <td>Maximum number of valid oriKv tokens in all batches.</td>
      <td>When 0 is passed in, interface derivation is used. The recommended value is 0.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>maxSeqlenCmpKv (int64_t)</td>
      <td>Input</td>
      <td>Maximum number of valid cmpKv tokens across all batches.</td>
      <td>When 0 is passed in, interface derivation is used. The recommended value is 0.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>oriTopk (int64_t)</td>
      <td>Input</td>
      <td>Number of sparse tokens selected from oriKv.</td>
      <td>In the SWA sparse ori_kv use case, this value is the last dimension K of the main operator oriSparseIndicesOptional and must be greater than 0; the recommended value is 0 in other scenarios.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cmpTopk (int64_t)</td>
      <td>Input</td>
      <td>Number of sparse tokens filtered from cmpKv.</td>
      <td>In the CSA use case of Atlas A2 products and Atlas A3 products, any integer within [1, 8192] is supported; the value is 0 in the SWA and HCA use cases, and the recommended value is 0.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cmpRatio (int64_t)</td>
      <td>Input</td>
      <td>Compression ratio of cmpKv relative to the KV length before compression, used for recovery of the pre-compression KV length used by the mask on the cmp side.</td>
      <td>Value range: [1, 128]. Not involved in compressed KV computation when only oriKv is passed in.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>oriMaskMode (int64_t)</td>
      <td>Input</td>
      <td>Mask mode for the calculation of q and oriKv.</td>
      <td>0: No Mask.<br>3: RightDownCausal mode.<br>4: Band mode.<br>The recommended value is 0.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cmpMaskMode (int64_t)</td>
      <td>Input</td>
      <td>Mask mode of the q and cmpKv computation.</td>
      <td>0: No Mask.<br>3: RightDownCausal mode.<br>The recommended value is 0.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>oriWinLeft (int64_t)</td>
      <td>Input</td>
      <td>Number of tokens that the sliding window extends to the left.</td>
      <td>Supports -1 or non-negative values, where -1 indicates an unrestricted window. The recommended value is -1.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>oriWinRight (int64_t)</td>
      <td>Input</td>
      <td>Number of tokens the sliding window extends to the right.</td>
      <td>Supports -1 or non-negative values, where -1 indicates an unrestricted window. The recommended value is -1.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layoutQOptional (char*)</td>
      <td>Input</td>
      <td>Specifies the data layout format of input q.</td>
      <td>Supports "BSND" and "TND". The recommended value is "BSND".</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layoutKvOptional (char*)</td>
      <td>Input</td>
      <td>Specifies the data layout format of the input KV.</td>
      <td>Supports "PA_BBND", "BSND", and "TND". The recommended value is "BSND".</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>hasOriKv (bool)</td>
      <td>Input</td>
      <td>Whether oriKv is passed in.</td>
      <td>Set based on whether oriKv is passed in. The recommended value is true.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>hasCmpKv (bool)</td>
      <td>Input</td>
      <td>Whether cmpKv is passed in.</td>
      <td>false for the SWA use case, and true for the CSA and HCA use cases. The recommended value is true depending on whether cmpKv is passed in.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>metaData (aclTensor*)</td>
      <td>Output</td>
      <td>Core partitioning metadata output, used by the SparseFlashMla operator.</td>
      <td>The shape is fixed to (1024, ).</td>
      <td>INT32</td>
      <td>ND</td>
      <td>1D</td>
      <td>×</td>
    </tr>
    <tr>
      <td>workspaceSize (uint64_t*)</td>
      <td>Output</td>
      <td>Returns the workspace size to be requested on the device.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor (aclOpExecutor**)</td>
      <td>Output</td>
      <td>Returns the op executor that contains the operator computation flow.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>

  <ul>
    <!-- npu="A3" id7 -->
    <li><term>Atlas A3 products</term>: sequsedQOptional and cmpTopkLengthOptional are not supported; numHeadsQ/numHeadsKv support any integer in [1, 128]; the SWA sparse ori_kv use case supports oriTopkLengthOptional, oriTopk greater than 0, and oriMaskMode of 0, and oriWinLeft and oriWinRight support non-negative values; in other SWA use cases, oriTopk is 0, oriMaskMode is 4, oriWinLeft is 127, and oriWinRight is 0; cmpTopk supports [0, 8192], where 0 indicates non-sparse compressed KV, cmpMaskMode supports only 3, cmpKv is not passed in for SWA, cmpRatio does not participate in the computation, CSA supports 1, 2, or 4, and HCA supports 128.</li>
    <!-- end id7 -->
    <!-- npu="910b" id8 -->
    <li><term>Atlas A2 products</term>: sequsedQOptional and cmpTopkLengthOptional are not supported; numHeadsQ/numHeadsKv support any integer in [1, 128]; the SWA sparse ori_kv use case supports oriTopkLengthOptional, oriTopk greater than 0, and oriMaskMode of 0, and oriWinLeft and oriWinRight support non-negative values; in other SWA use cases, oriTopk is 0, oriMaskMode is 4, oriWinLeft is 127, and oriWinRight is 0; cmpTopk supports [0, 8192], where 0 indicates non-sparse compressed KV, cmpMaskMode supports only 3, cmpKv is not passed in for SWA, cmpRatio does not participate in the computation, CSA supports 1, 2, or 4, and HCA supports 128.</li>
    <!-- end id8 -->
  </ul>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

    <!-- npu="950" id9 -->
  - <term>Ascend 950PR&950DT products</term>:

    <table style="undefined;table-layout: fixed;width: 1200px"><colgroup>
    <col style="width: 262px">
    <col style="width: 121px">
    <col style="width: 817px">
    </colgroup>
    <thead>
      <tr>
        <th>Return value</th>
        <th>Error code</th>
        <th>Description</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>ACLNN_ERR_INNER_CREATE_EXECUTOR</td>
        <td>561101</td>
        <td>Failed to create aclOpExecutor.</td>
      </tr>
      <tr>
        <td>ACLNN_ERR_INNER_NULLPTR</td>
        <td>561103</td>
        <td>workspaceSize or executor is a null pointer; the optional input is a null pointer after contiguity is performed; or the SparseFlashMlaMetadata AICPU task fails to be added.</td>
      </tr>
      <tr>
        <td rowspan="8">ACLNN_ERR_PARAM_INVALID</td>
        <td rowspan="8">161002</td>
        <td>batchSize or maxSeqlenQ is negative.</td>
      </tr>
      <tr>
        <td>numHeadsQ is not within range [1,128], numHeadsKv is not 1, numHeadsQ cannot be divided evenly by numHeadsKv, or numHeadsQ/numHeadsKv is not within range [1,128].</td>
      </tr>
      <tr>
        <td>headDim is not 512.</td>
      </tr>
      <tr>
        <td>oriMaskMode is not 0, 3, or 4, or cmpMaskMode is not 0 or 3.</td>
      </tr>
      <tr>
        <td>oriWinLeft or oriWinRight is less than -1.</td>
      </tr>
      <tr>
        <td>When hasCmpKv is true, cmpRatio is not within [1,128].</td>
      </tr>
      <tr>
        <td>When hasCmpKv is true, cmpTopk is negative.</td>
      </tr>
      <tr>
        <td>The shape, data type, or mandatory relationship of layoutQOptional, layoutKvOptional, cuSeqlens, seqused, or metaData is not supported.</td>
      </tr>
    </tbody>
    </table>
  <!-- end id9 -->
  <!-- npu="A3" id10 -->
  - <term>Atlas A3 products</term>:

    <table style="undefined;table-layout: fixed;width: 1200px"><colgroup>
    <col style="width: 262px">
    <col style="width: 121px">
    <col style="width: 817px">
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
        <td>ACLNN_ERR_INNER_CREATE_EXECUTOR</td>
        <td>561101</td>
        <td>Failed to create aclOpExecutor.</td>
      </tr>
      <tr>
        <td>ACLNN_ERR_INNER_NULLPTR</td>
        <td>561103</td>
        <td>workspaceSize or executor is a null pointer; the optional input is a null pointer after contiguity is performed; or adding the SparseFlashMlaMetadata AICPU task fails.</td>
      </tr>
      <tr>
        <td rowspan="9">ACLNN_ERR_PARAM_INVALID</td>
        <td rowspan="9">161002</td>
        <td>batchSize or maxSeqlenQ is negative.</td>
      </tr>
      <tr>
        <td>numHeadsQ is not within the range of [1,128], numHeadsKv is not 1, numHeadsQ cannot be divided by numHeadsKv, or numHeadsQ/numHeadsKv is not within the range of [1,128].</td>
      </tr>
      <tr>
        <td>headDim is not 512.</td>
      </tr>
      <tr>
        <td>In non-SWA sparse ori_kv use case, oriMaskMode is not 4; in SWA sparse ori_kv use case, oriMaskMode is not 0, or cmpMaskMode is not 3.</td>
      </tr>
      <tr>
        <td>In the non-SWA sparse ori_kv use case, oriWinLeft is not 127 or oriWinRight is not 0; in the SWA sparse ori_kv use case, oriWinLeft or oriWinRight is negative.</td>
      </tr>
      <tr>
        <td>cmpRatio does not match the CSA or HCA use cases.</td>
      </tr>
      <tr>
        <td>cmpTopk is not within the range of [0, 8192].</td>
      </tr>
      <tr>
        <td>In the SWA sparse ori_kv use case, oriTopkLengthOptional is not passed in, or the shape or data type of oriTopkLengthOptional does not meet the specifications; a non-empty Tensor is passed in for cmpTopkLengthOptional.</td>
      </tr>
      <tr>
        <td>The shape, data type, or mandatory relationship of layoutQOptional, layoutKvOptional, cuSeqlens, seqused, or metaData is out of the supported range.</td>
      </tr>
    </tbody>
    </table>
  <!-- end id10 -->
  <!-- npu="910b" id11 -->
  - <term>Atlas A2 products</term>:

    <table style="undefined;table-layout: fixed;width: 1200px"><colgroup>
    <col style="width: 262px">
    <col style="width: 121px">
    <col style="width: 817px">
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
        <td>ACLNN_ERR_INNER_CREATE_EXECUTOR</td>
        <td>561101</td>
        <td>Failed to create aclOpExecutor.</td>
      </tr>
      <tr>
        <td>ACLNN_ERR_INNER_NULLPTR</td>
        <td>561103</td>
        <td>workspaceSize or executor is a null pointer; the optional input is a null pointer after contiguity is performed; or adding the SparseFlashMlaMetadata AICPU task fails.</td>
      </tr>
      <tr>
        <td rowspan="9">ACLNN_ERR_PARAM_INVALID</td>
        <td rowspan="9">161002</td>
        <td>batchSize or maxSeqlenQ is negative.</td>
      </tr>
      <tr>
        <td>numHeadsQ is not within the range of [1,128], numHeadsKv is not 1, numHeadsQ cannot be evenly divided by numHeadsKv, or numHeadsQ/numHeadsKv is not within the range of [1,128].</td>
      </tr>
      <tr>
        <td>headDim is not 512.</td>
      </tr>
      <tr>
        <td>In the non-SWA sparse ori_kv use case, oriMaskMode is not 4; in the SWA sparse ori_kv use case, oriMaskMode is not 0, or cmpMaskMode is not 3.</td>
      </tr>
      <tr>
        <td>In the non-SWA sparse ori_kv use case, oriWinLeft is not 127 or oriWinRight is not 0; in the SWA sparse ori_kv use case, oriWinLeft or oriWinRight is negative.</td>
      </tr>
      <tr>
        <td>cmpRatio does not match the CSA or HCA use case.</td>
      </tr>
      <tr>
        <td>cmpTopk is not within the range of [0, 8192].</td>
      </tr>
      <tr>
        <td>In the SWA sparse ori_kv use case, oriTopkLengthOptional is not passed in, or the shape or data type of oriTopkLengthOptional does not meet the specification; cmpTopkLengthOptional is passed in as a non-empty Tensor.</td>
      </tr>
      <tr>
        <td>The shape, data type, or mandatory relationship of layoutQOptional, layoutKvOptional, cuSeqlens, seqused, or metaData is not within the supported range.</td>
      </tr>
    </tbody>
    </table>
  <!-- end id11 -->

## aclnnSparseFlashMlaMetadata

- **Parameters**

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
      <td>Workspace memory address requested on the device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Workspace size requested on the device, obtained from the first-phase API aclnnSparseFlashMlaMetadataGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Executor of the op, which contains the operator computation flow.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream for executing tasks.</td>
    </tr>
  </tbody>
  </table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation

  - aclnnSparseFlashMlaMetadata defaults to a deterministic implementation. The same inputs produce the same results across multiple calls.

- General specification constraints
  - B (Batch) indicates the batch size of input samples. q, oriKvOptional, and cmpKvOptional are input parameters of the paired aclnnSparseFlashMla operator. S1 indicates the S axis size in the q shape when layoutQOptional=BSND. T1 indicates the T axis size in the q shape when layoutQOptional=TND. S2 indicates the S axis size in the oriKvOptional shape when layoutKvOptional=BSND. S3 indicates the S axis size in the cmpKvOptional shape when layoutKvOptional=BSND. N2 indicates the N axis size in the oriKvOptional and cmpKvOptional shapes.
  - The values of cuSeqlensQOptional, cuSeqlensOriKvOptional, and cuSeqlensCmpKvOptional must be the cumulative number of valid tokens in the current and preceding batches. The first element is fixed to 0, and each subsequent element must be greater than or equal to the preceding element.
  - The values of sequsedQOptional, sequsedOriKvOptional, and sequsedCmpKvOptional must indicate the number of valid tokens in each batch.
  - Only the following layoutQOptional and layoutKvOptional combinations are supported: "BSND"/"BSND", "TND"/"TND", "BSND"/"PA_BBND", and "TND"/"PA_BBND". In non-PA_BBND use cases, layoutQOptional and layoutKvOptional must be consistent.
  - cmpResidualKvOptional must satisfy cmpResidualKvOptional[i] < cmpRatio.
<!-- npu="950" id12 -->
- Ascend 950PR&950DT products constraints:
  - When hasOriKv is true, the oriKvOptional part is considered sparse if oriTopk is greater than 0, and non-sparse if oriTopk is 0.
  - When hasCmpKv is true, the cmpKvOptional part is considered sparse if cmpTopk is greater than 0, and non-sparse if cmpTopk is 0.
  - When hasOriKv is true, oriTopk is not 0, and oriMaskMode is 0, oriTopkLengthOptional must be passed in. In this case, the valid seqlen of oriKvOptional corresponding to the current q token is the minimum of the oriMaskMode rule and the elements in oriTopkLengthOptional. In other oriKvOptional sparse scenarios, the valid seqlen of oriKvOptional corresponding to the current q token is the minimum of the oriMaskMode rule and oriTopk.
  - When hasCmpKv is true, cmpTopk is not 0, and cmpMaskMode is 0, cmpTopkLengthOptional must be passed in. In this case, the valid seqlen of cmpKvOptional corresponding to the current q token is the minimum of the cmpMaskMode rule and the elements in cmpTopkLengthOptional. In other cmpKvOptional sparse scenarios, the valid seqlen of cmpKvOptional corresponding to the current q token is the minimum of the cmpMaskMode rule and cmpTopk.
  - layoutQOptional=BSND use case
    - maxSeqlenQ must be passed in with the value of S1.
  - layoutKvOptional=BSND use case
    - When hasOriKv is true, maxSeqlenOriKv must be passed in with the value of S2.
    - When hasCmpKv is true, maxSeqlenCmpKv must be passed in with the value of S3.
  - layoutQOptional=TND use case
    - cuSeqlensQOptional must be passed in.
  - layoutKvOptional=TND use case
    - When hasOriKv is true, cuSeqlensOriKvOptional must be passed in.
    - When hasCmpKv is true, cuSeqlensCmpKvOptional must be passed in.
  - layoutKvOptional=PA_BBND use case
    - When hasOriKv is true, oriTopk is not 0, and oriMaskMode is 0 (oriTopkLengthOptional is mandatory in this scenario), sequsedOriKvOptional is optional. In other scenarios, sequsedOriKvOptional must be passed in.
    - When hasCmpKv is true, cmpTopk is not 0, and cmpMaskMode is 0 (cmpTopkLengthOptional is mandatory in this scenario), sequsedCmpKvOptional is optional. In other scenarios, sequsedCmpKvOptional must be passed in.
  - Batch value rule
    - When layoutQOptional is BSND, batch is derived preferentially from the shape of sequsedQOptional. If sequsedQOptional is not passed in, batch is obtained from batch_size.
    - When layoutQOptional is TND, batch is derived preferentially from the shape of sequsedQOptional. If sequsedQOptional is not passed in, batch is derived from the shape of cuSeqlensQOptional.
  - q Seqlen value rule
    - When layoutQOptional is BSND, seqlen is obtained preferentially from elements in sequsedQOptional. If sequsedQOptional is not passed in, seqlen is obtained from maxSeqlenQ.
    - When layoutQOptional is TND, seqlen is obtained preferentially from elements in sequsedQOptional. If sequsedQOptional is not passed in, seqlen is obtained from elements in cuSeqlensQOptional.
  - oriKvOptional Seqlen value rule
    - When layoutKvOptional is BSND, seqlen is obtained preferentially from elements in sequsedOriKvOptional. If sequsedOriKvOptional is not passed in, seqlen is obtained from maxSeqlenOriKv.
    - When layoutKvOptional is TND, seqlen is obtained preferentially from elements in sequsedOriKvOptional. If sequsedOriKvOptional is not passed in, seqlen is obtained from elements in cuSeqlensOriKvOptional.
    - When layoutKvOptional is PA_BBND, seqlen is obtained preferentially from elements in sequsedOriKvOptional. If sequsedOriKvOptional is not passed in, seqlen is obtained from oriTopkLengthOptional.
  - cmpKvOptional Seqlen value rule
    - When layoutKvOptional is BSND, seqlen is obtained preferentially from elements in sequsedCmpKvOptional. If sequsedCmpKvOptional is not passed in, seqlen is obtained from maxSeqlenCmpKv.
    - When layoutKvOptional is TND, seqlen is obtained preferentially from elements in sequsedCmpKvOptional. If sequsedCmpKvOptional is not passed in, seqlen is obtained from elements in cuSeqlensCmpKvOptional.
    - When layoutKvOptional is PA_BBND, seqlen is obtained preferentially from elements in sequsedCmpKvOptional. If sequsedCmpKvOptional is not passed in, seqlen is obtained from cmpTopkLengthOptional.
<!-- end id12 -->
<!-- npu="A3" id13 -->
- Atlas A3 products constraints:
  - In the SWA sparse ori_kv use case, only the SWA template is supported. `hasOriKv` must be true, `hasCmpKv` must be false, `oriTopk` must be greater than 0, `oriMaskMode` must be 0, `oriWinLeft` and `oriWinRight` must be non-negative, and `oriTopkLengthOptional` must be passed in. `oriTopk` must be consistent with the last dimension K of oriSparseIndicesOptional of the paired main operator. `oriTopkLengthOptional` specifies the number of left-aligned valid index entries for each q token and KV head. The value must be within the range [0, K]. Metadata uses only `oriTopkLengthOptional` to generate task partitioning. In the PA_BBND use case, the paired main operator still requires `sequsedOriKvOptional` to be passed in.
  - When layoutQOptional is TND, `cuSeqlensQOptional` must be passed in.
  - When layoutKvOptional is PA_BBND, `sequsedOriKvOptional` must be passed in. In the BSND use case, `sequsedOriKvOptional` is optional and overrides the valid oriKv length of each batch; in the TND use case, `cuSeqlensOriKvOptional` is used to express the oriKv sequence boundary.
  - When layoutKvOptional is TND, `cuSeqlensOriKvOptional` must be passed in; if hasCmpKv is true, `cuSeqlensCmpKvOptional` must also be passed in.
  - `sequsedCmpKvOptional` is an optional input for all layoutKvOptional values. When explicitly passed in, it is used to override the logical valid length on the cmp side.
  - `cmpResidualKvOptional` is an optional input of `aclnnSparseFlashMlaMetadata` and `aclnnSparseFlashMla`. It is mandatory in this scenario where CSA or HCA is used, cmpRatio is not equal to 1, and cmpMaskMode is 3. It is used to recover the pre-compression length used by the cmp-side mask.
<!-- end id13 -->
<!-- npu="910b" id14 -->
- Atlas A2 products constraints:
  - In the SWA sparse ori_kv use case, only the SWA template is supported. `hasOriKv` must be true, `hasCmpKv` must be false, `oriTopk` must be greater than 0, `oriMaskMode` must be 0, `oriWinLeft` and `oriWinRight` must be non-negative, and `oriTopkLengthOptional` must be passed in. `oriTopk` must be consistent with K, the last dimension of oriSparseIndicesOptional of the accompanying main operator. `oriTopkLengthOptional` indicates the number of left-aligned valid index entries for each q token and KV head, and its value must be within the range of [0, K]. Metadata uses only `oriTopkLengthOptional` to generate task partitioning. In the PA_BBND use case, the accompanying main operator still requires `sequsedOriKvOptional` to be passed in.
  - When layoutQOptional is TND, `cuSeqlensQOptional` must be passed in.
  - When layoutKvOptional is PA_BBND, `sequsedOriKvOptional` must be passed in. In the BSND use case, `sequsedOriKvOptional` is optional and overrides the valid oriKv length of each batch; in the TND use case, `cuSeqlensOriKvOptional` is used to express the oriKv sequence boundary.
  - When layoutKvOptional is TND, `cuSeqlensOriKvOptional` must be passed in; if hasCmpKv is true, `cuSeqlensCmpKvOptional` must also be passed in.
  - `sequsedCmpKvOptional` is an optional input for all layoutKvOptional values. When explicitly passed in, it is used to override the logical valid length on the cmp side.
  - `cmpResidualKvOptional` is an optional input of `aclnnSparseFlashMlaMetadata` and `aclnnSparseFlashMla`. It is mandatory in the CSA, HCA, cmpRatio not equal to 1, or cmpMaskMode equal to 3 use cases, and is used for recovery of the pre-compression length used by the cmp-side mask.
<!-- end id14 -->

## Example

The sample code is as follows, for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```c++
/**
 * @file test_aclnn_sparse_flash_mla_metadata.cpp
 */
#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include <limits>
#include <functional>
#include <utility>
#include "acl/acl.h"
#include "aclnnop/aclnn_sparse_flash_mla_metadata.h"

#define CHECK_LOG_RET(cond, ret_val, fmt, ...)      \
    do {                                            \
        if (!(cond)) {                              \
            printf(fmt "\n", ##__VA_ARGS__);        \
            return (ret_val);                       \
        }                                           \
    } while (0)

// Refer to sparse_flash_mla_metadata.h.
constexpr uint32_t AIC_CORE_MAX_NUM = 36;
constexpr uint32_t AIV_CORE_MAX_NUM = 72;
constexpr uint32_t SMLA_METADATA_TOTAL_SIZE = 1024;
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

struct SmlaMetadata {
    uint32_t faMetadata[AIC_CORE_MAX_NUM][FA_METADATA_SIZE];
    uint32_t fdMetadata[AIV_CORE_MAX_NUM][FD_METADATA_SIZE];
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
    // Boilerplate: initialization.
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
    // Call aclrtMallocHost to request host-side memory.
    auto ret = aclrtMallocHost(&(tensor.hostAddr), size);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "aclrtMallocHost failed. ERROR: %d", ret);
    memset(tensor.hostAddr, 0, size);
    // Call aclrtMalloc to request device-side memory.
    ret = aclrtMalloc(&(tensor.deviceAddr), size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "aclrtMalloc failed. ERROR: %d", ret);
    // Call aclCreateTensor to create aclTensor.
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
    ret = CreateTensor(aclDataType::ACL_INT32, { SMLA_METADATA_TOTAL_SIZE }, context.metadata);     // 1024: Fix size
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
    strcpy(context.layoutKvOptional, "BSND");               // BSND,TND,PA_BBND
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
    // 1. (Boilerplate) device/stream initialization. Refer to the external API list.
    // Fill in deviceId based on the actual device.
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
    // Call the first-phase API of aclnnSparseFlashMlaMetadata.
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    void *workspaceAddr = nullptr;
    ret = aclnnSparseFlashMlaMetadataGetWorkspaceSize(
        context.cuSeqlensQOptional.data, context.cuSeqlensOriKvOptional.data, context.cuSeqlensCmpKvOptional.data,
        context.sequsedQOptional.data, context.sequsedOriKvOptional.data, context.sequsedCmpKvOptional.data,
        context.cmpResidualKvOptional.data, context.oriTopkLengthOptional.data, context.cmpTopkLengthOptional.data,
        context.numHeadsQ, context.numHeadsKv, context.headDim, context.batchSize, context.maxSeqlenQ,
        context.maxSeqlenOriKv, context.maxSeqlenCmpKv, context.oriTopk, context.cmpTopk, context.cmpRatio,
        context.oriMaskMode, context.cmpMaskMode, context.oriWinLeft, context.oriWinRight, context.layoutQOptional,
        context.layoutKvOptional, context.hasOriKv, context.hasCmpKv, context.metadata.data, &workspaceSize, &executor);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret,
        "aclnnSparseFlashMlaMetadataGetWorkspaceSize failed. ERROR: %d\n", ret);

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

    // Call the second-phase API of aclnnSparseFlashMlaMetadata.
    ret = aclnnSparseFlashMlaMetadata(workspaceAddr, workspaceSize, executor, stream);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "aclnnSparseFlashMlaMetadata failed. ERROR: %d\n", ret);

    // 4. (Boilerplate) Wait for task execution to finish synchronously.
    ret = aclrtSynchronizeStream(stream);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "aclrtSynchronizeStream failed. ERROR: %d\n", ret);

    // 5. Print the output.
    SmlaMetadata result {};
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
        printf("    FD Task Workspace Idx   : %u\n", result.fdMetadata[i][FD_WORKSPACE_IDX_INDEX]);
        printf("    FD Task Workspace Num   : %u\n", result.fdMetadata[i][FD_WORKSPACE_NUM_INDEX]);
        printf("    FD Subtask M Start      : %u\n", result.fdMetadata[i][FD_M_START_INDEX]);
        printf("    FD Subtask M Num        : %u\n", result.fdMetadata[i][FD_M_NUM_INDEX]);
    }

    return 0;
}
```
