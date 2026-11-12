# aclnnSparseLightningIndexerKLLossGradMetadata

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-21T13:45:29.475Z -->

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

- Description: This operator is an AICPU operator and serves as the preceding operator of the aclnnSparseLightningIndexerKLLossGrad operator. Based on the input shape, layout, mask, and compression ratio information of the aclnnSparseLightningIndexerKLLossGrad operator, it computes and outputs core-splitting metadata. The output can be used as the metadataOptional input of the aclnnSparseLightningIndexerKLLossGrad operator, reducing host array access during the tiling phase of the main operator.

  **This operator is not recommended for standalone use. It is recommended to be used together with the aclnnSparseLightningIndexerKLLossGrad operator to form a complete workflow.**
    1. Receive the shape information of the main operator, including batchSize, maxSeqlenQ, maxSeqlenK, numHeadsQ, numHeadsK, headDim, topk, layout, and mask information.
    2. Estimate the workload based on the valid sparse length of each q, and evenly split the post-axis-aligned B/S1 tasks across the available AIC cores.
    3. After the metadata is output, it is used as the metadataOptional input of the aclnnSparseLightningIndexerKLLossGrad operator.

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). You must first call "aclnnSparseLightningIndexerKLLossGradMetadataGetWorkspaceSize" to obtain the workspace size, and then call "aclnnSparseLightningIndexerKLLossGradMetadata" to perform the computation.

``` cpp
aclnnStatus aclnnSparseLightningIndexerKLLossGradMetadataGetWorkspaceSize(
    const aclTensor *cuSeqLensQOptional,
    const aclTensor *cuSeqLensKOptional,
    const aclTensor *seqUsedQOptional,
    const aclTensor *seqUsedKOptional,
    const aclTensor *cmpResidualKOptional,
    int64_t batchSize,
    int64_t maxSeqlenQ,
    int64_t maxSeqlenK,
    int64_t numHeadsQ,
    int64_t numHeadsK,
    int64_t headDim,
    int64_t topk,
    char *layoutQOptional,
    char *layoutKOptional,
    int64_t maskMode,
    int64_t cmpRatio,
    const aclTensor *metadata,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);
```

``` cpp
aclnnStatus aclnnSparseLightningIndexerKLLossGradMetadata(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);
```

## aclnnSparseLightningIndexerKLLossGradMetadataGetWorkspaceSize

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1600px"><colgroup>
  <col style="width: 150px">
  <col style="width: 100px">
  <col style="width: 350px">
  <col style="width: 150px">
  <col style="width: 70px">
  <col style="width: 70px">
  <col style="width: 190px">
  <col style="width: 80px">
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
      <td>cuSeqLensQOptional (aclTensor*)</td>
      <td>Input</td>
      <td>Indicates the cumulative sequence length of q in different batches.</td>
      <td><ul><li>Supports empty Tensor</li><li>The shape is fixed to (B+1, ).</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>1D</td>
      <td>√</td>
    </tr>
    <tr>
      <td>cuSeqLensKOptional (aclTensor*)</td>
      <td>Input</td>
      <td>Indicates the cumulative sequence length of k in different batches.</td>
      <td><ul><li>Supports empty tensor</li><li>shape is fixed to (B+1, ).</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>1-dim</td>
      <td>√</td>
    </tr>
    <tr>
      <td>seqUsedQOptional (aclTensor*)</td>
      <td>Input</td>
      <td>Indicates the actual sequence length of q participating in computation in different batches.</td>
      <td><ul><li>Supports empty tensor.</li><li>shape is fixed to (B, ).</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>1-D</td>
      <td>√</td>
    </tr>
    <tr>
      <td>seqUsedKOptional(aclTensor*)</td>
      <td>Input</td>
      <td>Indicates the sequence length of k that actually participates in the computation in different batches.</td>
      <td><ul><li>Supports empty tensors.</li><li>The shape is fixed to (B, ).</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>1-D</td>
      <td>√</td>
    </tr>
    <tr>
      <td>cmpResidualKOptional (aclTensor*)</td>
      <td>Input</td>
      <td>Indicates the residual between the sequence length of k in different batches and cmpRatio.</td>
      <td><ul><li>Supports empty tensors.</li><li>shape is fixed to (B, ).</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>1-D</td>
      <td>√</td>
    </tr>
    <tr>
      <td>batchSize (int64_t)</td>
      <td>Input</td>
      <td>Indicates the batch quantity.</td>
      <td><ul><li>Supports non-negative values. In TND scenarios, 0 can be filled, and the value is derived via cuSeqLensQOptional.</li><li>The recommended value is 0.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>maxSeqlenQ (int64_t)</td>
      <td>Input</td>
      <td>Indicates the maximum sequence length of q.</td>
      <td><ul><li>Supports non-negative numbers. Must be positive in the BSND scenario.</li><li>The recommended value is 0.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>maxSeqlenK (int64_t)</td>
      <td>Input</td>
      <td>Indicates the maximum sequence length of k.</td>
      <td><ul><li>Supports non-negative numbers. Must be positive in the BSND scenario.</li><li>The recommended value is 0.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>numHeadsQ (int64_t)</td>
      <td>Input</td>
      <td>Indicates the number of q heads.</td>
      <td>Must be positive and divisible by numHeadsK. Currently supports [1, 128].</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>numHeadsK (int64_t)</td>
      <td>Input</td>
      <td>Indicates the number of heads of k.</td>
      <td>Must be positive. Currently only supports 1.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>headDim (int64_t)</td>
      <td>Input</td>
      <td>Indicates the head dimension of q/k.</td>
      <td>Must be positive; currently only supports 128.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>topk (int64_t)</td>
      <td>Input</td>
      <td>Indicates the number of key tokens filtered from k.</td>
      <td>Must be positive. Currently supports [1, 2048], 4096, and 8192.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layoutQOptional (char*)</td>
      <td>Input</td>
      <td>Indicates the layout of q.</td>
      <td><ul><li>Supports BSND and TND.</li><li>Recommended value is BSND.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layoutKOptional (char*)</td>
      <td>Input</td>
      <td>Indicates the layout of k.</td>
      <td><ul><li>Supports BSND and TND.</li><li>The recommended value is BSND.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>maskMode (int64_t)</td>
      <td>Input</td>
      <td>Indicates the sparse mask mode.</td>
      <td><ul><li>0: No mask.</li><li>3: mask in rightDownCausal mode, corresponding to the lower-triangular scenario split by the top-right vertex.</li><li>The recommended value is 0.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cmpRatio (int64_t)</td>
      <td>Input</td>
      <td>Indicates the compression rate of k.</td>
      <td><ul><li>Value range: [1, 128].</li><li>The recommended value is 1, indicating no compression.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>metadata (aclTensor*)</td>
      <td>Output</td>
      <td>Indicates the output of the load balancing result.</td>
      <td>The output is used as the metadataOptional input of aclnnSparseLightningIndexerKLLossGrad, and the shape is fixed to (64, ).</td>
      <td>INT32</td>
      <td>ND</td>
      <td>1D</td>
      <td>x</td>
    </tr>
    <tr>
      <td>workspaceSize (uint64_t*)</td>
      <td>Output</td>
      <td>Returns the workspace size to be allocated on the Device side.</td>
      <td>The current implementation returns 0.</td>
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

  <ul>
    <!-- npu="950" id7 -->
    <li><term>Ascend 950PR&950DT products</term>: topk only supports [1, 2048].</li>
    <!-- end id7 -->
    <!-- npu="A3" id8 -->
    <li><term>Atlas A3 products</term>: seqUsedQOptional, seqUsedKOptional, and cmpResidualKOptional are not supported; numHeadsQ only supports 8/16/32/64; topk only supports 512/1024/2048/4096/8192.</li>
    <!-- end id8 -->
    <!-- npu="910b" id9 -->
    <li><term>Atlas A2 products</term>: seqUsedQOptional, seqUsedKOptional, and cmpResidualKOptional are not supported; numHeadsQ only supports 8/16/32/64; topk only supports 512/1024/2048/4096/8192.</li>
    <!-- end id9 -->
  </ul>

- **Return value**

    Returns an aclnnStatus status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

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
        <td>Parameter workspaceSize or executor is a null pointer, or parameters cuSeqLensQOptional, cuSeqLensKOptional, seqUsedQOptional, seqUsedKOptional, and cmpResidualKOptional are null pointers after Contiguous processing.</td>
      </tr>
      <tr>
        <td rowspan="1">ACLNN_ERR_PARAM_INVALID</td>
        <td rowspan="1">161002</td>
        <td>The specifications of parameters cuSeqLensQOptional, cuSeqLensKOptional, seqUsedQOptional, seqUsedKOptional, cmpResidualKOptional, batchSize, maxSeqlenQ, maxSeqlenK, numHeadsQ, numHeadsK, headDim, topk, layoutQOptional, layoutKOptional, maskMode, and cmpRatio are not within the supported range.</td>
      </tr>
    </tbody>
    </table>

## aclnnSparseLightningIndexerKLLossGradMetadata

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
        <td>Workspace size allocated on the Device side, obtained by calling the first-phase API aclnnSparseLightningIndexerKLLossGradMetadataGetWorkspaceSize.</td>
        </tr>
        <tr>
        <td>executor</td>
        <td>Input</td>
        <td>Op executor that contains the computation process of the operator.</td>
        </tr>
        <tr>
        <td>stream</td>
        <td>Input</td>
        <td>Stream for executing the task.</td>
        </tr>
    </tbody>
    </table>

- **Return value**

    Returns an aclnnStatus status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

  - aclnnSparseLightningIndexerKLLossGradMetadata uses a deterministic implementation. The deterministic computation configuration does not change its output rules.
  - B (Batch) indicates the batch size of input samples, q is an input parameter of the companion aclnnSparseLightningIndexerKLLossGrad operator, and S1 indicates the size of the S axis in the q shape when layoutQOptional=BSND.
  - The values of cuSeqlensQOptional and cuSeqlensKOptional must be the cumulative sum of the number of valid tokens in the current and preceding batches. The first element is fixed to 0, and the value of each subsequent element must be greater than or equal to that of the preceding element.
  - The values of parameters sequsedQOptional and sequsedKOptional must indicate the number of valid tokens in each batch.
  - Parameter cmpResidualKOptional must satisfy cmpResidualKOptional[i] < cmpRatio.
  - layoutQOptional and layoutKOptional must be the same.
  - numHeadsQ must be divisible by numHeadsK.
  <!-- npu="950" id10 -->
  - Ascend 950PR&950DT products constraints:
    - layoutQOptional=BSND scenario
      - The value of S1 must be passed to maxSeqlenQ.
    - layoutQOptional=TND scenario
      - cuSeqlensQOptional is mandatory.
  <!-- end id10 -->
  <!-- npu="A3" id11 -->
  - <term>Atlas A3 products</term> constraints:
    - BSND scenario
      - batchSize, maxSeqlenQ, maxSeqlenK, and topk are mandatory to obtain shape information.
    - TND scenario
      - cuSeqLensQOptional, cuSeqLensKOptional, and topk are mandatory to obtain correct shape information.
      - When batchSize is 0, the batch is derived via the shape of cuSeqLensQOptional.
  <!-- end id11 -->
  <!-- npu="910b" id12 -->
  - <term>Atlas A2 products</term> constraints:
    - BSND scenario
      - batchSize, maxSeqlenQ, maxSeqlenK, and topk are mandatory to obtain shape information.
    - TND scenario
      - cuSeqLensQOptional, cuSeqLensKOptional, and topk are mandatory to obtain correct shape information.
      - When batchSize is 0, derive the batch from the shape of cuSeqLensQOptional.
  <!-- end id12 -->

<details>
<summary><a id="Mask"></a>Mask</summary>
    &nbsp;&nbsp;<table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
        <col style="width: 165px">
        <col style="width: 625px">
        <col style="width: 360px">
        </colgroup>
        <thead>
            <tr>
                <th>mask_mode</th>
                <th>Meaning</th>
                <th>Remarks</th>
            </tr>
        </thead>
        <tbody>
        <tr>
            <td>0</td>
            <td>No mask.</td>
            <td>Supported</td>
        </tr>
        <tr>
            <td>3</td>
            <td>Mask in rightDownCausal mode, corresponding to the lower triangular scenario partitioned by the right vertex.</td>
            <td>Supported</td>
        </tr>
        </tbody>
    </table>
</details>

<details>
<summary><a id="special-constraints"></a>Special Constraints</summary>

  <!-- npu="950" id13 -->
  - Ascend 950PR&950DT products constraints:
      - Batch value rules
        - When layoutQOptional is BSND, prefer to use the shape of sequsedQOptional to derive the batch; if sequsedQOptional is not passed, obtain the number of batches from batch_size.
        - When layoutQOptional is TND, prefer to use the shape of sequsedQOptional to derive the batch; if sequsedQOptional is not passed, derive the batch from the shape of cuSeqlensQOptional.
      - q Seqlen value rules
        - When layoutQOptional is BSND, prefer to use the elements in sequsedQOptional to obtain the seqlen; if sequsedQOptional is not passed, obtain the seqlen from maxSeqlenQ.
        - When layoutQOptional is TND, prefer to use the elements in sequsedQOptional to obtain the seqlen; if sequsedQOptional is not passed, obtain the seqlen from the elements in cuSeqlensQOptional.
  <!-- end id13 -->
  <!-- npu="A3" id14 -->
  - Atlas A3 products constraints:
    - Batch value rules
      - If batchSize is greater than 0, prefer to use batchSize.
      - If batchSize is less than or equal to 0 and layoutQOptional is TND, derive batch via the shape of cuSeqLensQOptional.
      - If batchSize is less than or equal to 0 and layoutQOptional is BSND, an error is reported.
    - Seqlen value rules
      - In TND scenarios, calculate the actual q/k length of each batch via cuSeqLensQOptional and cuSeqLensKOptional.
      - In BSND scenarios, obtain the q/k length via maxSeqlenQ and maxSeqlenK.
    - layout constraints
      - layoutQOptional must be BSND or TND.
      - layoutKOptional supports BSND and TND, and is recommended to be consistent with layoutQOptional.
    - head constraints
      - numHeadsQ, numHeadsK, and headDim must be positive.
      - numHeadsQ must be divisible by numHeadsK.
    - sparse constraints
      - topk must be positive.
      - The value range of cmpRatio is [0, 128].
      - maskMode currently only supports 0 and 3.
  <!-- end id14 -->
  <!-- npu="910b" id15 -->
  - Atlas A2 products constraints:
    - Batch value rules
      - If batchSize is greater than 0, prefer to use batchSize.
      - If batchSize is less than or equal to 0 and layoutQOptional is TND, derive the batch via the shape of cuSeqLensQOptional.
      - If batchSize is less than or equal to 0 and layoutQOptional is BSND, an error is reported.
    - Seqlen value rules
      - In TND scenarios, calculate the actual q/k length of each batch via cuSeqLensQOptional and cuSeqLensKOptional.
      - In BSND scenarios, obtain the q/k length via maxSeqlenQ and maxSeqlenK.
    - layout constraints
      - layoutQOptional must be BSND or TND.
      - layoutKOptional supports BSND and TND, and is recommended to be consistent with layoutQOptional.
    - head constraints
      - numHeadsQ, numHeadsK, and headDim must be positive.
      - numHeadsQ must be divisible by numHeadsK.
    - sparse constraints
      - topk must be positive.
      - The value range of cmpRatio is [0, 128].
      - maskMode currently only supports 0 and 3.
  <!-- end id15 -->

</details>

<details>
<summary><a id="Metadata"></a>Metadata Output Layout</summary>

  The metadata output is an INT32 tensor whose shape is fixed to (64,). The field layout is as follows.
  <ul>
  <!-- npu="950" id16 -->
  <li><term>Ascend 950PR&950DT products</term>:
  <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
    <col style="width: 180px">
    <col style="width: 150px">
    <col style="width: 820px">
    </colgroup>
    <thead>
        <tr>
        <th>Field</th>
        <th>Index</th>
        <th>Description</th>
        </tr>
    </thead>
    <tbody>
        <tr>
        <td>totalNum</td>
        <td>0</td>
        <td>Total number of task rows with post-axis-aligned B/S1, that is, the sum of q sequence lengths across all batches.</td>
        </tr>
        <tr>
        <td>formerCoreProcessNum</td>
        <td>1</td>
        <td>Number of task rows processed by each fully loaded core (front core), equal to ceil(totalNum / aicCoreNum).</td>
        </tr>
        <tr>
        <td>remainCoreProcessNum</td>
        <td>2</td>
        <td>Number of task rows processed by each tail core, equal to formerCoreProcessNum - 1.</td>
        </tr>
        <tr>
        <td>remainCoreNum</td>
        <td>3</td>
        <td>Number of tail cores, that is, the number of cores processing fewer tasks, equal to formerCoreProcessNum × aicCoreNum - totalNum.</td>
        </tr>
        <tr>
        <td>usedCoreNum</td>
        <td>4</td>
        <td>Number of AIC cores actually used, set to totalNum when totalNum < aicCoreNum, otherwise set to aicCoreNum.</td>
        </tr>
    </tbody>
  </table>
  </li>
  <!-- end id16 -->
  <!-- npu="A3" id17 -->
  <li><term>Atlas A3 products</term>:
  <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
    <col style="width: 180px">
    <col style="width: 150px">
    <col style="width: 820px">
    </colgroup>
    <thead>
        <tr>
        <th>Field</th>
        <th>index</th>
        <th>Description</th>
        </tr>
    </thead>
    <tbody>
        <tr>
        <td>coreNum</td>
        <td>0</td>
        <td>Number of cube cores actually used by the main kernel.</td>
        </tr>
        <tr>
        <td>totalSize</td>
        <td>1</td>
        <td>Total number of task rows after B/S1 merged axis.</td>
        </tr>
        <tr>
        <td>splitFactorSize</td>
        <td>2</td>
        <td>Initial core-splitting granularity in equal-split use cases.</td>
        </tr>
        <tr>
        <td>reserved</td>
        <td>3-7</td>
        <td>Reserved field, currently set to 0.</td>
        </tr>
        <tr>
        <td>bS1Index</td>
        <td>8-32</td>
        <td>Starting B/S1 merged axis index of each cube core; inactive cores are filled with totalSize.</td>
        </tr>
    </tbody>
  </table>
  </li>
  <!-- end id17 -->
  <!-- npu="910b" id18 -->
  <li><term>Atlas A2 products</term>:
  <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
    <col style="width: 180px">
    <col style="width: 150px">
    <col style="width: 820px">
    </colgroup>
    <thead>
        <tr>
        <th>Field</th>
        <th>index</th>
        <th>Description</th>
        </tr>
    </thead>
    <tbody>
        <tr>
        <td>coreNum</td>
        <td>0</td>
        <td>Number of cube cores actually used by the main kernel.</td>
        </tr>
        <tr>
        <td>totalSize</td>
        <td>1</td>
        <td>Total number of task rows after B/S1 merged axis.</td>
        </tr>
        <tr>
        <td>splitFactorSize</td>
        <td>2</td>
        <td>Initial core-splitting granularity in evenly split scenarios.</td>
        </tr>
        <tr>
        <td>reserved</td>
        <td>3-7</td>
        <td>Reserved fields, currently set to 0.</td>
        </tr>
        <tr>
        <td>bS1Index</td>
        <td>8-32</td>
        <td>Start B/S1 merged axis index of each cube core. Inactive cores are filled with totalSize.</td>
        </tr>
    </tbody>
  </table>
  </li>
  <!-- end id18 -->
  </ul>

</details>

## Example

The sample code is as follows and is for reference only. For the compilation and execution process, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

``` cpp
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <iostream>
#include <string>
#include <utility>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_sparse_lightning_indexer_kl_loss_grad_metadata.h"

#define CHECK_LOG_RET(cond, ret_val, fmt, ...)      \
    do {                                            \
        if (!(cond)) {                              \
            printf(fmt "\n", ##__VA_ARGS__);        \
            return (ret_val);                       \
        }                                           \
    } while (0)

constexpr uint32_t SLI_METADATA_MAX_CORE_NUM = 25;
constexpr uint32_t SLI_METADATA_HEADER_SIZE = 8;
constexpr uint32_t SLI_METADATA_SIZE = 64;

constexpr uint32_t GRAD_METADATA_SIZE = 5;
constexpr uint32_t TOTAL_NUM = 0;
constexpr uint32_t FORMER_CORE_PROCESS_NUM = 1;
constexpr uint32_t REMAIN_CORE_PROCESS_NUM = 2;
constexpr uint32_t REMAIN_CORE_NUM = 3;
constexpr uint32_t USED_CORE_NUM = 4;

struct SliGradKLLossMetaData {
    int32_t coreNum;
    int32_t totalSize;
    int32_t splitFactorSize;
    int32_t reserved[SLI_METADATA_HEADER_SIZE - 3];
    int32_t bS1Index[SLI_METADATA_MAX_CORE_NUM];
};

struct ScopeGuard
{
    explicit ScopeGuard(std::function<void()> onExitScope) : m_exitFunc(std::move(onExitScope)),
        m_isDismissed(false) {}
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
    bool hasCuSeq { true };
};

struct ArgContext {
    Tensor cuSeqLensQOptional {};
    Tensor cuSeqLensKOptional {};
    Tensor seqUsedQOptional {};
    Tensor seqUsedKOptional {};
    Tensor cmpResidualKOptional {};
    Tensor metadata {};
    int64_t batchSize { 0 };
    int64_t maxSeqlenQ { 0 };
    int64_t maxSeqlenK { 0 };
    int64_t numHeadsQ { 8 };
    int64_t numHeadsK { 1 };
    int64_t headDim { 128 };
    int64_t topk { 512 };
    char *layoutQOptional { nullptr };
    char *layoutKOptional { nullptr };
    int64_t maskMode { 0 };
    int64_t cmpRatio { 4 };
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
    auto ret = aclrtMallocHost(&(tensor.hostAddr), size);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "aclrtMallocHost failed. ERROR: %d", ret);
    memset(tensor.hostAddr, 0, size);

    ret = aclrtMalloc(&(tensor.deviceAddr), size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "aclrtMalloc failed. ERROR: %d", ret);
    tensor.data = aclCreateTensor(shape.data(), shape.size(), dataType, nullptr, 0, aclFormat::ACL_FORMAT_ND,
        shape.data(), shape.size(), tensor.deviceAddr);

    ret = aclrtMemcpy(tensor.deviceAddr, size, tensor.hostAddr, size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "aclrtMemcpy failed. ERROR: %d", ret);
    return ACL_SUCCESS;
}

void SetInt32TensorData(Tensor &tensor, const std::vector<int32_t> &hostData)
{
    auto size = hostData.size() * sizeof(int32_t);
    memcpy(tensor.hostAddr, hostData.data(), size);
    aclrtMemcpy(tensor.deviceAddr, size, tensor.hostAddr, size, ACL_MEMCPY_HOST_TO_DEVICE);
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
    DestroyTensor(context.cuSeqLensQOptional);
    DestroyTensor(context.cuSeqLensKOptional);
    DestroyTensor(context.seqUsedQOptional);
    DestroyTensor(context.seqUsedKOptional);
    DestroyTensor(context.cmpResidualKOptional);

    if (context.layoutQOptional != nullptr) {
        free(context.layoutQOptional);
        context.layoutQOptional = nullptr;
    }
    if (context.layoutKOptional != nullptr) {
        free(context.layoutKOptional);
        context.layoutKOptional = nullptr;
    }
}

aclnnStatus CreateArgs(const ArgScenario &scenario, ArgContext &context)
{
    ScopeGuard argsGuard([&] { DestroyArgs(context); });
    aclnnStatus ret;

    int64_t batchSize = 1;
    context.maxSeqlenQ = 16;
    context.maxSeqlenK = 4;
    context.layoutQOptional = (char *)malloc(sizeof(char) * 16);
    context.layoutKOptional = (char *)malloc(sizeof(char) * 16);
    strcpy(context.layoutQOptional, scenario.hasCuSeq ? "TND" : "BSND");
    strcpy(context.layoutKOptional, scenario.hasCuSeq ? "TND" : "BSND");

    ret = CreateTensor(aclDataType::ACL_INT32, { SLI_METADATA_SIZE }, context.metadata);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "Create metadata failed. Error: %d", ret);

    if (scenario.hasCuSeq) {
        ret = CreateTensor(aclDataType::ACL_INT32, { batchSize + 1 }, context.cuSeqLensQOptional);
        CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "Create cuSeqLensQOptional failed. Error: %d", ret);
        ret = CreateTensor(aclDataType::ACL_INT32, { batchSize + 1 }, context.cuSeqLensKOptional);
        CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "Create cuSeqLensKOptional failed. Error: %d", ret);
        SetInt32TensorData(context.cuSeqLensQOptional, { 0, static_cast<int32_t>(context.maxSeqlenQ) });
        SetInt32TensorData(context.cuSeqLensKOptional, { 0, static_cast<int32_t>(context.maxSeqlenK) });
        context.batchSize = 0;
    } else {
        context.batchSize = batchSize;
    }

    argsGuard.Dismiss();
    return ACL_SUCCESS;
}

void PrintMetadata(const SliGradKLLossMetaData &metadata)
{
    const char *socName = aclrtGetSocName();
    std::string socVersion = (socName != nullptr) ? std::string(socName) : std::string();
    bool isA5 = socVersion.find("Ascend910") == std::string::npos;
    if (isA5) {
        const int32_t *data = reinterpret_cast<const int32_t *>(&metadata);
        printf("TOTAL_NUM               : %d\n", data[TOTAL_NUM]);
        printf("FORMER_CORE_PROCESS_NUM : %d\n", data[FORMER_CORE_PROCESS_NUM]);
        printf("REMAIN_CORE_PROCESS_NUM : %d\n", data[REMAIN_CORE_PROCESS_NUM]);
        printf("REMAIN_CORE_NUM         : %d\n", data[REMAIN_CORE_NUM]);
        printf("USED_CORE_NUM           : %d\n", data[USED_CORE_NUM]);
        return;
    }
    printf("coreNum          : %d\n", metadata.coreNum);
    printf("totalSize        : %d\n", metadata.totalSize);
    printf("splitFactorSize  : %d\n", metadata.splitFactorSize);
    for (uint32_t i = 0; i < std::min<uint32_t>(metadata.coreNum, SLI_METADATA_MAX_CORE_NUM); ++i) {
        printf("bS1Index[%u]      : %d\n", i, metadata.bS1Index[i]);
    }
}

int main() {
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "Init acl failed. ERROR: %d", ret);
    ScopeGuard sysGuard([&] { Finalize(deviceId, stream); });

    ArgScenario scenario {};
    scenario.hasCuSeq = true;
    ArgContext context {};
    ret = CreateArgs(scenario, context);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "Create input arguments failed. ERROR: %d", ret);
    ScopeGuard argsGuard([&] { DestroyArgs(context); });

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    void *workspaceAddr = nullptr;
    ret = aclnnSparseLightningIndexerKLLossGradMetadataGetWorkspaceSize(
        context.cuSeqLensQOptional.data, context.cuSeqLensKOptional.data, context.seqUsedQOptional.data,
        context.seqUsedKOptional.data, context.cmpResidualKOptional.data, context.batchSize, context.maxSeqlenQ,
        context.maxSeqlenK, context.numHeadsQ, context.numHeadsK, context.headDim, context.topk, context.layoutQOptional,
        context.layoutKOptional, context.maskMode, context.cmpRatio, context.metadata.data, &workspaceSize, &executor);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret,
        "aclnnSparseLightningIndexerKLLossGradMetadataGetWorkspaceSize failed. ERROR: %d", ret);

    if (workspaceSize > static_cast<uint64_t>(0)) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "allocate workspace failed. ERROR: %d", ret);
    }
    ScopeGuard workspaceGuard([&] {
        if (workspaceAddr != nullptr) {
            aclrtFree(workspaceAddr);
            workspaceAddr = nullptr;
        }
    });

    ret = aclnnSparseLightningIndexerKLLossGradMetadata(workspaceAddr, workspaceSize, executor, stream);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret,
        "aclnnSparseLightningIndexerKLLossGradMetadata failed. ERROR: %d", ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "aclrtSynchronizeStream failed. ERROR: %d", ret);

    SliGradKLLossMetaData result {};
    ret = aclrtMemcpy(&result, sizeof(result), context.metadata.deviceAddr, sizeof(result), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "aclrtMemcpy failed. ERROR: %d", ret);
    PrintMetadata(result);

    return 0;
}
```
