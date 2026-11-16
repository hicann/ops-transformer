# aclnnInplaceFusedCausalConv1dV2

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-21T10:48:26.383Z -->

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

- Description: Performs causal 1D convolution on sequences. Along the sequence dimension, each sequence head is padded with cached data (with a length of the convolution kernel width minus 1), ensuring that the output depends on the current and historical inputs. After the convolution, the data of the current sequence part is updated to the cache. Based on the output of the causal 1D convolution, the original input is added to the output to implement a residual connection. This API supports features such as APC (Automatic Prefix Caching), MTP (Multi-Token Prediction, which can be used for speculative decoding), and residual connection. Compared with aclnnInplaceFusedCausalConv1dV1, this API adds support for the maxDraftTokens parameter, which indicates the maximum number of supported speculative tokens. Compared with the FusedCausalConv1d operator, this API supports in-place update.<br>

- The following use cases are supported:
  - Scenario 1 (prefill scenario):

    ```cpp
    x: [cuSeqLen, dim]
    weight: [K, dim], where K=3
    convStates: [-1, K-1, dim]
    queryStartLoc: [batch+1]
    cacheIndices: APC disabled: [batch] or None, APC enabled: [block, maxNumBlocks]
    initialStateMode: [batch]
    bias: [dim] (not supported currently)
    numAcceptedTokens: [batch] (not supported currently)
    numComputedTokens: [batch]
    blockIdxFirstScheduledToken: APC disabled: None, APC enabled: [batch]
    blockIdxLastScheduledToken: APC disabled: None, APC enabled: [batch]
    initialStateIdx: APC disabled: None, APC enabled: [batch]
    activationMode: (not supported currently)
    padSlotId: default value -1
    runMode: (not supported yet)
    residualConnection: without residual connection: 0, with residual connection: 1
    blockSize: typical values 128/256
    convMode: Qwen3-Next mode: 0, Pangu V2: 1
    ```

    where cuSeqLen is the total length after concatenating all variable-length sequences in the batch.

  - Scenario 2 (mixed prefill and decode scenario):

    ```cpp
    x: [cuSeqLen, dim]
    weight: [K, dim], where K=3
    convStates: [-1, K-1+m, dim]
    queryStartLoc: [batch+1]
    cacheIndices: APC disabled: [batch] or None, APC enabled: [block, maxNumBlocks]
    initialStateMode: [batch]
    bias: [dim] (not supported yet)
    numAcceptedTokens: [batch]
    numComputedTokens: [batch]
    blockIdxFirstScheduledToken: APC disabled: None, APC enabled: [batch]
    blockIdxLastScheduledToken: APC disabled: None, APC enabled: [batch]
    initialStateIdx: APC disabled: None, APC enabled: [batch]
    activationMode: (not supported yet)
    padSlotId: default value -1
    runMode: (not supported yet)
    residualConnection: residual disabled: 0, residual enabled: 1
    blockSize: typical value 128/256
    convMode: Qwen3-Next mode: 0, Pangu V2: 1
    maxDraftTokens: number of speculative tokens (multiTokenNum), value range [0, 16]
    ```

    where cuSeqLen is the total length of all variable-length sequences concatenated in a batch.

  - Scenario 3 (decode scenario - variable-length sequences):

    ```cpp
    x: [cuSeqLen, dim]
    weight: [K, dim], where K=3
    convStates: [-1, K-1+m, dim]
    queryStartLoc: [batch+1]
    cacheIndices: APC disabled: [batch] or None, APC enabled: [block, maxNumBlocks]
    initialStateMode: [batch]
    bias: [dim] (currently not supported)
    numAcceptedTokens: [batch]
    numComputedTokens: [batch]
    blockIdxFirstScheduledToken: APC disabled: None, APC enabled: [batch]
    blockIdxLastScheduledToken: APC disabled: None, APC enabled: [batch]
    initialStateIdx: APC disabled: None, APC enabled: [batch]
    activationMode: (currently not supported)
    padSlotId: default value -1
    runMode: (currently not supported)
    residualConnection: no residual: 0, residual: 1
    blockSize: typical value 128/256
    convMode: Qwen3-Next mode: 0, Pangu V2: 1
    maxDraftTokens: speculative count (multiTokenNum), value range [0, 16]
    ```

    where stateLen must be greater than the maximum token count in all batches plus 1.

  - Scenario 4 (decode scenario - fixed batch):

    ```cpp
    x: [batch, m+1, dim]
    weight: [K, dim], where K=3
    convStates: [-1, K-1+m, dim]
    queryStartLoc: [batch+1]
    cacheIndices: APC disabled: [batch] or None, APC enabled: [block, maxNumBlocks]
    initialStateMode: [batch]
    bias: [dim] (currently not supported)
    numAcceptedTokens: [batch]
    numComputedTokens: [batch]
    blockIdxFirstScheduledToken: APC disabled: None, APC enabled: [batch]
    blockIdxLastScheduledToken: without APC: None, with APC: [batch]
    initialStateIdx: without APC: None, with APC: [batch]
    activationMode: (not supported currently)
    padSlotId: default value -1
    runMode: (not supported currently)
    residualConnection: without residual: 0, with residual: 1
    blockSize: typical value 128/256
    convMode: Qwen3-Next mode: 0, Pangu V2: 1
    maxDraftTokens: number of speculative tokens (multiTokenNum), value range [0, 16]
    ```

- Formula:

  Formula: K is the convolution kernel width, L is the sequence length of the current batch, dim is the feature dimension, batchId is the index of the current variable-length sequence being processed, C is the length of the convStates cache space (convStates.size(1)), and B is the APC block size (bSize). For each batch in input x, perform the following operations:
  1. Cache read

      Cache line index:

      $$
      readCacheLine = \begin{cases}
      cacheIndices[batchId, \; initialStateIdx[batchId]], & \text{APC mode} \\
      cacheIndices[batchId], & \text{ non-APC and cacheIndices exist} \\
      batchId, & \text{ otherwise}
      \end{cases}
      $$

      Case 1: First computation (numComputedTokens[batchId] == 0)

      $$
      cachedState[i, dim] = 0, \quad 0 \leq i < K-1
      $$

      $$
      offset = 0
      $$

      Case 2: Speculative decoding mode (numAcceptedTokens exists)

      $$
      offset = numAcceptedTokens[batchId] - 1
      $$

      $$
      cachedState[i, dim] = convStates[readCacheLine][i, dim], \quad 0 \leq i <   offset + K - 1
      $$

      Case 3: Default mode

      $$
      offset = C - (K - 1)
      $$

      $$
      cachedState[i, dim] = convStates[readCacheLine][i, dim], \quad 0 \leq i < offset + K - 1
      $$

  2. Cache concatenation

      $$
      paddedInput[i, dim] =
      \begin{cases}
      cachedState[i, dim], & 0 \leq i < offset + K - 1 \\
      x[i - (offset + K - 1), dim], & offset + K - 1 \leq i < offset + K - 1 + L
      \end{cases}
      $$

  3. Cache update

      $$
      Len = offset + K - 1 + L
      $$

      $$
      M = \min(C, \; Len)
      $$

      $$
      writeCacheLine = \begin{cases}
      cacheIndices[batchId, \; idxLast], & \text{APC mode} \\
      cacheIndices[batchId], & \text{ not APC and cacheIndices exist} \\
      batchId, & \text{ otherwise}
      \end{cases}
      $$

      $$
      convStates[writeCacheLine][C - M + i, dim] = paddedInput[Len - M + i, dim], \quad i = 0, 1, \dots, M-1
      $$

  4. Offset clipping

      $$
      x'[i, dim] = paddedInput[i + offset, dim], \quad 0 \leq i < K - 1 + L
      $$

  5. APC cache filling (optional, in APC mode)

      $$
      seqCompletedOffsetToken = numComputedTokens[batchId] \mod B
      $$

      $$
      seqCompletedOffset = B - seqCompletedOffsetToken
      $$

      $$
      seqEndOffset = (L - seqCompletedOffset) \mod B
      $$

      $$
      lastFullBlockTokenIndex = \begin{cases}
      L - seqEndOffset - B, & seqEndOffset = 0 \\
      L - seqEndOffset, & \text{otherwise}
      \end{cases}
      $$

      $$
      nBlockToFill = idxLast - idxFirst
      $$

      For each chunk = 0, 1, ..., nBlockToFill - 1:

      $$
      boundaryIdx = lastFullBlockTokenIndex - (nBlockToFill - chunk - 1) \times B
      $$

      $$
      convStates[cacheIndices[batchId, \; idxFirst + chunk]][C-(K-1)+j, \; dim] = x'[boundaryIdx + j, \; dim], \quad j = 0, \dots, K-2
      $$

  6. Causal 1D convolution

      $$
      y[i, dim] = \sum_{k=0}^{K-1} w[k, dim] \cdot x'[i + k, dim], \quad i = 0, 1, \dots, L-1
      $$

  7. Zero-padding reset (optional, when convMode == 1 and numComputedTokens is not empty)

      $$
      resetIdx = \min\!\Big(\max\!\big(K - 1 - numComputedTokens[batchId], \; 0\big), \; L\Big)
      $$

      $$
      y[i, dim] = 0, \quad 0 \leq i < resetIdx
      $$

  8. Residual connection (optional)

      $$
      y[i, dim] = x[i, dim] + y[i, dim]
      $$

  9. In-place update

      $$
      x[i, dim] = y[i, dim]
      $$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). The `aclnnInplaceFusedCausalConv1dV2GetWorkspaceSize` API must be called first to obtain the input parameters, calculate the required workspace size, and obtain the executor that includes the operator calculation process. Then, call the `aclnnInplaceFusedCausalConv1dV2` API to perform the calculation.

```Cpp
aclnnStatus aclnnInplaceFusedCausalConv1dV2GetWorkspaceSize(
  aclTensor       *x,
  const aclTensor *weight,
  aclTensor       *convStates,
  const aclTensor *queryStartLoc,
  const aclTensor *cacheIndices,
  const aclTensor *initialStateMode,
  const aclTensor *bias,
  const aclTensor *numAcceptedTokens,
  const aclTensor *numComputedTokens,
  const aclTensor *blockIdxFirstScheduledToken,
  const aclTensor *blockIdxLastScheduledToken,
  const aclTensor *initialStateIdx,
  int64_t          activationMode,
  int64_t          padSlotId,
  int64_t          runMode,
  int64_t          maxQueryLen,
  int64_t          residualConnection,
  int64_t          blockSize,
  int64_t          convMode,
  int64_t          maxDraftTokens,
  uint64_t        *workspaceSize,
  aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnInplaceFusedCausalConv1dV2(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnInplaceFusedCausalConv1dV2GetWorkspaceSize

- **Parameters**:

  <table style="undefined;table-layout: fixed; width: 1550px"><colgroup>
  <col style="width: 250px">
  <col style="width: 100px">
  <col style="width: 333px">
  <col style="width: 400px">
  <col style="width: 212px">
  <col style="width: 100px">
  <col style="width: 107px">
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>x (aclTensor*)</td>
      <td>Input/Output</td>
      <td>x in the formula represents the input sequence. The convolution result updates x in place.</td>
      <td><ul><li>Does not support empty tensors.</li><li>prefill scenario: shape is [cuSeqLen, dim].</li><li>decode scenario: shape is [cuSeqLen, dim] or [batch, seqLen, dim].</li>
      <li>Only the token dimension (dim0 for 2D, dim1 for 3D) can be non-contiguous, with a stride greater than or equal to 1</li></ul></td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>2-3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>weight (aclTensor*)</td>
      <td>Input</td>
      <td>w in the formula represents the causal 1D convolution kernel.</td>
      <td><ul><li>Does not support empty tensors.</li><li>Shape is [K, dim].</li></ul></td>
      <td>Data type consistent with x</td>
      <td>ND</td>
      <td>2</td>
      <td>×</td>
    </tr>
    <tr>
      <td>convStates(aclTensor*)</td>
      <td>Input/Output</td>
      <td>cacheState in the formula. It indicates the cache state tensor, which stores the historical token data of each sequence and is updated in place after each sequence is calculated.</td>
      <td><ul><li>Does not support empty tensors.</li><li>Shape is [..., stateLen, dim]. The size of dimension 0 is not fixed.</li><li>stateLen == K-1+m. In the prefill scenario, m=0. In decode and PD mixed deployment scenarios, m=[0, maxDraftTokens].</li><li>m indicates the speculation count, which maps to the value of the numAcceptedTokens parameter.</li><li>Supports non-contiguous dim0 and dim1 with a stride greater than or equal to 1.</li></ul></td>
      <td>Data type consistent with x</td>
      <td>ND</td>
      <td>3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>queryStartLoc (aclTensor*)</td>
      <td>Input</td>
      <td>Sequence start position index, recording the start position of each sequence in the concatenated tensor x.</td>
      <td><ul><li>Does not support empty tensors.</li><li>The shape is [batch+1,].</li><li>queryStartLoc[i] indicates the start offset of the i-th sequence.</li><li>queryStartLoc[0] must be 0, queryStartLoc[-1] must be cuSeqLen, and adjacent values are not equal.</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>1</td>
      <td>×</td>
    </tr>
    <tr>
      <td>cacheIndices (aclTensor*)</td>
      <td>Input</td>
      <td>Cache indices, specifying the index in convStates of the cache state corresponding to each sequence.</td>
      <td><ul><li>Does not support empty tensors. When APC is enabled, the tensor cannot be None and must be 2-dimensional.</li><li>1D shape [batch,] or 2D shape [batch, maxNumBlocks], where maxNumBlocks is in range [1,1024].</li><li>Values must be unique (unless equal to padSlotId).</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>1-2</td>
      <td>×</td>
    </tr>
    <tr>
      <td>initialStateMode (aclTensor*)</td>
      <td>Input</td>
      <td>Initial state flag, indicating whether each sequence uses cached data.</td>
      <td>Not supported in the current version. Set it to nullptr or ignore it when passing parameters.</td>
      <td>INT32</td>
      <td>ND</td>
      <td>1</td>
      <td>×</td>
    </tr>
    <tr>
      <td>bias (aclTensor*)</td>
      <td>Input</td>
      <td>Bias of the convolution.</td>
      <td>Not supported in the current version. Set it to nullptr or ignore it when passing parameters.</td>
      <td>Data type consistent with x</td>
      <td>ND</td>
      <td>1</td>
      <td>×</td>
    </tr>
    <tr>
      <td>numAcceptedTokens (aclTensor*)</td>
      <td>Input</td>
      <td>numAcceptedTokens in the formula, indicating the number of random speculative decoding tokens for each batch.</td>
      <td><ul><li>For prefill, the element value is 0. During decode, 1 <= the value in numAcceptedTokens <= seqlen - 1, where seqlen indicates the sequence length of the batch.</li><li>The shape is [batch,].</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>1</td>
      <td>×</td>
    </tr>
    <tr>
      <td>numComputedTokens (aclTensor*)</td>
      <td>Input</td>
      <td>numComputedTokens in the formula, the total count of tokens processed in the current batch, used to determine the initial state.</td>
      <td><ul><li>Uses a zero-initialized cache for the first token; cannot be empty in Pangu V2 mode.</li><li>shape is [batch,].</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>1</td>
      <td>×</td>
    </tr>
    <tr>
      <td>blockIdxFirstScheduledToken (aclTensor*)</td>
      <td>Input</td>
      <td>Block index corresponding to the start position of the current batch.</td>
      <td><ul><li>Cannot be empty when APC is enabled.</li><li>shape is [batch,].</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>1</td>
      <td>×</td>
    </tr>
    <tr>
      <td>blockIdxLastScheduledToken (aclTensor*)</td>
      <td>Input</td>
      <td>Block index corresponding to seqLen-1 of the current batch.</td>
      <td><ul><li>Cannot be empty when APC is enabled.</li><li>The shape is [batch,].</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>1</td>
      <td>×</td>
    </tr>
    <tr>
      <td>initialStateIdx (aclTensor*)</td>
      <td>Input</td>
      <td>Index of the initial index block.</td>
      <td><ul><li>Cannot be empty when APC is enabled.</li><li>Shape is [batch,].</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>1</td>
      <td>×</td>
    </tr>
    <tr>
      <td>activationMode (int64_t)</td>
      <td>Input</td>
      <td>Activation function type.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>padSlotId (int64_t)</td>
      <td>Input</td>
      <td>Used to skip batches that do not need to participate in the calculation.</td>
      <td><ul><li>Only variable-length sequences that do not participate in the calculation are supported at the beginning or end of x.</li><li>The batch is skipped when cacheIndices[i]==padSlotId.</li></ul></td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>runMode (int64_t)</td>
      <td>Input</td>
      <td>Used to determine whether it is a prefill or decode scenario.</td>
      <td>Legacy API.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>maxQueryLen (int64_t)</td>
      <td>Input</td>
      <td>Maximum seqLen across all batches. Only the decode scenario (fixed batch) supports -1.</td>
      <td>-</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>residualConnection (int64_t)</td>
      <td>Input</td>
      <td>Whether to perform the residual connection.</td>
      <td><ul><li>Value: 0 or 1:<br>0: no residual connection;<br>1: y and x are added, and the result is output.</li></ul></td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>blockSize (int64_t)</td>
      <td>Input</td>
      <td>Size of the block.</td>
      <td><ul><li>>Value range: greater than or equal to 2 or equal to 0 (0 can be passed in non-APC scenarios), typical values: 128/256.</li></ul></td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>convMode (int64_t)</td>
      <td>Input</td>
      <td>convMode in the formula. Two implementations are supported: Qwen3-Next and Pangu V2.</td>
      <td><ul><li>The value can be 0 or 1:<br>0: Qwen3-Next community version implementation;<br>1: Pangu V2 implementation.</li></ul></td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>maxDraftTokens(int64_t)</td>
      <td>Input</td>
      <td>Maximum speculative count.</td>
      <td><ul><li>Value range: [0, 16].</li></ul></td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>workspaceSize(int64_t*)</td>
      <td>Output</td>
      <td>Returns the workspace size that the user needs to allocate on the Device side.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor (aclOpExecutor**)</td>
      <td>Output</td>
      <td>Returns the op executor that includes the calculation flow of the operator.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody></table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table style="undefined;table-layout: fixed;width: 1155px"><colgroup>
  <col style="width: 319px">
  <col style="width: 144px">
  <col style="width: 671px">
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
      <td>The passed-in x, weight, or convStates is a null pointer.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_INNER_TILING_ERROR</td>
      <td>561002</td>
      <td>      The data types of the input and output are not in the supported range.<br>
      The data types of x, weight, convStates, and bias are inconsistent.<br>
      The data types of queryStartLoc, cacheIndices, initialStateMode, numAcceptedTokens, numComputedTokens, blockIdxFirstScheduledToken, blockIdxLastScheduledToken, and initialStateIdx are inconsistent.<br>
      The shapes of the input and output tensors are not in the supported range.<br>
      The input attributes are not in the supported range.<br>
      dim is not in the specified value range.<br>
      </td>
    </tr>
  </tbody></table>

## aclnnInplaceFusedCausalConv1dV2

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
      <td>Memory address of the workspace applied on the Device side.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace applied on the Device side, obtained by the first-phase API aclnnInplaceFusedCausalConv1dV2GetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>The op executor that includes the calculation flow of the operator.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream on which the task is executed.</td>
    </tr>
  </tbody>
  </table>

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnInplaceFusedCausalConv1dV2 defaults to a deterministic implementation.

- Input shape constraints:
  - prefill use case:
    - x supports 2D [cuSeqLen, dim].
    - weight must be 2D [K, dim], where K is fixed to 3.
    - convStates must be 3D [..., K-1, dim]. The size of dimension 0 is not fixed and must be greater than or equal to the count of batches involved in the calculation (that is, the count of batches where cacheIndicesOptional is not equal to padSlotId).
    - queryStartLocOptional must exist.
    - cacheIndicesOptional is 1D [batch, ] or 2D [batch, maxNumBlocks], where 1D indicates that APC is not enabled and 2D indicates that APC is enabled.
    - cuSeqLen range [batch, 1024 * 1024], dim range [64, 16384] and dim must be a multiple of 16, and their product must be within [64 * batch, 4G]. batch range [1, 256].
    - maxNumBlocks >= ceil(maxQueryLen, blockSize).
    - maxQueryLen > maxDraftTokens + 1.
  - Mixed prefill and decode use case:
    - x supports 2D [cuSeqLen, dim].
    - weight must be 2D [K, dim], where K is fixed to 3.
    - convStates must be 3D [..., K-1+m, dim], where the size of dimension 0 is not fixed and must be greater than or equal to the batch count participating in the calculation (that is, the batch count where cacheIndicesOptional is not equal to padSlotId).
    - queryStartLocOptional must exist.
    - cacheIndicesOptional is 1D [batch, ] or 2D [batch, maxNumBlocks], where 1D indicates that APC is not enabled and 2D indicates that APC is enabled.
    - cuSeqLen is in the range [batch, 1024 * 1024]; dim is in the range [64, 16384] and must be a multiple of 16; the product of the two must be within [64 * batch, 4G]; batch is in the range [1, 256].
    - maxNumBlocks >= ceil(maxQueryLen, blockSize).
    - maxQueryLen > maxDraftTokens + 1.
  - Decode scenario (variable-length sequences):
    - x supports 2D [cuSeqLen, dim].
    - weight must be 2D [K, dim], where K is fixed to 3.
    - convStates must be 3D [..., K-1+m, dim]. Its dimension 0 size is not fixed and must be greater than or equal to the count of batches participating in the calculation (that is, the count of batches whose cacheIndicesOptional is not equal to padSlotId).
    - queryStartLocOptional must exist.
    - cacheIndicesOptional is 1D [batch, ] or 2D [batch, maxNumBlocks], where 1D indicates APC is disabled and 2D indicates APC is enabled.
    - The cuSeqLen range is [batch, batch * (maxDraftTokens + 1)], and the seqLen range of each batch is [1, maxDraftTokens + 1]. The dim range is [64, 16384] and dim is a multiple of 16. The batch range is [1, 256].
    - maxNumBlocks >= ceil(maxQueryLen, blockSize).
    - The maxQueryLen range is [1, maxDraftTokens + 1].
  - decode use case (fixed batch):
    - x supports 3D [batch, m+1, dim].
    - weight must be 2D [K, dim], where K is fixed to 3.
    - convStates must be 3D [..., K-1+m, dim]. Its dimension 0 size is not fixed and must be greater than or equal to the count of batches participating in the calculation (that is, the count of batches whose cacheIndicesOptional is not equal to padSlotId).
    - cacheIndicesOptional is 1D [batch, ] or 2D [batch, maxNumBlocks], where 1D indicates that APC is not enabled and 2D indicates that APC is enabled.
    - m: range [0, maxDraftTokens]; dim: range [64, 16384] and a multiple of 16; batch: range [1, 256].
    - maxNumBlocks >= ceil(maxQueryLen, blockSize).
    - maxQueryLen: range [1, maxDraftTokens + 1]; can be -1.

- Input value range restrictions:
  - queryStartLocOptional is the cumulative offset, with a value range of [0, cuSeqLen] and a length of batch+1. queryStartLocOptional[i] indicates the start offset of the i-th sequence, and queryStartLocOptional[batch+1] indicates the end position of the last sequence.
  - blockSize is 0 or greater than or equal to 2, and must not be 0 when APC is enabled.
  - When blockIdxFirstScheduledTokenOptional, blockIdxLastScheduledTokenOptional, initialStateIdxOptional, numComputedTokensOptional, and cacheIndicesOptional all exist, APC is enabled and the following conditions must be met (i is the batch index):
    - cacheIndices is 2D
    - initialStateIdxOptional[i] <= blockIdxFirstScheduledTokenOptional[i]+1
    - initialStateIdxOptional[i] <= blockIdxLastScheduledTokenOptional[i]
    - blockIdxFirstScheduledTokenOptional[i] <= blockIdxLastScheduledTokenOptional[i]
    - blockIdxLastScheduledTokenOptional[i] < maxNumBlocks
  - numAcceptedTokensOptional can be None or non-None. When not None, its length is batch. Elements corresponding to prefill are 0, and elements corresponding to decode are greater than 0 and less than or equal to seqLen-1 of the current batch.
  - Each element in numComputedTokensOptional is greater than or equal to 0.
  - The value range of cacheIndicesOptional is [0, convStates.dim[0]-1], and the values must not be equal to each other (unless equal to padSlotId).
  - maxQueryLen = the maximum seqLen in the batch.
  - The value range of maxDraftTokens is [0, 16].
  - In Pangu V2 mode (convMode = 1), numComputedTokensOptional cannot be None.
  - The operator input and intermediate calculation results do not exceed the value range of the corresponding data type (float16/bfloat16).
  - The operator input does not support ±inf or Nan.

## Example

The following sample code is for reference only. For details about the compilation and execution process, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include <cmath>
#include "acl/acl.h"
#include "aclnnop/aclnn_inplace_fused_causal_conv1d_v2.h"

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

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream *stream)
{
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

// IEEE 754 fp16 decode (uint16_t → float)
float Fp16ToFloat(uint16_t raw)
{
    uint32_t sign = (raw >> 15) & 0x1;
    uint32_t exp = (raw >> 10) & 0x1F;
    uint32_t mant = raw & 0x3FF;
    float val;
    if (exp == 0) {
        val = (sign ? -1.0f : 1.0f) * (static_cast<float>(mant) / 1024.0f) * (1.0f / 16384.0f);
    } else if (exp == 31) {
        val = (mant == 0) ? (sign ? -HUGE_VALF : HUGE_VALF) : NAN;
    } else {
        val = (sign ? -1.0f : 1.0f) * (1.0f + static_cast<float>(mant) / 1024.0f) *
              ldexpf(1.0f, static_cast<int>(exp) - 15);
    }
    return val;
}

template <typename T>
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor, aclFormat format)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    *tensor = aclCreateTensor(shape.data(), static_cast<int64_t>(shape.size()), dataType, strides.data(), 0, format,
                              shape.data(), static_cast<int64_t>(shape.size()), *deviceAddr);
    return 0;
}

void FreeTensorAndBuffer(aclTensor *tensor, void *deviceAddr)
{
    if (tensor != nullptr) {
        aclDestroyTensor(tensor);
    }
    if (deviceAddr != nullptr) {
        aclrtFree(deviceAddr);
    }
}

int main()
{
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    int64_t K = 3;            // kernel width
    int64_t dim = 128;        // feature dimension
    int64_t batch = 4;        // number of sequences
    int64_t numSlots = 8;     // total cache slots (>= batch for non-APC)
    int64_t stateLen = K - 1; // cache state length per slot (= 2)
    // prefill: seqLens = [5, 3, 7, 4], cuSeqLen = 19
    int64_t cuSeqLen = 19;

    // ---- Tensor shapes ----
    std::vector<int64_t> xShape = {cuSeqLen, dim};
    std::vector<int64_t> weightShape = {K, dim};
    std::vector<int64_t> convStatesShape = {numSlots, stateLen, dim};
    std::vector<int64_t> queryStartLocShape = {batch + 1};
    std::vector<int64_t> cacheIndicesShape = {batch}; // 1D: non-APC
    std::vector<int64_t> initialStateModeShape = {batch};
    std::vector<int64_t> biasShape = {dim};
    std::vector<int64_t> numAcceptedTokensShape = {batch};

    // ---- Host data ----
    // x: fp16 1.0, shape [19, 128]
    std::vector<uint16_t> hostX(cuSeqLen * dim, 0x3C00); // fp16 1.0

    // weight: pass-through at k=0, zero elsewhere
    std::vector<uint16_t> hostWeight(K * dim, 0);
    for (int64_t d = 0; d < dim; d++) {
        hostWeight[0 * dim + d] = 0x3C00; // k=0: 1.0
    }

    // convStates: zero-initialized, shape [8, 2, 128]
    std::vector<uint16_t> hostConvStates(numSlots * stateLen * dim, 0);

    // queryStartLoc: [0, 5, 8, 15, 19]
    std::vector<int32_t> hostQueryStartLoc = {0, 5, 8, 15, 19};

    // cacheIndices: each batch uses a distinct cache slot
    std::vector<int32_t> hostCacheIndices = {0, 3, 1, 5};

    // initialStateMode: 1=from cache, 0=zero-init, 2=from cache (MTP variant)
    std::vector<int32_t> hostInitialStateMode = {1, 0, 2, 1};

    // bias: zero (unused)
    std::vector<uint16_t> hostBias(dim, 0);

    // numAcceptedTokens: 0 for prefill (no speculative decoding)
    std::vector<int32_t> hostNumAcceptedTokens(batch, 0);

    // ---- Device pointers and aclTensors ----
    void *xDev = nullptr, *weightDev = nullptr, *convStatesDev = nullptr;
    void *queryStartLocDev = nullptr, *cacheIndicesDev = nullptr;
    void *initialStateModeDev = nullptr, *biasDev = nullptr;
    void *numAcceptedTokensDev = nullptr;

    aclTensor *xTensor = nullptr, *weightTensor = nullptr, *convStatesTensor = nullptr;
    aclTensor *queryStartLocTensor = nullptr, *cacheIndicesTensor = nullptr;
    aclTensor *initialStateModeTensor = nullptr, *biasTensor = nullptr;
    aclTensor *numAcceptedTokensTensor = nullptr;

    // ---- Create required tensors (x is non-const for inplace) ----
    ret = CreateAclTensor(hostX, xShape, &xDev, ACL_FLOAT16, &xTensor, ACL_FORMAT_ND);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor(hostWeight, weightShape, &weightDev, ACL_FLOAT16, &weightTensor, ACL_FORMAT_ND);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret =
        CreateAclTensor(hostConvStates, convStatesShape, &convStatesDev, ACL_FLOAT16, &convStatesTensor, ACL_FORMAT_ND);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor(hostQueryStartLoc, queryStartLocShape, &queryStartLocDev, ACL_INT32, &queryStartLocTensor,
                          ACL_FORMAT_ND);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor(hostCacheIndices, cacheIndicesShape, &cacheIndicesDev, ACL_INT32, &cacheIndicesTensor,
                          ACL_FORMAT_ND);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor(hostInitialStateMode, initialStateModeShape, &initialStateModeDev, ACL_INT32,
                          &initialStateModeTensor, ACL_FORMAT_ND);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor(hostBias, biasShape, &biasDev, ACL_FLOAT16, &biasTensor, ACL_FORMAT_ND);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor(hostNumAcceptedTokens, numAcceptedTokensShape, &numAcceptedTokensDev, ACL_INT32,
                          &numAcceptedTokensTensor, ACL_FORMAT_ND);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    int64_t activationMode = 0; // 0: None
    int64_t padSlotId = -1;     // -1: no pad-slot skipping
    int64_t runMode = 0;
    int64_t maxQueryLen = 7;
    int64_t residualConnection = 0; // 0: no residual
    int64_t blockSize = 0;          // 0: non-APC
    int64_t convMode = 0;           // 0: Qwen3-Next
    int64_t maxDraftTokens = 7;

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;

    ret = aclnnInplaceFusedCausalConv1dV2GetWorkspaceSize(
        xTensor, weightTensor, convStatesTensor, queryStartLocTensor, cacheIndicesTensor, initialStateModeTensor,
        biasTensor, numAcceptedTokensTensor,
        nullptr, // numComputedTokens (non-APC: nullptr)
        nullptr, // blockIdxFirstScheduledToken
        nullptr, // blockIdxLastScheduledToken
        nullptr, // initialStateIdx
        activationMode, padSlotId, runMode, maxQueryLen, residualConnection, blockSize, convMode, maxDraftTokens, &workspaceSize,
        &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceFusedCausalConv1dV2GetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);

    void *workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    ret = aclnnInplaceFusedCausalConv1dV2(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceFusedCausalConv1dV2 failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    auto xSize = GetShapeSize(xShape);
    std::vector<uint16_t> xResult(xSize, 0);
    ret = aclrtMemcpy(xResult.data(), xSize * sizeof(uint16_t), xDev, xSize * sizeof(uint16_t),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy x from device to host failed. ERROR: %d\n", ret); return ret);

    auto csSize = GetShapeSize(convStatesShape);
    std::vector<uint16_t> csResult(csSize, 0);
    ret = aclrtMemcpy(csResult.data(), csSize * sizeof(uint16_t), convStatesDev, csSize * sizeof(uint16_t),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy convStates from device to host failed. ERROR: %d\n", ret);
              return ret);

    LOG_PRINT("=== x output (inplace, first 10 elements) ===\n");
    for (int64_t i = 0; i < 10 && i < xSize; i++) {
        LOG_PRINT("  x[%lld] = %f\n", (long long)i, (double)Fp16ToFloat(xResult[i]));
    }
    LOG_PRINT("  ...\n");

    LOG_PRINT("=== convStates output (first 10 elements) ===\n");
    for (int64_t i = 0; i < 10 && i < csSize; i++) {
        LOG_PRINT("  convStates[%lld] = %f\n", (long long)i, (double)Fp16ToFloat(csResult[i]));
    }

    FreeTensorAndBuffer(xTensor, xDev);
    FreeTensorAndBuffer(weightTensor, weightDev);
    FreeTensorAndBuffer(convStatesTensor, convStatesDev);
    FreeTensorAndBuffer(queryStartLocTensor, queryStartLocDev);
    FreeTensorAndBuffer(cacheIndicesTensor, cacheIndicesDev);
    FreeTensorAndBuffer(initialStateModeTensor, initialStateModeDev);
    FreeTensorAndBuffer(biasTensor, biasDev);
    FreeTensorAndBuffer(numAcceptedTokensTensor, numAcceptedTokensDev);

    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    LOG_PRINT("Test completed successfully!\n");
    return 0;
}
```
