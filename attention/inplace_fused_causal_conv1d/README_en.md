# InplaceFusedCausalConv1d

English | [简体中文](./README.md)

## Product Support

| Product                                                         | Supported |
| :----------------------------------------------------------- | :------: |
|  <term>Ascend 950PR&950DT products</term>   |     √    |
| <term>Atlas A3 products</term>     |    ×     |
| <term>Atlas A2 products</term> |    ×     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                             |    ×     |
| <term>Atlas training products</term>                              |    ×     |

## Description

- Operator function: Performs a 1D causal convolution on sequences. Along the sequence dimension, cache data (with a length of the convolution kernel width minus 1) is used to pad the head of each sequence, ensuring that the output depends on the current and historical inputs. After the convolution is complete, part of the data of the current sequence is updated to the cache. Based on the output of the 1D causal convolution, the original input is added to the output to implement the residual connection. Features such as APC (Automatic Prefix Caching), MTP (speculative decoding), residual connection, and in-place update are supported.

- This operator supports the following scenarios:

  - Scenario 1 (prefill scenario):

    ```cpp
    x: [cuSeqLen, dim]
    weight: [K, dim], where K=3
    convStates: [-1, K-1, dim]
    queryStartLoc: [batch+1]
    cacheIndices: without APC: [batch] or None, with APC: [block, maxNumBlocks]
    initialStateMode: [batch]
    bias: [dim] (not supported currently)
    numAcceptedTokens: [batch] (not supported currently)
    numComputedTokens: [batch]
    blockIdxFirstScheduledToken: without APC: None, with APC: [batch]
    blockIdxLastScheduledToken: without APC: None, with APC: [batch]
    initialStateIdx: without APC: None, with APC: [batch]
    activationMode: (not supported currently)
    padSlotId: default value -1
    runMode: (not supported currently)
    residualConnection: no residual: 0, residual: 1
    blockSize: typical value 128/256
    convMode: Qwen3-Next mode: 0, Pangu V2: 1
    ```

    cuSeqLen is the total length of all variable-length sequences in the batch after concatenation.

  - Scenario 2 (prefill and decode mixed scenario):

    ```cpp
    x: [cuSeqLen, dim]
    weight: [K, dim], where K=3
    convStates: [-1, K-1+m, dim]
    queryStartLoc: [batch+1]
    cacheIndices: without APC: [batch] or None, with APC: [block, maxNumBlocks]
    initialStateMode: [batch]
    bias: [dim] (not supported currently)
    numAcceptedTokens: [batch]
    numComputedTokens: [batch]
    blockIdxFirstScheduledToken: without APC: None, with APC: [batch]
    blockIdxLastScheduledToken: without APC: None, with APC: [batch]
    initialStateIdx: without APC: None, with APC: [batch]
    activationMode: (not supported currently)
    padSlotId: default value -1
    runMode: (not supported currently)
    residualConnection: no residual: 0, residual: 1
    blockSize: typical value 128/256
    convMode: Qwen3-Next mode: 0, Pangu V2: 1
    maxDraftTokens: number of speculative tokens (multiTokenNum), value range [0, 16]
    ```

    cuSeqLen is the total length of all variable-length sequences in the batch after concatenation.

  - Scenario 3 (decode scenario - variable-length sequences):

    ```cpp
    x: [cuSeqLen, dim]
    weight: [K, dim], where K=3
    convStates: [-1, K-1+m, dim]
    queryStartLoc: [batch+1]
    cacheIndices: without APC: [batch] or None, with APC: [block, maxNumBlocks]
    initialStateMode: [batch]
    bias: [dim] (not supported currently)
    numAcceptedTokens: [batch]
    numComputedTokens: [batch]
    blockIdxFirstScheduledToken: without APC: None, with APC: [batch]
    blockIdxLastScheduledToken: without APC: None, with APC: [batch]
    initialStateIdx: without APC: None, with APC: [batch]
    activationMode: (not supported currently)
    padSlotId: default value -1
    runMode: (not supported currently)
    residualConnection: no residual: 0, residual: 1
    blockSize: typical value 128/256
    convMode: Qwen3-Next mode: 0, Pangu V2: 1
    maxDraftTokens: number of speculative tokens (multiTokenNum), value range [0, 16]
    ```

    stateLen must be greater than the maximum number of tokens in all batches plus 1.

  - Scenario 4 (decode scenario - fixed batch):

    ```cpp
    x: [batch, m+1, dim]
    weight: [K, dim], where K=3
    convStates: [-1, K-1+m, dim]
    queryStartLoc: [batch+1]
    cacheIndices: without APC: [batch] or None, with APC: [block, maxNumBlocks]
    initialStateMode: [batch]
    bias: [dim] (not supported currently)
    numAcceptedTokens: [batch]
    numComputedTokens: [batch]
    blockIdxFirstScheduledToken: without APC: None, with APC: [batch]
    blockIdxLastScheduledToken: without APC: None, with APC: [batch]
    initialStateIdx: without APC: None, with APC: [batch]
    activationMode: (not supported currently)
    padSlotId: default value -1
    runMode: (not supported currently)
    residualConnection: no residual: 0, residual: 1
    blockSize: typical value 128/256
    convMode: Qwen3-Next mode: 0, Pangu V2: 1
    maxDraftTokens: number of speculative tokens (multiTokenNum), value range [0, 16]
    ```

- Computation formula:

  Computation formula: K is the convolution kernel width, L is the sequence length of the current batch, dim is the feature dimension, batchId is the index of the variable-length sequence being processed, C is the length of the convStates cache space (convStates.size(1)), and B is the APC block size (bSize). Perform the following operations on each batch in input x:
  1. Cache read

      Cache line index:

      $$
      readCacheLine = \begin{cases}
      cacheIndices[batchId, \; initialStateIdx[batchId]], & \text{APC mode} \\
      cacheIndices[batchId], & \text{non-APC and cacheIndices exists} \\
      batchId, & \text{otherwise}
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
      cacheIndices[batchId], & \text{non-APC and cacheIndices exists} \\
      batchId, & \text{otherwise}
      \end{cases}
      $$

      $$
      convStates[writeCacheLine][C - M + i, dim] = paddedInput[Len - M + i, dim], \quad i = 0, 1, \dots, M-1
      $$

  4. Offset trimming

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

  6. 1D causal convolution

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

## Parameter Description

<table style="undefined;table-layout: fixed; width: 1200px">
  <colgroup>
    <col style="width: 160px">
    <col style="width: 100px">
    <col style="width: 260px">
    <col style="width: 260px">
    <col style="width: 140px">
    <col style="width: 80px">
    <col style="width: 280px">
    <col style="width: 100px">
  </colgroup>
  <thead>
    <tr>
      <th>Parameter</th>
      <th>Input/Output</th>
      <th>Description</th>
      <th>Data Type</th>
      <th>Data Format</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>x</td>
      <td>Input/Output</td>
      <td>Input sequence x in the formula. The convolution result is updated to x in place.</td>
      <td>BFLOAT16, FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>weight</td>
      <td>Input</td>
      <td>1D causal convolution kernel w in the formula.</td>
      <td>Same as x</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>convStates</td>
      <td>Input/Output</td>
      <td>
        <ul>
          <li>convStates in the formula, the cache state tensor, which stores the historical token data of each sequence.</li>
          <li>Updated in place after the computation of each sequence is complete.</li>
        </ul>
      </td>
      <td>Same as x</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>queryStartLoc</td>
      <td>Optional input</td>
      <td>
        <ul>
          <li>In the scenario where x is two-dimensional, the sequence start position index, which records the start position of each sequence in the concatenated tensor x.</li>
          <li>queryStartLoc[i] indicates the start offset of the i-th sequence. queryStartLoc[0] must be 0, queryStartLoc[-1] must be cuSeqLen, and adjacent values must not be equal.</li>
        </ul>
      </td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>cacheIndices</td>
      <td>Optional input</td>
      <td>Cache index, specifying the index of the cache state corresponding to each sequence in cacheState.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>initialStateMode</td>
      <td>Optional input</td>
      <td>Specifies the padding policy for each sequence.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>bias</td>
      <td>Optional input</td>
      <td>Bias of the convolution.</td>
      <td>Same as x</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>numAcceptedTokens</td>
      <td>Optional input</td>
      <td>numAcceptedTokens in the formula.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>numComputedTokens</td>
      <td>Optional input</td>
      <td>numComputedTokens in the formula, the total number of tokens already processed for the current batch, used to determine the initial state.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>blockIdxFirstScheduledToken</td>
      <td>Optional input</td>
      <td>Block index corresponding to the start position of the current batch.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>blockIdxLastScheduledToken</td>
      <td>Optional input</td>
      <td>Block index corresponding to seqLen-1 of the current batch.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>initialStateIdx</td>
      <td>Optional input</td>
      <td>Index of the initial index block.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>activationMode</td>
      <td>Optional input</td>
      <td>Type of the activation function.</td>
      <td>STR</td>
      <td>-</td>
    </tr>
    <tr>
      <td>padSlotId</td>
      <td>Optional input</td>
      <td>Used to specify variable-length sequences that do not participate in the computation (batches with cacheIndices[i]==padSlotId do not participate in the computation); the output region corresponding to non-participating batches is undefined, and its content should not be read or relied upon.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>runMode</td>
      <td>Optional input</td>
      <td>Indicates the prefill or decode scenario. This is a legacy field and is not supported currently.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>maxQueryLen</td>
      <td>Optional input</td>
      <td>Maximum seqLen among all batches. Only the decode scenario (fixed batch) supports -1.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>residualConnection</td>
      <td>Optional input</td>
      <td>Used for the residual connection.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>blockSize</td>
      <td>Optional input</td>
      <td>Size of a block.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>convMode</td>
      <td>Optional input</td>
      <td>convMode in the formula. Two implementations are supported: Qwen3-Next and Pangu V2.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>maxDraftTokens (int64_t)</td>
      <td>Attribute</td>
      <td>Maximum number of speculative tokens.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
  </tbody>
</table>

## Constraints

- Input shape restrictions:
  - prefill scenario:
    - x supports two dimensions: [cuSeqLen, dim].
    - weight must be two-dimensional: [K, dim], where K is fixed to 3.
    - convStates must be three-dimensional: [..., K-1, dim]. The size of dimension 0 is not fixed and must be greater than or equal to the number of batches participating in the computation (that is, the number of batches whose cacheIndices does not equal padSlotId).
    - queryStartLoc must exist.
    - cacheIndices is one-dimensional [batch, ] or two-dimensional [batch, maxNumBlocks], where one dimension indicates that APC is disabled and two dimensions indicate that APC is enabled.
    - The range of cuSeqLen is [batch, 1024 * 1024], the range of dim is [64, 16384] and dim must be a multiple of 16, and the product of the two must satisfy [64 * batch, 4G]. The range of batch is [1, 256].
    - maxNumBlocks >= ceiv(maxQueryLen, blockSize).
    - maxQueryLen > maxDraftTokens + 1.
  - prefill and decode mixed scenario:
    - x supports two dimensions: [cuSeqLen, dim].
    - weight must be two-dimensional: [K, dim], where K is fixed to 3.
    - convStates must be three-dimensional: [..., K-1+m, dim]. The size of dimension 0 is not fixed and must be greater than or equal to the number of batches participating in the computation (that is, the number of batches whose cacheIndices does not equal padSlotId).
    - queryStartLoc must exist.
    - cacheIndices is one-dimensional [batch, ] or two-dimensional [batch, maxNumBlocks], where one dimension indicates that APC is disabled and two dimensions indicate that APC is enabled.
    - The range of cuSeqLen is [batch, 1024 * 1024], the range of dim is [64, 16384] and dim must be a multiple of 16, and the product of the two must satisfy [64 * batch, 4G]. The range of batch is [1, 256].
    - maxNumBlocks >= ceiv(maxQueryLen, blockSize).
    - maxQueryLen > maxDraftTokens + 1.
  - decode scenario (variable-length sequences):
    - x supports two dimensions: [cuSeqLen, dim].
    - weight must be two-dimensional: [K, dim], where K is fixed to 3.
    - convStates must be three-dimensional: [..., K-1+m, dim]. The size of dimension 0 is not fixed and must be greater than or equal to the number of batches participating in the computation (that is, the number of batches whose cacheIndices does not equal padSlotId).
    - queryStartLoc must exist.
    - cacheIndices is one-dimensional [batch, ] or two-dimensional [batch, maxNumBlocks], where one dimension indicates that APC is disabled and two dimensions indicate that APC is enabled.
    - The range of cuSeqLen is [batch, batch * (maxDraftTokens + 1)], and the seqLen range of each batch is [1, maxDraftTokens + 1]. The range of dim is [64, 16384] and dim must be a multiple of 16. The range of batch is [1, 256].
    - maxNumBlocks >= ceiv(maxQueryLen, blockSize).
    - The range of maxQueryLen is [1, maxDraftTokens + 1].
  - decode scenario (fixed batch):
    - x supports three dimensions: [batch, m+1, dim].
    - weight must be two-dimensional: [K, dim], where K is fixed to 3.
    - convStates must be three-dimensional: [..., K-1+m, dim]. The size of dimension 0 is not fixed and must be greater than or equal to the number of batches participating in the computation (that is, the number of batches whose cacheIndices does not equal padSlotId).
    - cacheIndices is one-dimensional [batch, ] or two-dimensional [batch, maxNumBlocks], where one dimension indicates that APC is disabled and two dimensions indicate that APC is enabled.
    - The range of m is [0, maxDraftTokens], the range of dim is [64, 16384] and dim must be a multiple of 16, and the range of batch is [1, 256].
    - maxNumBlocks >= ceiv(maxQueryLen, blockSize).
    - The range of maxQueryLen is [1, maxDraftTokens + 1], and -1 is allowed.

- Input value range restrictions:
  - queryStartLoc is the cumulative offset. Its value range is [0, cuSeqLen] and its length is batch+1. queryStartLoc[i] indicates the start offset of the i-th sequence, and queryStartLoc[batch] indicates the end position of the last sequence.
  - blockSize is 0 or greater than or equal to 2. It cannot be 0 when APC is enabled.
  - When blockIdxFirstScheduledToken, blockIdxLastScheduledToken, initialStateIdx, numComputedTokens, and cacheIndices all exist, APC is enabled and the following conditions are satisfied (i is the batch index):
    - cacheIndices is two-dimensional
    - initialStateIdx[i] <= blockIdxFirstScheduledToken[i]+1
    - initialStateIdx[i] <= blockIdxLastScheduledToken[i]
    - blockIdxFirstScheduledToken[i] <= blockIdxLastScheduledToken[i]
    - blockIdxLastScheduledToken[i] < maxNumBlocks
  - numAcceptedTokens can be None or non-None. In the non-None case, its length is batch. The element value is 0 for prefill and greater than 0 and less than or equal to seqLen-1 of the current batch for decode.
  - Each element in numComputedTokens is greater than or equal to 0.
  - The value range of cacheIndices is [0, convStates.dim[0]-1], and the values must not be equal to each other (unless equal to padSlotId).
  - maxQueryLen = the maximum seqLen in the batch.
  - The value range of maxDraftTokens is [0, 16].
  - In Pangu V2 mode (convMode = 1), numComputedTokens cannot be None.
  - The operator inputs and intermediate computation results do not exceed the value range of the corresponding data type (float16/bfloat16).
  - Operator inputs with ±inf and NaN are not supported.

  ## Calling Instructions

  | Calling Method  | Sample Code                                                     | Description                                                         |
  | --------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
  | aclnn API | [test_aclnn_inplace_fused_causal_conv1d](./examples/test_aclnn_inplace_fused_causal_conv1d.cpp) / [test_aclnn_inplace_fused_causal_conv1d_v2](./examples/test_aclnn_inplace_fused_causal_conv1d_v2.cpp) | Calls the InplaceFusedCausalConv1d operator through [aclnnInplaceFusedCausalConv1d](docs/aclnnInplaceFusedCausalConv1d_en.md) or [aclnnInplaceFusedCausalConv1dV2](docs/aclnnInplaceFusedCausalConv1dV2_en.md) |
  | Graph mode | - | Calls the InplaceFusedCausalConv1d operator through graph construction with the [operator IR](./op_graph/inplace_fused_causal_conv1d_proto.h) |
  | PyTorch API | - | Calls the fused_causal_conv1d_ operator through the [fused_causal_conv1d_](docs/torchapi_fused_causal_conv1d__en.md) API. |
