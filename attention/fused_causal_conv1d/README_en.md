# FusedCausalConv1d

English | [简体中文](./README.md)

## Product Support

| Product | Supported |
| :----------------------------------------------------------- | :------: |
|  <term>Ascend 950PR&950DT products</term>   |     √    |
| <term>Atlas A3 products</term>     |    ×     |
| <term>Atlas A2 products</term> |    ×     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                             |    ×     |
| <term>Atlas training products</term>                              |    ×     |

## Function Description

- Operator function: Performs causal 1D convolution on sequences. Along the sequence dimension, cached data (with a length of the convolution kernel width minus 1) is used to pad the head of each sequence, ensuring that the output depends on the current and historical inputs. After the convolution completes, part of the current sequence data is updated to the cache. Based on the output of the causal 1D convolution, the original input is added to the output to implement a residual connection. This operator supports APC (Automatic Prefix Caching), MTP (speculative decoding), residual connection, and other features.

- This operator supports the following scenarios:
  - Scenario 1 (prefill scenario):

    ```cpp
    x: [cuSeqLen, dim]
    weight: [K, dim], where K=3
    convStates: [-1, K-1, dim]
    queryStartLoc: [batch+1]
    cacheIndices: APC disabled: [batch] or None, APC enabled: [block, maxNumBlocks]
    initialStateMode: [batch]
    bias: [dim] (not supported yet)
    numAcceptedTokens: [batch] (not supported yet)
    numComputedTokens: [batch]
    blockIdxFirstScheduledToken: APC disabled: None, APC enabled: [batch]
    blockIdxLastScheduledToken: APC disabled: None, APC enabled: [batch]
    initialStateIdx: APC disabled: None, APC enabled: [batch]
    activationMode: (not supported yet)
    padSlotId: default value -1
    runMode: (not supported yet)
    residualConnection: no residual: 0, with residual: 1
    blockSize: typical value 128/256
    convMode: Qwen3-Next mode: 0, Pangu V2: 1
    y: [cuSeqLen, dim]
    ```

    where cuSeqLen is the total length of all variable-length sequences in the batch concatenated together.

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
    residualConnection: no residual: 0, with residual: 1
    blockSize: typical value 128/256
    convMode: Qwen3-Next mode: 0, Pangu V2: 1
    maxDraftTokens: number of speculative tokens (multiTokenNum), value range [0, 16]
    y: [cuSeqLen, dim]
    ```

    where cuSeqLen is the total length of all variable-length sequences in the batch concatenated together.

  - Scenario 3 (decode scenario - variable-length sequences):

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
    residualConnection: no residual: 0, with residual: 1
    blockSize: typical value 128/256
    convMode: Qwen3-Next mode: 0, Pangu V2: 1
    maxDraftTokens: number of speculative tokens (multiTokenNum), value range [0, 16]
    y: [cuSeqLen, dim]
    ```

    where stateLen must be greater than the maximum token count plus 1 across all batches.

  - Scenario 4 (decode scenario - fixed batch):

    ```cpp
    x: [batch, m+1, dim]
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
    residualConnection: no residual: 0, with residual: 1
    blockSize: typical value 128/256
    convMode: Qwen3-Next mode: 0, Pangu V2: 1
    maxDraftTokens: number of speculative tokens (multiTokenNum), value range [0, 16]
    y: [batch, m+1, dim]
    ```


- Computation formula:

  Computation formula: K is the convolution kernel width, L is the sequence length of the current batch, dim is the feature dimension, batchId is the index of the variable-length sequence currently processed, C is the length of the convStates cache space (convStates.size(1)), and B is the APC block size (bSize). For each batch in the input x, perform the following operations:
  1. Cache read

      Cache line index:

      $$
      readCacheLine = \begin{cases}
      cacheIndices[batchId, \; initialStateIdx[batchId]], & \text{APC mode} \\
      cacheIndices[batchId], & \text{non-APC and cacheIndices exists} \\
      batchId, & \text{otherwise}
      \end{cases}
      $$

      Case 1: first computation (numComputedTokens[batchId] == 0)

      $$
      cachedState[i, dim] = 0, \quad 0 \leq i < K-1
      $$

      $$
      offset = 0
      $$

      Case 2: speculative decoding mode (numAcceptedTokens exists)

      $$
      offset = numAcceptedTokens[batchId] - 1
      $$

      $$
      cachedState[i, dim] = convStates[readCacheLine][i, dim], \quad 0 \leq i <   offset + K - 1
      $$

      Case 3: default mode

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

  4. Offset cropping

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

## Parameters

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
      <td>Input sequence x in the formula.</td>
      <td>BFLOAT16, FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>weight</td>
      <td>Input</td>
      <td>Causal 1D convolution kernel w in the formula.</td>
      <td>Same as x</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>convStates</td>
      <td>Input/Output</td>
      <td>
        <ul>
          <li>convStates in the formula; a cache state tensor that stores the historical token data of each sequence.</li>
          <li>Updated in place after the computation of each sequence completes.</li>
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
          <li>When x is 2D, the sequence start position index, which records the start position of each sequence in the concatenated tensor x.</li>
          <li>queryStartLoc[i] indicates the start offset of the i-th sequence. queryStartLoc[0] must be 0, queryStartLoc[-1] must be cuSeqLen, and adjacent values must not be equal.</li>
        </ul>
      </td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>cacheIndices</td>
      <td>Optional input</td>
      <td>Cache index, which specifies the index of the cache state of each sequence in cacheState.</td>
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
      <td>numComputedTokens in the formula; the total number of tokens already processed in the current batch, used to determine the initial state.</td>
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
      <td>Block index corresponding to position seqLen-1 of the current batch.</td>
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
      <td>Activation function type.</td>
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
      <td>Indicates the prefill or decode scenario. This is a legacy field and is not supported yet.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>maxQueryLen</td>
      <td>Optional input</td>
      <td>Maximum seqLen across all batches. Only the decode scenario (fixed batch) supports -1.</td>
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
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Result of x computed by conv1d.</td>
      <td>Same as x</td>
      <td>ND</td>
    </tr>
  </tbody>
</table>

## Constraints

- Input shape restrictions:
  - Prefill scenario:
    - x supports 2D [cuSeqLen, dim].
    - weight must be 2D [K, dim], where K is fixed to 3.
    - convStates must be 3D [..., K-1, dim]; the size of dimension 0 is not fixed and must be greater than or equal to the number of batches participating in the computation (that is, the number of batches whose cacheIndices does not equal padSlotId).
    - queryStartLoc must exist.
    - cacheIndices is 1D [batch, ] or 2D [batch, maxNumBlocks]; 1D indicates that APC is disabled, and 2D indicates that APC is enabled.
    - cuSeqLen is in the range [batch, 1024 * 1024]; dim is in the range [64, 16384] and is a multiple of 16; the product of the two must be in the range [64 * batch, 4G]; batch is in the range [1, 256].
    - maxNumBlocks >= ceiv(maxQueryLen, blockSize).
    - maxQueryLen > maxDraftTokens + 1.
  - Mixed prefill and decode scenario:
    - x supports 2D [cuSeqLen, dim].
    - weight must be 2D [K, dim], where K is fixed to 3.
    - convStates must be 3D [..., K-1+m, dim]; the size of dimension 0 is not fixed and must be greater than or equal to the number of batches participating in the computation (that is, the number of batches whose cacheIndices does not equal padSlotId).
    - queryStartLoc must exist.
    - cacheIndices is 1D [batch, ] or 2D [batch, maxNumBlocks]; 1D indicates that APC is disabled, and 2D indicates that APC is enabled.
    - cuSeqLen is in the range [batch, 1024 * 1024]; dim is in the range [64, 16384] and is a multiple of 16; the product of the two must be in the range [64 * batch, 4G]; batch is in the range [1, 256].
    - maxNumBlocks >= ceiv(maxQueryLen, blockSize).
    - maxQueryLen > maxDraftTokens + 1.
  - Decode scenario (variable-length sequences):
    - x supports 2D [cuSeqLen, dim].
    - weight must be 2D [K, dim], where K is fixed to 3.
    - convStates must be 3D [..., K-1+m, dim]; the size of dimension 0 is not fixed and must be greater than or equal to the number of batches participating in the computation (that is, the number of batches whose cacheIndices does not equal padSlotId).
    - queryStartLoc must exist.
    - cacheIndices is 1D [batch, ] or 2D [batch, maxNumBlocks]; 1D indicates that APC is disabled, and 2D indicates that APC is enabled.
    - cuSeqLen is in the range [batch, batch * (maxDraftTokens + 1)]; the seqLen of each batch is in the range [1, maxDraftTokens + 1]; dim is in the range [64, 16384] and is a multiple of 16; batch is in the range [1, 256].
    - maxNumBlocks >= ceiv(maxQueryLen, blockSize).
    - maxQueryLen is in the range [1, maxDraftTokens + 1].
  - Decode scenario (fixed batch):
    - x supports 3D [batch, m+1, dim].
    - weight must be 2D [K, dim], where K is fixed to 3.
    - convStates must be 3D [..., K-1+m, dim]; the size of dimension 0 is not fixed and must be greater than or equal to the number of batches participating in the computation (that is, the number of batches whose cacheIndices does not equal padSlotId).
    - cacheIndices is 1D [batch, ] or 2D [batch, maxNumBlocks]; 1D indicates that APC is disabled, and 2D indicates that APC is enabled.
    - m is in the range [0, maxDraftTokens]; dim is in the range [64, 16384] and is a multiple of 16; batch is in the range [1, 256].
    - maxNumBlocks >= ceiv(maxQueryLen, blockSize).
    - maxQueryLen is in the range [1, maxDraftTokens + 1], and can be -1.

- Input value range restrictions:
  - queryStartLoc is the cumulative offset. Its value range is [0, cuSeqLen] and its length is batch+1. queryStartLoc[i] indicates the start offset of the i-th sequence, and queryStartLoc[batch] indicates the end position of the last sequence.
  - blockSize is 0 or greater than or equal to 2, and is not 0 when APC is enabled.
  - When blockIdxFirstScheduledToken, blockIdxLastScheduledToken, initialStateIdx, numComputedTokens, and cacheIndices all exist, APC is enabled and the following conditions are met (i is the batch index):
    - cacheIndices is 2D
    - initialStateIdx[i] <= blockIdxFirstScheduledToken[i]+1
    - initialStateIdx[i] <= blockIdxLastScheduledToken[i]
    - blockIdxFirstScheduledToken[i] <= blockIdxLastScheduledToken[i]
    - blockIdxLastScheduledToken[i] < maxNumBlocks
  - numAcceptedTokens is either None or non-None. In the non-None case, its length is batch; the element value is 0 for prefill, and greater than 0 and less than or equal to seqLen-1 of the current batch for decode.
  - Each element in numComputedTokens has a value greater than or equal to 0.
  - cacheIndices is in the range [0, convStates.dim[0]-1], and the values must not be equal to each other (unless equal to padSlotId).
  - maxQueryLen is the maximum seqLen in the batch.
  - maxDraftTokens is in the range [0, 16].
  - In Pangu V2 mode (convMode = 1), numComputedTokens cannot be None.
  - The operator inputs and intermediate computation results do not exceed the value range of the corresponding data type (float16/bfloat16).
  - The operator inputs do not support ±inf or NaN.

  ## Invocation

  | Invocation Method | Sample Code | Description |
  | --------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
  | aclnn interface | [test_aclnn_fused_causal_conv1d](./examples/test_aclnn_fused_causal_conv1d.cpp) / [test_aclnn_fused_causal_conv1d_v2](./examples/test_aclnn_fused_causal_conv1d_v2.cpp)| Call the FusedCausalConv1d operator through [aclnnFusedCausalConv1d](docs/aclnnFusedCausalConv1d_en.md) or [aclnnFusedCausalConv1dV2](docs/aclnnFusedCausalConv1dV2_en.md) |
  | Graph mode | - | Call the FusedCausalConv1d operator by constructing a graph through the [operator IR](./op_graph/fused_causal_conv1d_proto.h) |
  | PyTorch API | - | Call the fused_causal_conv1d operator through the [fused_causal_conv1d](docs/torchapi_fused_causal_conv1d_en.md) interface. |
