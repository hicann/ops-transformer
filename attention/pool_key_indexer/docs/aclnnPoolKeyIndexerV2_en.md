# aclnnPoolKeyIndexerV2

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

- Description: `pool_key_indexer` packs multiple consecutive tokens into one pool, computes attention relevance scores and selects top-k positions on a per-pool basis, reducing indexing overhead while retaining the sparse attention advantage. This operator applies to sparse index selection in key pool-based sparse attention use cases. It supports FLOAT16 and BFLOAT16 floating-point inputs, and supports FLOAT8_E4M3FN quantized inputs with dequantization coefficients (FP8 per-token-head or mxFP8 quantization mode) as well as INT8 quantized inputs.

- Relationship with aclnnPoolKeyIndexer (V1): aclnnPoolKeyIndexerV2 is the full-prototype API. Its computation flow and formula are identical to those of aclnnPoolKeyIndexer. The only difference is that the sparseIndicesMode attribute is added to the API signature to control the write layout of the tail tokens in the output sparseIndicesOut. The meanings and constraints of all other parameters are the same as those of aclnnPoolKeyIndexer. aclnnPoolKeyIndexer is equivalent to aclnnPoolKeyIndexerV2 with sparseIndicesMode fixed to 0.

- Formula:

  The complete computation flow consists of three parts: input dequantization, score computation and selection, and index expansion and appending.

  **Input dequantization (quantization scenario only, quantMode is 0, 1, or 2):**

  When quantMode is -1 (no quantization), the original input is used directly:

  $$
  Q = query,\quad K_{pool} = poolKey
  $$

  When quantMode is 0, 1, or 2 (quantization), the quantized input is first dequantized with the dequantization coefficient:

  $$
  Q = \text{Dequant}(query, q\_descale)
  $$

  $$
  K_{pool} = \text{Dequant}(poolKey, k\_descale)
  $$

  Here, quantMode 0 indicates the FP8 per-token-head quantization mode, and the dequantization coefficient is of the FLOAT type; quantMode 1 indicates the mxFP8 quantization mode, and the dequantization coefficient is of the FLOAT8_E8M0 type; quantMode 2 indicates the int8 quantization mode, where query and poolKey are of the INT8 type and the dequantization coefficients are of the FLOAT16 type (q_descale is in per-token-head granularity, k_descale is in per-token granularity, and both are symmetric quantization scales).

  **Score computation and selection:**

  1. Scaled dot product:

      $$
      S = Q \otimes K_{pool}^T \odot \frac{1}{\sqrt{headDim}}
      $$

  2. ReLU activation to filter negative correlation signals:

      $$
      S' = \text{ReLU}(S)
      $$

  3. Multi-head weight aggregation:

      $$
      PoolScores = W \otimes S'
      $$

  4. Visibility filtering: Marks valid candidate pools based on causality and padding states, and sets invalid pool scores to $-\infty$.

  5. Top-k selection:

      $$
      TopK\_pool = \text{Top-}k(PoolScores)
      $$

      where $k = \lfloor topk / pool\_size \rfloor$, that is, the first $k$ pools are selected.

  **Index expansion and appending:**

  $$
  SparseIndices = \text{Append}(\text{Expand}(TopK\_pool, PoolIndices))
  $$

  where $\text{Expand}$ expands the selected pool indices into original token indices through $PoolIndices$, and $\text{Append}$ appends the valid tail tokens of incomplete pools to prevent omission of tail tokens in short sequences (expansion first, then appending). The number of valid tail tokens is specified by poolTailK. Incomplete tail pools are not contained in $poolKey$. $poolKey$ contains only complete pools, and the indices of the tail tokens are the consecutive $poolTailK$ original token indices starting from the number of complete pools $\times pool\_size$.

  When appending tail tokens, the layout of each output row is determined by sparseIndicesMode (expandTotal is the number of valid expanded tokens, that is, min(number of valid pools, $topk/pool\_size$) $\times pool\_size$; visibleTailK is the number of causally visible valid tail tokens):

  - If sparseIndicesMode is 0 (legacy layout, default), tail tokens are written starting from the fixed offset $topk$, that is, occupying $[topk,\ topk+visibleTailK)$. For short sequences (the number of valid pools is less than $topk/pool\_size$), a -1 gap $[expandTotal,\ topk)$ exists between the expansion region and the tail region.
  - If sparseIndicesMode is 1 (compact layout), tail tokens are written immediately after the valid expanded tokens, that is, occupying $[expandTotal,\ expandTotal+visibleTailK)$, and all -1 values are located at the end of the row, eliminating the gap.
  - The set and number of valid tokens in each row are completely identical in both layouts; only the position layout differs. For long sequences (the number of valid pools is not less than $topk/pool\_size$), expandTotal equals $topk$, and the two layouts are identical.

  **Output:**

  $$
  SparseValues = PoolScores[TopK\_pool]
  $$

  $SparseValues$ is output when returnValue is true, that is, the score values of the selected pools. When returnValue is false, $SparseValues$ is not output.

  The symbols are described as follows:

  - $SparseIndices$: Output sparse token indices.
  - $SparseValues$: Output scores of the selected pools (output when returnValue is true).
  - $PoolScores$: Overall score of each pool.
  - $TopK\_pool$: Indices of the selected top-k pools.
  - $Top$-$k$: Pool indices corresponding to the first $k$ maximum scores, where $k = \lfloor topk / pool\_size \rfloor$.
  - $W$: multi-head weight projection matrix.
  - $\otimes$: matrix multiplication (matmul).
  - $\text{ReLU}$: activation function $ReLU(x) = \max(0, x)$.
  - $Q$: Query vector (the dequantized Query in the quantization scenario).
  - $K_{pool}$: weighted aggregated key of the Pool. $K_{pool}^T$ is the transpose of its last two dimensions (the dequantized Key in the quantization scenario).
  - $\odot$: element-wise multiplication (Hadamard product).
  - $headDim$: head dimension. The scaling factor $\frac{1}{\sqrt{headDim}}$ prevents the dot product from being too large.
  - $PoolIndices$: raw token indices contained in each pool.
  - $query$ and $poolKey$: raw input tensors (FLOAT8_E4M3FN in the quantization scenario).
  - $q\_descale$ and $k\_descale$: dequantization coefficients of the Query and the Key.
  - $poolTailK$: Number of valid tokens in incomplete pools at the end of each batch.

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call the "aclnnPoolKeyIndexerV2GetWorkspaceSize" API first to obtain the workspace size required for computation and the executor that contains the operator computation process. Then call the "aclnnPoolKeyIndexerV2" API to perform the computation.

```Cpp
aclnnStatus aclnnPoolKeyIndexerV2GetWorkspaceSize(
    const aclTensor   *query,
    const aclTensor   *poolKey,
    const aclTensor   *weights,
    const aclIntArray *poolTailK,
    const aclIntArray *actualSeqQOptional,
    const aclIntArray *actualSeqKOptional,
    const aclTensor   *blockTableOptional,
    const aclTensor   *qDescaleOptional,
    const aclTensor   *kDescaleOptional,
    int64_t            topk,
    int64_t            poolSize,
    char              *layoutQOptional,
    char              *layoutKOptional,
    int64_t            maskMode,
    int64_t            quantMode,
    bool               returnValue,
    int64_t            keyStride0,
    int64_t            kDescaleStride0,
    int64_t            sparseIndicesMode,
    const aclTensor   *sparseIndicesOut,
    const aclTensor   *sparseValuesOut,
    uint64_t          *workspaceSize,
    aclOpExecutor    **executor)
```

```Cpp
aclnnStatus aclnnPoolKeyIndexerV2(
    void             *workspace,
    uint64_t          workspaceSize,
    aclOpExecutor    *executor,
    const aclrtStream stream)
```

## aclnnPoolKeyIndexerV2GetWorkspaceSize

- **Parameters:**

  > [!NOTE]
  >
  > - Dimension meanings of the query, poolKey, and weights parameters: B (Batch Size) indicates the batch size of input samples, S (Sequence Length) indicates the sequence length of input samples, N (Head Num) indicates the number of heads, D (Head Dim) indicates the dimension of each head, and T indicates the cumulative sum of the sequence lengths of input samples in all batches.
  > - S1 indicates the S in the query shape, S2 indicates the S in the poolKey shape, T1 indicates the T in the query shape, T2 indicates the T in the poolKey shape, N1 indicates the N in the query shape, and N2 indicates the N in the poolKey shape.
  > - block_num is the total number of blocks in PageAttention, and block_size is the number of tokens in a block.
  > - poolTailK, actualSeqLengthsQueryOptional, and actualSeqLengthsKeyOptional are host-side value inputs (ValueDepend), which are passed in as aclIntArray (element type INT64) at the API layer instead of aclTensor.

  <table style="undefined;table-layout: fixed; width: 1601px"><colgroup>
  <col style="width: 264px">
  <col style="width: 132px">
  <col style="width: 232px">
  <col style="width: 330px">
  <col style="width: 164px">
  <col style="width: 119px">
  <col style="width: 215px">
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
    </tr></thead>
  <tbody>
    <tr>
      <td>query (aclTensor*)</td>
      <td>Input</td>
      <td>Input Q in the formula.</td>
      <td>
          <ul>
                <li>Empty tensors are not supported.</li>
                <li>If layoutQueryOptional is BSND, the shape is (B,S1,N1,D).</li>
                <li>If layoutQueryOptional is TND, the shape is (T1,N1,D).</li>
          </ul>
      </td>
      <td>FLOAT16, BFLOAT16, FLOAT8_E4M3FN, INT8</td>
      <td>ND</td>
      <td>3-4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>poolKey (aclTensor*)</td>
      <td>Input</td>
      <td>Input K<sub>pool</sub> in the formula.</td>
      <td>
          <ul>
                <li>Empty "zero-pool" tensors are supported (pure tail-block mode: BSND S2=0, TND T2=0, PA block_num=0). In this case, all valid keys of each batch are expressed by poolTailK, and the tail tokens are written into sparseIndices according to causal visibility.</li>
                <li>block_num is the total number of blocks in PageAttention, and block_size is the number of pools per block (each row in a block is the key of one pool, the row stride is headDim, and tokens are not expanded).</li>
                <li>The data type must be the same as that of query.</li>
                <li>The data layout must be the same as that of query, except for the PageAttention use case (layoutKeyOptional is PA_BBND).</li>
                <li>If layoutKeyOptional is PA_BBND, the shape is (block_num, block_size, N2, D).</li>
                <li>If layoutKeyOptional is BSND, the shape is (B, S2, N2, D).</li>
                <li>If layoutKeyOptional is TND, the shape is (T2, N2, D).</li>
          </ul>
      </td>
      <td>FLOAT16, BFLOAT16, FLOAT8_E4M3FN, INT8</td>
      <td>ND</td>
      <td>3-4</td>
      <td>Supports non-contiguous axis 0 (PA_BBND layout only, used with the keyStride0 attribute; BSND/TND layouts require contiguous input)</td>
    </tr>
    <tr>
      <td>weights (aclTensor*)</td>
      <td>Input</td>
      <td>Input W in the formula.</td>
      <td>
          <ul>
                <li>Empty tensors are not supported.</li>
                <li>If layoutQueryOptional is BSND, the shape is (B,S1,N1).</li>
                <li>If layoutQueryOptional is TND, the shape is (T1,N1).</li>
                <li>When quantMode is 2 (int8 quantization), only the FLOAT16 data type is supported.</li>
          </ul>
      </td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>2-3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>poolTailK (aclIntArray*)</td>
      <td>Input</td>
      <td>Number of valid tokens in the incomplete tail pool of each Batch.</td>
      <td>
          <ul>
                <li>Empty tensors are not supported.</li>
                <li>Value range: [0, pool_size-1]. 0 indicates no valid tail tokens.</li>
                <li>The incomplete tail pool is not contained in poolKey. poolKey contains only complete pools. The total number of valid tokens of each Batch = number of complete pools x pool_size + poolTailK. The caller must ensure that the number of valid tokens in the downstream original KV is consistent with this value.</li>
                <li>The number of elements must equal B.</li>
                <li>The shape is (B,).</li>
          </ul>
      </td>
      <td>INT64</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>actualSeqLengthsQueryOptional (aclIntArray*)</td>
      <td>Input</td>
      <td>Number of valid tokens of the query in each batch.</td>
      <td>
          <ul>
                <li>Optional input. If seqlen is not specified, pass a null pointer to indicate that the value is the same as S in the query shape. When layoutQueryOptional is BSND (including the PageAttention use case where layoutKeyOptional is PA_BBND and query is organized in BSND), this input must be a null pointer.</li>
                <li>Empty tensors are not supported.</li>
                <li>In this input, the number of valid tokens in each batch must not exceed the size of dimension S of query and must not be less than 0. A one-dimensional tensor of length B is supported.</li>
                <li>When layoutQueryOptional is TND, this input is mandatory. The number of elements in this input is used as the value of B. The value of each element indicates the total number of tokens in the current batch and all preceding batches, that is, a prefix sum. Therefore, the value of each element must be greater than or equal to that of the preceding element.</li>
                <li>The shape is (B,).</li>
          </ul>
      </td>
      <td>INT64</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>actualSeqLengthsKeyOptional (aclIntArray*)</td>
      <td>Input</td>
      <td>Number of valid Key tokens in each Batch.</td>
      <td>
          <ul>
                <li>Optional input. If seqlen is not specified, a null pointer can be passed in, indicating that the value is the same as the length of S in the shape of poolKey. When layoutKeyOptional is BSND, this input must be a null pointer.</li>
                <li>Empty tensors are not supported.</li>
                <li>The number of valid pools of each Batch in this parameter must not exceed the dimension S of poolKey (BSND/TND) or the total number of pools of blockSize (PA_BBND), and must not be less than 0. A one-dimensional tensor of length B is supported.</li>
                <li>This parameter is mandatory when layoutKeyOptional is TND or PA_BBND. If layoutKeyOptional is TND, the value of each element in this parameter indicates the total number of tokens of the current batch and all preceding batches, that is, the prefix sum. Therefore, the value of an element must be greater than or equal to that of its preceding element. If layoutKeyOptional is PA_BBND, the value of each element in this parameter indicates the number of pools of the current batch (that is, the number of pools obtained by dividing the number of valid tokens of the batch by pool_size and rounding down, not in prefix-sum form).</li>
                <li>The shape is (B,).</li>
          </ul>
      </td>
      <td>INT64</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>blockTableOptional (aclTensor*)</td>
      <td>Input</td>
      <td>Represents the block mapping table used for KV storage in PageAttention.</td>
      <td>
          <ul>
                <li>Optional input. A null pointer can be passed in non-PageAttention use cases.</li>
                <li>Empty tensors are not supported.</li>
                <li>In PageAttention use cases, block_table must be 2D. The first dimension must equal B, and the second dimension must not be less than maxBlockNumPerSeq (maxBlockNumPerSeq is the number of blocks corresponding to the maximum actual_seq_lengths_key in each batch).</li>
                <li>The shape is (B, maxBlockNumPerSeq).</li>
          </ul>
      </td>
      <td>INT32</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>qDescaleOptional (aclTensor*)</td>
      <td>Input</td>
      <td>Dequantization coefficient of query.</td>
      <td>
          <ul>
                <li>Optional input. In non-quantization scenarios (quantMode is -1), a null pointer can be passed.</li>
                <li>Empty tensors are not supported.</li>
                <li>Required in quantization scenarios. The data type is determined by quantMode: FLOAT when quantMode is 0 (FP8 per-token-head quantization), FLOAT8_E8M0 when quantMode is 1 (mxFP8 quantization), and FLOAT16 when quantMode is 2.</li>
                <li>When quantMode is 0: the shape is (B,S1,N1) when layoutQueryOptional is BSND, and (T1,N1) when layoutQueryOptional is TND.</li>
                <li>When quantMode is 1 (mxFP8, every 32 elements along the D dimension share one E8M0 scale): the shape is (B,S1,N1,D/64,2) when layoutQueryOptional is BSND, and (T1,N1,D/64,2) when layoutQueryOptional is TND.</li>
                <li>When quantMode is 2 (int8): the shape is (B,S1,N1) when layoutQueryOptional is BSND, and (T1,N1) when layoutQueryOptional is TND. Each dimension is consistent with the corresponding dimension of query.</li>
                <li>When quantMode is 2, element values must be positive (symmetric quantization scales).</li>
          </ul>
      </td>
      <td>FLOAT, FLOAT8_E8M0, FLOAT16</td>
      <td>ND</td>
      <td>2-5</td>
      <td>√</td>
    </tr>
    <tr>
      <td>kDescaleOptional (aclTensor*)</td>
      <td>Input</td>
      <td>Dequantization coefficient of Key.</td>
      <td>
          <ul>
                <li>Optional input. In non-quantization scenarios (quantMode is -1), a null pointer can be passed in.</li>
                <li>Empty tensors are not supported.</li>
                <li>Mandatory in the quantization scenario. The data type is determined by quantMode: FLOAT when quantMode is 0 (FP8 per-token-head quantization), FLOAT8_E8M0 when quantMode is 1 (mxFP8 quantization), and FLOAT16 when quantMode is 2.</li>
                <li>When quantMode is 0: if layoutKeyOptional is BSND, the shape is (B,S2,N2); if layoutKeyOptional is TND, the shape is (T2,N2); if layoutKeyOptional is PA_BBND, the shape is (block_num, block_size, N2).</li>
                <li>When quantMode is 1 (mxFP8): if layoutKeyOptional is BSND, the shape is (B,S2,N2,D/64,2); if layoutKeyOptional is TND, the shape is (T2,N2,D/64,2); if layoutKeyOptional is PA_BBND, the shape is (block_num, block_size, N2, D/64,2).</li>
                <li>When quantMode is 2 (int8): the shape is poolKey with the D axis removed. That is, if layoutKeyOptional is BSND, the shape is (B,S2,N2); if layoutKeyOptional is TND, the shape is (T2,N2); if layoutKeyOptional is PA_BBND, the shape is (block_num, block_size, N2). Each dimension is consistent with the corresponding dimension of poolKey.</li>
                <li>When quantMode is 2, element values must be positive (symmetric quantization scales).</li>
          </ul>
      </td>
      <td>FLOAT, FLOAT8_E8M0, FLOAT16</td>
      <td>ND</td>
      <td>2-5</td>
      <td>Supports non-contiguous axis 0 (only for the PA_BBND layout in the quantization scenario; the kDescaleStride0 attribute is required; the BSND/TND layout requires contiguous inputs)</td>
    </tr>
    <tr>
      <td>topk (int64_t)</td>
      <td>Input</td>
      <td>Number of tokens to be retained after expansion, corresponding to topk in the formula. Optional attribute. The recommended value is 2048.</td>
      <td>
          <ul>
                <li>Supports [1, 2048] as well as 3072, 4096, 5120, 6144, 7168, and 8192.</li>
                <li>topk % pool_size == 0 must be satisfied.</li>
          </ul>
      </td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>poolSize (int64_t)</td>
      <td>Input</td>
      <td>Number of tokens in each pool, corresponding to pool_size in the formula. Optional attribute. The recommended value is 16.</td>
      <td>Supported range: [1, 128].</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layoutQueryOptional (char*)</td>
      <td>Input</td>
      <td>Identifies the data layout format of the input Query.</td>
      <td>
          <ul>
                <li>If not specified by the user, pass in the recommended value "BSND".</li>
                <li>Currently, BSND and TND are supported.</li>
          </ul>
      </td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layoutKeyOptional (char*)</td>
      <td>Input</td>
      <td>Identifies the data layout format of the input Key.</td>
      <td>
          <ul>
                <li>If not specified by the user, pass in the recommended value "BSND".</li>
                <li>Currently, PA_BBND, BSND, and TND are supported.</li>
          </ul>
      </td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>maskMode (int64_t)</td>
      <td>Input</td>
      <td>Indicates the mask mode.</td>
      <td>
          <ul>
                <li>When maskMode is 0, the mask mode is defaultMask.</li>
                <li>When maskMode is 3, the mask mode is rightDownCausal, corresponding to the lower-triangular use case partitioned by the right vertex.</li>
          </ul>
      </td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>quantMode (int64_t)</td>
      <td>Input</td>
      <td>Quantization mode of Query/Key.</td>
      <td>
          <ul>
                <li>If quantMode is -1, no quantization is performed.</li>
                <li>If quantMode is 0, the FP8 per-token-head quantization mode is used, and the dequantization coefficient is FLOAT.</li>
                <li>If quantMode is 1, the mxFP8 quantization mode is used, and the dequantization coefficient is FLOAT8_E8M0.</li>
                <li>If quantMode is 2, the int8 quantization mode is used. query and poolKey are of the INT8 type, and the dequantization coefficients qDescaleOptional and kDescaleOptional are of the FLOAT16 type (q is in per-token-head granularity, and k is in per-token granularity).</li>
          </ul>
      </td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>returnValue (bool)</td>
      <td>Input</td>
      <td>Specifies whether to output sparseValuesOut. Optional attribute. The recommended value is false.</td>
      <td>
          <ul>
                <li>If returnValue is false, sparseValuesOut is not output.</li>
                <li>If returnValue is true, sparseValuesOut is output.</li>
          </ul>
      </td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>keyStride0 (int64_t)</td>
      <td>Input</td>
      <td>Stride of axis 0 of poolKey (element unit), used for addressing poolKey with non-contiguous axis 0 when layoutKeyOptional is PA_BBND. Optional attribute. The recommended value is -1.</td>
      <td>
          <ul>
                <li>-1 indicates unspecified, and the stride is derived from the shape as a contiguous input.</li>
                <li>If specified, the value must be no less than block_size * N2 * D (contiguous stride).</li>
                <li>An error is reported if the specified value conflicts with the actual runtime stride of poolKey.</li>
          </ul>
      </td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>kDescaleStride0 (int64_t)</td>
      <td>Input</td>
      <td>Stride of axis 0 of kDescaleOptional (element unit), used for addressing kDescaleOptional with non-contiguous axis 0 when layoutKeyOptional is PA_BBND. Optional attribute. The recommended value is -1.</td>
      <td>
          <ul>
                <li>Valid only in quantization scenarios (quantMode is 0, 1, or 2).</li>
                <li>-1 indicates unspecified, and the stride is derived from the shape as a contiguous input.</li>
                <li>If specified, the value must be no less than the contiguous stride: block_size * N2 when quantMode is 0, block_size * N2 * (D / 32) when quantMode is 1, and block_size * N2 when quantMode is 2 (FLOAT16 scale, in element units).</li>
                <li>An error is reported if the specified value conflicts with the actual runtime stride of kDescaleOptional.</li>
          </ul>
      </td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>sparseIndicesMode (int64_t)</td>
      <td>Input</td>
      <td>Index output layout mode (tail-token appending mode), which controls the write position of the tail tokens in sparseIndicesOut. Optional attribute. The recommended value is 0.</td>
      <td>
          <ul>
                <li>Only 0 and 1 are supported. An error is reported for other values.</li>
                <li>If sparseIndicesMode is 0, the legacy layout (default) is used: tail tokens are written starting from the fixed offset topk. For short sequences (the number of valid pools is less than topk / pool_size), a -1 gap exists between the expansion region and the tail region. The output is bit-for-bit identical to that of aclnnPoolKeyIndexer (V1).</li>
                <li>If sparseIndicesMode is 1, the compact layout is used: tail tokens are written immediately after the valid expanded tokens, and all -1 values are located at the end of the row, eliminating the gap.</li>
                <li>The set and number of valid tokens in each row are completely identical in both layouts; only the position layout differs.</li>
          </ul>
      </td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>sparseIndicesOut (aclTensor*)</td>
      <td>Output</td>
      <td>The SparseIndices output in the formula, which is the expanded sparse token indices.</td>
      <td>
          <ul>
                <li>Empty tensors are not supported.</li>
                <li>Elements are original KV token indices (per-batch local scope), with the valid domain [0, number of complete pools x pool_size + poolTailK), where the number of complete pools is the S2 (BSND) / T2 (TND) dimension size of poolKey or the corresponding element of actualSeqLengthsKeyOptional (PA_BBND). The indices do not point into poolKey. The tail poolTailK tokens are retrieved from the original KV by the downstream based on the indices.</li>
                <li>The in-row layout of the output is determined by sparseIndicesMode: when it is 0 (legacy), tail tokens are written starting from the fixed offset topk, and -1 is filled between the expansion region and the tail region for short sequences; when it is 1 (compact), tail tokens are written immediately after the valid expanded tokens, and all -1 values are located at the end of the row. The set and number of valid tokens in each row are completely identical in both layouts.</li>
                <li>Positions with no tail tokens (poolTailK is 0) or with causally invisible tail tokens are filled with -1.</li>
                <li>If layoutQueryOptional is BSND, the output shape is (B, S1, topk + pool_size - 1).</li>
                <li>If layoutQueryOptional is TND, the output shape is (T1, topk + pool_size - 1).</li>
          </ul>
      </td>
      <td>INT32</td>
      <td>ND</td>
      <td>2-3</td>
      <td>x</td>
    </tr>
    <tr>
      <td>sparseValuesOut (aclTensor*)</td>
      <td>Output</td>
      <td>SparseValues output in the formula, that is, the score values corresponding to the selected pools.</td>
      <td>
          <ul>
                <li>When returnValue is true, valid results are output. Empty tensors are not supported.</li>
                <li>When returnValue is false, an empty tensor with shape (0,) is output.</li>
                <li>When layoutQueryOptional is BSND, the output shape is (B, S1, topk // pool_size).</li>
                <li>When layoutQueryOptional is TND, the output shape is (T1, topk // pool_size).</li>
          </ul>
      </td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>2-3</td>
      <td>x</td>
    </tr>
    <tr>
      <td>workspaceSize (uint64_t*)</td>
      <td>Output</td>
      <td>Returns the size of the workspace that needs to be applied for on the Device side.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor (aclOpExecutor**)</td>
      <td>Output</td>
      <td>Returns the op executor, including the operator calculation process.</td>
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
                <td>query, poolKey, weights, poolTailK, sparseIndicesOut, or sparseValuesOut is a null pointer (except when an optional input is a null pointer).</td>
            </tr>
            <tr>
                <td rowspan="5">ACLNN_ERR_PARAM_INVALID</td>
                <td rowspan="5">161002</td>
                <td>The data type or data format of query, poolKey, weights, poolTailK, sparseIndicesOut, or sparseValuesOut is not in the supported range.</td>
            </tr>
            <tr>
                <td>The shape dimension of query, poolKey, or weights is not in the supported range.</td>
            </tr>
            <tr>
                <td>topk and poolSize do not meet the constraint topk % pool_size == 0.</td>
            </tr>
            <tr>
                <td>The value of topk or poolSize is not in the supported range.</td>
            </tr>
            <tr>
                <td>The value of sparseIndicesMode is not in the supported range (only 0 and 1 are supported).</td>
            </tr>
        </tbody>
    </table>

## aclnnPoolKeyIndexerV2

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1151px"><colgroup>
  <col style="width: 184px">
  <col style="width: 134px">
  <col style="width: 833px">
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
      <td>Memory address of the workspace allocated on the Device side.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace allocated on the Device side, obtained by the first-phase API aclnnPoolKeyIndexerV2GetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>op executor, which contains the operator computation flow.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Specifies the Stream for task execution.</td>
    </tr>
  </tbody>
  </table>

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic implementation:
  - aclnnPoolKeyIndexerV2 defaults to a deterministic implementation.
- Input shape constraints:
  - N1 of query supports values less than or equal to 64, N2 of poolKey must equal 1, headDim supports 128, and D of poolKey must be the same as the headDim of query.
  - Invalid data padded on S1 or S2 of query or poolKey is not supported.
  - The last dimension of output sparseIndicesOut is topk + pool_size - 1, and the last dimension of output sparseValuesOut is topk // pool_size.
  - Dimension consistency constraint: Each dimension of weights and qDescaleOptional (B, S1, and N1 for BSND; T1 and N1 for TND) must be consistent with the corresponding dimension of query; each dimension of kDescaleOptional must be consistent with the corresponding dimension of poolKey (blockNum, blockSize, and N2 must be consistent with poolKey for PA_BBND); the number of elements in poolTailK must equal B; the number of elements in actualSeqLengthsKeyOptional must equal B.
  - Null pointer constraint of actualSeqLengths: When both layoutQueryOptional and layoutKeyOptional are BSND (non-PageAttention use case), actualSeqLengthsQueryOptional and actualSeqLengthsKeyOptional must be null pointers. In the PageAttention use case where layoutKeyOptional is PA_BBND and query is BSND, actualSeqLengthsQueryOptional must be a null pointer.
  - Prefix sum constraint: When layoutQueryOptional is TND, the value of the last element in the prefix sum of actualSeqLengthsQueryOptional must equal the T1 of query; when layoutKeyOptional is TND, the value of the last element in the prefix sum of actualSeqLengthsKeyOptional must equal the T2 of poolKey.
  - Valid index domain constraint: Valid indices in sparseIndicesOut are original KV token indices (per-batch local scope), with the value range [0, number of complete pools x pool_size + poolTailK). poolKey contains only complete pools. The poolTailK tokens of the incomplete tail pool are not in poolKey and are retrieved from the original KV by index by the downstream. The caller must ensure that the number of valid tokens of each batch in the original KV is consistent with this upper bound. Invalid positions are filled with -1.
  - Index output layout constraint: When sparseIndicesMode is 0 (legacy), tail tokens are written starting from the fixed offset topk, and a -1 gap exists between the expansion region and the tail region for short sequences. When sparseIndicesMode is 1 (compact), tail tokens are written immediately after the valid expanded tokens, and all -1 values are located at the end of the row. The set and number of valid tokens in each row are completely identical in both layouts.
  - Output dimension constraint: The B and S1 (or T1) dimensions of sparseIndicesOut and sparseValuesOut must be consistent with the corresponding dimensions of query.
- Input attribute restrictions:
  - topk must satisfy topk % pool_size == 0. pool_size supports [1, 128]. topk supports [1, 2048], as well as 3072, 4096, 5120, 6144, 7168, and 8192.
  - sparseIndicesMode supports only 0 (legacy layout) or 1 (compact layout). An error is reported for other values.
  - block_size must be a multiple of 16, and the maximum supported value is 2048.
  - When layoutKeyOptional is PA_BBND and poolKey is a non-contiguous input on axis 0, pass the actual stride(0) of poolKey to keyStride0 (in element units, and not less than block_size * N2 * D); keyStride0 of -1 indicates unspecified, and the stride is derived as for a contiguous input; if the specified value of keyStride0 conflicts with the actual runtime stride of poolKey, an error is reported.
  - When layoutKeyOptional is PA_BBND and kDescaleOptional is a non-contiguous input on axis 0 (quantization scenario), pass the actual stride(0) of kDescaleOptional to kDescaleStride0 (in element units, and not less than the contiguous stride: block_size * N2 when quantMode is 0, and block_size * N2 * (D / 32) when quantMode is 1); kDescaleStride0 of -1 indicates unspecified, and the stride is derived as for a contiguous input; if the specified value of kDescaleStride0 conflicts with the actual runtime stride of kDescaleOptional, an error is reported.
- Input data type restrictions:
  - The data types of query and poolKey must be the same.
  - In the non-quantization scenario (quantMode is -1), the data type of weights must be the same as those of query and poolKey; in the quantization scenario (quantMode is 0 or 1), weights is FLOAT16 or BFLOAT16; in the int8 quantization scenario (quantMode is 2), weights supports FLOAT16 only.
  <!-- npu="950" id8 -->
  - <term>Ascend 950PR&950DT products</term>: supports the FLOAT8_E4M3FN (query and poolKey) data type, and supports the quantization scenario (the dequantization coefficient is FLOAT when quantMode is 0, and FLOAT8_E8M0 when quantMode is 1); supports the INT8 quantization scenario (query and poolKey are INT8, and the dequantization coefficient is FLOAT16 when quantMode is 2).
  <!-- end id8 -->
  <!-- npu="A3,910b" id7 -->
  - <term>Atlas A3 products</term>, <term>Atlas A2 products</term>: do not support the FLOAT8_E4M3FN and FLOAT8_E8M0 data types, or the FP8 quantization scenario (quantMode is 0 or 1); supports the INT8 quantization scenario (query and poolKey are INT8, weights and the dequantization coefficients are FLOAT16 when quantMode is 2).
  <!-- end id7 -->
- Input data layout restrictions:
  - The data layouts of query and poolKey (layoutQueryOptional and layoutKeyOptional) must be the same, except in the PageAttention use case (when layoutKeyOptional is PA_BBND).
- Other constraints:
  - poolKey supports a non-contiguous axis 0 (only the PA_BBND layout, which requires the keyStride0 attribute; BSND/TND layouts require contiguous input).
  - kDescaleOptional supports a non-contiguous axis 0 (only the PA_BBND layout in the quantization scenario, which requires the kDescaleStride0 attribute; BSND/TND layouts require contiguous input).

## Example

The following sample code is for reference only. For details about the compilation and execution process, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include "securec.h"
#include "acl/acl.h"
#include "aclnnop/aclnn_pool_key_indexer_v2.h"

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
    void* poolKeyDeviceAddr = nullptr;
    void* weightsDeviceAddr = nullptr;
    void* sparseIndicesDeviceAddr = nullptr;
    void* sparseValuesDeviceAddr = nullptr;

    aclTensor* queryTensor = nullptr;
    aclTensor* poolKeyTensor = nullptr;
    aclTensor* weightsTensor = nullptr;
    aclTensor* sparseIndicesTensor = nullptr;
    aclTensor* sparseValuesTensor = nullptr;
};

int InitializeTensors(TensorResources& resources) {
    // topk=2048, pool_size=16
    // The last dimension of sparseIndicesOut = topk + pool_size - 1 = 2063
    // The last dimension of sparseValuesOut = topk // pool_size = 128
    std::vector<int64_t> queryShape = {1, 2, 1, 128};
    std::vector<int64_t> poolKeyShape = {1, 2, 1, 128};
    std::vector<int64_t> weightsShape = {1, 2, 1};
    std::vector<int64_t> sparseIndicesShape = {1, 2, 2063};
    std::vector<int64_t> sparseValuesShape = {1, 2, 128};

    int64_t queryShapeSize = GetShapeSize(queryShape);
    int64_t poolKeyShapeSize = GetShapeSize(poolKeyShape);
    int64_t weightsShapeSize = GetShapeSize(weightsShape);
    int64_t sparseIndicesShapeSize = GetShapeSize(sparseIndicesShape);
    int64_t sparseValuesShapeSize = GetShapeSize(sparseValuesShape);

    // Host data of FP16 tensors is filled with the half bit pattern (uint16_t) to match
    // aclDataType::ACL_FLOAT16; 0x3C00 is FP16 1.0. Do not fill with a raw float byte stream,
    // which would be interpreted as misaligned half values (the bytes of 1.0f, 00 00 80 3F,
    // become [0.0, 1.875] alternately).
    constexpr uint16_t kFp16One = 0x3C00;
    std::vector<uint16_t> queryHostData(queryShapeSize, kFp16One);
    std::vector<uint16_t> poolKeyHostData(poolKeyShapeSize, kFp16One);
    std::vector<uint16_t> weightsHostData(weightsShapeSize, kFp16One);
    std::vector<int32_t> sparseIndicesHostData(sparseIndicesShapeSize, 1);
    std::vector<float> sparseValuesHostData(sparseValuesShapeSize, 1);

    int ret = CreateAclTensor(queryHostData, queryShape, &resources.queryDeviceAddr,
                              aclDataType::ACL_FLOAT16, &resources.queryTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }

    ret = CreateAclTensor(poolKeyHostData, poolKeyShape, &resources.poolKeyDeviceAddr,
                          aclDataType::ACL_FLOAT16, &resources.poolKeyTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }

    ret = CreateAclTensor(weightsHostData, weightsShape, &resources.weightsDeviceAddr,
                          aclDataType::ACL_FLOAT16, &resources.weightsTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }

    ret = CreateAclTensor(sparseIndicesHostData, sparseIndicesShape, &resources.sparseIndicesDeviceAddr,
                          aclDataType::ACL_INT32, &resources.sparseIndicesTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }

    ret = CreateAclTensor(sparseValuesHostData, sparseValuesShape, &resources.sparseValuesDeviceAddr,
                         aclDataType::ACL_FLOAT, &resources.sparseValuesTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }
    return ACL_SUCCESS;
}

int ExecutePoolKeyIndexer(TensorResources& resources, aclrtStream stream,
                        void** workspaceAddr, uint64_t* workspaceSize) {
    int64_t topk = 2048;
    int64_t poolSize = 16;
    int64_t maskMode = 3;
    int64_t quantMode = -1;
    bool returnValue = true;
    int64_t keyStride0 = -1;  // -1 indicates unspecified. Derive the stride based on the contiguous input.
    int64_t kDescaleStride0 = -1;  // -1 indicates unspecified. Derive the stride based on the contiguous input (k_descale uses non-contiguous addressing on axis 0 in the quantization scenario).
    int64_t sparseIndicesMode = 1;  // Index output layout: 0 = legacy (identical to V1), 1 = compact (tail follows valid expansion)
    constexpr const char layerOutStr[] = "BSND";
    constexpr size_t layerOutLen = sizeof(layerOutStr);
    char layoutQuery[layerOutLen];
    char layoutKey[layerOutLen];
    errno_t memcpyRet = memcpy_s(layoutQuery, sizeof(layoutQuery), layerOutStr, layerOutLen);
    if (!CHECK_RET(memcpyRet == 0)) {
        LOG_PRINT("memcpy_s layoutQuery failed. ERROR: %d\n", memcpyRet);
        return -1;
    }
    memcpyRet = memcpy_s(layoutKey, sizeof(layoutKey), layerOutStr, layerOutLen);
    if (!CHECK_RET(memcpyRet == 0)) {
        LOG_PRINT("memcpy_s layoutKey failed. ERROR: %d\n", memcpyRet);
        return -1;
    }
    // poolTailK is a host-side value input (ValueDepend), passed in as the aclIntArray type.
    std::vector<int64_t> poolTailKHostData = {0};
    aclIntArray* poolTailK = aclCreateIntArray(poolTailKHostData.data(), poolTailKHostData.size());
    aclOpExecutor* executor;

    int ret = aclnnPoolKeyIndexerV2GetWorkspaceSize(resources.queryTensor, resources.poolKeyTensor,
                                                    resources.weightsTensor, poolTailK,
                                                    nullptr, nullptr, nullptr, nullptr, nullptr,
                                                    topk, poolSize, layoutQuery, layoutKey,
                                                    maskMode, quantMode, returnValue, keyStride0, kDescaleStride0,
                                                    sparseIndicesMode,
                                                    resources.sparseIndicesTensor, resources.sparseValuesTensor,
                                                    workspaceSize, &executor);

    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclnnPoolKeyIndexerV2GetWorkspaceSize failed. ERROR: %d\n", ret);
        return ret;
    }

    if (*workspaceSize > 0ULL) {
        ret = aclrtMalloc(workspaceAddr, *workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (!CHECK_RET(ret == ACL_SUCCESS)) {
            LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
            return ret;
        }
    }

    ret = aclnnPoolKeyIndexerV2(*workspaceAddr, *workspaceSize, executor, stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclnnPoolKeyIndexerV2 failed. ERROR: %d\n", ret);
        return ret;
    }
    aclDestroyIntArray(poolTailK);

    return ACL_SUCCESS;
}

int PrintValuesOutResult(std::vector<int64_t> &shape, void** deviceAddr) {
  auto size = GetShapeSize(shape);
  std::vector<float> resultData(size, 0);
  auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]),
                         *deviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret);
        return ret;
  }
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("values result[%ld] is: %f\n", i, resultData[i]);
  }
  return ACL_SUCCESS;
}

int PrintIndicesOutResult(std::vector<int64_t> &shape, void** deviceAddr) {
  auto size = GetShapeSize(shape);
  std::vector<int32_t> resultData(size, 0);
  auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]),
                         *deviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret);
        return ret;
  }
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("indices result[%ld] is: %d\n", i, resultData[i]);
  }
  return ACL_SUCCESS;
}

void CleanupResources(TensorResources& resources, void* workspaceAddr,
                     aclrtStream stream, int32_t deviceId) {
    if (resources.queryTensor) {
        aclDestroyTensor(resources.queryTensor);
    }
    if (resources.poolKeyTensor) {
        aclDestroyTensor(resources.poolKeyTensor);
    }
    if (resources.weightsTensor) {
        aclDestroyTensor(resources.weightsTensor);
    }
    if (resources.sparseIndicesTensor) {
        aclDestroyTensor(resources.sparseIndicesTensor);
    }
    if (resources.sparseValuesTensor) {
        aclDestroyTensor(resources.sparseValuesTensor);
    }

    if (resources.queryDeviceAddr) {
        aclrtFree(resources.queryDeviceAddr);
    }
    if (resources.poolKeyDeviceAddr) {
        aclrtFree(resources.poolKeyDeviceAddr);
    }
    if (resources.weightsDeviceAddr) {
        aclrtFree(resources.weightsDeviceAddr);
    }
    if (resources.sparseIndicesDeviceAddr) {
        aclrtFree(resources.sparseIndicesDeviceAddr);
    }
    if (resources.sparseValuesDeviceAddr) {
        aclrtFree(resources.sparseValuesDeviceAddr);
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
    std::vector<int64_t> sparseIndicesShape = {1, 2, 2063};
    std::vector<int64_t> sparseValuesShape = {1, 2, 128};
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
    ret = ExecutePoolKeyIndexer(resources, stream, &workspaceAddr, &workspaceSize);
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
    PrintIndicesOutResult(sparseIndicesShape, &resources.sparseIndicesDeviceAddr);
    PrintValuesOutResult(sparseValuesShape, &resources.sparseValuesDeviceAddr);

    // 6. Cleanup resources
    CleanupResources(resources, workspaceAddr, stream, deviceId);
    return 0;
}
```
