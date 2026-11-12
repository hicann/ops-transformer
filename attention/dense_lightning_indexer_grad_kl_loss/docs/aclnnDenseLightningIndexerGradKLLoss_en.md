# aclnnDenseLightningIndexerGradKLLoss

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-21T08:33:04.751Z -->

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

- Description: The DenseLightningIndexerGradKlLoss operator is the backward operator of LightningIndexer and additionally integrates loss computation. The LightningIndexer operator selects the TopK items with the highest intrinsic relevance between QueryToken and KeyToken, reducing the Attention computation in long-sequence use cases and improving the inference and training performance of long-sequence networks. In the dense use case, the query, key, query_index, and key_index inputs of LightningIndexerGrad do not need to be sparsified.

- Formula:

  1. Formula for calculating the Top-k value:

     $$
     I_{t,:}=W_{t,:}@ReLU(\tilde{q}_{t,:}@\tilde{K}_{:t,:}^\top)
     $$

     - $W_{t,:}$ is the $weights$ corresponding to the $t$-th token;
     - $\tilde{q}_{t,:}$ is the result of merging the $G$ query heads corresponding to the $t$-th token in the $\tilde{q}$ matrix;
     - $\tilde{K}_{:t,:}$ is the $\tilde{K}$ matrix with $t$ rows.

  2. Formula for the forward Softmax:

     $$
     p_{t,:} = \text{Softmax}(q_{t,:} @ K_{:t,:}^\top/\sqrt{d})
     $$

     - $p_{t,:}$ is the Softmax result corresponding to the $t$-th token;
     - $q_{t,:}$ is the result of merging the $G$ query heads corresponding to the $t$-th token in the $q$ matrix;
     - ${K}_{:t,:}$ is the $K$ matrix with $t$ rows.

  3. npu_lightning_indexer is trained separately, and the corresponding loss function is:

      $$
      Loss{=}\sum_tD_{KL}(p_{t,:}||Softmax(I_{t,:}))
      $$

      where $p_{t,:}$ is the target distribution, obtained by summing the main attention score over all heads and then applying L1 normalization to the sum along the context direction. $D_{KL}$ denotes the KL divergence, expressed as:

      $$
      D_{KL}(a||b){=}\sum_ia_i\mathrm{log}{\left(\frac{a_i}{b_i}\right)}
      $$

  4. The gradient expression of the loss can be obtained by derivation:

      $$
      dI\mathop{{}}\nolimits_{{t,:}}=Softmax \left( I\mathop{{}}\nolimits_{{t,:}} \left) -p\mathop{{}}\nolimits_{{t,:}}\right. \right.
      $$

      The gradients of the weights, query, and key matrices can be computed using the chain rule:

      $$
      dW\mathop{{}}\nolimits_{{t,:}}=dI\mathop{{}}\nolimits_{{t,:}}\text{@} \left( ReLU \left( S\mathop{{}}\nolimits_{{t,:}} \left) \left) \mathop{{}}\nolimits^{\top}\right. \right. \right. \right.
      $$

      $$
      d\mathop{{\tilde{q}}}\nolimits_{{t,:}}=dS\mathop{{}}\nolimits_{{t,:}}@\tilde{K}\mathop{{}}\nolimits_{{:t,:}}
      $$

      $$
      d\tilde{K}\mathop{{}}\nolimits_{{:t,:}}=\left(dS\mathop{{}}\nolimits_{{t,:}} \left) \mathop{{}}\nolimits^{\top}@\tilde{q}\mathop{{}}\nolimits_{{:t, :}}\right. \right.
      $$

      where $S$ is the result of the matrix multiplication between $\tilde{q}$ and $K$.

<!-- - **NOTE**:

   <blockquote>The query, key, and value data layout can be interpreted from multiple dimensions, where B (Batch) indicates the input batch size, S (Seq-Length) indicates the input sequence length, H (Head-Size) indicates the size of the hidden layer, N (Head-Num) indicates the number of heads, D (Head-Dim) indicates the size of the smallest unit of the hidden layer, D=H/N, and T indicates the cumulative sum of the sequence lengths of all Batch input samples.
   </blockquote> -->

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). You must first call the "aclnnDenseLightningIndexerGradKLLossGetWorkspaceSize" API to obtain the input parameters and calculate the required workspace size based on the computation flow, and then call the "aclnnDenseLightningIndexerGradKLLoss" API to perform the computation.

```c++
aclnnStatus aclnnDenseLightningIndexerGradKLLossGetWorkspaceSize(
    const aclTensor     *query,
    const aclTensor     *key,
    const aclTensor     *queryIndex,
    const aclTensor     *keyIndex,
    const aclTensor     *weights,
    const aclTensor     *softmaxMax,
    const aclTensor     *softmaxSum,
    const aclTensor     *softmaxMaxIndex,
    const aclTensor     *softmaxSumIndex,
    const aclTensor     *queryRope,
    const aclTensor     *keyRope,
    const aclIntArray   *actualSeqLengthsQuery,
    const aclIntArray   *actualSeqLengthsKey,
    double               scaleValue,
    char                *layout,
    int64_t              sparseMode,
    int64_t              preTokens,
    int64_t              nextTokens,
    const aclTensor     *dQueryIndex,
    const aclTensor     *dKeyIndex,
    const aclTensor     *dWeights,
    const aclTensor     *loss,
    uint64_t            *workspaceSize,
    aclOpExecutor       **executor)
```

```c++
aclnnStatus aclnnDenseLightningIndexerGradKLLoss(
    void             *workspace,
    uint64_t          workspaceSize,
    aclOpExecutor    *executor,
    aclrtStream       stream)
```

## aclnnDenseLightningIndexerGradKLLossGetWorkspaceSize

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1550px">
      <colgroup>
          <col style="width: 320px">
          <col style="width: 120px">
          <col style="width: 200px">
          <col style="width: 400px">
          <col style="width: 212px">
          <col style="width: 100px">
          <col style="width: 190px">
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
          <th>Shape</th>
          <th>Non-contiguous Tensor</th>
      </tr></thead>
      <tbody>
      <tr>
       <td>query (aclTensor*)</td>
       <td>Input</td>
       <td>Input Q of the attention structure.</td>
       <td><ul><li>B: Generalization supported.</li><li>S1: Generalization supported.</li><li>N1: 128, 64, or 32.</li><li>D: 128.</li><li>T1: Sum of S1 across multiple batches.</li></ul></td>
       <td>FLOAT16, BFLOAT16</td>
       <td>ND</td>
       <td>(B,S1,N1,D);(T1,N1,D)</td>
       <td>×</td>
      </tr>
      <tr>
       <td>key (aclTensor*)</td>
       <td>Input</td>
       <td>Input K of the attention structure.</td>
       <td><ul><li>B: Supports generalization and must be the same as B of query.</li><li>S2: Supports generalization.</li><li>N2: Equal to N1.</li><li>D: 128.</li><li>T2: Sum of S2 across multiple batches.</li></ul></td>
       <td>FLOAT16, BFLOAT16</td>
       <td>ND</td>
       <td>(B,S2,N2,D);(T2,N2,D)</td>
       <td>×</td>
      </tr>
      <tr>
       <td>queryIndex (aclTensor*)</td>
       <td>Input</td>
       <td>queryIndex input of the lightningIndexer structure.</td>
       <td><ul><li>B: Supports generalization and must be the same as B of query.</li><li>S1: Supports generalization.</li><li>Nidx1: 64, 32, 16, 8.</li><li>D: 128.</li><li>T1: Sum of S1 across multiple batches.</li></ul></td>
       <td>FLOAT16, BFLOAT16</td>
       <td>ND</td>
       <td>(B,S1,Nidx1,D);(T1,Nidx1,D)</td>
       <td>×</td>
      </tr>
      <tr>
       <td>keyIndex (aclTensor*)</td>
       <td>Input</td>
       <td>Input keyIndex of the lightningIndexer structure.</td>
       <td><ul><li>B: Supports generalization and is consistent with the B of query.</li> <li>S2: Supports generalization.</li><li>Nidx2: 1.</li><li>D: 128.</li><li>T2: Sum of S2 across multiple batches.</li></ul>
       </td>
       <td>FLOAT16, BFLOAT16</td>
       <td>ND</td>
       <td>(B,S2,Nidx2,D);(T2,Nidx2,D)</td>
       <td>×</td>
      </tr>
      <tr>
       <td>weights (aclTensor*)</td>
       <td>Input</td>
       <td>Weight</td>
       <td><ul><li>B: supports generalization and is the same as the B of query.</li><li>S1: supports generalization and is the same as the S1 of query.</li><li>Nidx1: 64, 32, 16, 8.</li><li>T1: sum of S1 across multiple batches.</li></ul></td>
       <td>FLOAT16, BFLOAT16, FLOAT32</td>
       <td>ND</td>
       <td>(B,S1,Nidx1);(T1,Nidx1)</td>
       <td>×</td>
      </tr>
      <tr>
       <td>softmaxMax (aclTensor*)</td>
       <td>Input</td>
       <td>Device-side aclTensor, intermediate output of attention forward computation.</td>
       <td><ul><li>B: Supports generalization and is consistent with the B of query.</li><li>N2: Equals N1.</li><li>S1: Supports generalization and is consistent with the S1 of query.</li><li>G: N1/N2.</li><li>T1: Cumulative S1 of multiple batches.</li></ul></td>
       <td>FLOAT32</td>
       <td>ND</td>
       <td>(B,N2,S1,G);(N2,T1,G)</td>
       <td>×</td>
      </tr>
      <tr>
       <td>softmaxSum (aclTensor*)</td>
       <td>Input</td>
       <td>Device-side aclTensor, intermediate output of attention forward computation.</td>
       <td><ul><li>B: supports generalization and is the same as the B of query.</li><li>N2: equals N1.</li><li>S1: supports generalization and is the same as the S1 of query.</li><li>G: N1/N2.</li><li>T1: sum of S1 across multiple batches.</li></ul></td>
       <td>FLOAT32</td>
       <td>ND</td>
       <td>(B,N2,S1,G);(N2,T1,G)</td>
       <td>×</td>
      </tr>
      <tr>
       <td>softmaxMaxIndex(aclTensor*)</td>
       <td>Input</td>
       <td>aclTensor on the Device side, intermediate output of attention forward computation.</td>
       <td><ul><li>B: supports generalization and is the same as the B of query.</li><li>Nidx2: 1.</li><li>S1: supports generalization and is the same as the S1 of query.</li><li>T1: sum of S1 across multiple batches.</li></ul></td>
       <td>FLOAT32</td>
       <td>ND</td>
       <td>(B,Nidx2,S1);(Nidx2,T1)</td>
       <td>×</td>
      </tr>
      <tr>
       <td>softmaxSumIndex (aclTensor*)</td>
       <td>Input</td>
       <td>aclTensor on the Device side, intermediate output of the forward attention computation.</td>
       <td><ul><li>B: Supports generalization and must be the same as the B of query.</li><li>Nidx2: 1.</li><li>S1: Supports generalization and must be the same as the S1 of query.</li><li>T1: Sum of S1 across multiple batches.</li></ul></td>
       <td>FLOAT32</td>
       <td>ND</td>
       <td>(B,Nidx2,S1);(Nidx2,T1)</td>
       <td>√</td>
      </tr>
      <tr>
       <td>queryRope (aclTensor*)</td>
       <td>Input</td>
       <td>MLA rope part: output of the Query positional encoding.</td>
       <td><ul><li>Dimensions consistent with the layout of query.</li><li>B: Supports generalization and is consistent with the B of query.</li><li>S1: Supports generalization and is consistent with the S1 of query.</li><li>N1: 128, 64, 32.</li><li>Dr: 64.</li><li>T1: Sum of S1 over multiple batches.</li></ul>
       </td>
       <td>FLOAT16, BFLOAT16</td>
       <td>ND</td>
       <td>(B,S1,N1,Dr);(T1,N1,Dr)</td>
       <td>√</td>
      </tr>
      <tr>
       <td>keyRope (aclTensor*)</td>
       <td>Input</td>
       <td>MLA rope part: output of Key position encoding.</td>
       <td><ul><li>Consistent with the layout dimensions of key.</li><li>B: Generalization supported, consistent with the B of query.</li><li>S2: Generalization supported, consistent with S1 of key.</li><li>N2: Equal to N1.</li><li>Dr: 64.</li><li>T2: Cumulative sum of S2 across multiple batches.</li></ul></td>
       <td>FLOAT16, BFLOAT16</td>
       <td>ND</td>
       <td>(B,S2,N2,Dr);(T2,N2,Dr)</td>
       <td>√</td>
      </tr>
      <tr>
       <td>actualSeqLengthsQuery (aclIntArray*)</td>
       <td>Input</td>
       <td>Number of valid Query tokens in each batch.</td>
       <td><ul><li>Value dependency.</li><li>The length is the same as B.</li><li>The cumulative sum is the same as T1.</li></ul></td>
       <td>INT64</td>
       <td>ND</td>
       <td>(B,)</td>
       <td>-</td>
      </tr>
      <tr>
       <td>actualSeqLengthsKey (aclIntArray*)</td>
       <td>Input</td>
       <td>Number of valid Key tokens in each Batch.</td>
       <td><ul><li>Value dependency.</li><li>The length is the same as B.</li><li>The cumulative sum is the same as T2.</li></ul></td>
       <td>INT64</td>
       <td>ND</td>
       <td>(B,)</td>
       <td>-</td>
      </tr>
      <tr>
       <td>scaleValue (double)</td>
       <td>Input</td>
       <td>Scale factor</td>
       <td><ul><li>Recommended value: reciprocal of the square root of d in the formula.</li></ul></td>
       <td>-</td>
       <td>-</td>
       <td>-</td>
       <td>-</td>
      </tr>
      <tr>
       <td>layout (char*)</td>
       <td>Input</td>
       <td>layout format.</td>
       <td>Only BSND and TND formats are supported.</td>
       <td>-</td>
       <td>-</td>
       <td>-</td>
       <td>-</td>
      </tr>
      <tr>
       <td>sparseMode (int64_t)</td>
       <td>Input</td>
       <td>Sparse mode</td>
       <td><ul><li>Specifies the sparse mode. For details about different sparse modes, see <a href="#constraints">Constraints</a>.</li><li>Only mode 3 is supported.</li></ul></td>
       <td>-</td>
       <td>-</td>
       <td>-</td>
       <td>-</td>
      </tr>
      <tr>
       <td>preTokens (int64_t)</td>
       <td>Input</td>
       <td>Used for sparse computation, indicating the number of preceding tokens that Attention needs to compute correlations with.</td>
        <td>Same as preTokens in Attention. Takes effect when sparseMode = 0 or 4. Default value: 2147483647.</td>
       <td>-</td>
       <td>-</td>
       <td>-</td>
       <td>-</td>
      </tr>
      <tr>
       <td>nextTokens (int64_t)</td>
       <td>Input</td>
       <td>Used for sparse computation, indicating the number of subsequent tokens that Attention needs to compute correlations with.</td>
        <td>Same as nextTokens in Attention. Takes effect when sparseMode = 0 or 4. Default value: 2147483647.</td>
       <td>-</td>
       <td>-</td>
       <td>-</td>
       <td>-</td>
      </tr>
      <tr>
       <td>dQueryIndex (aclTensor*)</td>
       <td>Output</td>
       <td>Gradient of QueryIndex.</td>
       <td><ul><li>B: Supports generalization and must be the same as the B of query.</li><li>S1: Supports generalization and must be the same as the S1 of query.</li><li>Nidx1: 64, 32, 16, 8.</li><li>D: 128.</li><li>T1: Cumulative sum of S1 across multiple batches.</li></ul></td>
       <td>FLOAT16, BFLOAT16</td>
       <td>ND</td>
       <td>(B,S1,Nidx1,D);(T1,Nidx1,D)</td>
       <td>√</td>
      </tr>
      <tr>
       <td>dKeyIndex (aclTensor*)</td>
       <td>Output</td>
       <td>Gradient of KeyIndex.</td>
       <td><ul><li>B: Supports generalization and must be the same as the B of query.</li><li>S2: Supports generalization and must be the same as the S2 of key.</li><li>Nidx2: 1.</li><li>D: 128.</li><li>T2: Sum of S2 across multiple batches.</li></ul></td>
       <td>FLOAT16, BFLOAT16</td>
       <td>ND</td>
       <td>(B,S2,Nidx2,D);(T2,Nidx2,D)</td>
       <td>√</td>
      </tr>
      <tr>
       <td>dWeights (aclTensor*)</td>
       <td>Output</td>
       <td>Gradient of Weights.</td>
       <td><ul><li>B: supports generalization.</li><li>S1: supports generalization and cannot be the M dimension of Matmul.</li><li>Nidx1: 64, 32, 16, 8.</li><li>T1: accumulated S1 across multiple batches.</li></ul></td>
       <td>FLOAT16, BFLOAT16, FLOAT32</td>
       <td>ND</td>
       <td>(B,S1,Nidx1);(T1,Nidx1)</td>
       <td>√</td>
      </tr>
      <tr>
       <td>loss (aclTensor*)</td>
       <td>Output</td>
       <td>Loss function value.</td>
       <td>-</td>
       <td>FLOAT32</td>
       <td>ND</td>
       <td>(1,)</td>
       <td>-</td>
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
      <td>executor (aclOpExecutor**)</td>
      <td>Output</td>
      <td>Returns the op executor, which contains the computation process of the operator.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
      </tbody>
  </table>

- **Return value**

  Returns an aclnnStatus status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

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
       <td>A required parameter or output is a null pointer.</td>
      </tr>
      <tr>
       <td>ACLNN_ERR_PARAM_INVALID</td>
       <td>161002</td>
       <td>The data types or data formats of input variables such as query, key, queryIndex, keyIndex, weights, and softmaxMax are not in the supported range.</td>
      </tr>
      <tr>
       <td>ACLNN_ERR_INNER_TILING_ERROR</td>
       <td>561002</td>
       <td>The shapes of multiple input tensors do not match (see Parameters for details).</td>
      </tr>
      </tbody>
  </table>

## aclnnDenseLightningIndexerGradKLLoss

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
      <td>Workspace memory address allocated on the Device side.</td>
      </tr>
      <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Workspace size allocated on the Device side, obtained by calling the first-phase API aclnnDenseLightningIndexerGradKLLossGetWorkspaceSize.</td>
      </tr>
      <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Op executor containing the operator computation flow.</td>
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

- The data types of query, key, queryIndex, and keyIndex must be the same.
- If weights is not float32, the data types of query, key, queryIndex, keyIndex, and weights must be the same.

- General Constraints

  - Deterministic computation:
    aclnnDenseLightningIndexerGradKLLoss defaults to a non-deterministic implementation, and supports enabling deterministic computation through alcrtCtxSetSysParamOpt.
  - Handling of empty input parameters:
    - If query, key, queryIndex, keyIndex, or weights is an empty tensor: currently not supported, and an error is reported.

  <table style="undefined;table-layout: fixed; width: 942px"><colgroup>
      <col style="width: 100px">
      <col style="width: 740px">
      <col style="width: 360px">
      </colgroup>
      <thead>
       <tr>
       <th>sparseMode</th>
       <th>Description</th>
       <th>Remarks</th>
       </tr>
      </thead>
      <tbody>
      <tr>
       <td>0</td>
       <td>defaultMask mode. If attenmask is not passed in, no mask operation is performed, and preTokens and nextTokens are ignored. If attenmask is passed in, the complete attenmask matrix must be passed in, indicating that the part between preTokens and nextTokens needs to be computed.</td>
       <td>Not supported</td>
      </tr>
      <tr>
       <td>1</td>
       <td>allMask. The complete attenmask matrix must be passed in.</td>
       <td>Not supported</td>
      </tr>
      <tr>
       <td>2</td>
       <td>Mask of the leftUpCausal mode. The optimized attenmask matrix must be passed in.</td>
       <td>Not supported</td>
      </tr>
      <tr>
       <td>3</td>
       <td>Mask of the rightDownCausal mode, corresponding to the lower triangular use case divided by the right vertex. The optimized attenmask matrix must be passed in.</td>
       <td>Supported</td>
  </tr>
      <tr>
       <td>4</td>
       <td>Mask of the band mode. The optimized attenmask matrix must be passed in.</td>
       <td>Not supported</td>
      </tr>
      <tr>
       <td>5</td>
       <td>prefix</td>
       <td>Not supported</td>
      </tr>
      <tr>
       <td>6</td>
       <td>global</td>
       <td>Not supported</td>
      </tr>
      <tr>
       <td>7</td>
       <td>dilated</td>
       <td>Not supported</td>
      </tr>
      <tr>
       <td>8</td>
       <td>block_local</td>
       <td>Not supported</td>
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
       <th>Specification</th>
       <th>Specification Description</th>
      </tr>
      </thead>
      <tbody>
      <tr>
       <td>B</td>
       <td>1-256</td>
       <td>-</td>
      </tr>
      <tr>
       <td>S1, S2</td>
       <td>1-1M</td>
       <td>S1 and S2 support different lengths.</td>
      </tr>
      <tr>
       <td>N1</td>
       <td>32, 64, 128</td>
       <td>-</td>
      </tr>
      <tr>
       <td>Nidx1</td>
       <td>8, 16, 32, 64</td>
       <td>-</td>
      </tr>
      <tr>
       <td>N2</td>
       <td>32, 64, 128</td>
       <td>-</td>
      </tr>
      <tr>
       <td>Nidx2</td>
       <td>1</td>
       <td>-</td>
      </tr>
      <tr>
       <td>D</td>
       <td>128</td>
       <td>The D of query is the same as that of query_index.</td>
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

- Typical values

  <table style="undefined;table-layout: fixed; width: 942px"><colgroup>
      <col style="width: 100px">
      <col style="width: 660px">
      </colgroup>
      <thead>
      <tr>
      <th>Item</th>
      <th>Typical Value</th>
      </tr>
      </thead>
      <tbody>
      <tr>
       <td>query</td>
       <td>N1=128/64/32; D=128</td>
      </tr>
      <tr>
       <td>queryIndex</td>
       <td>Nidx1 = 64/32/16/8;  D = 128 ; S1 = 64k/128k</td>
      </tr>
      <tr>
       <td>keyIndex</td>
       <td>D = 128</td>
      </tr>
      <tr>
       <td>qRope</td>
       <td>D = 64</td>
      </tr>
      </tbody>
  </table>

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```c++
#include <iostream>
#include <vector>
#include <cstdint>
#include <cmath>
#include "acl/acl.h"
#include "aclnnop/aclnn_dense_lightning_indexer_grad_kl_loss.h"

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
  std::vector<aclFloat16> resultData(size, 0);
  auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]),
                         *deviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("mean result[%ld] is: %f\n", i, aclFloat16ToFloat(resultData[i]));
  }
}

int Init(int32_t deviceId, aclrtContext* context, aclrtStream* stream) {
  // Boilerplate: AscendCL initialization.
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
  // Call aclrtMemcpy to copy host-side data to device-side memory.
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // Calculate the strides of a contiguous tensor.
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  // Call aclCreateTensor to create an aclTensor.
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

int main() {
  // 1. (Boilerplate) Initialize device/context/stream. See the AscendCL external API list.
  // Set deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtContext context;
  aclrtStream stream;
  auto ret = Init(deviceId, &context, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the inputs and outputs. Customize the construction based on the API.
  int64_t s1 = 1;
  int64_t s2 = 1;
  int64_t n1 = 32;
  int64_t n2 = n1;
  int64_t n1Index = 8;
  int64_t n2Index = 1;
  int64_t dQuery = 128;
  int64_t dRope = 64;
  int64_t dQueryIndex = 128;
  int64_t t1 = s1;
  int64_t t2 = s2;
  int64_t G = n1 / n2;

  std::vector<int64_t> qShape = {t1, n1, dQuery};
  std::vector<int64_t> kShape = {t2, n2, dQuery};
  std::vector<int64_t> qRopeShape = {t1, n1, dRope};
  std::vector<int64_t> kRopeShape = {t2, n2, dRope};
  std::vector<int64_t> qIndexShape = {t1, n1Index, dQueryIndex};
  std::vector<int64_t> kIndexShape = {t2, n2Index, dQueryIndex};
  std::vector<int64_t> weightShape = {t1, n1Index};
  std::vector<int64_t> softmaxMaxShape = {n2, t1, G};
  std::vector<int64_t> softmaxSumShape = {n2, t1, G};
  std::vector<int64_t> softmaxMaxIndexShape = {n2Index, t1};
  std::vector<int64_t> softmaxSumIndexShape = {n2Index, t1};

  std::vector<int64_t> dQIndexShape = {t1, n1Index, dQueryIndex};
  std::vector<int64_t> dKIndexShape = {t2, n2Index, dQueryIndex};
  std::vector<int64_t> dWeightShape = {t1, n1Index};
  std::vector<int64_t> lossShape = {1};

  void* qDeviceAddr = nullptr;
  void* kDeviceAddr = nullptr;
  void* qRopeDeviceAddr = nullptr;
  void* kRopeDeviceAddr = nullptr;
  void* qIndexDeviceAddr = nullptr;
  void* kIndexDeviceAddr = nullptr;
  void* weightDeviceAddr = nullptr;
  void* softmaxMaxDeviceAddr = nullptr;
  void* softmaxSumDeviceAddr = nullptr;
  void* softmaxMaxIndexDeviceAddr = nullptr;
  void* softmaxSumIndexDeviceAddr = nullptr;

  void* dQIndexDeviceAddr = nullptr;
  void* dKIndexDeviceAddr = nullptr;
  void* dWeightDeviceAddr = nullptr;
  void* lossDeviceAddr = nullptr;

  aclTensor* q = nullptr;
  aclTensor* k = nullptr;
  aclTensor* qRope = nullptr;
  aclTensor* kRope = nullptr;
  aclTensor* qIndex = nullptr;
  aclTensor* kIndex = nullptr;
  aclTensor* weight = nullptr;
  aclTensor* softmaxMax = nullptr;
  aclTensor* softmaxSum = nullptr;
  aclTensor* softmaxMaxIndex = nullptr;
  aclTensor* softmaxSumIndex = nullptr;

  aclTensor* dQIndex = nullptr;
  aclTensor* dKIndex = nullptr;
  aclTensor* dWeight = nullptr;
  aclTensor* loss = nullptr;

  std::vector<aclFloat16> qHostData(t1 * n1 * dQuery, aclFloatToFloat16(0.1));
  std::vector<aclFloat16> kHostData(t2 * n2 * dQuery, aclFloatToFloat16(0.2));
  std::vector<aclFloat16> qRopeHostData(t1 * n1 * dRope, aclFloatToFloat16(0.1));
  std::vector<aclFloat16> kRopeHostData(t2 * n2 * dRope, aclFloatToFloat16(0.2));
  std::vector<aclFloat16> qIndexHostData(t1 * n1Index * dQueryIndex, aclFloatToFloat16(0.2));
  std::vector<aclFloat16> kIndexHostData(t2 * n2Index * dQueryIndex, aclFloatToFloat16(0.1));
  std::vector<aclFloat16> weightHostData(t1 * n1Index, aclFloatToFloat16(0.005));

  std::vector<float> softmaxMaxHostData(t1 * n2, 25.4483f);
  std::vector<float> softmaxSumHostData(t1 * n2, 1.0f);
  std::vector<float> softmaxMaxIndexHostData(t1 * n2Index, 25.4483f);
  std::vector<float> softmaxSumIndexHostData(t1 * n2Index, 1.0f);

  std::vector<aclFloat16> dQIndexHostData(t1 * n1Index * dQueryIndex);
  std::vector<aclFloat16> dKIndexHostData(t2 * n2Index * dQueryIndex);
  std::vector<aclFloat16> dWeightHostData(t1 * n1Index);
  std::vector<float> lossHostData(1, 1.0f);

  ret = CreateAclTensor(qHostData, qShape, &qDeviceAddr, aclDataType::ACL_FLOAT16, &q);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(kHostData, kShape, &kDeviceAddr, aclDataType::ACL_FLOAT16, &k);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(qRopeHostData, qRopeShape, &qRopeDeviceAddr, aclDataType::ACL_FLOAT16, &qRope);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(kRopeHostData, kRopeShape, &kRopeDeviceAddr, aclDataType::ACL_FLOAT16, &kRope);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(qIndexHostData, qIndexShape, &qIndexDeviceAddr, aclDataType::ACL_FLOAT16, &qIndex);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(kIndexHostData, kIndexShape, &kIndexDeviceAddr, aclDataType::ACL_FLOAT16, &kIndex);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(weightHostData, weightShape, &weightDeviceAddr, aclDataType::ACL_FLOAT16, &weight);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(softmaxMaxHostData, softmaxMaxShape, &softmaxMaxDeviceAddr, aclDataType::ACL_FLOAT, &softmaxMax);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(softmaxSumHostData, softmaxSumShape, &softmaxSumDeviceAddr, aclDataType::ACL_FLOAT, &softmaxSum);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(softmaxMaxIndexHostData, softmaxMaxIndexShape, &softmaxMaxIndexDeviceAddr, aclDataType::ACL_FLOAT, &softmaxMaxIndex);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(softmaxSumIndexHostData, softmaxSumIndexShape, &softmaxSumIndexDeviceAddr, aclDataType::ACL_FLOAT, &softmaxSumIndex);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(dQIndexHostData, dQIndexShape, &dQIndexDeviceAddr, aclDataType::ACL_FLOAT16, &dQIndex);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(dKIndexHostData, dKIndexShape, &dKIndexDeviceAddr, aclDataType::ACL_FLOAT16, &dKIndex);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(dWeightHostData, dWeightShape, &dWeightDeviceAddr, aclDataType::ACL_FLOAT16, &dWeight);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(lossHostData, lossShape, &lossDeviceAddr, aclDataType::ACL_FLOAT, &loss);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  std::vector<int64_t>  acSeqQLenOp = {t1};
  std::vector<int64_t>  acSeqKvLenOp = {t2};
  aclIntArray* acSeqQLen = aclCreateIntArray(acSeqQLenOp.data(), acSeqQLenOp.size());
  aclIntArray* acSeqKvLen = aclCreateIntArray(acSeqKvLenOp.data(), acSeqKvLenOp.size());
  float scaleValue = 1.0 / sqrt(dQuery);
  int64_t preTokens = 2147483647;
  int64_t nextTokens = 2147483647;
  int64_t sparseMode = 3;
  bool deterministic = false;

  char layout[5] = {'T', 'N', 'D', 0};

  // 3. Call the CANN operator library API. Change it to the specific API name.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  // Call the first-phase API aclnnDenseLightningIndexerGradKLLossGetWorkspaceSize.
  ret = aclnnDenseLightningIndexerGradKLLossGetWorkspaceSize(
            q, k, qIndex, kIndex, weight, softmaxMax, softmaxSum, softmaxMaxIndex, softmaxSumIndex, qRope, kRope,
            acSeqQLen, acSeqKvLen, scaleValue, layout, sparseMode, preTokens, nextTokens, dQIndex, dKIndex, dWeight, loss,
            &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnDenseLightningIndexerGradKLLossGetWorkspaceSize failed. ERROR: %d\n", ret);
            return ret);

  // Allocate device memory based on the workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }

  // Call the second-phase API aclnnDenseLightningIndexerGradKLLoss.
  ret = aclnnDenseLightningIndexerGradKLLoss(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnDenseLightningIndexerGradKLLoss failed. ERROR: %d\n", ret); return ret);

  // 4. (boilerplate) Synchronize and wait for the task execution to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output values. Copy the result from the device side to the host side. Modify the code based on the specific API definition.
  PrintOutResult(dQIndexShape, &dQIndexDeviceAddr);
  PrintOutResult(dKIndexShape, &dKIndexDeviceAddr);
  PrintOutResult(dWeightShape, &dWeightDeviceAddr);
  PrintOutResult(lossShape, &lossDeviceAddr);

  // 6. Release aclTensor and aclScalar. Modify the code based on the specific API definition.
  aclDestroyTensor(q);
  aclDestroyTensor(k);
  aclDestroyTensor(qIndex);
  aclDestroyTensor(kIndex);
  aclDestroyTensor(qRope);
  aclDestroyTensor(kRope);
  aclDestroyTensor(weight);
  aclDestroyTensor(softmaxMax);
  aclDestroyTensor(softmaxSum);
  aclDestroyTensor(softmaxMaxIndex);
  aclDestroyTensor(softmaxSumIndex);

  aclDestroyTensor(dQIndex);
  aclDestroyTensor(dKIndex);
  aclDestroyTensor(dWeight);
  aclDestroyTensor(loss);

  // 7. Release device resources.
  aclrtFree(qDeviceAddr);
  aclrtFree(kDeviceAddr);
  aclrtFree(qIndexDeviceAddr);
  aclrtFree(kIndexDeviceAddr);
  aclrtFree(qRopeDeviceAddr);
  aclrtFree(kRopeDeviceAddr);
  aclrtFree(weightDeviceAddr);
  aclrtFree(softmaxMaxDeviceAddr);
  aclrtFree(softmaxSumDeviceAddr);
  aclrtFree(softmaxMaxIndexDeviceAddr);
  aclrtFree(softmaxSumIndexDeviceAddr);

  aclrtFree(dQIndexDeviceAddr);
  aclrtFree(dKIndexDeviceAddr);
  aclrtFree(dWeightDeviceAddr);
  aclrtFree(lossDeviceAddr);
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
