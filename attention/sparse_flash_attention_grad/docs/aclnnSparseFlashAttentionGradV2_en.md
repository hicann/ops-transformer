# aclnnSparseFlashAttentionGradV2

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-21T13:14:23.371Z -->

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

- **Description**: Rearranges data of size selectedBlockSize selected from key and value based on topkIndices, and then computes the backward output of attention in training scenarios.
- **Relationship with the V1 API**: Based on V1 (aclnnSparseFlashAttentionGrad, compatible with commercial releases, 5 outputs), V2 only adds the optional learnable OSS Sink input `sinksOptional` and the corresponding gradient output `dSinksOptional` (6 outputs). The other parameters are exactly the same as those of V1. V1/V2 share the same OpDef and inner implementation.
- **OSS Sink (optional)**: V2 supports the learnable OSS Sink input (`sinksOptional`) and the corresponding gradient output (`dSinksOptional`). When `sinksOptional` is passed in, the `dSinksOptional` output must be provided at the same time. `sinksOptional` is supported only on <term>Ascend 950PR&950DT products</term>.

- **Formula**: Rearranges data of selectedBlockCount blocks, each of size selectedBlockSize, selected from keyIn and value based on the input topkIndices. The formula is as follows:

  $$
   selectedKey\text{ }=\text{ }Gather \left( key,topkIndices \left[ i \left]  \left) ,\text{ }0\text{ } < =i < \text{ }selectBlockCount\right. \right. \right. \right.
  $$

  $$
   selectedValue\text{ }=\text{ }Gather \left( value,topkIndices \left[ i \left]  \left) ,\text{ }0\text{ } < =i < \text{ }selectBlockCount\right. \right. \right. \right.
  $$

  In the KV merge use case, if value is a null pointer, value in the preceding formula is processed as key, and selectedValue is equivalent to selectedKey. In this use case, the backward output does not return dValue separately; dKeyOut indicates the merged gradient:

  $$
   dKeyOut = dK + dV
  $$

<div style="padding-left:40px;">

  Phase 1: Compute $dP$ and $dV$ based on the matrix multiplication derivative rule:

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

<div style="padding-left:40px;">

  Stage 4: Compute $dSink$ (only when the sinksOptional input is provided):

</div>

  Sink acts as a "virtual key" in the forward pass. Its corresponding "virtual value" is 0. It does not directly contribute to the output O, but indirectly affects all attention weights $P_j$ through the softmax normalization factor $\ell$. Therefore, Sink participates in computing the max and sum of the forward softmax:

  $$
   m \mathop{{}}=\mathop{{}}\nolimits max \left( max \mathop{{}}\nolimits_{{j}} \left( S \mathop{{}}\nolimits_{{t,j}} \right) ,\mathop{{}}\nolimits sink \right)
  $$

  $$
   e \mathop{{}}\nolimits_{{sink}} = exp \left( sink - m \right)
  $$

  $$
   \ell = e \mathop{{}}\nolimits_{{sink}} + \sum \mathop{{}}\nolimits_{{j}} exp \left( S \mathop{{}}\nolimits_{{t,j}} - m \right)
  $$

  $$
   P \mathop{{}}\nolimits_{{sink}} = e \mathop{{}}\nolimits_{{sink}} / \ell
  $$

  After Sink is introduced, the common term $D_{i}$ in the softmax backward remains unchanged: $D_{i} = \sum \mathop{{}}\nolimits_{{j}} P \mathop{{}}\nolimits_{{t,j}} \cdot dP \mathop{{}}\nolimits_{{t,j}}$(where $dP \mathop{{}}\nolimits_{{sink}} = dO \text{@} V^{T} = 0$, Sink does not directly contribute to the output and therefore does not produce dP). The gradient of Sink is as follows:

  $$
   dSink \left[ h \right] = P \mathop{{}}\nolimits_{{sink}} \odot \left( dP \mathop{{}}\nolimits_{{sink}} - D_{i} \right) = \sum \mathop{{}}\nolimits_{{b,i}} P \mathop{{}}\nolimits_{{sink}} \left[ b,i,h \right] \cdot \left( - D_{i} \left[ b,i,h \right] \right)
  $$

  > NOTE: The score gradient of Sink follows the "squeeze effect" of the softmax backward — when the Sink probability increases, it squeezes the probabilities of other tokens. Therefore, $dSink$ depends on the average gain $D_{i}$ of all tokens. Summing over all Batch and Query positions yields $dSink$ with dimension $[N1]$.

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). First call the "aclnnSparseFlashAttentionGradV2GetWorkspaceSize" API to obtain the workspace size required for the computation and the executor that contains the operator computation process. Then call the "aclnnSparseFlashAttentionGradV2" API to perform the computation.

> [!NOTE]
> V2 is a new API. It shares the same OpDef (SparseFlashAttentionGrad) and the same inner aclnn implementation as V1. To use the commercial-release-compatible signature (5 outputs, no sinks/dSinks), use [aclnnSparseFlashAttentionGrad](aclnnSparseFlashAttentionGrad_en.md).

**Comparison with the V1 API (aclnnSparseFlashAttentionGrad)**

| Comparison item | V1: aclnnSparseFlashAttentionGrad | V2: aclnnSparseFlashAttentionGradV2 |
| :--- | :--- | :--- |
| API positioning | Commercial-release compatibility API. Positional parameters are strictly aligned with the commercial `npu_sparse_flash_attention_grad` | New API that supports OSS Sink |
| Number of outputs | 5 (dQuery, dKey, dValue, dQueryRope, dKeyRope) | 6 (adds dSinks based on V1) |
| sinksOptional input | None. | New (optional; supported only on <term>Ascend 950PR&950DT products</term>) |
| dSinksOptional output | None. | New (optional; provided in pair with sinksOptional) |
| Other parameters | - | Exactly the same as V1 (parameter order, types, and meanings unchanged) |
| OpDef | SparseFlashAttentionGrad | Shares the same OpDef and inner aclnn as V1 |

> In summary, compared with V1, V2 only adds the optional sinksOptional input and the corresponding dSinksOptional output. Other parameters are exactly the same as those of V1. If sinksOptional/dSinksOptional are not provided, the inputs and outputs of V2 remain consistent with the V1 behavior.

```c++
aclnnStatus aclnnSparseFlashAttentionGradV2GetWorkspaceSize(
    const aclTensor     *query,
    const aclTensor     *key,
    const aclTensor     *value,
    const aclTensor     *sparseIndices,
    const aclTensor     *dOut,
    const aclTensor     *out,
    const aclTensor     *softmaxMax,
    const aclTensor     *softmaxSum,
    const aclTensor     *sinksOptional,            // OSS Sink input [N1], OPTIONAL, paired with dSinksOptional
    const aclTensor     *actualSeqLengthsQueryOptional,
    const aclTensor     *actualSeqLengthsKvOptional,
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
    const aclTensor     *dSinksOptional,           // Sink gradient [N1], OPTIONAL, paired with sinksOptional
    uint64_t            *workspaceSize,
    aclOpExecutor      **executor)
```

```c++
aclnnStatus aclnnSparseFlashAttentionGradV2(
    void             *workspace,
    uint64_t          workspaceSize,
    aclOpExecutor    *executor,
    aclrtStream       stream)
```

## aclnnSparseFlashAttentionGradV2GetWorkspaceSize

- **Parameters**

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
            <td>query</td>
            <td>Input</td>
            <td>Input Q of the attention structure.</td>
            <td>            The shape dimensions of query, key, value, sparseIndices, dOut, out, softmaxMax, and softmaxSum must remain consistent.
            </td>
            <td>BFLOAT16, FLOAT16</td>
            <td>ND</td>
            <td>(B,S1,N1,D), (T1,N1,D)<br>
            B: supports generalization; S1: supports generalization; D: 512; T1: B × S1<br>
            N1: <term>Atlas A2 products</term> and <term>Atlas A3 products</term> support 1 to 128; <term>Ascend 950PR&950DT products</term> support 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, and 128
            </td>
            <td>√</td>
        </tr>
        <tr>
            <td>key</td>
            <td>Input</td>
            <td>Input K of the attention structure.</td>
            <td>-</td>
            <td>BFLOAT16, FLOAT16</td>
            <td>ND</td>
            <td>(B,S2,N2,D), (T2,N2,D)<br>
            N2: 1; T2: B × S2
            </td>
            <td>√</td>
        </tr>
        <tr>
            <td>value</td>
            <td>Input</td>
            <td>Input v of the attention structure.</td>
            <td>            Optional item. When a null pointer is passed in, KV merge is enabled and value is processed as key internally. In this case, a null pointer must also be passed in for dValueOut, and dKeyOut returns the merged result of dK+dV.
            </td>
            <td>BFLOAT16, FLOAT16</td>
            <td>ND</td>
            <td>(B,S2,N2,D), (T2,N2,D)
            </td>
            <td>√</td>
        </tr>
        <tr>
            <td>sparseIndices</td>
            <td>Input</td>
            <td>Attention indices with higher weights selected in sparse use cases.</td>
            <td>            -
            </td>
            <td>INT32</td>
            <td>ND</td>
            <td>(B,S1,N2,K), (T1,N2,K)<br>
            K: 1024, 2048, 3072, 4096, 5120, 6144, 7168, 8192
            </td>
            <td>√</td>
        </tr>
        <tr>
            <td>dOut</td>
            <td>Input</td>
            <td>Gradient of the attention output matrix.</td>
            <td>            -
            </td>
            <td>BFLOAT16, FLOAT16</td>
            <td>ND</td>
            <td>(B,S1,N1,D), (T1,N1,D)
            </td>
            <td>√</td>
        </tr>
        <tr>
            <td>out</td>
            <td>Input</td>
            <td>Attention output matrix.</td>
            <td>            -
            </td>
            <td>BFLOAT16, FLOAT16</td>
            <td>ND</td>
            <td>(B,S1,N1,D), (T1,N1,D)
            </td>
            <td>√</td>
        </tr>
        <tr>
            <td>softmaxMax</td>
            <td>Input</td>
            <td>Intermediate output of the attention forward computation.</td>
            <td>-</td>
            <td>FLOAT32</td>
            <td>ND</td>
            <td>(B,N2,S1,G), (N2,T1,G)<br>
            G: N1/N2
            </td>
            <td>√</td>
        </tr>
        <tr>
            <td>softmaxSum</td>
            <td>Input</td>
            <td>Intermediate output of the forward attention computation.</td>
            <td>-</td>
            <td>FLOAT32</td>
            <td>ND</td>
            <td>(B,N2,S1,G) or (N2,T1,G)
            </td>
            <td>√</td>
        </tr>
        <tr>
            <td>sinksOptional</td>
            <td>Input</td>
            <td>Learnable OSS Sink input.</td>
            <td>            <ul>
                <li>Optional item. Defaults to a null pointer. Not involved in the computation if not provided. Supported only on <term>Ascend 950PR&950DT products</term>.</li>
                <li>When sinksOptional is passed in, the dSinksOptional output must also be provided.</li>
                <li>Sink is involved in the max/sum computation of forward softmax. For details, see stage 4 in <a href="#function">Function</a>.</li>
            </ul>
            </td>
            <td>FLOAT32</td>
            <td>ND</td>
            <td>(N1,)<br>
            N1: remains consistent with the N1 of query
            </td>
            <td>√</td>
        </tr>
        <tr>
            <td>actualSeqLengthsQueryOptional</td>
            <td>Input</td>
            <td>Number of valid tokens in Query for each batch.</td>
            <td>            <ul>
                <li>Optional item: exists when layout is TND.</li>
                <li>The length must remain consistent with B.</li>
                <li>The cumulative sum must remain consistent with T1 (the cumulative sum can be less than T1 in the EOD scenario).</li>
                <li>The value must be non-negative. Passing in a negative value may trigger a chip alarm.</li>
            </ul>
            </td>
            <td>INT32</td>
            <td>ND</td>
            <td>(B,)</td>
            <td>√</td>
        </tr>
        <tr>
            <td>actualSeqLengthsKvOptional</td>
            <td>Input</td>
            <td>Number of valid tokens for Key and value in each batch.</td>
            <td>            <ul>
                <li>Optional item: This variable exists when layout is TND.</li>
                <li>The length remains consistent with B.</li>
                <li>The cumulative sum remains consistent with T2 (in the EOD scenario, the cumulative sum can be less than T2).</li>
            </ul>
            </td>
            <td>INT32</td>
            <td>ND</td>
            <td>(B,)</td>
            <td>√</td>
        </tr>
        <tr>
            <td>queryRopeOptional</td>
            <td>Input</td>
            <td>MLA rope part: output of Query position encoding.</td>
            <td>            -
            </td>
            <td>BFLOAT16, FLOAT16</td>
            <td>ND</td>
            <td>(B,S1,N1,Dr), (T1,N1,Dr)<br>
            Dr: 64
            </td>
            <td>√</td>
        </tr>
        <tr>
            <td>keyRopeOptional</td>
            <td>Input</td>
            <td>MLA rope part: output of Key position encoding.</td>
            <td>            -
            </td>
            <td>BFLOAT16, FLOAT16</td>
            <td>ND</td>
            <td>(B,S2,N2,Dr),(T2,N2,Dr)
            </td>
            <td>√</td>
        </tr>
        <tr>
            <td>scaleValue</td>
            <td>Input</td>
            <td>Scaling factor.</td>
            <td>Recommended value: reciprocal of the square root of d in the formula.</td>
            <td>FLOAT32</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
        </tr>
        <tr>
            <td>sparseBlockSize</td>
            <td>Input</td>
            <td>Size of the selected block.</td>
            <td>            <term>Atlas A2 products</term>, <term>Atlas A3 products</term> support 1, 8, 16, 32, 64<br>
            <term>Ascend 950PR&950DT products</term> supports 1
            </td>
            <td>INT64</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
        </tr>
        <tr>
            <td>layout</td>
            <td>Input</td>
            <td>Layout format of the input data.</td>
            <td>            BSND and TND are supported.
            </td>
            <td>STRING</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
        </tr>
        <tr>
            <td>sparseMode</td>
            <td>Input</td>
            <td>Sparse mode.</td>
        <td>              <ul>
                <li>Indicates the sparse mode. For details about different sparse modes, see <a href="#constraints">Constraints</a>.</li>
                <li>Only modes 0 and 3 are supported.</li>
              </ul>
        </td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        </tr>
        <tr>
        <td>preTokens</td>
            <td>Input</td>
            <td>Start position of the sliding window on the S matrix in the Attention operator.</td>
            <td>            <ul>
                <li>When sparseMode=4, pre_tokens takes effect.</li>
                <li>Only supported value: 2147483647</li>
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
            <td>End position of the sliding window on the S matrix in the Attention operator.</td>
            <td>            <ul>
                <li>next_tokens takes effect when sparseMode=4.</li>
                <li>Supported value: 2147483647 only.</li>
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
            <td>deterministic computation.</td>
            <td>            Must remain consistent with the whole-network deterministic parameter use_deterministic_algorithms.
            </td>
            <td>BOOL</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
        </tr>
        <tr>
            <td>dQuery</td>
            <td>Output</td>
            <td>Indicates the gradient of query.</td>
            <td>            The dimensions remain consistent with the Shape of the input query.
            </td>
            <td>BFLOAT16, FLOAT16</td>
            <td>ND</td>
            <td>(B,S1,N1,D), (T1,N1,D)
            </td>
            <td>x</td>
        </tr>
        <tr>
            <td>dKey</td>
            <td>Output</td>
            <td>Indicates the gradient of key.</td>
            <td>            In the KV merge scenario, indicates the merged gradient of key and value, that is, dK+dV.
            </td>
            <td>BFLOAT16, FLOAT16</td>
            <td>ND</td>
            <td>(B,S2,N2,D), (T2,N2,D)
            </td>
            <td>x</td>
        </tr>
        <tr>
            <td>dValue</td>
            <td>Output</td>
            <td>Indicates the gradient of value.</td>
            <td>            Optional item. In the normal scenario, the Shape dimensions remain consistent with those of the input value. In the KV merge scenario, a null pointer must be passed in. dValue is not output separately, and its gradient is merged into dKey (supported by Ascend 950PR&950DT products and Atlas A2 training/inference products; on 950, dValueOut must be passed in as a null pointer).
            </td>
            <td>BFLOAT16, FLOAT16</td>
            <td>ND</td>
            <td>(B,S2,N2,D), (T2,N2,D)</td>
            <td>x</td>
        </tr>
          <tr>
            <td>dQueryRopeOptional</td>
            <td>Output</td>
            <td>Indicates the gradient of queryRope.</td>
            <td>            <ul>
                <li>This variable is output only when the input queryRope exists.</li>
                <li>The shape dimensions remain consistent with the input query.</li>
            </ul>
            </td>
            <td>BFLOAT16, FLOAT16</td>
            <td>ND</td>
            <td>(B,S1,N1,Dr), (T1,N1,Dr)
            </td>
            <td>x</td>
        </tr>
        <tr>
            <td>dKeyRopeOptional</td>
            <td>Output</td>
            <td>Indicates the gradient of keyRope.</td>
            <td>            This variable is output only when the input keyRope exists.
            </td>
            <td>BFLOAT16, FLOAT16</td>
            <td>ND</td>
            <td>(B,S2,N2,Dr), (T2,N2,Dr)
            </td>
            <td>x</td>
        </tr>
        <tr>
            <td>dSinksOptional</td>
            <td>Output</td>
            <td>Indicates the gradient of sinksOptional.</td>
            <td>            <ul>
                <li>This variable is output only when the input sinksOptional exists.</li>
                <li>The shape is consistent with sinksOptional.</li>
            </ul>
            </td>
            <td>FLOAT32</td>
            <td>ND</td>
            <td>(N1,)</td>
            <td>x</td>
        </tr>
        </tbody>
    </table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

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
                <td>A required parameter or output is a null pointer. In the KV merge use case, value and dValueOut can be null pointers.</td>
            </tr>
            <tr>
                <td>ACLNN_ERR_PARAM_INVALID</td>
                <td>161002</td>
                <td>The data type or data format of input variables such as query, key, value, and sparseIndices is out of the supported range.</td>
            </tr>
            <tr>
                <td>ACLNN_ERR_RUNTIME_ERROR</td>
                <td>361001</td>
                <td>An exception occurs when the API calls the npu runtime API for memory operations.</td>
            </tr>
            <tr>
                <td>ACLNN_ERR_INNER_TILING_ERROR</td>
                <td>561002</td>
                <td>The value of an input parameter (such as layout or sparseMode) exceeds the supported range, or the shape dimensions of the input tensor do not meet the constraints.</td>
            </tr>
        </tbody>
    </table>

## aclnnSparseFlashAttentionGradV2

- **Parameters**

    <table style="undefined;table-layout: fixed; width: 1155px"><colgroup>
    <col style="width: 144px">
    <col style="width: 125px">
    <col style="width: 700px">
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
        <td>Workspace memory address allocated on the Device.</td>
        </tr>
        <tr>
        <td>workspaceSize</td>
        <td>Input</td>
        <td>Workspace size allocated on the Device, obtained by calling the first-phase API aclnnSparseFlashAttentionGradV2GetWorkspaceSize.</td>
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

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnSparseFlashAttentionGradV2 defaults to a non-deterministic implementation. To enable deterministic computation, use aclrtSetSysParamOpt.
- General constraints:
    - Handling of empty input parameters:
        - query is an empty tensor: return directly.
    - Currently, only the scenario where value and key are identical is supported.
    - In the KV merge use case (supported by Ascend 950PR&950DT products and Atlas A2 training/inference products), null pointers must be passed in for value and dValueOut. dKeyOut returns dK+dV, and dValue is no longer returned separately.
    - OSS Sink: If sinksOptional is passed in but dSinksOptional is not provided, ACLNN_ERR_PARAM_INVALID is returned.
    - OSS Sink: sinksOptional is supported only on <term>Ascend 950PR&950DT products</term>. Passing in sinksOptional on other platforms returns ACLNN_ERR_RUNTIME_ERROR.

- Mask
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
            <td>No mask operation.</td>
            <td>Supported</td>
        </tr>
        <tr>
            <td>1</td>
            <td>allMask, the complete attenmask matrix must be passed in</td>
            <td>Not supported</td>
        </tr>
        <tr>
            <td>2</td>
            <td>Mask of leftUpCausal mode, the optimized attenmask matrix must be passed in</td>
            <td>Not supported</td>
        </tr>
        <tr>
            <td>3</td>
            <td>Mask of rightDownCausal mode, corresponding to the lower-triangle use case with the right vertex as the division point.</td>
            <td>Supported</td>
  </tr>
        <tr>
            <td>4</td>
            <td>Mask of band mode, the optimized attenmask matrix must be passed in</td>
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
                <th>Item</th>
                <th>Specification</th>
                <th>Description</th>
            </tr>
        </thead>
        <tbody>
        <tr>
            <td>deterministic</td>
            <td>bool</td>
            <td>            Supports deterministic computation.
            </td>
        </tr>
        <tr>
            <td>B</td>
            <td>1-256</td>
            <td>-</td>
        </tr>
        <tr>
            <td>S1, S2</td>
            <td>1-128K</td>
            <td>S1 and S2 support unequal lengths.</td>
        </tr>
        <tr>
            <td>N1</td>
            <td>
            <term>Atlas A2 products</term> and <term>Atlas A3 products</term>: 1 to 128<br>
            <term>Ascend 950PR&950DT products</term>: 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 128
            </td>
            <td>SparseFA is MQA.</td>
        </tr>
        <tr>
            <td>N2</td>
            <td>1</td>
            <td>SparseFA is MQA, Nidx2=1.</td>
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
            <td>K</td>
            <td>1024, 2048, 2051, 3072, 4096, 5120, 6144, 7168, 8192</td>
            <td>K * sparseBlockSize exceeding 100k is not recommended. Otherwise, OOM may occur due to the hardware limit of the internal algorithm.</td>
        </tr>
        <tr>
            <td>layout</td>
            <td>BSND/TND</td>
            <td>query and key/value share the same layout.</td>
        </tr>
        </tbody>
    </table>

## Example

The sample code is as follows (taking Atlas A2 products as an example). It is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```c++
#include <iostream>
#include <vector>
#include <numeric>
#include "acl/acl.h"
#include "aclnnop/aclnn_sparse_flash_attention_grad_v2.h"

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
  // Boilerplate: AscendCL initialization
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
  // Call aclrtMalloc to allocate device-side memory.
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  // Call aclrtMemcpy to copy host-side data to the device-side memory.
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // Calculate the strides of the contiguous tensor.
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
  // 1. (Boilerplate) Initialize device/context/stream. For details, see the list of AscendCL external APIs.
  // Set deviceId according to the actual device.
  int32_t deviceId = 0;
  aclrtContext context;
  aclrtStream stream;
  auto ret = Init(deviceId, &context, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the input and output based on the API definition.
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
  std::vector<int64_t> sinksShape = {16};                   // N1

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
  void* sinksDeviceAddr = nullptr;
  void* dSinksDeviceAddr = nullptr;

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
  aclTensor* sinks = nullptr;
  aclTensor* dSinks = nullptr;

  std::vector<short> qHostData(1 * 16 * 512, 1.0);
  std::vector<short> kHostData(2048 * 1 * 512, 1.0);
  std::vector<short> vHostData(2048 * 1 * 512, 1.0);
  std::vector<int32_t> sparseIndicesHostData(2048);
  std::iota(sparseIndicesHostData.begin(), sparseIndicesHostData.end(), 0);
  std::vector<short> outHostData(1 * 16 * 512, 1.0);
  std::vector<short> dOutHostData(1 * 16 * 512, 1.0);
  std::vector<float> softmaxMaxHostData(16, 3.0);
  std::vector<float> softmaxSumHostData(16, 3.0);
  std::vector<int32_t> actSeqQLenHostData(1, 1);
  std::vector<int32_t> actSeqKvLenHostData(1, 2048);
  std::vector<short> qRopeHostData(1 * 16 * 64, 1.0);
  std::vector<short> kRopeHostData(2048 * 1 * 64, 1.0);
  std::vector<short> dqHostData(1 * 16 * 512, 0);
  std::vector<short> dkHostData(2048 * 1 * 512, 0);
  std::vector<short> dvHostData(2048 * 1 * 512, 0);
  std::vector<short> dqRopeHostData(1 * 16 * 64, 0);
  std::vector<short> dkRopeHostData(2048 * 1 * 64, 0);
  std::vector<float> sinksHostData(16, 0.0);
  std::vector<float> dSinksHostData(16, 0.0);

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
  ret = CreateAclTensor(actSeqQLenHostData, actSeqQLenshape, &actSeqQLenDeviceAddr, aclDataType::ACL_INT32, &actSeqQLen);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(actSeqKvLenHostData, actSeqKvLenshape, &actSeqKvLenDeviceAddr, aclDataType::ACL_INT32, &actSeqKvLen);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(qRopeHostData, qRopeShape, &qRopeDeviceAddr, aclDataType::ACL_FLOAT16, &qRope);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(kRopeHostData, kRopeShape, &kRopeDeviceAddr, aclDataType::ACL_FLOAT16, &kRope);
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
  ret = CreateAclTensor(sinksHostData, sinksShape, &sinksDeviceAddr, aclDataType::ACL_FLOAT, &sinks);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(dSinksHostData, sinksShape, &dSinksDeviceAddr, aclDataType::ACL_FLOAT, &dSinks);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  double scaleValue = 0.044194;
  int64_t sparseBlockSize = 1;
  int64_t sparseMode = 0;
  int64_t preTokens = 2147483647;
  int64_t nextTokens = 2147483647;
  bool deterministic = false;
  char layout[5] = {'T', 'N', 'D', 0};

  // 3. Call the CANN operator library API. Replace it with the specific API name.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  // Call the first-phase API of aclnnSparseFlashAttentionGradV2.
  ret = aclnnSparseFlashAttentionGradV2GetWorkspaceSize(q, k, v, sparseIndices, dOut, out, softmaxMax, softmaxSum, sinks,
            actSeqQLen, actSeqKvLen, qRope, kRope, scaleValue, sparseBlockSize, layout, sparseMode,
            preTokens, nextTokens, deterministic, dq, dk, dv, dqRope, dkRope, dSinks, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSparseFlashAttentionGradV2GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  // Allocate device memory using the workspaceSize calculated by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }

  // Call the second-phase API of aclnnSparseFlashAttentionGradV2.
  ret = aclnnSparseFlashAttentionGradV2(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSparseFlashAttentionGradV2 failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate) Synchronize and wait for the task to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value. Copy the result from the device memory to the host memory. Modify the code based on the specific API definition.
  PrintOutResult(qShape, &dqDeviceAddr);
  PrintOutResult(kShape, &dkDeviceAddr);
  PrintOutResult(vShape, &dvDeviceAddr);
  PrintOutResult(qRopeShape, &dqRopeDeviceAddr);
  PrintOutResult(kRopeShape, &dkRopeDeviceAddr);
  PrintOutResult(sinksShape, &dSinksDeviceAddr);

  // 6. Release aclTensor and aclScalar. Modify the code based on the specific API definition.
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
  aclDestroyTensor(sinks);
  aclDestroyTensor(dSinks);

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
  aclrtFree(sinksDeviceAddr);
  aclrtFree(dSinksDeviceAddr);
  aclrtFree(actSeqQLenDeviceAddr);
  aclrtFree(actSeqKvLenDeviceAddr);
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
