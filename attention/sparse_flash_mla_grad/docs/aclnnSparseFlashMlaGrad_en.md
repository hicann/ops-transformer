# aclnnSparseFlashMlaGrad

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-21T13:24:17.540Z -->

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

- **Description**: Computes the backward output of attention in the `SparseFlashMla` training use case. Supports Sliding Window Attention, Compressed Attention, and Sparse Compressed Attention.

- **Formula**:

    Phase 1: Selects the computation use case based on whether cmp_kv and cmp_sparse_indices are passed in, and selects ori_kv or cmp_kv as the input.

    - SWA (cmp_kv is empty):

    $$
    selectedKv\text{ }=\text{ }orikv
    $$

    - SCFA (Both cmp_kv and cmp_sparse_indices are not empty):

    $$
    selectedKv\text{ }=concat(oriKv, \text{ }Gather \left( cmpkv,topkIndices \left[ i \left]  \left)) ,\text{ }0\text{ } < =i < \text{ }selectBlockCount\right. \right. \right. \right.
    $$

    - CFA (cmp_kv is not empty and cmp_sparse_indices is empty):

    $$
    selectedKv\text{ }=concat(oriKv, \text{ }cmpkv)
    $$

    Stage 2: Compute P, dP, dS

    $$
    P = SimpleSoftmax(Mask(Q \text{ }@\text{ } selectedKv^{{T}} \cdot \text{ } scale), lse)
    $$

    $$
    dP = dO \text{ }@\text{ } selectedKv^{{T}}
    $$

    $$
    dS = P \times (dP\text{ } -\text{ } SoftmaxGrad(dO, O))
    $$

    Stage 3: Compute dQ, dKV, dSinks

    $$
    dQ = dS \text{ } @ \text{ } selectedKv \text{ }  \cdot \text{ } scale
    $$

    $$
    dKV = dS^{{T}} \text{ } @ \text{ } Q \text{ } \cdot \text{ } scale + P^{{T}}  @ \text{ } dO
    $$

    $$
    dSinks = ReduceSum(-P \text{ }\times\text{ } dP \text{ }\times\text{ } SimpleSoftmax(sinks, lse), dim=-1)
    $$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call the "aclnnSparseFlashMlaGradGetWorkspaceSize" API first to obtain the workspace size required for computation and the executor containing the operator computation process. Then call the "aclnnSparseFlashMlaGrad" API to perform the computation.

```c++
aclnnStatus aclnnSparseFlashMlaGradGetWorkspaceSize(
    const aclTensor   *query,
    const aclTensor   *dOut,
    const aclTensor   *out,
    const aclTensor   *lse,
    const aclTensor   *oriKvOptional,
    const aclTensor   *cmpKvOptional,
    const aclTensor   *oriSparseIndicesOptional,
    const aclTensor   *cmpSparseIndicesOptional,
    const aclTensor   *cuSeqlensQOptional,
    const aclTensor   *cuSeqlensOriKvOptional,
    const aclTensor   *cuSeqlensCmpKvOptional,
    const aclTensor   *sequsedQOptional,
    const aclTensor   *sequsedOriKvOptional,
    const aclTensor   *sequsedCmpKvOptional,
    const aclTensor   *cmpResidualKvOptional,
    const aclTensor   *oriTopkLengthOptional,
    const aclTensor   *cmpTopkLengthOptional,
    const aclTensor   *sinksOptional,
    const aclTensor   *metadataOptional,
    double             scaleValue,
    int64_t            cmpRatio,
    int64_t            oriMaskMode,
    int64_t            cmpMaskMode,
    int64_t            oriWinLeft,
    int64_t            oriWinRight,
    char              *layoutQOptional,
    char              *layoutKvOptional,
    const aclTensor   *dQueryOut,
    const aclTensor   *dOriKvOutOptional,
    const aclTensor   *dCmpKvOutOptional,
    const aclTensor   *dSinksOutOptional,
    const aclTensor   *oriSoftmaxL1NormOptional,
    const aclTensor   *cmpSoftmaxL1NormOptional,
    uint64_t          *workspaceSize,
    aclOpExecutor    **executor);
```

```c++
aclnnStatus aclnnSparseFlashMlaGrad(
    void             *workspace,
    uint64_t          workspaceSize,
    aclOpExecutor    *executor,
    aclrtStream       stream);
```

## aclnnSparseFlashMlaGradGetWorkspaceSize

- **Parameters:**

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
            <td>query (aclTensor*)</td>
            <td>Input</td>
            <td>Input Q of the attention structure.</td>
            <td>            The shape dimensions of query, oriKvOptional, cmpKvOptional, dOut, out, lse, oriSparseIndicesOptional, and cmpSparseIndicesOptional must be consistent.
            </td>
            <td>BFLOAT16, FLOAT16</td>
            <td>ND</td>
            <td>(B,S1,N1,D), (T1,N1,D)<br>
            B: support generalization; S1: support generalization; N1: support 1-128; D: 512; T1: B × S1
            </td>
            <td>√</td>
        </tr>
        <tr>
            <td>dOut(aclTensor*)</td>
            <td>Input</td>
            <td>Gradient of the attention output matrix.</td>
            <td>            -
            </td>
            <td>BFLOAT16, FLOAT16</td>
            <td>ND</td>
            <td>(B,S1,N1,D), (T1,N1,D)<br>
            B: Keep consistent with the B of query; S1: Keep consistent with the S1 of query; N1: Keep consistent with the N1 of query; D: 512; T1: B × S1
            </td>
            <td>√</td>
        </tr>
        <tr>
            <td>out (aclTensor*)</td>
            <td>Input</td>
            <td>Attention output matrix.</td>
            <td>            -
            </td>
            <td>BFLOAT16, FLOAT16</td>
            <td>ND</td>
            <td>(B,S1,N1,D),(T1,N1,D)<br>
            Shape is consistent with dOut.
            </td>
            <td>√</td>
        </tr>
        <tr>
            <td>lse (aclTensor*)</td>
            <td>Input</td>
            <td>Output lse of forward attention computation. For the formula, see the sparse_flash_mla document.</td>
            <td>            -
            </td>
            <td>FLOAT32</td>
            <td>ND</td>
            <td>(B,N2,S1,G),(N2,T1,G)<br>
            B: consistent with B of query; N2: 1; S1: consistent with S1 of query; G: N1/N2; T1: B × S1
            </td>
            <td>√</td>
        </tr>
        <tr>
            <td>oriKvOptional (aclTensor*)</td>
            <td>Input</td>
            <td>Original input K(V) of the attention structure.</td>
            <td>            Currently, empty tensors are not supported.
            </td>
            <td>BFLOAT16, FLOAT16</td>
            <td>ND</td>
            <td>(B,S2,N2,D), (T2,N2,D)<br>
            B: Keep Consistent with the B of query; S2: Support Generalization; N2: 1; D: 512; T2: B × S2
            </td>
            <td>√</td>
        </tr>
        <tr>
            <td>cmpKvOptional (aclTensor*)</td>
            <td>Input</td>
            <td>K(V) compressed by the Compressor.</td>
            <td>            Supports empty tensors. In this case, computation follows the SWA use case in the formula.
            </td>
            <td>BFLOAT16, FLOAT16</td>
            <td>ND</td>
            <td>(B,S3,N2,D), (T3,N2,D)<br>
            B: Keep consistent with the B of query; S3 = S2 // cmpRatio; N2: 1; D: 512; T3: B × S3
            </td>
            <td>√</td>
        </tr>
        <tr>
            <td>oriSparseIndicesOptional (aclTensor*)</td>
            <td>Input</td>
            <td>Attention indices with higher weights selected from oriKvOptional in the sparse use case.</td>
            <td>            Sparse computation on ori_kv is currently not supported. Therefore, this parameter does not take effect.
            </td>
            <td>INT32</td>
            <td>ND</td>
            <td>(B,S1,N2,K1), (T1,N2,K1)<br>
            B: Keep Consistent with the B of query; S1: Keep Consistent with the S1 of query; N2: 1; K1: Support Generalization; T1: B × S1
            </td>
            <td>√</td>
        </tr>
        <tr>
            <td>cmpSparseIndicesOptional(aclTensor*)</td>
            <td>Input</td>
            <td>Attention indices with higher weights selected from cmpKvOptional in sparse use cases.</td>
            <td>            Supports empty tensors:
            <ul>
                <li>If cmpKvOptional is not empty, perform CFA scenario computation.</li>
                <li>If cmpKvOptional is empty, perform SWA scenario computation.</li>
            </ul>
            </td>
            <td>INT32</td>
            <td>ND</td>
            <td>(B,S1,N2,K2), (T1,N2,K2)<br>
            B: keep consistent with B of query; S1: keep consistent with S1 of query; N2: 1; K2: Support Generalization; T1: B × S1
            </td>
            <td>√</td>
        </tr>
        <tr>
            <td>cuSeqlensQOptional (aclTensor*)</td>
            <td>Input</td>
            <td>Number of effective Query tokens in each Batch.</td>
            <td>            <ul>
                <li>Optional Item: This variable exists when layout is TND.</li>
                <li>Length and B+1 keep consistent.</li>
                <li>Accumulated sum and T1 keep consistent.</li>
            </ul>
            </td>
            <td>INT32</td>
            <td>ND</td>
            <td>(B+1,)</td>
            <td>√</td>
        </tr>
        <tr>
            <td>cuSeqlensOriKvOptional (aclTensor*)</td>
            <td>Input</td>
            <td>Number of effective tokens in oriKvOptional for each batch.</td>
            <td>            <ul>
                <li>Optional Item: this variable exists when layout is TND.</li>
                <li>The length is consistent with B+1.</li>
                <li>The accumulated sum is consistent with T2.</li>
            </ul>
            </td>
            <td>INT32</td>
            <td>ND</td>
            <td>(B+1,)</td>
            <td>√</td>
        </tr>
        <tr>
            <td>cuSeqlensCmpKvOptional (aclTensor*)</td>
            <td>Input</td>
            <td>Number of effective tokens in cmpKvOptional for each Batch.</td>
            <td>            <ul>
                <li>Optional item: this variable exists when layout is TND.</li>
                <li>Length and B+1 keep consistent.</li>
                <li>Accumulated sum and T3 keep consistent.</li>
            </ul>
            </td>
            <td>INT32</td>
            <td>ND</td>
            <td>(B+1,)</td>
            <td>√</td>
        </tr>
        <tr>
            <td>sequsedQOptional (aclTensor*)</td>
            <td>Input</td>
            <td>Indicates the number of query tokens that actually participate in the operation in different batches.</td>
            <td>            Length is B.
            </td>
            <td>INT32</td>
            <td>ND</td>
            <td>(B,)<br>
            </td>
            <td>√</td>
        </tr>
        <tr>
            <td>sequsedOriKvOptional (aclTensor*)</td>
            <td>Input</td>
            <td>Indicates the number of tokens of oriKvOptional that actually participate in the operational computation in different batches.</td>
            <td>            Length is B.
            </td>
            <td>INT32</td>
            <td>ND</td>
            <td>(B,)<br>
            </td>
            <td>√</td>
        </tr>
        <tr>
            <td>sequsedCmpKvOptional(aclTensor*)</td>
            <td>Input</td>
            <td>Indicates the number of tokens of cmpKvOptional that actually participate in the computation in different batches.</td>
            <td>            Length is B.
            </td>
            <td>INT32</td>
            <td>ND</td>
            <td>(B,)<br>
            </td>
            <td>√</td>
        </tr>
        <tr>
            <td>cmpResidualKvOptional (aclTensor*)</td>
            <td>Input</td>
            <td>Indicates the remainder of S2 // cmpRatio for each batch.</td>
            <td>            Must be passed in when cmpKvOptional is not empty and cmpMaskMode=3.
            </td>
            <td>INT32</td>
            <td>ND</td>
            <td>(B,)<br>
            </td>
            <td>√</td>
        </tr>
        <tr>
            <td>oriTopkLengthOptional (aclTensor*)</td>
            <td>Input</td>
            <td>Indicates the actual optional topk length of oriKvOptional for each query row.</td>
            <td>            Required when oriMaskMode=0 and oriSparseIndicesOptional is not empty. The value must be exact.
            </td>
            <td>INT32</td>
            <td>ND</td>
            <td>(B,S1,N2), (T1,N2)<br>
            </td>
            <td>√</td>
        </tr>
        <tr>
            <td>cmpTopkLengthOptional (aclTensor*)</td>
            <td>Input</td>
            <td>Indicates the actual optional topk length of cmpKvOptional for each query row.</td>
            <td>Required when cmpMaskMode=0 and cmpSparseIndicesOptional is not empty. The value must be accurate.
            </td>
            <td>INT32</td>
            <td>ND</td>
            <td>(B,S1,N2) or (T1,N2)<br>
            </td>
            <td>√</td>
        </tr>
        <tr>
            <td>sinksOptional (aclTensor*)</td>
            <td>Input</td>
            <td>Attention sink tensor.</td>
            <td>            -
            </td>
            <td>FLOAT32</td>
            <td>ND</td>
            <td>(N1)<br>
            N1: Keep consistent with the N1 of query.
            </td>
            <td>√</td>
        </tr>
        <tr>
            <td>metadataOptional (aclTensor*)</td>
            <td>Input</td>
            <td>Indicates the output result of the aicpu operator with tiling sinking.</td>
            <td>            -
            </td>
            <td>INT32</td>
            <td>ND</td>
            <td>(x)<br>
            </td>
            <td>√</td>
        </tr>
        <tr>
            <td>scaleValue (double)</td>
            <td>Input</td>
            <td>Scaling factor.</td>
            <td>            Recommended value: reciprocal of the square root of d in the formula.
            </td>
            <td>FLOAT32</td>
            <td>N/A</td>
            <td>-</td>
            <td>-</td>
        </tr>
        <tr>
            <td>cmpRatio (int64_t)</td>
            <td>Input</td>
            <td>Indicates the compression rate of oriKvOptional.</td>
            <td>            Value range: 0-128.
            </td>
            <td>INT64</td>
            <td>N/A</td>
            <td>-</td>
            <td>-</td>
        </tr>
        <tr>
            <td>oriMaskMode (int64_t)</td>
            <td>Input</td>
            <td>Indicates the mask mode for the computation of query and oriKvOptional.</td>
        <td>              <ul>
                <li>Indicates the sparse mode. For details about different sparse modes, see <a href="#constraints">Constraints</a>.</li>
              </ul>
        </td>
        <td>INT64</td>
        <td>N/A</td>
        <td>-</td>
        <td>-</td>
        </tr>
        <tr>
            <td>cmpMaskMode (int64_t)</td>
            <td>Input</td>
            <td>Indicates the mask mode for the computation of query and cmpKvOptional.</td>
        <td>              <ul>
                <li>Indicates the sparse mode. For details about different sparse modes, see <a href="#constraints">Constraints</a>.</li>
              </ul>
        </td>
        <td>INT64</td>
        <td>N/A</td>
        <td>-</td>
        <td>-</td>
        </tr>
        <tr>
        <td>oriWinLeft (int64_t)</td>
            <td>Input</td>
            <td>Indicates the number of past tokens that query attends to in the computation of query and oriKvOptional.</td>
            <td>            <ul>
                <li>Supported value: 127</li>
            </ul>
            </td>
            <td>INT64</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
        </tr>
        <tr>
        <td>oriWinRight (int64_t)</td>
            <td>Input</td>
            <td>Indicate the number of future tokens that query attends to in the computation of query and oriKvOptional.</td>
            <td>            <ul>
                <li>Only 0 is supported.</li>
            </ul>
            </td>
            <td>INT64</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
        </tr>
        <tr>
            <td>layoutQOptional (char*)</td>
            <td>Input</td>
            <td>Indicates the data layout format of the input query.</td>
            <td>            Supported values: "BSND" and "TND"
            </td>
            <td>STRING</td>
            <td>N/A</td>
            <td>-</td>
            <td>-</td>
        </tr>
        <tr>
            <td>layoutKvOptional (char*)</td>
            <td>Input</td>
            <td>Indicates the data layout format of the input ori_kv and cmp_kv.</td>
            <td>            Supports "BSND" and "TND".
            </td>
            <td>STRING</td>
            <td>N/A</td>
            <td>-</td>
            <td>-</td>
        </tr>
        <tr>
            <td>dQueryOut (aclTensor*)</td>
            <td>Output</td>
            <td>Indicates the gradient of query.</td>
            <td>            Keep consistent with the Shape of input query
            </td>
            <td>BFLOAT16, FLOAT16</td>
            <td>ND</td>
            <td>(B,S1,N1,D), (T1,N1,D)<br>
            </td>
            <td>√</td>
        </tr>
        <tr>
            <td>dOriKvOutOptional (aclTensor*)</td>
            <td>Output</td>
            <td>Indicates the gradient of oriKvOptional.</td>
            <td>            Keep Consistent With Input oriKvOptional in Shape Dimensions
            </td>
            <td>BFLOAT16, FLOAT16</td>
            <td>ND</td>
            <td>(B,S2,N2,D), (T2,N2,D)<br>
            </td>
            <td>√</td>
        </tr>
        <tr>
            <td>dCmpKvOptional (aclTensor*)</td>
            <td>Output</td>
            <td>Indicates the gradient of cmpKvOptional.</td>
            <td>            The shape dimensions are the same as those of input cmpKvOptional.
            </td>
            <td>BFLOAT16, FLOAT16</td>
            <td>ND</td>
            <td>(B,S3,N2,D), (T3,N2,D)<br>
            </td>
            <td>√</td>
        </tr>
        <tr>
            <td>dSinksOutOptional (aclTensor*)</td>
            <td>Output</td>
            <td>Indicates the gradient of sinksOptional.</td>
            <td>            Keep consistent with the Shape dimension of input sinksOptional.
            </td>
            <td>FLOAT32</td>
            <td>ND</td>
            <td>(N1)<br>
            </td>
            <td>√</td>
        </tr>
        <tr>
            <td>oriSoftmaxL1NormOptional (aclTensor*)</td>
            <td>Output</td>
            <td>Indicates the softmax L1Norm result computed from query and oriKvOptional. The formula is reduceG(softmax)/G.</td>
            <td>            If oriSparseIndicesOptional exists, this output is not empty; otherwise, this parameter output is empty.
            </td>
            <td>FLOAT32</td>
            <td>ND</td>
            <td>(B,S1,N2,K1), (T1,N2,K1)<br>
            </td>
            <td>√</td>
        </tr>
        <tr>
            <td>cmpSoftmaxL1NormOptional (aclTensor*)</td>
            <td>Output</td>
            <td>Indicates the softmax L1Norm result computed from query and cmpKvOptional. The formula is reduceG(softmax)/G.</td>
            <td>            If cmpSparseIndicesOptional exists, this output is not empty; otherwise, this parameter output is empty.
            </td>
            <td>FLOAT32</td>
            <td>ND</td>
            <td>(B,S1,N2,K2)/(T1,N2,K2)<br>
            </td>
            <td>√</td>
        </tr>
        </tbody>
    </table>

    <!-- npu="A3,910b" id7 -->
    - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>: The oriTopkLengthOptional, cmpTopkLengthOptional, and metadataOptional fields are not supported currently.

    <!-- end id7 -->

- **Return value**

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
                <td>A required parameter or the output is a null pointer.</td>
            </tr>
            <tr>
                <td>ACLNN_ERR_PARAM_INVALID</td>
                <td>161002</td>
                <td>The data type or data format of input variables, such as query, oriKvOptional, and cmpKvOptional, is not within the supported range.</td>
            </tr>
            <tr>
                <td>ACLNN_ERR_RUNTIME_ERROR</td>
                <td>361001</td>
                <td>An exception occurs when the API calls the npu runtime interface for memory.</td>
            </tr>
            <tr>
                <td>ACLNN_ERR_INNER_TILING_ERROR</td>
                <td>561002</td>
                <td>The value of input parameters (such as layout and maskMode) is out of the supported range, or the shape dimensions of the input Tensor do not meet the constraints.</td>
            </tr>
        </tbody>
    </table>

## aclnnSparseFlashMlaGrad

- **Parameters:**

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
        <td>Pointer to the workspace memory allocated on the Device.</td>
        </tr>
        <tr>
        <td>workspaceSize</td>
        <td>Input</td>
        <td>Size of the workspace allocated on the Device, obtained by calling the first-phase API aclnnSparseFlashMlaGradGetWorkspaceSize.</td>
        </tr>
        <tr>
        <td>executor</td>
        <td>Input</td>
        <td>Op executor that contains the computation process of the operator.</td>
        </tr>
        <tr>
        <td>stream</td>
        <td>Input</td>
        <td>Stream for task execution.</td>
        </tr>
    </tbody>
    </table>

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:

    <!-- npu="A3,910b" id8 -->
    - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>: aclnnSparseFlashMlaGrad defaults to a non-deterministic implementation. Enabling determinism through aclrtSetSysParamOpt is not supported.

    <!-- end id8 -->
    <!-- npu="950" id9 -->
    - <term>Ascend 950PR&950DT products</term>: aclnnSparseFlashMlaGrad defaults to a non-deterministic implementation. Enabling determinism through aclrtSetSysParamOpt is supported.
    <!-- end id9 -->

- General Constraints
    - This API does not support non-contiguous output.
    - Handling of empty input parameters:
        - query is an empty tensor: return directly.

- Mask
    <table style="undefined;table-layout: fixed; width: 942px"><colgroup>
        <col style="width: 100px">
        <col style="width: 740px">
        <col style="width: 360px">
        </colgroup>
        <thead>
            <tr>
                <th>maskMode</th>
                <th>Meaning</th>
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
            <td>3</td>
            <td>Mask for the rightDownCausal mode, corresponding to the lower-triangle use case with the right vertex as the dividing point.</td>
            <td>Supported</td>
        </tr>
        <tr>
            <td>4</td>
            <td>band-mode mask. The sliding window range is controlled by oriWinLeft and oriWinRight. The parameters start from the bottom-right corner.</td>
            <td>Supported for ori</td>
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
            <td>B</td>
            <td>Support Generalization</td>
            <td>-</td>
        </tr>
        <tr>
            <td>S1 and S2</td>
            <td>Support Generalization</td>
            <td>Support S1 and S2 with unequal lengths</td>
        </tr>
        <tr>
            <td>N1</td>
            <td>1-128</td>
            <td>-</td>
        </tr>
        <tr>
            <td>N2</td>
            <td>1</td>
            <td>-</td>
        </tr>
        <tr>
            <td>D</td>
            <td>512</td>
            <td>-</td>
        </tr>
        <tr>
            <td>layout</td>
            <td>BSND/TND</td>
            <td>-</td>
        </tr>
        </tbody>
    </table>

  - Supported values of cmpMaskMode:

    <!-- npu="A3,910b" id10 -->
    - <term>Atlas A2 products</term> and <term>Atlas A3 products</term>: cmpMaskMode supports 3.
    <!-- end id10 -->
    <!-- npu="950" id11 -->
    - <term>Ascend 950PR&950DT products</term>: cmpMaskMode supports 0 and 3.

    <!-- end id11 -->

  - Supported values of oriMaskMode:

    <!-- npu="A3,910b" id12 -->
    - <term>Atlas A2 products</term> and <term>Atlas A3 products</term>: oriMaskMode supports 4.
    <!-- end id12 -->
    <!-- npu="950" id13 -->
    - <term>Ascend 950PR&950DT products</term>: oriMaskMode supports 0, 3, and 4.

    <!-- end id13 -->

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```c++
#include <iostream>
#include <vector>
#include <numeric>
#include "acl/acl.h"
#include "aclnnop/aclnn_sparse_flash_mla_grad.h"
#include "aclnnop/aclnn_sparse_flash_mla_grad_metadata.h"

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
  // Call aclrtMalloc to allocate memory on the device.
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  // Call aclrtMemcpy to copy data from the host to the device memory.
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // Compute the strides of the contiguous tensor.
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  // Call the aclCreateTensor API to create an aclTensor.
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

int main() {
  // 1. (Boilerplate) Initialize the device/context/stream. For details, see the acl API manual.
  // Set deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtContext context;
  aclrtStream stream;
  auto ret = Init(deviceId, &context, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the input and output. Customize the construction based on the API.
  std::vector<int64_t> qShape = {1, 16, 512};                // T1, N1, D
  std::vector<int64_t> oriKvShape = {2048, 1, 512};          // T2, N2, D
  std::vector<int64_t> cmpKvShape = {16, 1, 512};            // T3, N2, D
  std::vector<int64_t> dOutShape = {1, 16, 512};             // T1, N1, D
  std::vector<int64_t> outShape = {1, 16, 512};              // T1, N1, D
  std::vector<int64_t> lseShape = {1, 1, 16};                // N2, T1, G
  std::vector<int64_t> cuSeqQLenshape = {2};                 // B + 1
  std::vector<int64_t> cuSeqOriKvLenshape = {2};             // B + 1
  std::vector<int64_t> cuSeqCmpKvLenshape = {2};             // B + 1
  std::vector<int64_t> cmpResidualKvShape = {1};             // B
  std::vector<int64_t> sinksShape = {16};                    // N1
  std::vector<int64_t> metadataShape = {1024};               // Fixed size

  void* qDeviceAddr = nullptr;
  void* oriKvDeviceAddr = nullptr;
  void* cmpKvDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  void* dOutDeviceAddr = nullptr;
  void* lseDeviceAddr = nullptr;
  void* cuSeqQLenDeviceAddr = nullptr;
  void* cuSeqOriKvLenDeviceAddr = nullptr;
  void* cuSeqCmpKvLenDeviceAddr = nullptr;
  void* cmpResidualKvDeviceAddr = nullptr;
  void* sinksDeviceAddr = nullptr;
  void* dqDeviceAddr = nullptr;
  void* dOriKvDeviceAddr = nullptr;
  void* dCmpKvDeviceAddr = nullptr;
  void* dSinksDeviceAddr = nullptr;
  void* metadataDeviceAddr = nullptr;
  void* oriSoftmaxL1NormDeviceAddr = nullptr;
  void* cmpSoftmaxL1NormDeviceAddr = nullptr;

  aclTensor* q = nullptr;
  aclTensor* oriKv = nullptr;
  aclTensor* cmpKv = nullptr;
  aclTensor* out = nullptr;
  aclTensor* dOut = nullptr;
  aclTensor* lse = nullptr;
  aclTensor* cuSeqQLen = nullptr;
  aclTensor* cuSeqOriKvLen = nullptr;
  aclTensor* cuSeqCmpKvLen = nullptr;
  aclTensor* cmpResidualKv = nullptr;
  aclTensor* sinks = nullptr;
  aclTensor* dq = nullptr;
  aclTensor* dOriKv = nullptr;
  aclTensor* dCmpKv = nullptr;
  aclTensor* dSinks = nullptr;
  aclTensor* metadata = nullptr;
  aclTensor* oriSoftmaxL1Norm = nullptr;
  aclTensor* cmpSoftmaxL1Norm = nullptr;

  std::vector<short> qHostData(1 * 16 * 512, 1.0);
  std::vector<short> oriKvHostData(2048 * 1 * 512, 1.0);
  std::vector<short> cmpKvHostData(16 * 1 * 512, 1.0);
  std::vector<short> outHostData(1 * 16 * 512, 1.0);
  std::vector<short> dOutHostData(1 * 16 * 512, 1.0);
  std::vector<float> lseHostData(16, 3.0);
  std::vector<int32_t> cuSeqQLenHostData = {0, 1};
  std::vector<int32_t> cuSeqOriKvLenHostData = {0, 2048};
  std::vector<int32_t> cuSeqCmpKvLenHostData = {0, 16};
  std::vector<int32_t> cmpResidualKvHostData = {0};
  std::vector<float> sinksHostData(16, 128.0);
  std::vector<short> dqHostData(1 * 16 * 512, 0);
  std::vector<short> dOriKvHostData(2048 * 1 * 512, 0);
  std::vector<short> dCmpKvHostData(16 * 1 * 512, 0);
  std::vector<float> dSinksHostData(16, 0);
  std::vector<int32_t> metadataHostData(1024, 0);

  ret = CreateAclTensor(qHostData, qShape, &qDeviceAddr, aclDataType::ACL_FLOAT16, &q);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(oriKvHostData, oriKvShape, &oriKvDeviceAddr, aclDataType::ACL_FLOAT16, &oriKv);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(cmpKvHostData, cmpKvShape, &cmpKvDeviceAddr, aclDataType::ACL_FLOAT16, &cmpKv);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT16, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(dOutHostData, dOutShape, &dOutDeviceAddr, aclDataType::ACL_FLOAT16, &dOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(lseHostData, lseShape, &lseDeviceAddr, aclDataType::ACL_FLOAT, &lse);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(cuSeqQLenHostData, cuSeqQLenshape, &cuSeqQLenDeviceAddr, aclDataType::ACL_INT32, &cuSeqQLen);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(cuSeqOriKvLenHostData, cuSeqOriKvLenshape, &cuSeqOriKvLenDeviceAddr, aclDataType::ACL_INT32, &cuSeqOriKvLen);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(cuSeqCmpKvLenHostData, cuSeqCmpKvLenshape, &cuSeqCmpKvLenDeviceAddr, aclDataType::ACL_INT32, &cuSeqCmpKvLen);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(cmpResidualKvHostData, cmpResidualKvShape, &cmpResidualKvDeviceAddr, aclDataType::ACL_INT32, &cmpResidualKv);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(sinksHostData, sinksShape, &sinksDeviceAddr, aclDataType::ACL_FLOAT, &sinks);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(dqHostData, qShape, &dqDeviceAddr, aclDataType::ACL_FLOAT16, &dq);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(dOriKvHostData, oriKvShape, &dOriKvDeviceAddr, aclDataType::ACL_FLOAT16, &dOriKv);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(dCmpKvHostData, cmpKvShape, &dCmpKvDeviceAddr, aclDataType::ACL_FLOAT16, &dCmpKv);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(dSinksHostData, sinksShape, &dSinksDeviceAddr, aclDataType::ACL_FLOAT, &dSinks);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(metadataHostData, metadataShape, &metadataDeviceAddr, aclDataType::ACL_INT32, &metadata);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  double scaleValue = 0.088388;
  int64_t cmpRatio = 128;
  int64_t oriMaskMode = 4;
  int64_t cmpMaskMode = 3;
  int64_t oriWinLeft = 127;
  int64_t oriWinRight = 0;
  char layoutQ[5] = {'T', 'N', 'D', 0};
  char layoutKv[5] = {'T', 'N', 'D', 0};

  // 3. Call the CANN operator library API and change it to the specific API name.
  // 3.1 Call SparseFlashMlaGradMetadata to obtain the metadata.
  uint64_t metaWorkspaceSize = 0;
  aclOpExecutor* metaExecutor;
  ret = aclnnSparseFlashMlaGradMetadataGetWorkspaceSize(
      cuSeqQLen, cuSeqOriKvLen, cuSeqCmpKvLen,
      nullptr, nullptr, nullptr,
      cmpResidualKv,
      nullptr, nullptr,
      16, 1, 512,
      0, 0, 0, 0,
      0, 0,
      cmpRatio, oriMaskMode, cmpMaskMode, oriWinLeft, oriWinRight,
      layoutQ, layoutKv,
      true, true,
      metadata, &metaWorkspaceSize, &metaExecutor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("SparseFlashMlaGradMetadataGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  void* metaWorkspaceAddr = nullptr;
  if (metaWorkspaceSize > 0) {
    ret = aclrtMalloc(&metaWorkspaceAddr, metaWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate metadata workspace failed. ERROR: %d\n", ret); return ret);
  }

  ret = aclnnSparseFlashMlaGradMetadata(metaWorkspaceAddr, metaWorkspaceSize, metaExecutor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("SparseFlashMlaGradMetadata failed. ERROR: %d\n", ret); return ret);

  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 3.2 Call SparseFlashMlaGrad to compute the gradient.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  ret = aclnnSparseFlashMlaGradGetWorkspaceSize(q, dOut, out, lse, oriKv, cmpKv, nullptr, nullptr, cuSeqQLen, cuSeqOriKvLen, cuSeqCmpKvLen,
            nullptr, nullptr, nullptr, cmpResidualKv, nullptr, nullptr, sinks, metadata, scaleValue, cmpRatio, oriMaskMode, cmpMaskMode, oriWinLeft, oriWinRight,
            layoutQ, layoutKv, dq, dOriKv, dCmpKv, dSinks, oriSoftmaxL1Norm, cmpSoftmaxL1Norm,
            &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("SparseFlashMlaGradGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }

  ret = aclnnSparseFlashMlaGrad(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("SparseFlashMlaGrad failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate) Wait synchronously for the task execution to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value, and copy the result from the device memory to the host. Modify it based on the interface definition of the specific API.
  PrintOutResult(qShape, &dqDeviceAddr);
  PrintOutResult(cmpKvShape, &dCmpKvDeviceAddr);
  PrintOutResult(sinksShape, &dSinksDeviceAddr);

  // 6. Release the aclTensor and aclScalar. Modify it based on the interface definition of the specific API.
  aclDestroyTensor(q);
  aclDestroyTensor(oriKv);
  aclDestroyTensor(cmpKv);
  aclDestroyTensor(out);
  aclDestroyTensor(dOut);
  aclDestroyTensor(lse);
  aclDestroyTensor(cuSeqQLen);
  aclDestroyTensor(cuSeqOriKvLen);
  aclDestroyTensor(cuSeqCmpKvLen);
  aclDestroyTensor(cmpResidualKv);
  aclDestroyTensor(sinks);
  aclDestroyTensor(dq);
  aclDestroyTensor(dOriKv);
  aclDestroyTensor(dCmpKv);
  aclDestroyTensor(dSinks);
  aclDestroyTensor(metadata);
  aclDestroyTensor(oriSoftmaxL1Norm);
  aclDestroyTensor(cmpSoftmaxL1Norm);

  // 7. Release the device resources.
  aclrtFree(qDeviceAddr);
  aclrtFree(oriKvDeviceAddr);
  aclrtFree(cmpKvDeviceAddr);
  aclrtFree(outDeviceAddr);
  aclrtFree(dOutDeviceAddr);
  aclrtFree(lseDeviceAddr);
  aclrtFree(cuSeqQLenDeviceAddr);
  aclrtFree(cuSeqOriKvLenDeviceAddr);
  aclrtFree(cuSeqCmpKvLenDeviceAddr);
  aclrtFree(cmpResidualKvDeviceAddr);
  aclrtFree(sinksDeviceAddr);
  aclrtFree(dqDeviceAddr);
  aclrtFree(dOriKvDeviceAddr);
  aclrtFree(dCmpKvDeviceAddr);
  aclrtFree(dSinksDeviceAddr);
  aclrtFree(metadataDeviceAddr);
  aclrtFree(oriSoftmaxL1NormDeviceAddr);
  aclrtFree(cmpSoftmaxL1NormDeviceAddr);
  if (metaWorkspaceSize > 0) {
    aclrtFree(metaWorkspaceAddr);
  }
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
