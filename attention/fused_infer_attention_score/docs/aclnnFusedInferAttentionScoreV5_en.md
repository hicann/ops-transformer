# aclnnFusedInferAttentionScoreV5

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-21T10:45:29.833Z -->

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

- Description: A FlashAttention operator suitable for both decode and prefill scenarios. It supports the prefill computation scenario (PromptFlashAttention) and the decode computation scenario (IncreFlashAttention).

  Compared with FusedInferAttentionScoreV4, this API adds the qStartIdxOptional, kvStartIdxOptional, and pseType parameters.

  **NOTE**

  KV Cache specific to the decode scenario: KV Cache is a common technique for optimizing inference performance in large models. During sampling, the Transformer model performs inference using the given prompt/context as the initial input (which can be processed in parallel), and then generates additional tokens one by one to further refine the generated sequence (reflecting the autoregressive nature of the model). During the sampling process, the Transformer performs self-attention operations, which requires extracting key-value (KV) vectors for each item in the current sequence (whether prompt/context or generated tokens). These vectors are stored in a matrix, commonly referred to as the KV cache.
- Formula:

  Self-attention builds an attention model using the relationships within the input samples themselves. The principle is as follows: assume there is an input sample sequence $x$ of length $n$, where each element of $x$ is a $d$-dimensional vector. Each $d$-dimensional vector can be regarded as a token embedding. After passing such a sequence through three weight matrices, three matrices of dimension $n*d$ are obtained.

  The self-attention formula is generally defined as follows, where $Q$, $K$, and $V$ are important property elements of the input sample, obtained by spatially transforming the input sample and unified into a single feature space. "Attention" in the formula and operator name is an abbreviation of "self-attention".

  $$
  Attention(Q,K,V)=Score(Q,K)V
  $$

  In this operator, the Score function uses the Softmax function, and the self-attention calculation formula is:

  $$
  Attention(Q,K,V)=Softmax(\frac{QK^T}{\sqrt{d}})V
  $$

  Where the product of $Q$ and $K^T$ represents the attention of input $x$. To prevent this value from becoming too large, it is typically divided by the square root of $d$ for scaling, and each row is normalized by softmax. After multiplying with $V$, an $n*d$ matrix is obtained.

  **Note**:

  <blockquote>The query, key, and value data layout formats can be interpreted from multiple dimensions, where B (Batch) indicates the input sample batch size, S (Seq-Length) indicates the input sample sequence length, H (Hidden-Size) indicates the hidden layer size, N (Head-Num) indicates the number of attention heads, D (Head-Dim) indicates the minimum unit size of the hidden layer, and D=H/N is satisfied, and T indicates the cumulative sum of the sequence lengths of all batch input samples.
    <br>Q_S indicates S in the query shape, KV_S indicates S in the key and value shapes, Q_N indicates num_query_heads, KV_N indicates num_key_value_heads. P indicates the computation result of Softmax(<span>(QK<sup class="superscript">T</sup>) / <span class="sqrt">d</span></span>).</blockquote>

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call "aclnnFusedInferAttentionScoreV5GetWorkspaceSize" first to obtain the input parameters and calculate the required workspace size based on the computation flow, and then call "aclnnFusedInferAttentionScoreV5" to execute the computation.

```c++
aclnnStatus aclnnFusedInferAttentionScoreV5GetWorkspaceSize(
    const aclTensor     *query,
    const aclTensorList *key,
    const aclTensorList *value,
    const aclTensor     *pseShiftOptional,
    const aclTensor     *attenMaskOptional,
    const aclIntArray   *actualSeqLengthsOptional,
    const aclIntArray   *actualSeqLengthsKvOptional,
    const aclTensor     *deqScale1Optional,
    const aclTensor     *quantScale1Optional,
    const aclTensor     *deqScale2Optional,
    const aclTensor     *quantScale2Optional,
    const aclTensor     *quantOffset2Optional,
    const aclTensor     *antiquantScaleOptional,
    const aclTensor     *antiquantOffsetOptional,
    const aclTensor     *blockTableOptional,
    const aclTensor     *queryPaddingSizeOptional,
    const aclTensor     *kvPaddingSizeOptional,
    const aclTensor     *keyAntiquantScaleOptional,
    const aclTensor     *keyAntiquantOffsetOptional,
    const aclTensor     *valueAntiquantScaleOptional,
    const aclTensor     *valueAntiquantOffsetOptional,
    const aclTensor     *keySharedPrefixOptional,
    const aclTensor     *valueSharedPrefixOptional,
    const aclIntArray   *actualSharedPrefixLenOptional,
    const aclTensor     *queryRopeOptional,
    const aclTensor     *keyRopeOptional,
    const aclTensor     *keyRopeAntiquantScaleOptional,
    const aclTensor     *dequantScaleQueryOptional,
    const aclTensor     *learnableSinkOptional,
    const aclIntArray   *qStartIdxOptional,
    const aclIntArray   *kvStartIdxOptional,
    int64_t             numHeads,
    double              scaleValue,
    int64_t             preTokens,
    int64_t             nextTokens,
    char                *inputLayout,
    int64_t             numKeyValueHeads,
    int64_t             sparseMode,
    int64_t             innerPrecise,
    int64_t             blockSize,
    int64_t             antiquantMode,
    bool                softmaxLseFlag,
    int64_t             keyAntiquantMode,
    int64_t             valueAntiquantMode,
    int64_t             queryQuantMode,
    int64_t             pseType,
    const aclTensor     *attentionOut,
    const aclTensor     *softmaxLse,
    uint64_t            *workspaceSize,
    aclOpExecutor       **executor)
```

```c++
aclnnStatus aclnnFusedInferAttentionScoreV5(
    void                *workspace,
    uint64_t            workspaceSize,
    aclOpExecutor       *executor,
    const aclrtStream   stream)
```

## aclnnFusedInferAttentionScoreV5GetWorkspaceSize

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1625px"><colgroup>
    <col style="width: 247px">
    <col style="width: 132px">
    <col style="width: 232px">
    <col style="width: 293px">
    <col style="width: 185px">
    <col style="width: 119px">
    <col style="width: 272px">
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
        <td>query</td>
        <td>Input</td>
        <td>Input Q in the formula.</td>
        <td>-</td>
        <td>FLOAT16, BFLOAT16, INT8, HIFLOAT8, FLOAT8_E4M3FN</td>
        <td>ND</td>
        <td>See the inputLayout parameter</td>
        <td>×</td>
    </tr>
    <tr>
        <td>key</td>
        <td>Input</td>
        <td>Input K in the formula.</td>
        <td>        <ul>
            <li>Partial scenarios support non-contiguous tensors. For details, see <a href="#constraints">Constraints</a>.</li>
        </ul>
        </td>
        <td>FLOAT16, BFLOAT16, INT8, HIFLOAT8, FLOAT8_E4M3FN, INT4(INT32), FLOAT4_E2M1</td>
        <td>ND</td>
        <td>See parameter inputLayout</td>
        <td>×</td>
    </tr>
    <tr>
        <td>value</td>
        <td>Input</td>
        <td>Input V in the formula.</td>
        <td>        <ul>
            <li>Partial scenarios support non-contiguous tensors. For details, see <a href="#constraints">Constraints</a>.</li>
        </ul>
        </td>
        <td>FLOAT16, BFLOAT16, INT8, HIFLOAT8, FLOAT8_E4M3FN, INT4(INT32), FLOAT4_E2M1</td>
        <td>ND</td>
        <td>See the inputLayout parameter.</td>
        <td>×</td>
    </tr>
    <tr>
        <td>pseShiftOptional</td>
        <td>Input</td>
        <td>Position encoding.</td>
        <td>        <ul>
            <li>Does not support empty Tensor.</li>
            <li>Pass in nullptr when this function is not used.</li>
            <li>For comprehensive constraints, see <a href="#constraints">Constraints</a>.</li>
        </ul>
        </td>
        <td>FLOAT16, BFLOAT16</td>
        <td>ND</td>
        <td>Recommended shapes: (B,Q_N,Q_S,KV_S) or (1,Q_N,Q_S,KV_S)</td>
        <td>×</td>
    </tr>
    <tr>
        <td>attenMaskOptional</td>
        <td>Input</td>
        <td>Mask matrix.</td>
        <td>        <ul>
            <li>Does not support empty tensors.</li>
            <li>Pass in nullptr if this function is not used.</li>
            <li>For comprehensive constraints, see <a href="#constraints">Constraints</a>.</li>
        </ul>
        </td>
        <td>BOOL, INT8, UINT8</td>
        <td>ND</td>
        <td>See <a href="#MaskChecker">comprehensive limitations in the Attention Mask parameter group</a>.</td>
        <td>×</td>
    </tr>
    <tr>
        <td>actualSeqLengthsOptional</td>
        <td>Input</td>
        <td>Effective sequence length of query in each batch.</td>
        <td>        <ul>
            <li>If the sequence length is not specified, pass in nullptr, indicating that the sequence length is the same as the S dimension of the query shape.</li>
            <li>For comprehensive constraints, see <a href="#constraints">Constraints</a>.</li>
        </ul>
        </td>
        <td>INT64</td>
        <td>-</td>
        <td>(1) or (B) or (>B)</td>
        <td>-</td>
    </tr>
    <tr>
        <td>actualSeqLengthsKvOptional</td>
        <td>Input</td>
        <td>Effective sequence length of key/value in each batch.</td>
        <td>        <ul>
            <li>If the sequence length is not specified, nullptr can be passed in, indicating that it has the same length as the S dimension in the shape of key/value.</li>
            <li>For comprehensive constraints, see <a href="#constraints">Constraints</a>.</li>
        </ul>
        </td>
        <td>INT64</td>
        <td>-</td>
        <td>(1) or (B) or (>B)</td>
        <td>-</td>
    </tr>
    <tr>
        <td>deqScale1Optional</td>
        <td>Input</td>
        <td>Dequantization factor after BMM1.</td>
        <td>        <ul>
            <li>Does not support empty tensors.</li>
            <li>Supports per-tensor.</li>
            <li>When the full quantization function is used, this parameter is calculated from the actual quantization process.</li>
            <li>When this function is not used, nullptr can be passed in.</li>
            <li>For comprehensive constraints, see <a href="#constraints">Constraints</a>.</li>
        </ul>
        </td>
        <td>UINT64, FLOAT32</td>
        <td>ND</td>
        <td>See <a href="#DequantChecker">Comprehensive Limitations in the Pseudo Quantization/Full Quantization Parameters</a>.</td>
        <td>-</td>
    </tr>
    <tr>
        <td>quantScale1Optional</td>
        <td>Input</td>
        <td>Quantization factor before BMM2.</td>
        <td>        <ul>
            <li>Does not support empty tensor.</li>
            <li>Supports per-tensor. </li>
            <li>Supports MxFP8. </li>
            <li>The supported range must be [1, 448]. </li>
            <li>If you do not use this function, you can pass in nullptr.</li>
            <li>For comprehensive constraints, see <a href="#constraints">Constraints</a>.</li>
        </ul>
        </td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>See <a href="#DequantChecker">Comprehensive Limitations in the Pseudo Quantization/Full Quantization Parameters</a>.</td>
        <td>-</td>
    </tr>
    <tr>
        <td>deqScale2Optional</td>
        <td>Input</td>
        <td>Dequantization factor after BMM2.</td>
        <td>        <ul>
            <li>Does not support empty tensors.</li>
            <li>Supports per-tensor. </li>
            <li>When the full quantization function is used, this parameter is obtained from the actual quantization process.</li>
            <li>Pass in nullptr if this function is not used.</li>
            <li>For comprehensive constraints, see <a href="#constraints">Constraints</a>.</li>
        </ul>
        </td>
        <td>UINT64, FLOAT32</td>
        <td>ND</td>
        <td>See <a href="#DequantChecker">Comprehensive Limitations in Pseudo Quantization Parameters/Full Quantization Parameters</a>.</td>
        <td>-</td>
    </tr>
    <tr>
        <td>quantScale2Optional</td>
        <td>Input</td>
        <td>Quantization factor of the output.</td>
        <td>        <ul>
            <li>Does not support empty tensors.</li>
            <li>Supports per-tensor and per-channel. </li>
            <li>Pass in nullptr when this function is not used.</li>
            <li>For comprehensive constraints, see <a href="#constraints">Constraints</a>.</li>
        </ul>
        </td>
        <td>FLOAT32, BFLOAT16</td>
        <td>ND</td>
        <td>See <a href="#PostQuantChecker">Comprehensive Limitations in the Post-quantization Parameter Group</a>.</td>
        <td>-</td>
    </tr>
    <tr>
        <td>quantOffset2Optional</td>
        <td>Input</td>
        <td>Quantization offset of the output.</td>
        <td>        <ul>
            <li>Does not support empty tensors.</li>
            <li>Supports per-tensor and per-channel.</li>
            <li>Pass in nullptr when this function is not used.</li>
            <li>For comprehensive constraints, see <a href="#constraints">Constraints</a>.</li>
        </ul>
        </td>
        <td>FLOAT32, BFLOAT16</td>
        <td>ND</td>
        <td>Keep consistent with quantScale2Optional.</td>
        <td>-</td>
    </tr>
    <tr>
        <td>antiquantScaleOptional</td>
        <td>Input</td>
        <td>Pseudo Quantization Factor.</td>
        <td>Not supported</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>antiquantOffsetOptional</td>
        <td>Input</td>
        <td>Pseudo quantization offset.</td>
        <td>Not supported</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>blockTableOptional</td>
        <td>Input</td>
        <td>Block mapping table used for KV storage in PageAttention.</td>
        <td>        <ul>
            <li>Does not support empty tensor.</li>
            <li>Pass in nullptr when this function is not used.</li>
        </ul>
        </td>
        <td>INT32</td>
        <td>ND</td>
        <td>The first dimension length must be equal to B, and the second dimension length cannot be less than maxBlockNumPerSeq (maxBlockNumPerSeq is the number of blocks corresponding to the maximum actualSeqLengthsKv among different batches).</td>
        <td>-</td>
    </tr>
    <tr>
        <td>queryPaddingSizeOptional</td>
        <td>Input</td>
        <td>Indicates whether the data of each batch in Query is right-aligned, and the number of right-aligned elements.</td>
        <td>        <ul>
            <li>Does not support empty tensors.</li>
            <li>Supported only when Q_S is greater than 1. This parameter is invalid in other use cases.</li>
            <li>Pass in nullptr when this function is not used.</li>
        </ul>
        </td>
        <td>INT64</td>
        <td>ND</td>
        <td>(1)</td>
        <td>-</td>
    </tr>
    <tr>
        <td>kvPaddingSizeOptional</td>
        <td>Input</td>
        <td>Indicates whether the data of each batch in key/value is right-aligned, and the number of right-aligned elements.</td>
        <td>        <ul>
            <li>Does not support empty tensors.</li>
            <li>Can pass in nullptr when this function is not used.</li>
        </ul>
        </td>
        <td>INT64</td>
        <td>ND</td>
        <td>(1)</td>
        <td>-</td>
    </tr>
    <tr>
        <td>keyAntiquantScaleOptional</td>
        <td>Input</td>
        <td>Indicates the dequantization factor of key, used in the KV pseudo quantization parameter separation use case.</td>
        <td>        <ul>
            <li>Does not support empty tensors.</li>
            <li>Can pass in nullptr when this function is not used.</li>
            <li>Supports non-contiguous tensors in partial scenarios. For details, see <a href="#constraints">Constraints</a>.</li>
            <li>Supports per-tensor, per-channel, per-token, per-token-group, per-tensor stacked with per-head, per-token stacked with per-head, per-token stacked with the page attention mode for managing scale/offset, and per-token stacked with per-head and the page attention mode for managing scale/offset and per-token-group.</li>
            <li>For comprehensive constraints, see <a href="#constraints">Constraints</a>.</li>
        </ul>
        </td>
        <td>FLOAT16, BFLOAT16, FLOAT32, FLOAT8_E8M0</td>
        <td>ND</td>
        <td>See <a href="#constraints">Constraints</a>.</td>
        <td>×</td>
    </tr>
    <tr>
        <td>keyAntiquantOffsetOptional</td>
        <td>Input</td>
        <td>Indicates the dequantization offset of key when KV pseudo quantization parameters are separated.</td>
        <td>        <ul>
            <li>Does not support empty tensors.</li>
            <li>If used, the shape must keep consistent with that of keyAntiquantScaleOptional.</li>
            <li>Pass in nullptr when this function is not used.</li>
            <li>Supports non-contiguous tensors in partial scenarios. For details, see <a href="#constraints">Constraints</a>.</li>
            <li>Supports per-tensor, per-channel, per-token, per-channel-group, per-tensor stacked with per-head, per-token stacked with per-head, per-token stacked with page attention mode for scale/offset management, and per-token stacked with per-head and page attention mode for scale/offset management.</li>
            <li>For comprehensive constraints, see <a href="#constraints">Constraints</a>.</li>
        </ul>
        </td>
        <td>FLOAT16, BFLOAT16, FLOAT32</td>
        <td>ND</td>
        <td>See <a href="#constraints">Constraints</a>.</td>
        <td>-</td>
    </tr>
    <tr>
        <td>valueAntiquantScaleOptional</td>
        <td>Input</td>
        <td>Indicates the dequantization factor of value, used in the KV pseudo quantization parameter separation use case.</td>
        <td>        <ul>
            <li>Does not support empty tensors.</li>
            <li>Pass in nullptr when this function is not used.</li>
            <li>Supports non-contiguous tensors in partial scenarios. For details, see <a href="#constraints">Constraints</a>.</li>
            <li>Supports per-tensor, per-channel, per-token, per-tensor stacked with per-head, per-token stacked with per-head, per-token stacked with the use of page attention mode to manage scale/offset, per-token stacked with per-head and page attention mode to manage scale/offset, and per-token-group.</li>
            <li>For comprehensive constraints, see <a href="#constraints">Constraints</a>.</li>
        </ul>
        </td>
        <td>FLOAT16, BFLOAT16, FLOAT32, FLOAT8_E8M0</td>
        <td>ND</td>
        <td>See <a href="#constraints">Constraints</a>.</td>
        <td>×</td>
    </tr>
    <tr>
        <td>valueAntiquantOffsetOptional</td>
        <td>Input</td>
        <td>Indicates the dequantization factor of value when KV pseudo quantization parameters are separated.</td>
        <td>        <ul>
            <li>Does not support empty tensors.</li>
            <li>When used, the shape must keep consistent with valueAntiquantScaleOptional.</li>
            <li>Pass in nullptr when this function is not used.</li>
            <li>Supports non-contiguous tensors in partial scenarios. For details, see <a href="#constraints">Constraints</a>.</li>
            <li>Supports per-tensor, per-channel, per-token, per-tensor stacked with per-head, per-token stacked with per-head, per-token stacked with PagedAttention mode for scale/offset management, and per-token stacked with per-head and PagedAttention mode for scale/offset management.</li>
            <li>For comprehensive constraints, see <a href="#constraints">Constraints</a>.</li>
        </ul>
        </td>
        <td>FLOAT16, BFLOAT16, FLOAT32</td>
        <td>ND</td>
        <td>See <a href="#constraints">Constraints</a>.</td>
        <td>-</td>
    </tr>
    <tr>
        <td>keySharedPrefixOptional</td>
        <td>Input</td>
        <td>Parameters of the system prefix part of the key in the attention structure.</td>
        <td>        <ul>
            <li>Does not support empty tensors.</li>
            <li>Can be nullptr when this function is not used.</li>
            <li>For comprehensive constraints, see <a href="#constraints">Constraints</a>.</li>
        </ul>
        </td>
        <td>FLOAT16, BFLOAT16, INT8, INT4/INT32</td>
        <td>ND</td>
        <td>        <ul>
            <li>When input_layout is BSH, the shape is (1, prefix_S, H=KV_N*KV_D)</li>
            <li>When input_layout is BSND, the shape is (1, prefix_S, KV_N, KV_D)</li>
            <li>When input_layout is BNSD or BNSD_BSND, the shape is (1, KV_N, prefix_S, KV_D)</li>
        </ul>
        </td>
        <td>×</td>
    </tr>
    <tr>
        <td>valueSharedPrefixOptional</td>
        <td>Input</td>
        <td>Input of the system prefix part of Value in the attention structure.</td>
        <td>        <ul>
            <li>Does not support empty tensors.</li>
            <li>Can be nullptr when this function is not used.</li>
            <li>For comprehensive constraints, see <a href="#constraints">Constraints</a>.</li>
        </ul>
        </td>
        <td>FLOAT16, BFLOAT16, INT8, INT4/INT32</td>
        <td>ND</td>
        <td>        <ul>
            <li>When input_layout is BSH, the shape is (1, prefix_S, H=KV_N*KV_D)</li>
            <li>When input_layout is BSND, the shape is (1, prefix_S, KV_N, KV_D)</li>
            <li>When input_layout is BNSD or BNSD_BSND, the shape is (1, KV_N, prefix_S, KV_D)</li>
        </ul>
        </td>
        <td>×</td>
    </tr>
    <tr>
        <td>actualSharedPrefixLenOptional</td>
        <td>Input</td>
        <td>Valid Sequence Length of keySharedPrefix/valueSharedPrefix.</td>
        <td>        <ul>
            <li>If you do not use this function, you can pass in nullptr, indicating that it is the same length as the s of keySharedPrefix/valueSharedPrefix.</li>
            <li>The valid Sequence Length in this parameter should not be greater than the Sequence Length in keySharedPrefix/valueSharedPrefix.</li>
            <li>For comprehensive constraints, see <a href="#constraints">Constraints</a>.</li>
        </ul>
        </td>
        <td>INT64</td>
        <td>-</td>
        <td>(1)</td>
        <td>-</td>
    </tr>
    <tr>
        <td>queryRopeOptional</td>
        <td>Input</td>
        <td>Rope information of the query in the MLA structure.</td>
        <td>        <ul>
            <li>Does not support empty tensor.</li>
            <li>For comprehensive constraints, see <a href="#constraints">Constraints</a>.</li>
        </ul>
        </td>
        <td>FLOAT16, BFLOAT16</td>
        <td>ND</td>
        <td>d is 64 in the shape of queryRope, and the remaining dimensions are the same as those of query.</td>
        <td>×</td>
    </tr>
    <tr>
        <td>keyRopeOptional</td>
        <td>Input</td>
        <td>Rope information of the key in the MLA structure.</td>
        <td>        <ul>
            <li>Does not support empty tensors.</li>
            <li>Does not support non-contiguous tensors.</li>
            <li>See <a href="#constraints">Constraints</a> for comprehensive constraints.</li>
        </ul>
        </td>
        <td>FLOAT16, BFLOAT16</td>
        <td>ND</td>
        <td>d in the shape of keyRope is 64, and the remaining dimensions are the same as those of key</td>
        <td>×</td>
    </tr>
    <tr>
        <td>keyRopeAntiquantScaleOptional</td>
        <td>Input</td>
        <td>Dequantization factor of the rope information of the key in the MLA structure.</td>
        <td>        <ul>
            <li>Does not support empty tensors.</li>
            <li>Reserved parameter. It does not take effect in the current version. Pass in nullptr.</li>
        </ul>
        </td>
        <td>FLOAT16, BFLOAT16</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>dequantScaleQueryOptional</td>
        <td>Input</td>
        <td>Dequantization factor for query.</td>
        <td>        <ul>
            <li>Does not support empty tensors.</li>
            <li>Involved in the full quantization scenario. Quantization modes support per-token-group, and per-token stacked with per-head.</li>
            <li>When this function is not used, nullptr can be passed in.</li>
        </ul>
        </td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>See <a href="#constraints">Constraints</a>.</td>
        <td>-</td>
    </tr>
    <tr>
        <td>learnableSinkOptional</td>
        <td>Input</td>
        <td>Indicates the learnable "Sink Token" used to absorb the Attention Score.</td>
        <td>        <ul>
            <li>Supported only in the non-quantization field.</li>
            <li>When this function is not used, you can pass in nullptr.</li>
            <li>For comprehensive constraints, see <a href="#constraints">Constraints</a>.</li>
        </ul>
        </td>
        <td>FLOAT16, BFLOAT16</td>
        <td>ND</td>
        <td>(Q_N)</td>
        <td>×</td>
    </tr>
    <tr>
        <td>qStartIdxOptional</td>
        <td>Input</td>
        <td>Indicates the start index of the query sequence of the current block in the global sequence for the outer cut use case.</td>
        <td>        <ul>
            <li>nullptr can be passed in.</li>
            <li>For comprehensive constraints, see <a href="#constraints">Constraints</a>.</li>
        </ul>
        </td>
        <td>INT64</td>
        <td>ND</td>
        <td>See <a href="#constraints">Constraints</a></td>
        <td>-</td>
    </tr>
    <tr>
        <td>kvStartIdxOptional</td>
        <td>Input</td>
        <td>Indicates the start index of the key and value sequences of the current chunk in the global sequence in the out-slicing use case.</td>
        <td>        <ul>
            <li>Can be set to nullptr.</li>
            <li>For comprehensive constraints, see <a href="#constraints">Constraints</a>.</li>
        </ul>
        </td>
        <td>INT64</td>
        <td>ND</td>
        <td>See <a href="#constraints">Constraints</a></td>
        <td>-</td>
    </tr>
    <tr>
        <td>numHeads</td>
        <td>Input</td>
        <td>Number of query heads.</td>
        <td>In the BNSD, BSND, BNSD_BSND, BSND_BNSD, BNSD_NBSD, BSND_NBSD, TND, NTD, NTD_TND, TND_NTD use cases, the value must be the same as the N-axis shape value of query in shape. Otherwise, an execution exception occurs.</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>scaleValue</td>
        <td>Input</td>
        <td>Reciprocal of the square root of d in the formula.</td>
        <td>        <ul>
            <li>The data type must satisfy the data type deduction rule with the data type of query. </li>
            <li>If the user does not specifically specify it, it is recommended to pass in 1.0. </li>
        </ul>
        </td>
        <td>DOUBLE</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>preTokens</td>
        <td>Input</td>
        <td>Used for sparse computation. Indicates that attention needs to compute correlations with the preceding Tokens.</td>
        <td>        <ul>
            <li>If not specifically specified, it is recommended to pass in 2147483647.</li>
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
        <td>Indicates the number of subsequent tokens to be associated with the attention computation.</td>
        <td>        <ul>
            <li>If not specifically specified, it is recommended to pass in 2147483647.</li>
        </ul>
        </td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>inputLayout</td>
        <td>Input</td>
        <td>Indicates the data layout formats of input query, key, and value.</td>
        <td>        <ul>
            <li>Currently, BSH, BSND, BNSD, BNSD_BSND (input BNSD, output BSND), BSND_BNSD (input BSND, output BNSD), BSH_BNSD (input BSH, output BNSD), BNSD_NBSD (input BNSD, output NBSD), BSND_NBSD (input BSND, output NBSD), BSH_NBSD (input BSH, output NBSD), TND (for comprehensive constraints on TND-related use cases, see <a href="#constraints">Constraints</a>), NTD, NTD_TND (input NTD, output TND), and TND_NTD (input TND, output NTD) are supported. If not specifically specified, it is recommended to pass in "BSH".</li>
            <li>Note that when the layout format contains an underscore, the part on the left of the underscore indicates the layout of input query, and the part on the right indicates the format of output.</li>
            <li>The data layout formats of query, key, and value can be interpreted in multiple dimensions: B (Batch) indicates the input batch size, S (Seq-Length) indicates the input sequence length, H (Hidden-Size) indicates the hidden layer size, N (Head-Num) indicates the number of heads, D (Head-Dim) indicates the smallest unit size of the hidden layer, where D=H/N, and T indicates the accumulated sum of input sequence lengths of all batches.</li>
            <li>inputLayout=BSH_BNSD, BSND_BNSD, NTD, and NTD_TND support only Q_D=K_D=V_D all equal to 64 or 128, or Q_D=K_D equal to 192 and V_D equal to 128.</li>
            <li>inputLayout=BNSD_BSND supports only Q_D=K_D=V_D all 16-aligned (32-aligned when the output dtype is int8), or Q_D=K_D equal to 192 and V_D equal to 128<br></li>
        </ul>
        </td>
        <td>STRING</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>numKeyValueHeads</td>
        <td>Input</td>
        <td>Number of heads in key and value.</td>
        <td>        <ul>
            <li>If not specifically specified by the user, it is recommended to pass in 0, indicating that the number of heads of key/value is equal to that of query.</li>
            <li>numHeads must be divisible by numKeyValueHeads. In GQA non-quantization scenarios and the Prefill MLA use case, the ratio of numHeads to numKeyValueHeads has no limit; in the Decode MLA use case, the ratio of numHeads to numKeyValueHeads has no limit in non-quantization scenarios. In full quantization scenarios, when the data type of query/key/value is FLOAT8_E4M3FN and inputLayout is TND, only the ratios of numHeads to numKeyValueHeads of 1, 2, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, and 128 are supported; in other full quantization scenarios, only the ratios of numHeads to numKeyValueHeads of 1, 2, 4, 8, 16, 32, 64, and 128 are supported.</li>
            <li>In the BNSD, BSND, BNSD_BSND, BSND_BNSD, BNSD_NBSD, BSND_NBSD, TND, NTD, NTD_TND, and TND_NTD use cases, this parameter must also be the same as the N-dimension shape value of key/value in shape. Otherwise, an execution error occurs.</li>
        </ul>
        </td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>sparseMode</td>
        <td>Input</td>
        <td>Sparse mode.</td>
        <td>        <ul>
            <li>When inputLayout is TND, TND_NTD, or NTD_TND, see <a href="#constraints">Constraints</a> for the comprehensive constraints.</li>
            <li>When sparseMode is 0, the mode is defaultMask. If attenMask is not passed in, the mask operation is not performed, and preTokens and nextTokens are ignored (internally set to INT_MAX). If attenMask is passed in, the complete attenMask matrix (S1 * S2) must be passed in, indicating that the part between preTokens and nextTokens needs to be calculated. preTokens + nextTokens >= 0 is required. </li>
            <li>When sparseMode is 1, the mode is allMask. The complete attenMask matrix (S1 * S2) must be passed in.</li>
            <li>When sparseMode is 2, the mode is the leftUpCausal mask. The optimized attenMask matrix (2048*2048) needs to be passed in.</li>
            <li>When sparseMode is 3, the mode is the rightDownCausal mask, corresponding to the lower-triangle use case partitioned by the right vertex. The optimized attenMask matrix (2048*2048) needs to be passed in.</li>
            <li>When sparseMode is 4, the mode is the band mask. The optimized attenMask matrix (2048*2048) needs to be passed in. preTokens + nextTokens >= 0 is required.</li>
            <li>When sparseMode is 5, 6, 7, or 8, the modes are prefix, global, dilated, and block_local respectively. These modes are not supported currently.</li>
            <li>If not specifically specified by the user, it is recommended to pass in 0.</li>
        </ul>
        </td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>innerPrecise</td>
        <td>Input</td>
        <td>Indicates the selection of high precision or high performance.</td>
        <td>        <ul>
            <li>When innerPrecise is 0, the high-precision mode is enabled and row invalid correction is not performed.</li>
            <li>When innerPrecise is 1, the high-performance mode is used and row invalid correction is not performed.</li>
            <li>When innerPrecise is 2, the high-precision mode is enabled and row invalid correction is performed.</li>
            <li>When innerPrecise is 3, the high-performance mode is used and row invalid correction is performed.</li>
            <li>When sparse_mode is 0 or 1 and a user-defined mask is passed in, enabling row invalid correction is recommended.</li>
            <li>BFLOAT16 and INT8 do not distinguish between high precision and high performance. Row invalid correction takes effect on FLOAT16, BFLOAT16, and INT8.</li>
            <li>Currently, 0 and 1 are reserved configuration values. If a whole row in the "mask part involved in the computation" is all 1s, precision loss may occur. In this case, you can try setting this parameter to 2 or 3 to enable the row invalid function for higher precision, but this configuration degrades performance.</li>
            <li>If the operator can detect invalid rows, invalid-row computation is automatically enabled, for example, when sparse_mode is 3 and Sq > Skv.</li>
        </ul>
        </td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>blockSize</td>
        <td>Input</td>
        <td>Maximum number of tokens in each block of KV storage in PagedAttention.</td>
        <td>Processed as 0 if not passed in.</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>antiquantMode</td>
        <td>Input</td>
        <td>Pseudo quantization method</td>
        <td>Not supported</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>softmaxLseFlag</td>
        <td>Input</td>
        <td>Whether to output softmax_lse.</td>
        <td>        <ul>
            <li>Supports outer splitting of the S axis (increases output).</li>
            <li>It is recommended to pass in false if the user does not specifically specify it.</li>
        </ul>
        </td>
        <td>BOOL</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>keyAntiquantMode</td>
        <td>Input</td>
        <td>Dequantization method of key.</td>
        <td>        <ul>
            <li>It is recommended to pass in 0 if not specifically specified.</li>
            <li>Except for the use case where keyAntiquantMode is 0 and valueAntiquantMode is 1, this parameter must be kept consistent with valueAntiquantMode.</li>
            <li>When keyAntiquantMode is 0, it indicates the per-channel mode (per-channel includes per-tensor).</li>
            <li>When keyAntiquantMode is 1, it indicates the per-token mode.</li>
            <li>When keyAntiquantMode is 2, it indicates the per-tensor stacked with per-head mode.</li>
            <li>When keyAntiquantMode is 3, it indicates the per-token stacked with per-head mode.</li>
            <li>When keyAntiquantMode is 4, it indicates the per-token mode stacked with page attention for scale/offset management.</li>
            <li>When keyAntiquantMode is 5, it indicates the per-token mode stacked with per-head and page attention for scale/offset management.</li>
            <li>When keyAntiquantMode is 6, it indicates the per-token-group mode.</li>
            <li>Values other than 0-6 cause an execution exception.</li>
            <li>For comprehensive constraints, see <a href="#constraints">Constraints</a>.</li>
        </ul>
        </td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>valueAntiquantMode</td>
        <td>Input</td>
        <td>Dequantization method of value.</td>
        <td>        <ul>
            <li>Mode numbers 0-6 of valueAntiquantMode are the same as those of keyAntiquantMode.</li>
            <li>If valueAntiquantMode is 8, it indicates the per-channel-group mode.</li>
            <li>It is recommended to pass in 0 if the user does not specifically specify it.</li>
            <li>For comprehensive constraints, see <a href="#constraints">Constraints</a>.</li>
        </ul>
        </td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>queryQuantMode</td>
        <td>Input</td>
        <td>Query dequantization mode.</td>
        <td>        <ul>
            <li>The mode number is the same as that of keyAntiquantMode.</li>
            <li>If the user does not specifically specify it, it is recommended to pass in 0.</li>
            <li>See <a href="#constraints">Constraints</a> for comprehensive constraints.</li>
        </ul>
        </td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>pseType</td>
        <td>Input</td>
        <td>pse mode.</td>
        <td>        <ul>
            <li>Configuring the value to 0 is supported (pseType = 1 is not supported in the inference use case).</li>
            <li>When pseType is 0, the externally passed-in pse is multiplied and then added.</li>
        </ul>
        </td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>attentionOut</td>
        <td>Output</td>
        <td>Output in the formula.</td>
        <td>-</td>
        <td>FLOAT16, BFLOAT16, INT8, FLOAT8_E4M3FN, HIFLOAT8</td>
        <td>ND</td>
        <td>The D dimension of this parameter must be consistent with the D dimension of value, and the remaining dimensions must be consistent with the shape of the input query.</td>
        <td>-</td>
    </tr>
    <tr>
        <td>softmaxLse</td>
        <td>Output</td>
        <td>In the ring attention algorithm, the max of the result of query multiplied by key is first taken to obtain softmax_max. Then softmax_max is subtracted from the result of query multiplied by key, exp is taken, and the sum is computed to obtain softmax_sum. Finally, the log of softmax_sum is taken, and softmax_max is added to obtain the result.</td>
        <td>        <ul>
            <li>It is recommended to pass in nullptr if the user does not specifically specify.</li>
            <li>Data with inf value indicates invalid data; when softmaxLseFlag is False, if the Tensor passed in for softmaxLse is not empty, the Tensor data is returned directly; if nullptr is passed in for softmaxLse, a Tensor of all zeros with shape {1} is returned.</li>
        </ul>
        </td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>When softmaxLseFlag is True, the shape must be [B, N, Q_S, 1] in general. When inputLayout is TND/NTD/NTD_TND/TND_NTD, the shape must be [T, N, 1].</td>
        <td>-</td>
    </tr>
    <tr>
        <td>workspaceSize</td>
        <td>Output</td>
        <td>Returns the workspace size that the user needs to allocate on the Device side.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>executor</td>
        <td>Output</td>
        <td>Returns the op executor, which contains the operator computation flow.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
    </tbody></table>

- **Return value**

  Returns an aclnnStatus status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and tiling validation and returns an error in the following scenarios:

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
            <td>query, key, value, or attentionOut is a null pointer.</td>
        </tr>
        <tr>
            <td>ACLNN_ERR_PARAM_INVALID</td>
            <td>161002</td>
            <td>The data types and data formats of query, key, value, pseShiftOptional, attenMaskOptional, and attentionOut are not in the supported range.</td>
        </tr>
        <tr>
            <td>ACLNN_ERR_RUNTIME_ERROR</td>
            <td>361001</td>
            <td>An exception occurs when the API calls the NPU runtime interface for memory.</td>
        </tr>
        <tr>
            <td>ACLNN_ERR_INNER_TILING_ERROR</td>
            <td>561002</td>
            <td>When the shape (dimensions/sizes), attribute values, or parameter combinations of the input parameters do not meet the constraints in "Constraints", an exception occurs when the API internally performs tiling.</td>
        </tr>
    </tbody>
    </table>

## aclnnFusedInferAttentionScoreV5

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
        <td>Workspace memory address applied on the Device side.</td>
        </tr>
        <tr>
        <td>workspaceSize</td>
        <td>Input</td>
        <td>Size of the workspace applied on the Device side, obtained by calling the first-phase API aclnnFusedInferAttentionScoreV5GetWorkspaceSize.</td>
        </tr>
        <tr>
        <td>executor</td>
        <td>Input</td>
        <td>Executor of the op that contains the operator computation flow.</td>
        </tr>
        <tr>
        <td>stream</td>
        <td>Input</td>
        <td>Stream for executing tasks.</td>
        </tr>
    </tbody>
    </table>

- **Return value**

  Returns an aclnnStatus status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The second-phase API dispatches the operator for execution. Input parameter validation and tiling validation have been completed in the first-phase API aclnnFusedInferAttentionScoreV5GetWorkspaceSize. An error is returned in the following scenarios:

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
            <td>ACLNN_ERR_RUNTIME_ERROR</td>
            <td>361001</td>
            <td>An exception occurs when the API calls the NPU runtime interface for memory.</td>
        </tr>
    </tbody>
    </table>

## Constraints

### Constraint Types

The constraints of the FusedInferAttentionScore operator are divided into 4 tiers in ascending order of constraint complexity: single parameter constraint, existence constraint, consistency constraint, and cross-feature constraint. The constraints and examples of each tier are as follows:

- Single parameter constraint: constraint on a single API parameter, covering the Tensor, TensorList, Array, and Attributes in the FusedInferAttentionScore operator API
  - For Tensors, TensorLists, and Arrays, the single parameter constraint includes the following checks
    - Check the shape, including the shape dimension dim and the dim value of each dimension
    - Check the dtype
    - Check the format
  - For Attributes
    - Check the attribute value
- Existence constraint: in a specific use case, a parameter in the feature parameter group must be passed in, or passing in a parameter is not supported
- Consistency constraint: constraints between parameters within a feature parameter group.
  - Example 1: The attribute sparseMode and the input tensor attenMask both belong to the Attention Mask parameter group. When the value of sparseMode is 2/3/4, the shape of attenMask must be (2048, 2048). Such a constraint is a consistency constraint within a parameter group
- Feature cross constraint: involves multiple parameter groups; constraints across different parameter groups
  - Example 1: The input tensor blockTable and the attribute blockSize belong to the Paged Attention (PA) parameter group, and the input tensor attenMask belongs to the Mask parameter group. In the PA use case, the last dimension of the attenMask input (attenMaskS2) must be greater than or equal to maxBlockNumPerSeq * blockSize. Such a constraint is a cross constraint between multiple parameter groups. **To keep the style consistent, this constraint is placed in the Paged Attention parameter group that comes later in the input parameter order**

### Feature Parameter Groups

<table><thead>
  <tr>
    <th>Feature Parameter Group</th>
    <th>Parameter Field Name</th>
    <th>Field Group</th>
    <th>Field Type</th>
  </tr></thead>
<tbody>
  <tr>
    <td rowspan="9">Common Parameter Group</td>
    <td>query</td>
    <td>INPUT</td>
    <td>Tensor</td>
  </tr>
  <tr>
    <td>key</td>
    <td>INPUT</td>
    <td>TensorList</td>
  </tr>
  <tr>
    <td>value</td>
    <td>INPUT</td>
    <td>TensorList</td>
  </tr>
  <tr>
    <td>numHeads</td>
    <td>ATTR</td>
    <td>int64</td>
  </tr>
  <tr>
    <td>scaleValue</td>
    <td>ATTR(OPTIONAL)</td>
    <td>double</td>
  </tr>
  <tr>
    <td>inputLayout</td>
    <td>ATTR(OPTIONAL)</td>
    <td>string</td>
  </tr>
  <tr>
    <td>numKeyValueHeads</td>
    <td>ATTR(OPTIONAL)</td>
    <td>int64</td>
  </tr>
  <tr>
    <td>innerPrecise</td>
    <td>ATTR(OPTIONAL)</td>
    <td>int64</td>
  </tr>
  <tr>
    <td>attentionOut</td>
    <td>OUTPUT</td>
    <td>Tensor</td>
  </tr>
  <tr>
    <td rowspan="4">PSE Parameter Group</td>
    <td>pseShift</td>
    <td>INPUT(OPTIONAL)</td>
    <td>Tensor</td>
  </tr>
  <tr>
    <td>qStartIdxOptional</td>
    <td>INPUT(OPTIONAL)</td>
    <td>IntArray</td>
  </tr>
  <tr>
    <td>kvStartIdxOptional</td>
    <td>INPUT(OPTIONAL)</td>
    <td>IntArray</td>
  </tr>
  <tr>
    <td>pseType</td>
    <td>ATTR(OPTIONAL)</td>
    <td>int64</td>
  </tr>
  <tr>
    <td rowspan="4">Mask parameter group</td>
    <td>attenMaskOptional</td>
    <td>INPUT(OPTIONAL)</td>
    <td>Tensor</td>
  </tr>
  <tr>
    <td>preTokens</td>
    <td>ATTR(OPTIONAL)</td>
    <td>int64</td>
  </tr>
  <tr>
    <td>nextTokens</td>
    <td>ATTR(OPTIONAL)</td>
    <td>int64</td>
  </tr>
  <tr>
    <td>sparseMode</td>
    <td>ATTR(OPTIONAL)</td>
    <td>int64</td>
  </tr>
  <tr>
    <td rowspan="2">ActualSeqLens parameter group</td>
    <td>actualSeqLengthsOptional</td>
    <td>INPUT(OPTIONAL)</td>
    <td>IntArray</td>
  </tr>
  <tr>
    <td>actualSeqLengthsKvOptional</td>
    <td>INPUT(OPTIONAL)</td>
    <td>IntArray</td>
  </tr>
  <tr>
    <td rowspan="14">Pseudo Quantization &amp; Full Quantization Parameter Group</td>
    <td>deqScale1Optional</td>
    <td>INPUT(OPTIONAL)</td>
    <td>Tensor</td>
  </tr>
  <tr>
    <td>quantScale1Optional</td>
    <td>INPUT(OPTIONAL)</td>
    <td>Tensor</td>
  </tr>
  <tr>
    <td>deqScale2Optional</td>
    <td>INPUT(OPTIONAL)</td>
    <td>Tensor</td>
  </tr>
  <tr>
    <td>antiquantScaleOptional</td>
    <td>INPUT(OPTIONAL)</td>
    <td>Tensor</td>
  </tr>
  <tr>
    <td>antiquantOffsetOptional</td>
    <td>INPUT(OPTIONAL)</td>
    <td>Tensor</td>
  </tr>
  <tr>
    <td>keyAntiquantScaleOptional</td>
    <td>INPUT(OPTIONAL)</td>
    <td>Tensor</td>
  </tr>
  <tr>
    <td>keyAntiquantOffsetOptional</td>
    <td>INPUT(OPTIONAL)</td>
    <td>Tensor</td>
  </tr>
  <tr>
    <td>valueAntiquantScaleOptional</td>
    <td>INPUT(OPTIONAL)</td>
    <td>Tensor</td>
  </tr>
  <tr>
    <td>valueAntiquantOffsetOptional</td>
    <td>INPUT(OPTIONAL)</td>
    <td>Tensor</td>
  </tr>
  <tr>
    <td>dequantScaleQueryOptional</td>
    <td>INPUT(OPTIONAL)</td>
    <td>Tensor</td>
  </tr>
  <tr>
    <td>antiquantMode</td>
    <td>ATTR(OPTIONAL)</td>
    <td>int64</td>
  </tr>
  <tr>
    <td>keyAntiquantMode</td>
    <td>ATTR(OPTIONAL)</td>
    <td>int64</td>
  </tr>
  <tr>
    <td>valueAntiquantMode</td>
    <td>ATTR(OPTIONAL)</td>
    <td>int64</td>
  </tr>
  <tr>
    <td>queryQuantMode</td>
    <td>ATTR(OPTIONAL)</td>
    <td>int64</td>
  </tr>
  <tr>
    <td rowspan="3">PostQuant Parameter Group</td>
    <td>quantScale2Optional</td>
    <td>INPUT(OPTIONAL)</td>
    <td>Tensor</td>
  </tr>
  <tr>
    <td>quantOffset2Optional</td>
    <td>INPUT(OPTIONAL)</td>
    <td>Tensor</td>
  </tr>
  <tr>
    <td>outType</td>
    <td>ATTR(OPTIONAL)</td>
    <td>int64</td>
  </tr>
  <tr>
    <td rowspan="2">Paged Attention parameter group</td>
    <td>blockTableOptional</td>
    <td>INPUT(OPTIONAL)</td>
    <td>Tensor</td>
  </tr>
  <tr>
    <td>blockSize</td>
    <td>ATTR(OPTIONAL)</td>
    <td>int64</td>
  </tr>
  <tr>
    <td rowspan="2">LeftPadding parameter group</td>
    <td>queryPaddingSizeOptional</td>
    <td>INPUT(OPTIONAL)</td>
    <td>Tensor</td>
  </tr>
  <tr>
    <td>kvPaddingSizeOptional</td>
    <td>INPUT(OPTIONAL)</td>
    <td>Tensor</td>
  </tr>
  <tr>
    <td rowspan="3">SystemPrefix parameter group</td>
    <td>keySharedPrefixOptional</td>
    <td>INPUT(OPTIONAL)</td>
    <td>Tensor</td>
  </tr>
  <tr>
    <td>valueSharedPrefixOptional</td>
    <td>INPUT(OPTIONAL)</td>
    <td>Tensor</td>
  </tr>
  <tr>
    <td>actualSharedPrefixLenOptional</td>
    <td>INPUT(OPTIONAL)</td>
    <td>IntArray</td>
  </tr>
  <tr>
    <td rowspan="3">Rope parameter group</td>
    <td>queryRopeOptional</td>
    <td>INPUT(OPTIONAL)</td>
    <td>Tensor</td>
  </tr>
  <tr>
    <td>keyRopeOptional</td>
    <td>INPUT(OPTIONAL)</td>
    <td>Tensor</td>
  </tr>
  <tr>
    <td>keyRopeAntiquantScaleOptional</td>
    <td>INPUT(OPTIONAL)</td>
    <td>Tensor</td>
  </tr>
  <tr>
    <td>LearnableSink Parameter Group</td>
    <td>learnableSinkOptional</td>
    <td>INPUT(OPTIONAL)</td>
    <td>Tensor</td>
  </tr>
  <tr>
    <td rowspan="2">SoftmaxLSE Parameter Group</td>
    <td>softmaxLseFlag</td>
    <td>ATTR(OPTIONAL)</td>
    <td>bool</td>
  </tr>
  <tr>
    <td>softmaxLse</td>
    <td>OUTPUT</td>
    <td>Tensor</td>
  </tr>
</tbody></table>

### Baseline Information

In the documented constraints, common fields are described as follows:

|    Name    |                            Meaning                            |
| :---------: | :---------------------------------------------------------: |
|     GQA     |           In the documented constraints, generally refers to all use cases where ROPE is not passed in, including use cases with G=1 and G>1           |
| Prefill MLA |  ROPE is passed in (including two modes: merged and separated, with a headdim of 64): in the separated mode, the headdim of input Q/K/V is 128, and ROPE is passed in separately; in the merged mode, the headdim of input Q/K is 192 (merged with ROPE), the headdim of V is 128, and ROPE is not passed in separately |
| Decode MLA |             The headdim of input Q/K/V is 512, key/value is reused, where value does not take effect, and ROPE is passed in separately.             |
| GQA full quantization | GQA full quantization use case where the data type of query/key/value is FLOAT8_E4M3FN, queryQuantMode is 3 (per-token stacked with per-head mode), keyAntiquantMode is 3 (per-token stacked with per-head mode), and valueAntiquantMode is 2 (per-tensor stacked with per-head mode). |
|      B      |                Batch, indicating the input batch size.                |
|     Q_N     |        Number of heads of the input query tensor, corresponding to N in the query shape.        |
|    KV_N    |    Number of heads of the input key/value tensor, corresponding to N in the key/value shape.    |
|     Q_S     |      Sequence length of the input query tensor, corresponding to S in the query shape.      |
|    KV_S    |  Sequence length of the input key/value tensor, corresponding to S in the key/value shape.  |
|     Q_T     |          Accumulated sum of the sequence lengths of all batches in the input query tensor.          |
|      G     |        Group, indicating the number of query head groups, corresponding to the ratio of Q_N to KV_N.         |

### Parameter Group Constraints

<!--
#### Overall Rules
1. Maintain by hierarchy: feature parameter group -> four-layer constraints -> quantization type (non-quantization/pseudo quantization/full quantization) -> FA use case (GQA/Prefill MLA/Decode MLA) -> chip generation (Atlas A2/Ascend 950PR)
2. Use a consistent indentation style for each feature group. No fixed constraints on table width and height; keep them concise and clear
3. For parameter naming in constraints, the first occurrence must be consistent with the parameter name in the API; subsequent occurrences can be abbreviated, e.g., "attenMask (same as mask, hereinafter referred to as mask)"
4. Abbreviations of input axes must be self-explanatory, such as QueryS, KeyNumHead, PseShiftS2; abbreviations such as S1, B, n2 are prohibited. It is best to supplement the meaning of each variable in the baseline information description
-->

#### Common Parameter Group (CommonChecker)

- Single Parameter Constraint

  - General Constraints

    <table style="undefined;table-layout: fixed; width: 1100px">
        <colgroup>
            <col style="width: 200px">
            <col style="width: 250px">
            <col style="width: 200px">
            <col style="width: 450px">
        </colgroup>
        <thead>
            <tr>
                <th>Use Case</th>
                <th>Query/Key/Value</th>
                <th>numHeads</th>
                <th>inputLayout</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>GQA</td>
                <td rowspan="3"><ul>
                                    <li>Only supports the data type FLOAT16/BFLOAT16.</li>
                                    <li>Only supports the data format ND.</li>
                                </ul>
                </td>
                <td>Must be an integer less than 2^32</td>
                <td>Supports BNSD, BSND, BSH, TND, NTD, BSH_BNSD, BSND_BNSD, NTD_TND, BNSD_BSND</td>
            </tr>
            <tr>
                <td>Prefill MLA</td>
                <td>Must be an integer, less than 2^32.</td>
                <td>Supports BNSD, BSND, BSH, TND, NTD, BSH_BNSD, BSND_BNSD, NTD_TND, and BNSD_BSND.</td>
            </tr>
            <tr>
                <td>Decode MLA</td>
                <td>Must be an integer, less than 2^32.</td>
                <td>Supports BNSD, BSND, BSH, TND, BNSD_NBSD, BSND_NBSD, BSH_NBSD, and TND_NTD.</td>
            </tr>
        </tbody>
    </table>

- Existence Constraint

  <table style="undefined;table-layout: fixed; width: 700px">
        <colgroup>
            <col style="width: 400px">
            <col style="width: 300px">
        </colgroup>
        <thead>
            <tr>
                <th>Use Case</th>
                <th>Expected Result</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>query, key, value, or attentionOut contains nullptr.</td>
                <td>Normal interception</td>
            </tr>
            <tr>
                <td>The shapeSize of the query and attentionOut tensors is 0.</td>
                <td>attentionOut returns an empty tensor.</td>
            </tr>
            <tr>
                <td>The shapeSize of the query and attentionOut tensors is not 0, and the shapeSize of the key and value tensors is 0.</td>
                <td>attentionOut returns all 0s.</td>
            </tr>
        </tbody>
    </table>

- Consistency Constraint

  <table style="undefined;table-layout: fixed; width: 1500px">
        <colgroup>
            <col style="width: 200px">
            <col style="width: 300px">
            <col style="width: 300px">
            <col style="width: 350px">
            <col style="width: 350px">
        </colgroup>
        <thead>
            <tr>
                <th>Use Case</th>
                <th>B</th>
                <th>N</th>
                <th>D</th>
                <th>numHeads/numKeyValueHeads</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>GQA</td>
                <td rowspan="3">                    <ul>
                        <li>Supports a B dimension less than or equal to 65536.</li>
                        <li>In non-contiguous use cases, the batch in the tensorlist of key and value can only be 1, and B cannot be greater than 256.</li>
                    </ul>
                </td>
                <td rowspan="3">                    When the layout of query is not BSH, the N dimension of query/attentionOut must keep consistent with numHeads, and the N dimension of key/value must keep consistent with numKeyValueHeads.
                </td>
                <td>                    <ul>
                        <li>Supports a D dimension less than or equal to 512.</li>
                        <li>In non-quantization fields, when the HeadDim of query, key, and value are all less than or equal to 128 and the layout is not NTD/NTD_TND/BSH_BNSD/BSND_BNSD/BNSD_BSND, the HeadDim of query/key can be different from that of value.
                        </li>
                        <li>In MxFP8 use cases, the HeadDim supports only 64 or 128.</li>
                        <li>In GQA full quantization fields, the HeadDim supports only 128.</li>
                        <li>When the layout is NTD/NTD_TND/BSH_BNSD/BSND_BNSD, the HeadDim supports only 64 or 128.</li>
                        <li>When the layout is BNSD_BSND, the HeadDim must be aligned to 16.</li>
                        <li>When the layout is BNSD_BSND and the data type of AttentionOut is INT8, the HeadDim must be aligned to 32.</li>
                    </ul>
                </td>
                <td rowspan="2">                    numHeads must be divisible by numKeyValueHeads, and the ratio of numHeads to numKeyValueHeads has no limit.
                </td>
            </tr>
            <tr>
                <td>Prefill MLA</td>
                <td>Supports the use case where the HeadDim of query/key is 192 and the HeadDim of value is 128. In other use cases, the HeadDim of query/key/value/attentionOut must keep consistent.</td>
            </tr>
            <tr>
                <td>Decode MLA</td>
                <td>The HeadDim of query/key/value/attentionOut must keep consistent.</td>
                <td>In non-quantization fields, numHeads must be divisible by numKeyValueHeads, and the ratio of numHeads to numKeyValueHeads has no limit; in full quantization fields, when the type of query/key/value is FLOAT8_E4M3FN and inputLayout is TND, numHeads supports only 1, 2, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128; in other full quantization fields, numHeads supports only 1, 2, 4, 8, 16, 32, 64, 128; numKeyValueHeads is 1.</td>
            </tr>
        </tbody>
    </table>

- Cross-Feature Constraints

  - Non-quantization Field
    <table style="undefined;table-layout: fixed; width: 1000px">
        <colgroup>
            <col style="width: 200px">
            <col style="width: 400px">
            <col style="width: 400px">
        </colgroup>
        <thead>
            <tr>
                <th>Use Case</th>
                <th>Unsupported Use Case</th>
                <th>PagedAttention use case</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>Common</td>
                <td>                    When the layout of query is TND/NTD, pseShift and tensorlist are not supported.
                </td>
                <td rowspan="4">                    <ul>
                        <li>When inputLayout is BNSD, TND, BSH, or BSND, the Key/Value layout supports three formats: BnBsH (blockNum, blockSize, H), BnNBsD (blockNum, KV_N,
                            blockSize, D), and NZ (blockNum, KV_N, D/16, blockSize, 16).</li>
                    </ul>
                </td>
            </tr>
            <tr>
                <td>GQA</td>
                <td>When the HeadDim of query/key is not equal to that of value, all parameter groups except the attenMask parameter group are not supported.</td>
            </tr>
            <tr>
                <td>Prefill MLA</td>
                <td>Full quantization and pseudo quantization are not supported.</td>
            </tr>
            <tr>
                <td>Decode MLA</td>
                <td>tensorlist, left padding, pseudo quantization, and prefix are not supported.</td>
            </tr>
        </tbody>
    </table>

#### PSE Parameter Group

- Single Parameter Constraint
  - Common
    - Input parameter pseType must satisfy the following conditions:
      - pseType must be 0.
      - When pseType is 1, the FA inference use case is not supported; only the FA training use case is supported.
    - Input parameter pseShift must satisfy the following conditions:
      - The data type of tensor must satisfy the following conditions:
        - When pseType is 0 and the data type of query is FLOAT16 or INT8, the data type of tensor must be FLOAT16.
        - When pseType is 0 and the data type of query is BFLOAT16, the data type of tensor must be BFLOAT16.
      - tensor shape must satisfy the following conditions:
        - When pseType is 0:
          - tensor shape must have 4 dimensions
          - When P_S1 (the third dimension of tensor shape) > 1:
            - The first dimension of tensor shape must be 1 or B
            - The second dimension of tensor shape must be Q_N
            - The third dimension of tensor shape must be greater than or equal to Q_S
            - In the non-prefix use case, the fourth dimension of tensor shape must be greater than or equal to KV_S
            - In the prefix use case, the fourth dimension of tensor shape must be greater than or equal to KV_S + actualSharedPrefixLen
          - When P_S1 (the third dimension of tensor shape) = 1:
            - The 1st dimension of tensor shape must be equal to 1 or B
            - The 2nd dimension of tensor shape must be equal to Q_N
            - In non-prefix use cases, the 4th dimension of tensor shape must be greater than or equal to KV_S
            - In prefix use cases, the 4th dimension of tensor shape must be greater than or equal to KV_S + actualSharedPrefixLen
- Feature Cross Constraints
  - Common
    - In PagedAttention use cases, the input parameter pseShift must satisfy the following conditions:
      - The last dimension of tensor shape must be greater than or equal to maxBlockNumPerBatch * blockSize
    - In alibi use cases, Q_S must be equal to KV_S
    - In MLA use cases, the input parameter pseShift must satisfy the following conditions:
      - pse is not supported, and pseShift cannot be passed in
    - In the use case where D values are not of the same length, input parameter pseShift must satisfy the following conditions:
      - pse is not supported, and pseShift cannot be passed in
    - In the pseudo quantization use case, when pseType is 0 and Q_S is 1, only 1 is supported for the third dimension of pseShift
    - In the GQA full quantization use case, pse is not supported, and pseShift cannot be passed in

#### <span id="MaskChecker">Attention Mask Parameter Group</span>

- Single Parameter Constraint
  - Common
    - The input attenMask must satisfy the following conditions:

      - tensor dtype is INT8/UINT8/BOOL
      - tensor format is ND/NCHW/NHWC/NCDHW
      - If Q_S and KV_S in the input attenMask shape are not 32B-aligned, they can be rounded up to aligned values
    - The input sparseMode must satisfy the following conditions:

      - sparseMode supports an input range of 0-4, and the default value is 0
      - When mask is not enabled, sparseMode supports only an input value of 0
      - The meanings of sparseMode are as follows (Note: In the attenMask matrix examples, 1 = masked out, 0 = keep)

      | sparseMode |                                                                               Meaning                                                                               |      attenMask Matrix Example      |                                             Note                                             |
      | :--------: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------: | :-------------------------: | :-------------------------------------------------------------------------------------------: |
      |     0     | defaultMask mode. If attenMask is not passed in, no mask operation is performed and preTokens and nextTokens are ignored;<br />if attenMask is passed in, the complete attenMask matrix needs to be passed in to compute the part between preTokens and nextTokens. | 11111<br />01111<br />00111 |          Complete attenMask matrix, that is, the full Q_S*KV_S matrix,<br />so different mask use cases can be customized.          |
      |     1     |                                                allMask mode. The complete attenMask matrix must be passed in and preTokens and nextTokens are ignored.                                                | 00101<br />10111<br />10101 |                                             Same as above                                             |
      |     2     |                                            leftUpCausal mode. The optimized attenMask matrix needs to be passed in and preTokens and nextTokens are ignored.                                            | 01111<br />00111<br />00011 | Optimized attenMask matrix, fixed as a 2048*2048 lower triangular matrix,<br />divided starting from the upper-left vertex, with all 0s on the diagonal. |
      |     3     |                                           rightDownCausal mode. The optimized attenMask matrix needs to be passed in and preTokens and nextTokens are ignored.                                           | 00011<br />00001<br />00000 | Optimized attenMask matrix, fixed as a 2048*2048 lower triangular matrix,<br />divided starting from the lower-right vertex, with all 0s on the diagonal. |
      |     4     |                                           band mode. The optimized attenMask matrix needs to be passed in to compute the part between preTokens and nextTokens.                                           | 00011<br />10001<br />11000 | Optimized attenMask matrix, fixed as a 2048*2048 lower triangular matrix,<br />divided starting from the lower-right vertex, with all 0s on the diagonal. |
      |     5     |                                                                              prefix                                                                              |              -              |                                            Not supported.                                            |
      |     6     |                                                                              global                                                                              |              -              |                                            Not supported.                                            |
      |     7     |                                                                              dilated                                                                              |              -              |                                            Not supported.                                            |
      |     8     |                                                                            block_local                                                                            |              -              |                                            Not supported.                                            |

- Existence Constraint
  - None.
- Consistency Constraint
  - Common
    - The input parameter attenMask supports only 2/3/4 input dimensions

      <!-- npu="950" id8 -->
      - <term>Ascend 950PR&950DT products</term>:
        - When the number of input dimensions is 2, sparseMode 0/1 modes are not supported
      <!-- end id8 -->

    - When input parameter sparseMode is 0/1, the shape of the attenMask matrix must satisfy [batchSize/1, >=Q_S, >=KV_S + actualSharedPrefixLen]; when the number of input dimensions is 4, the shape must satisfy [batchSize/1, 1, >=Q_S, >=KV_S + actualSharedPrefixLen]. That is, the second dimension must be 1, and extension by numHeads is not supported.
    - When input parameter sparseMode is 2/3/4, the last two dimensions of the attenMask matrix shape must be 2048. That is, (2048,2048), (1,2048,2048), or (1,1,2048,2048) can be passed in.
    - When pseudo quantization is not used or Q_S is greater than 1, preTokens and nextTokens must satisfy nextTokens * (-1) <= preTokens to ensure valid data exists.
  - Pseudo quantization
    - When Q_S is 1, only 3 or 4 dimensions are supported for the attenMask input, and the shape of the attenMask input must satisfy the following requirements: the first dimension equals batchSize or 1, the second-to-last dimension is greater than or equal to Q_S, and the last dimension is greater than or equal to the second dimension of blockTable * blockSize + actualSharedPrefixLen.
    - When Q_S is greater than 1, if sparseMode is 4 and the data type of attentionOut is int8, neither preTokens nor nextTokens can be negative.
- Feature cross constraints
  - Non-quantization
    - In the Decode MLA use case, sparseMode supports only modes 0/3/4.
    - In the GQA use case, when the head dims of query, key, and value are not equal, sparseMode supports only modes 0/2/3.
  - Full quantization
    - In the Decode MLA use case, if the data type of query/key/value is INT8 and the layout is TND/TND_NTD, sparseMode supports only modes 0 and 3, and attenMask is not supported in mode 0.
    - In the Decode MLA use case, if the data type of query/key/value is INT8, when the layout is not TND/TND_NTD and Q_S is greater than 1, sparseMode supports only mode 3 and attenMask must be passed in; when Q_S is 1, sparseMode supports only mode 0 and attenMask cannot be passed in.
    - In the Decode MLA use case, if the data type of query/key/value is FLOAT8_E4M3FN/HIFLOAT8, sparseMode supports only modes 0 and 3, and attenMask is not supported in mode 0.
    - In the MxFP8 use case, sparseMode supports only modes 0 and 3, and attenMask is not supported in mode 0.
    - In the GQA full quantization use case, sparseMode supports only modes 0 and 3, and attenMask is not supported in mode 0.

#### ActualSeqLen Parameter Group

- Single Parameter Constraints
  - Common
    - The input actualSeqLengths (actualSeqLengths of query) must satisfy the following conditions:
      - The length must satisfy the following conditions:
        - When the layout of query is TND/NTD, the length must be equal to the number of batches
        - When the layout of query is not TND/NTD, the length must be equal to 1 or greater than or equal to the batch value of query
      - The values in the input parameter must satisfy the following conditions:
        - When the layout of query is TND/NTD, the values must be in increasing order (greater than or equal to the previous value)
        - When the layout of query is TND/NTD, the last element must be equal to T
        - When the query layout is not TND/NTD, the value must not be greater than Q_S.
        - The value must be non-negative.
    - Input parameter actualSeqLengthsKv (actualSeqLengths of key/value) must meet the following requirements:
      - The length must meet the following requirements:
        - When the key/value layout is TND/NTD, the length must be equal to the number of batches.
        - When the key/value layout is not TND/NTD, the length must be equal to the batch value of key/value.
      - The values in the input parameter must meet the following requirements:
        - When the key/value layout is TND/NTD, the last element must be equal to T.
        - When the key/value layout is TND/NTD, the values must be in ascending order (greater than or equal to the previous value).
        - When the key/value layout is not TND/NTD, the value must not be greater than KV_S.
        - Its value must be a non-negative number
- Existence Constraint
  - Common
    - The input actualSeqLengths (the actualSeqLengths of query) must satisfy the following conditions:
      - When the layout of query is TND/NTD, actualSeqLengths must be passed in
    - The input actualSeqLengthsKv (the actualSeqLengths of key/value) must satisfy the following conditions:
      - When the layout of key/value is TND/NTD, actualSeqLengthsKv must be passed in
      - In the PagedAttention use case, actualSeqLengthsKv must be passed in
- Consistency Constraint
  - None.
- Cross-Feature Constraint
  - Full quantization
    - In the Decode MLA use case, if actualSeqLengths is passed in, the layout of query must be TND/NTD.

#### <span id="DequantChecker">Pseudo Quantization/Full Quantization Parameter Group (DequantChecker)</span>

- Single Parameter Constraint
  - Pseudo Quantization Scenario
    - The input parameters keyAntiquantMode and valueAntiquantMode must satisfy the following conditions:
      - The values in the input parameters must satisfy the following conditions:
        - The values must be 0 (per-channel/per-tensor), 1 (per-token), 2 (per-tensor stacked with per-head), 3 (per-token stacked with per-head),
          4 (per-token mode uses page attention to manage scale/offset),
          5 (per-token stacked with per-head mode and uses page attention to manage scale/offset), 6 (per-token-group).
        - Except when key supports per-channel and value supports per-token, keyAntiquantMode and valueAntiquantMode must be equal.
    - The input parameters keyAntiquantScale and valueAntiquantScale must satisfy the following conditions:
      - The data types of the input parameters must satisfy the following conditions:
        - per-channel (per-tensor), the data type must be the same as that of query
        - per-token, only the FLOAT32 data type is supported
        - per-tensor stacked with per-head, the data type must be the same as that of query
        - per-token stacked with per-head, only the FLOAT32 data type is supported
        - per-token mode uses page attention to manage scale/offset, only the FLOAT32 data type is supported
        - key supports per-channel stacked with per-token for value, only the FLOAT32 data type is supported
        - per-token-group, only the FLOAT8_E8M0 data type is supported
      - The shapes of input parameters must satisfy the following conditions:
        - per-channel, shape must be (1, N, 1, D), (1, 1, N, D), (1, N, D), (1, H), (N, 1, D), (N, D), (H)
        - per-tensor, shape must be (1)
        - per-token, shape must be (1, B, >=KV_S) or (B, >=KV_S)
        - per-tensor stacked with per-head, shape must be (N)
        - per-token stacked with per-head, shape must be (B, N, >=KV_S)
        - per-token mode using page attention to manage scale/offset, shape must be (blockNum, blockSize)
        - per-token stacked with per-head mode using page attention to manage scale/offset, shape must be (blockNum, N, blockSize)
        - key supports per-channel stacked with value supporting per-token, keyAntiquantScale shape must be (1, N, 1, D), (1, N, D), (1, H), (N, 1, D), (N, D) or (H)
        - key supports per-channel stacked with value supporting per-token, valueAntiquantScale shape must be (1, B, >=KV_S) or (B, >=KV_S)
        - per-token-group, shape must be (1, B, N, >=KV_S, D/32)
    - The input parameters key and value must meet the following conditions:
      - The data types of the input parameters must meet the following conditions:
        - per-tensor mode, whose supported data type is INT8
        - per-channel mode, whose supported data types are INT8, INT4(INT32), HIFLOAT8, and FLOAT8_E4M3FN
        - per-token mode, whose supported data types are INT8, INT4(INT32), and FLOAT8_E4M3FN
        - per-tensor mode stacked with per-head mode, whose supported data type is INT8
        - per-token mode stacked with per-head mode, whose supported data types are INT8 and INT4(INT32)
        - per-token mode using page attention to manage scale/offset, whose supported data types are INT8 and FLOAT8_E4M3FN
        - per-channel for key stacked with per-token for value, whose supported data types are INT8 and INT4(INT32)
        - per-token-group, whose supported data type is FLOAT4_E2M1
  - Full quantization scenario
    - Decode MLA full quantization
      - The dtype of input parameters dequantScaleQuery, keyAntiquantScale, and valueAntiquantScale is FLOAT32.
      - The shape of input parameters keyAntiquantScale and valueAntiquantScale is (1).
      - Input parameters keyAntiquantMode and valueAntiquantMode are 0 (per-tensor mode), and queryQuantMode is 3 (per-token stacked with per-head mode).
    - MxFP8 full quantization
      - The dtype of input parameters dequantScaleQuery, keyAntiquantScale, and valueAntiquantScale is FLOAT8_E8M0.
      - In the Q_S *G > 80 use case, the recommended shape of input parameter dequantScaleQuery is (Q_T, Q_N, D/64, 2); in the Q_S* G <= 80 use case, the recommended shape of input parameter dequantScaleQuery is (KV_N, Q_T, G, D/64, 2). In non-PagedAttention use cases, the shape of keyAntiquantScale is (KV_T, KV_N, D/64, 2), and the shape of valueAntiquantScale is (KV_T/64, KV_N, D, 2). In PagedAttention use cases, when the kv cache layout is BnNBsD, the shape of keyAntiquantScale is (Bn, KV_N, Bs, D/64, 2), and the shape of valueAntiquantScale is (Bn, KV_N, Bs/64, D, 2); when the kv cache layout is NZ, the shape of keyAntiquantScale is (Bn, KV_N, Bs/16, D/64, 16, 2), and the shape of valueAntiquantScale is (Bn, KV_N, D/16, Bs/64, 16, 2).
      - Input parameters queryQuantMode and keyAntiquantMode are 6 (per-token-group mode), and valueAntiquantMode is 8 (per-channel-group mode).
      - The shape of input parameter quantScale1 is (1), and the supported dtype is FLOAT32. quantScale1 is reused as pScale.
    - GQA full quantization
      - The dtype of input parameters dequantScaleQuery, keyAntiquantScale, and valueAntiquantScale is FLOAT32.
      - The shape of input dequantScaleQuery is (Q_N, Q_T), and the layout is NT; the shape of input keyAntiquantScale is (Bn, KV_N, Bs), and the layout is BnNBs; the shape of input valueAntiquantScale is (KV_N), and the layout is N
      - The input queryQuantMode is 3 (per-token stacked with per-head mode), keyAntiquantMode is 3 (per-token stacked with per-head mode), and valueAntiquantMode is 2 (per-tensor stacked with per-head mode)
      - The shape of input quantScale1 is (1), the dtype is FLOAT32, and quantScale1 is reused as pScale
      - The kv cache layout supports only BnNBsD (blocknum, KV_N, blocksize, D)
- Existence Constraint
  - Pseudo Quantization Scenario
    - Input keyAntiquantScale and valueAntiquantScale must satisfy the following conditions:
      - Must pass in keyAntiquantScale
      - Must pass in valueAntiquantScale
    - Input keyAntiquantOffset and valueAntiquantOffset must satisfy the following conditions:
      - When keyAntiquantOffset is passed in, valueAntiquantOffset must be passed in.
      - When valueAntiquantOffset is passed in, keyAntiquantOffset must be passed in.
      - When the data type of key/value is FLOAT8_E4M3FN, HIFLOAT8, or FLOAT4_E2M1, the offset is not supported and keyAntiquantOffset cannot be passed in.
      - When the data type of key/value is FLOAT8_E4M3FN, HIFLOAT8, or FLOAT4_E2M1, the offset is not supported and valueAntiquantOffset cannot be passed in.
  - Full Quantization Scenario
    - Decode MLA full quantization
      - dequantScaleQuery, keyAntiquantScale, and valueAntiquantScale must be passed in at the same time.
      - keyAntiquantOffset and valueAntiquantOffset cannot be passed in.
      - deqScale1, quantScale1, and deqScale2 cannot be passed in.
      - antiquantScale and antiquantOffset cannot be passed in.
      - Passing in keyRopeAntiquantScale is not supported
    - MxFP8 full quantization
      - dequantScaleQuery, keyAntiquantScale, and valueAntiquantScale must exist at the same time
      - Passing in keyAntiquantOffset and valueAntiquantOffset is not supported
      - Passing in deqScale1 and deqScale2 is not supported
      - Passing in antiquantScale and antiquantOffset is not supported
      - Passing in keyRopeAntiquantScale is not supported
    - GQA full quantization
      - dequantScaleQuery, keyAntiquantScale, and valueAntiquantScale must exist at the same time
      - quantScale1 must exist (quantScale1 is reused as pScale)
      - Does not support passing in keyAntiquantOffset and valueAntiquantOffset
      - Does not support passing in deqScale1 and deqScale2
      - Does not support passing in antiquantScale and antiquantOffset
      - Does not support passing in keyRopeAntiquantScale
- Consistency Constraints
  - None.
- Cross-Feature Constraints
  - Pseudo Quantization Scenario
    - inputLayout supports only BSH, BNSD, BSND, BNSD_BSND, and TND
    - If the data type of key/value is INT8, inputLayout does not support TND
    - If Q_S = 1:
      - inputLayout does not support BNSD_BSND
      - When the data type of key/value is INT8, keyAntiquantMode = 0 and valueAntiquantMode = 1, the data type of query and output supports only FLOAT16 (except in the use case where the KV cache layout is NZ and the last dimension D0 is 32; in this use case, query and output support only BFLOAT16)
    - When Q_S > 1:
      - When the data type of key/value is INT8, keyAntiquantMode does not support 2, 3, 4, or 5
      - When the data type of key/value is INT8 and keyAntiquantMode is 0 or 1, the data type of query and output supports only BFLOAT16
      - When the data type of key/value is INT8 and keyAntiquantMode is 0 or 1, the length of Q_S cannot be greater than 16
      - When the data type of key/value is INT8 and keyAntiquantMode is 0 or 1, tensor list is not supported
      - When the data type of key/value is INT8 and keyAntiquantMode is 0 or 1, left padding is not supported
      - When the data type of key/value is INT8 and keyAntiquantMode is 0 or 1, page attention is not supported (except in the use case where the KV cache layout is NZ and the last dimension D0 is 32; in this use case, PagedAttention must be enabled)
      - The data type of key/value does not support INT4 or INT32
    - In the page attention use case, the input parameters keyAntiquantScale and valueAntiquantScale must satisfy the following conditions:
      - The tensor shape must satisfy the following conditions:
        - In per-token mode, the last dimension of the shape must be greater than or equal to maxBlockNumPerBatch * blockSize.
        - In per-token stacked with per-head mode, the last dimension of the shape must be greater than or equal to maxBlockNumPerBatch * blockSize.
        - In per-token-group mode, the second-to-last dimension of the shape must be greater than or equal to maxBlockNumPerBatch * blockSize.
    - In the PagedAttention use case, kv cache and scale/offset can be passed in as non-contiguous tensors. The constraints are as follows:
      - If the kv cache layout is BnNBsD or NZ, non-contiguous axis 0 and axis 1 are supported; if the layout is BnBsH, only non-contiguous axis 0 is supported. Non-contiguous is not supported in non-PagedAttention use cases.
      - Non-contiguous scale/offset is supported only when page attention manages scale/offset in per-token mode (mode 4) or in per-token stacked with per-head mode (mode 5): in mode 4, the shape is (blockNum, blockSize) and only non-contiguous axis 0 is supported; in mode 5, the shape is (blockNum, N, blockSize) and non-contiguous axis 0 and axis 1 are supported. Non-contiguous is not supported in non-PagedAttention use cases.
    - Merged rope is not supported.
    - For the use case where the kv cache layout is NZ and the last dimension D0 equals 32 (kv cache layout is [blockNum, KV_N, D/32, blockSize, 32]):
      - query: BFLOAT16, Q_S supports [1,16], and Q_D=128.
      - key/value: INT8, K_D=V_D=128.
      - inputLayout supports only BSH, BSND, and BNSD.
      - innerPrecise supports only 1 (high-performance mode).
      - Only separate quantization of KV parameters is supported. Asymmetric quantization is not supported, and antiquantOffset, keyAntiquantOffset, and valueAntiquantOffset cannot be passed in.
      - Configuring queryRope and keyRope is not supported.
      - PagedAttention must be enabled, and blockSize supports only 128 or 512.
      - Mask: When MTP is 0, sparseMode=0 with attenMask set to nullptr is supported; when MTP is greater than 0 and less than 16, sparseMode=3 is supported (pass in the optimized attenMask matrix with shape 2048*2048).
      - per-channel (keyAntiquantMode=0, valueAntiquantMode=0): shape of keyAntiquantScale is [H] when inputLayout is BSH, [N,1,D] when inputLayout is BNSD, and [N,D] when inputLayout is BSND; valueAntiquantScale is the same as keyAntiquantScale.
      - per-token (keyAntiquantMode=1, valueAntiquantMode=1): shape of keyAntiquantScale is [B,S], and S must be greater than or equal to the product of the second dimension size of blockTable (shape[1]) and blockSize; valueAntiquantScale is the same as keyAntiquantScale.
      - Does not support left padding, tensorlist, pse, prefix, or post-quantization.
  - Full quantization scenario
    - Decode MLA full quantization
      - The dtype of query, key, and value is FLOAT8_E4M3FN/INT8/HIFLOAT8.
      - The dtype of attenOut is BFLOAT16.
      - The dtype of queryRope and keyRope is BFLOAT16.
      - When query/key/value is FLOAT8_E4M3FN/HIFLOAT8, inputLayout supports only BSH, BSND, BNSD, TND, BSH_NBSD, BSND_NBSD, BNSD_NBSD, and TND_NTD; when the type of query/key/value is INT8, inputLayout supports only BSH, BSND, TND, BSH_NBSD, BSND_NBSD, and TND_NTD.
      - When the type of query/key/value is FLOAT8_E4M3FN and inputLayout is TND, Q_N supports 1, 2, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, and 128. In other full quantization scenarios, Q_N supports 1, 2, 4, 8, 16, 32, 64, and 128.
      - KV_N must be 1; G supports [1, 128]; Q_S supports [1,16].
      - When the inputLayout of query is BSH, the shape of dequantScaleQuery should be (B, Q_S, Q_N); when the inputLayout of query is BSND, BNSD, or TND, the shape of dequantScaleQuery has only one fewer dimension (D) than query, and each dimension must be kept consistent with the corresponding dimension of query.
      - Does not support the common prefix use case, pse use case, alibi use case, or left padding use case
      - When the data type of query/key/value is INT8, only the PagedAttention use case is supported, and the kv cache layout must be in NZ format
    - MxFP8 full quantization
      - The dtype of query, key, and value is FLOAT8_E4M3FN
      - The dtype of attentionOut supports FLOAT16 and BFLOAT16
      - inputLayout supports only TND
      - Supports the PagedAttention and non-PagedAttention use cases. In the PagedAttention use case, only BnNBsD (blocknum, KV_N, blocksize, D) and NZ (blocknum, KV_N, D/D0, blocksize, D0) are supported
      - Does not support the alibi use case, the existence of Rope, post quantization, the use case where D is not of equal length, or the common prefix use case
    - GQA full quantization
      - The dtype of query, key, and value supports only FLOAT8_E4M3FN
      - The dtype of attentionOut supports FLOAT16 and BFLOAT16
      - inputLayout supports only NTD_TND
      - Only the PagedAttention use case is supported, and the kv cache layout supports only BnNBsD (blocknum, KV_N, blocksize, D)
      - Only 128 is supported for D
      - Q_S supports [1, 64K], and KV_S supports [0, 256K]
      - The shared prefix, pse, alibi, left-padding, tensorlist, rope, and post-quantization use cases are not supported

#### <span id="PostQuantChecker">Post-Quantization Parameter Group (PostQuantChecker)</span>

- Single Parameter Constraint
  - Common
    - Input parameter quantScale2 must satisfy the following conditions:
      - The tensor dtype is BF16/FP32
    - Input parameter quantOffset2 must satisfy the following conditions:
      - The tensor dtype is BF16/FP32
    - In the PostQuant use case, the data type of output attenOut supports only INT8/FP8_E4M3FN/HIFLOAT8
- Existence Constraint
  - Common
    - In the PostQuant use case, you must pass in quantScale2.
- Consistency Constraint
  - Common
    - In the PostQuant use case, when the number of dimensions of quantScale2 is greater than 1 and the quantization mode is per-channel, if layout is BSH/BSND/BNSD/BNSD_BSND, quantScale2 supports only the shape queryN*valueD; otherwise, only [numHeads, vHeadDim] is supported.
    - In the PostQuant use case, when the number of dimensions of quantScale2 equals 1 and the quantization mode is per-tensor, quantScale2 supports only the shape (1).
    - In the PostQuant use case, when quantOffset2 exists, quantOffset2 must have the same shape and data type as quantScale2.
- Cross-Feature Constraints
  - Common
    - In the PostQuant use case, when the input data type of query is not BF16, quantScale2 supports only the FP32 type.
  - Non-quantization
    - In the PostQuant use case, when prefix exists, only INT8 is supported for the data type of output attenOut
  - Pseudo quantization
    - In the PostQuant use case, the data type of output attenOut only supports being the same as the data types of input Key and Value
    - When keyAntiquantMode and valueAntiquantMode are per-token or per-token using page attention to manage scale/offset, and the query data type is FP16/BF16 and the key/value data type is FLOAT8_E4M3FN, stacked post-quantization is not supported

#### Paged Attention Parameter Group

- Single Parameter Constraint
  - Common
    - The input parameter blockTable must satisfy the following conditions:
      - The tensor dtype is int32.
      - The tensor shape is 2D and each dim value cannot be 0. The first dimension length must equal the batch size, and the second dimension length cannot be less than maxBlockNumPerSeq (maxBlockNumPerSeq is the number of blocks corresponding to the maximum actualSeqLengthsKv in each batch).
    - The input parameter blockSize must satisfy the following conditions:
      - blockSize must be greater than 0.
      - blockSize is a custom parameter whose value affects PagedAttention performance. Generally, PagedAttention improves throughput but may degrade performance. Increasing blockSize provides certain performance gains.
  - Non-quantization
    - The input parameter blockSize must satisfy the following conditions:
      - Decode MLA/Prefill MLA use case: blockSize is 16-aligned, with a maximum of 1024
      - GQA use case: If QueryHeadDim/KeyHeadDim/ValueHeadDim are all 64 or 128, blockSize is 16-aligned, with a maximum of 1024; otherwise, if Q_S> 1, blockSize is 128-aligned, with a maximum of 1024; if Q_S= 1, blockSize is 16-aligned, with a maximum of 512
  - Pseudo quantization
    - The input parameter blockSize must satisfy the following conditions: blockSize must be 32B-aligned based on the dtype size of key and value, with a maximum of 512. That is, when the key and value data type is INT8/HIFLOAT8/FLOAT8_E4M3FN, blockSize must be 32-aligned; when the key and value data type is INT4(INT32) or FLOAT4_E2M1, blockSize must be 64-aligned
  - Full quantization
    - In the Decode MLA full quantization use case, only blockSize 128 is supported
    - In the MxFP8 full quantization use case, only blockSize values 64, 128, 256, 512, or 1024 are supported
    - In the GQA full quantization use case, only blockSize 128 is supported
- Existence Constraint
  - Common
    - When PagedAttention is enabled, actualSeqLengthsKv must be passed in
    - PagedAttention does not support tensorlist use cases, left padding use cases, common prefix use cases, or use cases where the D lengths are different
- Consistency Constraint
  - None.
- Cross-Feature Constraints
  - Common
    - In PagedAttention use cases, if attenMask is also enabled, the last dimension of the passed-in attenMask must be greater than or equal to maxBlockNumPerSeq * blockSize
    - In PagedAttention use cases, the performance of the kv cache in the BnNBsD layout is generally better than that in the BnBsH layout. The BnNBsD format is recommended.
    - In PagedAttention use cases, when the input kv cache layout is BnBsH (blocknum, blocksize, H) and KV_N * D exceeds 65535, an error is reported due to hardware instruction constraints. To resolve this issue, enable GQA (to reduce KV_N) or change the kv cache layout to BnNBsD (blocknum, KV_N, blocksize, D).
  - Pseudo Quantization
    - In the PagedAttention use case, when the kv cache layout is 4-dimensional, inputLayout must be BNSD, BNSD_BSND, or TND
  - Full quantization
    - In the Decode MLA use case:
      - When the data type of query is FP8_E4M3FN/HIFLOAT8 and inputLayout is BSH, BSND, BSH_NBSD, or BSND_NBSD, the kv cache layout supports only two formats: BnBsH (blocknum, blocksize, H) and NZ (blocknum, KV_N, D/D0, blocksize, D0)
      - When the data type of query is FP8_E4M3FN/HIFLOAT8 and inputLayout is BNSD, TND, BNSD_NBSD, or TND_NTD, the kv cache layout supports three formats: BnBsH (blocknum, blocksize, H), BnNBsD (blocknum, KV_N, blocksize, D), and NZ (blocknum, KV_N, D/D0, blocksize, D0)
      - When the data type of query is INT8, the kv cache layout supports only NZ, and the layout is (blocknum, KV_N, D/D0, blocksize, D0)
      - When the kv cache layout is NZ, the last dimension D0 is 32, and the last dimension D0 of keyRope is 16
    - In the GQA full quantization use case, PagedAttention is not supported except for MxFP8 and GQA full quantization (queryQuantMode = 3, keyAntiquantMode = 3, valueAntiquantMode = 2)
    - MxFP8 full quantization supports the non-PagedAttention use case. The kv cache layout supports BnNBsD (blocknum, KV_N, blocksize, D) and NZ (blocknum, KV_N, D/D0, blocksize, D0)

#### Left Padding Parameter Group

- Single Parameter Constraint
  - Non-quantization
    - In the Query left padding use case, the shape of queryPaddingSize must be (1).
    - In the Key and value left padding use case, the shape of kvPaddingSize must be (1).
- Existence Constraint
  - Common
    - In the Query left padding use case, queryPaddingSize must be passed in.
    - In the Key and value left padding use case, kvPaddingSize must be passed in.
- Consistency Constraint
  - None.
- Cross-Feature Constraints
  - Common
    - In the left padding use case, PagedAttention is not supported.
    - In the left padding use case, the BSH_BNSD, BSND_BNSD, TND, NTD, NTD_TND, and TND_NTD use cases are not supported.
    - In the left padding use case, you must pass in actualSeqLengths/actualSeqLengthsKv.

#### Shared Prefix Parameter Group

- Single Parameter Constraint
  - The input parameters keySharedPrefix and valueSharedPrefix must meet the following conditions:
    - The tensor shape must meet the following conditions:
      - The shape must be (1).
      - If the layout is BNSD or BSND, the N and D dimensions must be the same as those of key/value.
      - If the layout is BSH, the H dimension must be the same as that of key/value.
      - The S dimensions of keySharedPrefix and valueSharedPrefix must be the same.
    - The tensor data type must meet the following conditions:
      - The data type must be the same as that of key/value.
  - The input parameter actualSharedPrefixLen must satisfy the following conditions:
    - The shape must satisfy the following conditions:
      The shape must be 1.
    - The values in the input parameter must satisfy the following conditions:
      - The value cannot be greater than the S dimension of the shapes of keySharedPrefix and valueSharedPrefix.
- Existence Constraint
  - Common
    - The input parameters keySharedPrefix and valueSharedPrefix must satisfy the following conditions:
      - If keySharedPrefix is passed in, valueSharedPrefix must be passed in.
      - If valueSharedPrefix is passed in, keySharedPrefix must be passed in.
- Consistency Constraint
  - None.
- Cross-Feature Constraint
  - Common
    - Does not support the PagedAttention use case
    - Does not support the tensorlist use case
    - Does not support the left padding use case
    - Does not support the alibi use case
    - Does not support the TND use case
    - Does not support the Prefill MLA use case (including unequal D length and independent ROPE input)
    - Does not support the Decode MLA use case
  - Full quantization
    - In the full quantization use case (including MLA full quantization and GQA full quantization), prefix is not supported
    - In the post-quantization use case, only the INT8 data type is supported
  - Pseudo quantization
    - In the pseudo quantization key/value combined use case, prefix is supported in all quantization modes
    - In the pseudo quantization key/value separated use case, prefix supports only the following quantization modes:
      - When Q_S > 1 and the pseudo quantization mode is per-channel(per-tensor) or per-token, only the INT8 key/value data type is supported
      - When Q_S = 1 and the pseudo quantization mode is per-tensor, per-tensor stacked per-head, or per-token stacked with scale/offset
        managed by page attention mode, or per-token stacked per-head with scale/offset managed by page attention mode, only the INT8 key/value data type is supported
      - When Q_S = 1 and the pseudo quantization mode is per-channel, per-token, or per-token stacked per-head, the key supports per-channel
        stacked with value supports per-token. The data types of key/value support INT8 and INT4 (INT32).

#### Rope Parameter Group

- Single Parameter Constraint
  - Common
    - The input parameters queryRope and keyRope must satisfy the following conditions
      - The tensor dtype is FLOAT16/BFLOAT16
      - The D dimension in the tensor shape is 64
- Existence Constraint
  - Common
    - The input parameters queryRope and keyRope must both exist
- Consistency Constraint
  - None.
- Cross-Feature Constraints
  - Common
    - The D dimension of query shape supports only 128 and 512.
    - In non-tensorlist use cases, the dimensions of queryRope shape need to keep consistent with query. Except for D being 64 in queryRope shape, the remaining dimensions need to be consistent with query; the dimensions of keyRope shape need to keep consistent with key. Except for D being 64 in keyRope shape, the remaining dimensions need to be consistent with key.
    - In non-quantization fields, the dtype of input parameters queryRope and keyRope needs to keep consistent with query and key.
    - Common prefix, pse, and alibi use cases are not supported.
    - Pseudo quantization use cases are not supported.
  - Decode MLA
    - Layout supports only BNSD, BSND, BSH, TND, BNSD_NBSD, BSND_NBSD, BSH_NBSD, and TND_NTD.
    - In non-quantization fields, Q_N has no limit. In full quantization fields, when the type of query/key/value is FLOAT8_E4M3FN and inputLayout is TND, Q_N supports only 1/2/4/6/8/12/16/24/32/48/64/96/128; in other full quantization fields, Q_N supports only 1/2/4/8/16/32/64/128. KV_N supports only 1.
    - In the non-quantization field, Q_S is unlimited; in the full quantization field, Q_S supports [1,16]
    - The left padding use case and the tensorlist use case are not supported
  - Prefill MLA
    - Layout supports only BNSD, BSND, BSH, TND, NTD, BSH_BNSD, BSND_BNSD, NTD_TND, and BNSD_BSND
    - In the tensorlist use case, D in the passed-in keyRope shape is 64, B must keep consistent with the length of the key tensorlist, and N and S must be equal to the N and S of each tensor in the key tensorlist
    - The full quantization field is not supported
  - GQA full quantization
    - The rope use case is not supported

#### LearnableSink Parameter Group

- Single Parameter Constraint
  - Common
    - The input parameter learnableSink must satisfy the following conditions:
      - The tensor data type is FLOAT16/BFLOAT16
      - The tensor shape is (Q_N)
- Existence Constraint
  - None.
- Consistency Constraint
  - None.
- Cross-Feature Constraint
  - Common
    - In the LearnableSink-enabled use case, the tensor dtype must keep consistent with the query dtype
    - In the LearnableSink-enabled use case, QueryHeadDim/KeyHeadDim supports only 64, 128, and 192, and ValueHeadDim supports only 64 and 128
    - In the LearnableSink-enabled use case, innerPrecise must be high-precision mode (0)
    - LearnableSink does not support the left-padding use case, common prefix use case, pse use case, alibi use case, or post-quantization use case
    - LearnableSink does not support the full quantization use case, pseudo quantization use case, or Decode MLA use case

#### SoftmaxLSE Parameter Group

- Single Parameter Constraint
  - Common
    - Output lseOut supports only the FP32 data type
- Existence Constraint
  - Common
    - In the softmaxLSE use case, output lseOut must not be empty
- Consistency Constraint
  - None.
- Cross-Feature Constraint
  - Common
    - In the softmaxLSE non-empty tensor use case, if the output layout is TND or NTD, the lseOut input must have 3 dimensions, and the shape must match [Q_T, Q_N, 1]
    - In the softmaxLSE non-empty tensor use case, if the output layout is not TND or NTD, the lseOut input must have 4 dimensions, and the shape must match [B, Q_N, Q_S, 1]

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```c++
  #include <iostream>
  #include <vector>
  #include <math.h>
  #include <cstring>
  #include "acl/acl.h"
  #include "aclnn/opdev/fp16_t.h"
  #include "aclnnop/aclnn_fused_infer_attention_score_v5.h"

  using namespace std;

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

  int Init(int32_t deviceId, aclrtStream* stream) {
      // Boilerplate: resource initialization
      auto ret = aclInit(nullptr);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
      ret = aclrtSetDevice(deviceId);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
      ret = aclrtCreateStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
      return 0;
  }

  template <typename T>
  int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                      aclDataType dataType, aclTensor** tensor) {
      auto size = GetShapeSize(shape) * aclDataTypeSize(dataType);
      // Call aclrtMalloc to allocate memory on the device.
      auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
      // Call aclrtMemcpy to copy host-side data to device-side memory.
      ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

      // Compute the strides of the contiguous tensor.
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
      // 1. (Boilerplate) Device/stream initialization. For details, see the acl API manual.
      // Set deviceId based on the actual device.
      int32_t deviceId = 0;
      aclrtStream stream;
      auto ret = Init(deviceId, &stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

      // 2. Construct the input and output. Custom construction is required based on the API.
      int32_t batchSize = 1;
      int32_t numHeads = 2;
      int32_t sequenceLengthQ = 1;
      int32_t headDims = 16;
      int32_t keyNumHeads = 2;
      int32_t sequenceLengthKV = 16;
      std::vector<int64_t> queryShape = {batchSize, numHeads, sequenceLengthQ, headDims}; // BNSD
      std::vector<int64_t> keyShape = {batchSize, keyNumHeads, sequenceLengthKV, headDims}; // BNSD
      std::vector<int64_t> valueShape = {batchSize, keyNumHeads, sequenceLengthKV, headDims}; // BNSD
      std::vector<int64_t> attenShape = {batchSize, 1, 1, sequenceLengthKV}; // B11S
      std::vector<int64_t> outShape = {batchSize, numHeads, sequenceLengthQ, headDims}; // BNSD
      void *queryDeviceAddr = nullptr;
      void *keyDeviceAddr = nullptr;
      void *valueDeviceAddr = nullptr;
      void *attenDeviceAddr = nullptr;
      void *outDeviceAddr = nullptr;
      aclTensor *queryTensor = nullptr;
      aclTensor *keyTensor = nullptr;
      aclTensor *valueTensor = nullptr;
      aclTensor *attenTensor = nullptr;
      aclTensor *outTensor = nullptr;
      std::vector<float> queryHostData(batchSize * numHeads * sequenceLengthQ * headDims, 1.0f);
      std::vector<float> keyHostData(batchSize * keyNumHeads * sequenceLengthKV * headDims, 1.0f);
      std::vector<float> valueHostData(batchSize * keyNumHeads * sequenceLengthKV * headDims, 1.0f);
      std::vector<int8_t> attenHostData(batchSize * sequenceLengthKV, 0);
      std::vector<float> outHostData(batchSize * numHeads * sequenceLengthQ * headDims, 1.0f);

      // Create the query aclTensor.
      ret = CreateAclTensor(queryHostData, queryShape, &queryDeviceAddr, aclDataType::ACL_FLOAT16, &queryTensor);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create the key aclTensor.
      ret = CreateAclTensor(keyHostData, keyShape, &keyDeviceAddr, aclDataType::ACL_FLOAT16, &keyTensor);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      int kvTensorNum = 1;
      aclTensor *tensorsOfKey[kvTensorNum];
      tensorsOfKey[0] = keyTensor;
      auto tensorKeyList = aclCreateTensorList(tensorsOfKey, kvTensorNum);
      // Create the value aclTensor.
      ret = CreateAclTensor(valueHostData, valueShape, &valueDeviceAddr, aclDataType::ACL_FLOAT16, &valueTensor);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      aclTensor *tensorsOfValue[kvTensorNum];
      tensorsOfValue[0] = valueTensor;
      auto tensorValueList = aclCreateTensorList(tensorsOfValue, kvTensorNum);
      // Create the atten aclTensor.
      ret = CreateAclTensor(attenHostData, attenShape, &attenDeviceAddr, aclDataType::ACL_BOOL, &attenTensor);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create the out aclTensor.
      ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT16, &outTensor);
      CHECK_RET(ret == ACL_SUCCESS, return ret);

      std::vector<int64_t> actualSeqlenVector = {sequenceLengthKV};
      auto actualSeqLengths = aclCreateIntArray(actualSeqlenVector.data(), actualSeqlenVector.size());

      int64_t numKeyValueHeads = numHeads;
      double scaleValue = 1 / sqrt(headDims); // 1/sqrt(d)
      int64_t preTokens = 65535;
      int64_t nextTokens = 65535;
      string sLayerOut = "BNSD";
      char layerOut[sLayerOut.length()+1];
      strcpy(layerOut, sLayerOut.c_str());
      int64_t sparseMode = 0;
      int64_t innerPrecise = 1;
      int blockSize = 0;
      int antiquantMode = 0;
      bool softmaxLseFlag = false;
      int keyAntiquantMode = 0;
      int valueAntiquantMode = 0;
      int queryAntiquantMode = 0;
      // 3. Call the CANN operator library API.
      uint64_t workspaceSize = 0;
      int64_t pseType = 0;
      aclOpExecutor* executor;
      // Call the first-phase API.
      ret = aclnnFusedInferAttentionScoreV5GetWorkspaceSize(queryTensor, tensorKeyList, tensorValueList, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, numHeads, scaleValue, preTokens, nextTokens, layerOut, numKeyValueHeads, sparseMode, innerPrecise, blockSize, antiquantMode, softmaxLseFlag, keyAntiquantMode, valueAntiquantMode, queryAntiquantMode, pseType, outTensor, nullptr, &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFusedInferAttentionScoreV5GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
      // Allocate device memory based on the workspaceSize calculated by the first-phase API.
      void* workspaceAddr = nullptr;
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
      }
      // Call the second-phase API.
      ret = aclnnFusedInferAttentionScoreV5(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFusedInferAttentionScoreV5 failed. ERROR: %d\n", ret); return ret);

      // 4. (boilerplate) Wait synchronously for the task execution to finish.
      ret = aclrtSynchronizeStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

      // 5. Obtain the output values and copy the results from device memory to the host. Modify them based on the definition of the specific API.
      auto size = GetShapeSize(outShape);
      std::vector<op::fp16_t> resultData(size, 0);
      ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                        size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
      for (int64_t i = 0; i < size; i++) {
          std::cout << "index: " << i << ": " << static_cast<float>(resultData[i]) << std::endl;
      }

      // 6. Release resources.
      aclDestroyTensor(queryTensor);
      aclDestroyTensor(keyTensor);
      aclDestroyTensor(valueTensor);
      aclDestroyTensor(attenTensor);
      aclDestroyTensor(outTensor);
      aclDestroyIntArray(actualSeqLengths);
      aclrtFree(queryDeviceAddr);
      aclrtFree(keyDeviceAddr);
      aclrtFree(valueDeviceAddr);
      aclrtFree(attenDeviceAddr);
      aclrtFree(outDeviceAddr);
      if (workspaceSize > 0) {
          aclrtFree(workspaceAddr);
      }
      aclrtDestroyStream(stream);
      aclrtResetDevice(deviceId);
      aclFinalize();
      return 0;
  }
```
