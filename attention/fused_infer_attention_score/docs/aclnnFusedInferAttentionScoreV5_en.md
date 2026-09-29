
# aclnnFusedInferAttentionScoreV5

[📄 View Source Code](https://gitcode.com/cann/ops-transformer/tree/master/attention/fused_infer_attention_score)

## Supported Products

|Product     | Supported|
|:----------------------------|:-----------:|
|Ascend 950PR/Ascend 950DT|      √     |
|<term>Atlas A3 training products/Atlas A3 inference products</term>|      ×     |
|<term>Atlas A2 training products/Atlas A2 inference products</term>|      ×     |
|<term>Atlas 200I/500 A2 inference products</term>|      ×     |
|<term>Atlas inference products</term>|      ×     |
|<term>Atlas training products</term>|      ×     |

## Function

- Adapts to the `FlashAttention` operator in the decode (`IncreFlashAttention`) and prefill (`PromptFlashAttention`) inference scenarios.

  Compared with FusedInferAttentionScoreV4, this API adds the qStartIdxOptional, kvStartIdxOptional, and pseType parameters.

  **NOTE**

  KV cache specific to the decode scenario: KV cache is a common technology for optimizing the inference performance of foundation models. During sampling, the transformer model uses the given prompt/context as the initial input for inference (parallel processing supported), and then generates additional tokens one by one to improve the generated sequence (reflecting the auto-regressive property of the model). The transformer performs the self-attention operation during sampling. Therefore, KV vectors need to be extracted for each item (regardless of the prompt/context or generated token) in the current sequence. These vectors are stored in a matrix called KV cache.

- Formulas:

    Self-attention constructs an attention model by leveraging the relationships within the input samples. The principle assumes there is an input sample sequence $x$ of length $n$, where each element of $x$ is a $d$-dimensional vector. Each $d$-dimensional vector can be regarded as a token embedding. Such a sequence is transformed by three weight matrices to obtain three $n × d$ matrices.

    The computation formula for self-attention is generally defined as follows, where $Q$, $K$, and $V$ are key attribute elements of the input sample, obtained through spatial transformation and unified into a single feature space. "Attention" in the formula and operator name is an abbreviation for "self-attention."

    $$
    Attention(Q,K,V)=Score(Q,K)V
    $$

    The `Score` function in this operator employs the `Softmax` function. The self-attention calculation formula is as follows:

    $$
    Attention(Q,K,V)=Softmax(\frac{QK^T}{\sqrt{d}})V
    $$

    The product of $Q$ and $K^T$ represents the attention to the input $x$. To prevent this value from becoming excessively large, it is typically scaled by dividing by the square root of $d$, followed by row-wise softmax normalization. The result is then multiplied by $V$ to produce an $n × d$ matrix.

    **Note:**

    <blockquote>The data layout of <code>query</code>, <code>key</code>, and <code>value</code> can be interpreted from multiple dimensions. To be specific, <code>B</code> (<code>Batch</code>) indicates the size of an input sample batch, <code>S</code> (<code>Seq-Length</code>) indicates the length of the input sample sequence, <code>H</code> (<code>Hidden-Size</code>) indicates the size of the hidden layer, <code>N</code> (<code>Head-Num</code>) indicates the number of heads, and <code>D</code> (<code>Head-Dim</code>) indicates the minimum unit size of the hidden layer (<code>D</code> = <code>H</code>/<code>N</code>). <code>T</code> indicates the total length of all input sample sequences.
    <br><code>Q_S</code> indicates <code>S</code> in the shape of <code>query</code>, <code>KV_S</code> indicates <code>S</code> in the shapes of <code>key</code> and <code>value</code>, <code>Q_N</code> indicates <code>num_query_heads</code>, and <code>KV_N</code> indicates <code>num_key_value_heads</code>. <code>P</code> indicates the computation result of Softmax(<span>(QK<sup class="superscript">T</sup>)/<span class="sqrt">d</span></span>).</blockquote>

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnFusedInferAttentionScoreV5GetWorkspaceSize` is called to obtain the input parameters and compute the required workspace size based on the process. Then, `aclnnFusedInferAttentionScoreV5` is called to perform computation.

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
        <th>Name</th>
        <th>Input/Output</th>
        <th>Description</th>
        <th>Usage</th>
        <th>Data Type</th>
        <th>Data Format</th>
        <th>Dimension (Shape)</th>
        <th>Non-contiguous Tensor</th>
    </tr>
    </thead>
    <tbody>
    <tr>
        <td>query</td>
        <td>Input</td>
        <td>Input <code>Q</code> in the formula.</td>
        <td>-</td>
        <td>FLOAT16, BFLOAT16, INT8, HIFLOAT8, FLOAT8_E4M3FN</td>
        <td>ND</td>
        <td>See the <code>inputLayout</code> parameter.</td>
        <td>×</td>
    </tr>
    <tr>
        <td>key</td>
        <td>Input</td>
        <td>Input <code>K</code> in the formula.</td>
        <td>-</td>
        <td>FLOAT16, BFLOAT16, INT8, HIFLOAT8, FLOAT8_E4M3FN, INT4(INT32), FLOAT4_E2M1</td>
        <td>ND</td>
        <td>See the <code>inputLayout</code> parameter.</td>
        <td>×</td>
    </tr>
    <tr>
        <td>value</td>
        <td>Input</td>
        <td>Input <code>V</code> in the formula.</td>
        <td>-</td>
        <td>FLOAT16, BFLOAT16, INT8, HIFLOAT8, FLOAT8_E4M3FN, INT4(INT32), FLOAT4_E2M1</td>
        <td>ND</td>
        <td>See the <code>inputLayout</code> parameter.</td>
        <td>×</td>
    </tr>
    <tr>
        <td>pseShiftOptional</td>
        <td>Input</td>
        <td>Positional encoding.</td>
        <td>
        <ul>
            <li>Empty tensors are not supported.</li>
            <li>If this parameter is not used, pass <code>nullptr</code>.</li>
            <li>For details about the constraints, see <a href="#constraints">Constraints</a>.</li>
        </ul>
        </td>
        <td>FLOAT16, BFLOAT16</td>
        <td>ND</td>
        <td>Recommended shape input: <code>(B, Q_N, Q_S, KV_S)</code> or <code>(1, Q_N, Q_S, KV_S)</code>.</td>
        <td>×</td>
    </tr>
    <tr>
        <td>attenMaskOptional</td>
        <td>Input</td>
        <td>Mask matrix.</td>
        <td>
        <ul>
            <li>Empty tensors are not supported.</li>
            <li>If this parameter is not used, pass <code>nullptr</code>.</li>
            <li>For details about the constraints, see <a href="#constraints">Constraints</a>.</li>
        </ul>
        </td>
        <td>BOOL, INT8, UINT8</td>
        <td>ND</td>
        <td>
        <ul>
            <li>When sparseMode is set to 0 or 1, the shape of attenMaskOptional can be (B, Q_S, KV_S), (1,Q_S,KV_S), (B,1,Q_S,KV_S), or (1,1,Q_S,KV_S).</li>
            <li>When sparseMode is set to 2, 3, or 4, the shape of attenMaskOptional can be (2048,2048), (1,2048,2048), or (1,1,2048,2048)</li>.
            <li>Q_S is S in the shape of the query, and KV_S is S in the shape of the key and value. If Q_S and KV_S in the shape of the input attenMask are not 32-byte aligned, the aligned Q_S and KV_S can be rounded up.</li>
        </ul>
        </td>
        <td>×</td>
    </tr>
    <tr>
        <td>actualSeqLengthsOptional</td>
        <td>Input</td>
        <td>Valid sequence length of <code>query</code> in different batches.</td>
        <td>
        <ul>
            <li>If the sequence length is not specified, nullptr can be passed, indicating that the length is the same as the S length in the shape of the query.</li>
            <li>For details about the constraints, see <a href="#constraints">Constraints</a>.</li>
        </ul>
        </td>
        <td>INT64</td>
        <td>-</td>
        <td><code>(1)</code>, <code>(B)</code>, or <code>(>B)</code> </td>
        <td>-</td>
    </tr>
    <tr>
        <td>actualSeqLengthsKvOptional</td>
        <td>Input</td>
        <td>Valid sequence lengths of <code>key</code>/<code>value</code> in different batches.</td>
        <td>
        <ul>
            <li>If the sequence length is not specified, nullptr can be passed, indicating that the length is the same as the S length in the shape of the key/value.</li>
            <li>For details about the constraints, see <a href="#constraints">Constraints</a>.</li>
        </ul>
        </td>
        <td>INT64</td>
        <td>-</td>
        <td><code>(1)</code>, <code>(B)</code>, or <code>(>B)</code> </td>
        <td>-</td>
    </tr>
    <tr>
        <td>deqScale1Optional</td>
        <td>Input</td>
        <td>Dequantization factor after BMM1.</td>
        <td>
        <ul>
            <li>Empty tensors are not supported.</li>
            <li>Per-tensor is supported.</li>
            <li>When the full quantization function is used, this parameter is calculated based on the actual quantization process.</li>
            <li>If this parameter is not used, pass <code>nullptr</code>.</li>
            <li>For details about the constraints, see <a href="#constraints">Constraints</a>.</li>
        </ul>
        </td>
        <td>UINT64, FLOAT32</td>
        <td>ND</td>
        <td>See <a href="#INT8">Restrictions on the number of input parameters and input and output data formats related to INT8/FP8 quantization</a>.</td>
        <td>-</td>
    </tr>
    <tr>
        <td>quantScale1Optional</td>
        <td>Input</td>
        <td>Quantization factor before BMM2.</td>
        <td>
        <ul>
            <li>Empty tensors are not supported.</li>
            <li>Per-tensor is supported. </li>
            <li>If this parameter is not used, pass <code>nullptr</code>.</li>
            <li>For details about the constraints, see <a href="#constraints">Constraints</a>.</li>
        </ul>
        </td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>See <a href="#INT8">Restrictions on the number of input parameters and input and output data formats related to INT8/FP8 quantization</a>.</td>
        <td>-</td>
    </tr>
    <tr>
        <td>deqScale2Optional</td>
        <td>Input</td>
        <td>Dequantization factor after BMM2.</td>
        <td>
        <ul>
            <li>Empty tensors are not supported.</li>
            <li>Per-tensor is supported. </li>
            <li>When the full quantization function is used, the value of this parameter is calculated based on the actual quantization process.</li>
            <li>If this parameter is not used, pass <code>nullptr</code>.</li>
            <li>For details about the constraints, see <a href="#constraints">Constraints</a>.</li>
        </ul>
        </td>
        <td>UINT64, FLOAT32</td>
        <td>ND</td>
        <td>See <a href="#INT8">Restrictions on the number of input parameters and input and output data formats related to INT8/FP8 quantization</a>.</td>
        <td>-</td>
    </tr>
    <tr>
        <td>quantScale2Optional</td>
        <td>Input</td>
        <td>Output quantization factor.</td>
        <td>
        <ul>
            <li>Empty tensors are not supported.</li>
            <li>Per-tensor and per-channel are supported. </li>
            <li>If this parameter is not used, pass <code>nullptr</code>.</li>
            <li>For details about the constraints, see <a href="#constraints">Constraints</a>.</li>
        </ul>
        </td>
        <td>FLOAT32, BFLOAT16</td>
        <td>ND</td>
        <td>See <a href="#INT8">Restrictions on the number of input parameters and input and output data formats related to INT8/FP8 quantization</a>.</td>
        <td>-</td>
    </tr>
    <tr>
        <td>quantOffset2Optional</td>
        <td>Input</td>
        <td>Output quantization offset.</td>
        <td>
        <ul>
            <li>Empty tensors are not supported.</li>
            <li>Per-tensor and per-channel are supported. </li>
            <li>If this parameter is not used, pass <code>nullptr</code>.</li>
            <li>For details about the constraints, see <a href="#constraints">Constraints</a>.</li>
        </ul>
        </td>
        <td>FLOAT32, BFLOAT16</td>
        <td>ND</td>
        <td>Same as <code>quantScale2Optional</code>.</td>
        <td>-</td>
    </tr>
    <tr>
        <td>antiquantScaleOptional</td>
        <td>Input</td>
        <td>Fake-quantization factor.</td>
        <td>Not supported.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>antiquantOffsetOptional</td>
        <td>Input</td>
        <td>Fake-quantization offset.</td>
        <td>Not supported.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr> 
    <tr>
        <td>blockTableOptional</td>
        <td>Input</td>
        <td>Block mapping table used for KV storage in paged attention.</td>
        <td>
        <ul>
            <li>Empty tensors are not supported.</li>
            <li>If this parameter is not used, pass <code>nullptr</code>.</li>
        </ul>
        </td>
        <td>INT32</td>
        <td>ND</td>
        <td>The length of the first dimension must be equal to <code>B</code>, and the length of the second dimension must be greater than or equal to <code>maxBlockNumPerSeq</code> (the maximum number of blocks corresponding to <code>actualSeqLengthsKv</code> in different batches).</td>
        <td>-</td>
    </tr>
    <tr> 
        <td>queryPaddingSizeOptional</td>
        <td>Input</td>
        <td>Whether the data in each batch of <code>query</code> is right-aligned and the number of right-aligned elements.</td>
        <td>
        <ul>
            <li>Empty tensors are not supported.</li>
            <li>This parameter is valid only when <code>Q_S</code> is greater than <code>1</code>. In other scenarios, it is invalid.</li>
            <li>If this parameter is not used, pass <code>nullptr</code>.</li>
        </ul>
        </td>
        <td>INT64</td>
        <td>ND</td>
        <td><code>(1)</code></td>
        <td>-</td>
    </tr>
    <tr> 
        <td>kvPaddingSizeOptional</td>
        <td>Input</td>
        <td>Whether the data in each batch of <code>key</code>/<code>value</code> is right-aligned and the number of right-aligned elements.</td>
        <td>
        <ul>
            <li>Empty tensors are not supported.</li>
            <li>If this parameter is not used, pass <code>nullptr</code>.</li>
        </ul>
        </td>
        <td>INT64</td>
        <td>ND</td>
        <td><code>(1)</code></td>
        <td>-</td>
    </tr>
    <tr> 
        <td>keyAntiquantScaleOptional</td>
        <td>Input</td>
        <td>Indicates the dequantization factor of the key, which is used in the scenario where the KV fake-quantization parameters are separated.</td>
        <td>
        <ul>
            <li>Empty tensors are not supported.</li>
            <li>If this parameter is not used, pass <code>nullptr</code>.</li>
            <li>The per-tensor, per-channel, and per-token modes are supported. The per-tensor mode can be combined with the per-head mode, the per-token mode can be combined with the per-head mode, the per-token mode can be combined with the paged attention mode to manage scale/offset, the per-token mode can be combined with the per-head mode and the paged attention mode to manage scale/offset, and the per-token-group mode is supported.</li>
            <li>For details about the constraints, see <a href="#constraints">Constraints</a>.</li>
        </ul>
        </td>
        <td>FLOAT16, BFLOAT16, FLOAT32, FLOAT8_E8M0</td>
        <td>ND</td>
        <td>See <a href="#constraints">Constraints</a>.</td>
        <td>-</td>
    </tr>
    <tr> 
        <td>keyAntiquantOffsetOptional</td>
        <td>Input</td>
        <td>Key dequantization offset when the KV fake-quantization parameters are separated.</td>
        <td>
        <ul>
            <li>Empty tensors are not supported.</li>
            <li>The shape must be the same as that of <code>keyAntiquantScaleOptional</code>.</li>
            <li>If this parameter is not used, pass <code>nullptr</code>.</li>
            <li>The per-tensor, per-channel, and per-token modes are supported. The per-tensor mode can be combined with the per-head mode, the per-token mode can be combined with the per-head mode, the per-token mode can be combined with the paged attention mode to manage scale/offset, and the per-token mode can be combined with the per-head mode and the paged attention mode to manage scale/offset.</li>
            <li>For details about the constraints, see <a href="#constraints">Constraints</a>.</li>
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
        <td>Dequantization factor of the value, used in the scenario where the KV fake-quantization parameters are separated.</td>
        <td>
        <ul>
            <li>Empty tensors are not supported.</li>
            <li>If this parameter is not used, pass <code>nullptr</code>.</li>
            <li>The per-tensor, per-channel, and per-token modes are supported. The per-tensor mode can be combined with the per-head mode, the per-token mode can be combined with the per-head mode, the per-token mode can be combined with the paged attention mode to manage scale/offset, the per-token mode can be combined with the per-head mode and the paged attention mode to manage scale/offset, and the per-token-group mode is supported.</li>
            <li>For details about the constraints, see <a href="#constraints">Constraints</a>.</li>
        </ul>
        </td>
        <td>FLOAT16, BFLOAT16, FLOAT32, FLOAT8_E8M0</td>
        <td>ND</td>
        <td>See <a href="#constraints">Constraints</a>.</td>
        <td>-</td>
    </tr>
    <tr> 
        <td>valueAntiquantOffsetOptional</td>
        <td>Input</td>
        <td>Fake-dequantization factor of the value when the KV fake-quantization parameters are separated.</td>
        <td>
        <ul>
            <li>Empty tensors are not supported.</li>
            <li>The shape must be the same as that of <code>valueAntiquantScaleOptional</code>.</li>
            <li>If this parameter is not used, pass <code>nullptr</code>.</li>
            <li>The per-tensor, per-channel, and per-token modes are supported. The per-tensor mode can be combined with the per-head mode, the per-token mode can be combined with the per-head mode, the per-token mode can be combined with the paged attention mode to manage scale/offset, and the per-token mode can be combined with the per-head mode and the paged attention mode to manage scale/offset.</li>
            <li>For details about the constraints, see <a href="#constraints">Constraints</a>.</li>
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
        <td>System prefix of <code>key</code> in the attention structure.</td>
        <td>
        <ul>
            <li>Empty tensors are not supported.</li>
            <li>If this parameter is not used, pass <code>nullptr</code>.</li>
            <li>For details about the constraints, see <a href="#constraints">Constraints</a>.</li>
        </ul>
        </td>
        <td>FLOAT16, BFLOAT16, INT8, INT4/INT32</td>
        <td>ND</td>
        <td>
        <ul>
            <li>When the input layout is <code>BSH</code>, the shape is <code>(1, prefix_S, H = KV_N × KV_D)</code>.</li>
            <li>When the input layout is <code>BSND</code>, the shape is <code>(1, prefix_S, KV_N, KV_D)</code>.</li>
            <li>When the input layout is <code>BNSD</code> or <code>BNSD_BSND</code>, the shape is <code>(1, KV_N, prefix_S, KV_D)</code>.</li>
        </ul>
        </td>
        <td>×</td>
    </tr>
    <tr> 
        <td>valueSharedPrefixOptional</td>
        <td>Input</td>
        <td>System prefix of <code>value</code> in the attention structure.</td>
        <td>
        <ul>
            <li>Empty tensors are not supported.</li>
            <li>If this parameter is not used, pass <code>nullptr</code>.</li>
            <li>For details about the constraints, see <a href="#constraints">Constraints</a>.</li>
        </ul>
        </td>
        <td>FLOAT16, BFLOAT16, INT8, INT4/INT32</td>
        <td>ND</td>
        <td>
        <ul>
            <li>When the input layout is <code>BSH</code>, the shape is <code>(1, prefix_S, H = KV_N × KV_D)</code>.</li>
            <li>When the input layout is <code>BSND</code>, the shape is <code>(1, prefix_S, KV_N, KV_D)</code>.</li>
            <li>When the input layout is <code>BNSD</code> or <code>BNSD_BSND</code>, the shape is <code>(1, KV_N, prefix_S, KV_D)</code>.</li>
        </ul>
        </td>
        <td>×</td>
    </tr>
    <tr> 
        <td>actualSharedPrefixLenOptional</td>
        <td>Input</td>
        <td>Valid sequence length of <code>keySharedPrefix</code>/<code>valueSharedPrefix</code>.</td>
        <td>
        <ul>
            <li>If this parameter is not used, pass <code>nullptr</code>, indicating that the sequence length is the same as <code>S</code> of <code>keySharedPrefix</code>/<code>valueSharedPrefix</code>.</li>
            <li>The valid sequence length in this input parameter must be less than or equal to that in <code>keySharedPrefix</code>/<code>valueSharedPrefix</code>.</li>
            <li>For details about the constraints, see <a href="#constraints">Constraints</a>.</li>
        </ul>
        </td>
        <td>INT64</td>
        <td>-</td>
        <td><code>(1)</code></td>
        <td>-</td>
    </tr>
    <tr>
        <td>queryRopeOptional</td>
        <td>Input</td>
        <td>Rope information of <code>query</code> in the MLA structure.</td>
        <td>
        <ul>
            <li>Empty tensors are not supported.</li>
            <li>For details about the constraints, see <a href="#constraints">Constraints</a>.</li>
        </ul>
        </td>
        <td>FLOAT16, BFLOAT16</td>
        <td>ND</td>
        <td> In <code>queryRope</code>, dimension <code>d</code> of the shape is <code>64</code>, and the values of other dimensions are the same as those of <code>query</code>.</td>
        <td>×</td>
    </tr>
    <tr>
        <td>keyRopeOptional</td>
        <td>Input</td>
        <td>Rope information of <code>key</code> in the MLA structure.</td>
        <td>
        <ul>
            <li>Empty tensors are not supported.</li>
            <li>For details about the constraints, see <a href="#constraints">Constraints</a>.</li>
        </ul>
        </td>
        <td>FLOAT16, BFLOAT16</td>
        <td>ND</td>
        <td> In <code>keyRope</code>, dimension <code>d</code> of the shape is <code>64</code>, and the values of other dimensions are the same as those of <code>key</code>.</td>
        <td>×</td>
    </tr>
    <tr>
        <td>keyRopeAntiquantScaleOptional</td>
        <td>Input</td>
        <td>Dequantization factor of the rope information of <code>key</code> in the MLA structure.</td>
        <td>
        <ul>
            <li>Empty tensors are not supported.</li>
            <li>This parameter is reserved and does not take effect in the current version. Pass <code>nullptr</code>.</li>
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
        <td>Factor for dequantizing the query.</td>
        <td>
        <ul>
            <li>Empty tensors are not supported.</li>
            <li>This parameter is involved in the full quantization scenario. The quantization mode supports the per-token + per-head mode.</li>
            <li>If this parameter is not used, pass <code>nullptr</code>.</li>
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
        <td>Absorbs attention scores through learnable sink tokens.</td>
        <td>
        <ul>
            <li>Only non-quantization scenarios are supported.</li>
            <li><code>V_D</code> can only be <code>128</code>/<code>64</code>.</li>
            <li>PSE, left padding, common prefix, and post-quantization are not supported.</li>
        </ul>
        </td>
        <td>BFLOAT16</td>
        <td>ND</td>
        <td>(Q_N)</td>
        <td>×</td>
    </tr>
    <tr> 
        <td>qStartIdxOptional</td>
        <td>Input</td>
        <td>Global start index of the query sequence for the current chunk in an outer splitting scenario.</td>
        <td>
        <ul>
            <li>This parameter is valid in the internal PSE generation scenario (pseType is 2 or 3). In other scenarios, nullptr can be passed.</li>
            <li>If pseType is 2 or 3, this parameter is not passed and the value 0 is used.</li>
            <li>For details about the constraints, see <a href="#constraints">Constraints</a>.</li>
        </ul>
        </td>
        <td>INT64</td>
        <td>ND</td>
        <td>See <a href="#constraints">Constraints</a>.</td>
        <td>-</td>
    </tr>
    <tr> 
        <td>kvStartIdxOptional</td>
        <td>Input</td>
        <td>Global start index of the key/value sequence for the current chunk in an outer splitting scenario.</td>
        <td>
        <ul>
            <li>This parameter is valid in the internal PSE generation scenario (pseType is 2 or 3). In other scenarios, nullptr can be passed.</li>
            <li>If pseType is 2 or 3, this parameter is not passed and the value 0 is used.</li>
            <li>For details about the constraints, see <a href="#constraints">Constraints</a>.</li>
        </ul>
        </td>
        <td>INT64</td>
        <td>ND</td>
        <td>See <a href="#constraints">Constraints</a>.</td>
        <td>-</td>
    </tr>
    <tr>
        <td>numHeads</td>
        <td>Input</td>
        <td>Number of heads in <code>query</code>.</td>
        <td>In the BNSD, BSND, BNSD_BSND, BSND_BNSD, BNSD_NBSD, BSND_NBSD, TND, NTD, NTD_TND, and TND_NTD scenarios, the value must be the same as the N-axis shape value of the query in the shape. Otherwise, an exception occurs.</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>scaleValue</td>
        <td>Input</td>
        <td>Reciprocal of the square root of <code>d</code> in the formula.</td>
        <td>
        <ul>
            <li>Its data type and the data type of `query` must meet the type deduction rules. </li>
            <li>If no specific value is required, <code>1.0</code> is recommended. </li>
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
        <td>Number of preceding tokens to associate in attention computation for sparse computation.</td>
        <td>
        <ul>
            <li>If the value is not specified, you are advised to pass 2147483647.</li>
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
        <td>Number of succeeding tokens to associate in attention computation.</td>
        <td>
        <ul>
            <li>If this parameter is not specified, you are advised to set it to 2147483647.</li>
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
        <td>Layout of the input <code>query</code>,<code>key</code>, and <code>value</code>.</td>
        <td>
        <ul>
            <li>Currently, the following formats are supported: BSH, BSND, BNSD, BNSD_BSND (when the input is BNSD, the output format is BSND), BSND_BNSD (when the input is BSND, the output format is BNSD), BSH_BNSD (when the input is BSH, the output format is BNSD), BNSD_NBSD (when the input is BNSD, the output format is NBSD), BSND_NBSD (when the input is BSND, the output format is NBSD), BSH_NBSD (when the input is BSH, the output format is NBSD), TND (for details about the restrictions in TND-related scenarios, see <a href="#constraints"> Restrictions </a>), NTD, NTD_TND (when the input is NTD, the output format is TND), and TND_NTD (when the input is TND, the output format is NTD). If no specific value is required, <code>BSH</code> is recommended.</li>
            <li>Note that when the layout format contains an underscore (_), the part on the left of the underscore indicates the layout of the input query, and the part on the right of the underscore indicates the output format.</li>
            <li>The query, key, and value data formats can be interpreted from multiple dimensions. B (Batch) indicates the batch size of input samples, S (Seq-Length) indicates the sequence length of input samples, H (Hidden-Size) indicates the size of the hidden layer, N (Head-Num) indicates the number of heads, D (Head-Dim) indicates the minimum unit size of the hidden layer, and D = H/N. T indicates the sum of the sequence lengths of all batch input samples.</li>
            <li>inputLayout=BSH_BNSD, BSND_BNSD, NTD, and NTD_TND support only the following cases: Q_D = K_D = V_D = 64 or 128, or Q_D = K_D = 192 and V_D = 128.</li>
            <li>inputLayout=BNSD_BSND supports only the following cases: Q_D = K_D = V_D is 16-aligned (32-aligned when the output dtype is int8), or Q_D = K_D = 192 and V_D = 128.<br></li>
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
        <td>Number of heads in <code>key</code> and <code>value</code>.</td>
        <td>
        <ul>
            <li>If the user does not specify the value, it is recommended that 0 be passed, indicating that the number of heads in the key/value is the same as that in the query.</li>
            <li>numHeads must be exactly divisible by numKeyValueHeads. In GQA non-quantization and Prefill MLA non-quantization scenarios, the ratio of numHeads to numKeyValueHeads is not limited. In other scenarios, the ratio of numHeads to numKeyValueHeads cannot be greater than 64.</li>
            <li>In the BNSD, BSND, BNSD_BSND, BSND_BNSD, BNSD_NBSD, BSND_NBSD, TND, NTD, NTD_TND, and TND_NTD scenarios, the value must be the same as the N-axis shape value of the key/value in the shape. Otherwise, an exception occurs.</li>
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
        <td>
        <ul>
            <li>If inputLayout is set to TND, TND_NTD, or NTD_TND, see <a href="#constraints">Constraints</a> for the comprehensive constraints.</li>
            <li>When sparseMode is set to 0, it indicates the defaultMask mode. If attenmask is not passed, the mask operation is not performed, and preTokens and nextTokens are ignored (the internal value is set to INT_MAX). If attenmask is passed, the complete attenmask matrix (S1 x S2) must be passed, indicating that the part between preTokens and nextTokens needs to be calculated. The value of preTokens + nextTokens must be greater than or equal to 0. </li>
            <li>When sparseMode is set to 1, it indicates the allMask mode. The complete attenmask matrix (S1 x S2) must be passed.</li>
            <li>When sparseMode is set to 2, it indicates the mask in leftUpCausal mode. The optimized attenmask matrix (2048 x 2048) must be passed.</li>
            <li>When sparseMode is set to 3, it indicates the mask in rightDownCausal mode. The optimized attenmask matrix (2048 x 2048) must be passed, which corresponds to the lower triangular scenario where the right vertex is used for division.</li>
            <li>When sparseMode is set to 4, it indicates the mask in band mode. The optimized attenmask matrix (2048 x 2048) needs to be passed. The value of preTokens + nextTokens must be greater than or equal to 0.</li>
            <li>When sparseMode is set to 5, 6, 7, or 8, it indicates prefix, global, dilated, or block_local, respectively. Currently, these modes are not supported.</li>
            <li>If no specific value is required, <code>0</code> is recommended.</li>
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
        <td>Choice between the high-precision and high-performance modes.</td>
        <td>
        <ul>
            <li>When innerPrecise is set to 0, the high precision mode is enabled, and row invalidation correction is not performed.</li>
            <li>When innerPrecise is set to 1, the high performance mode is enabled, and row invalidation correction is not performed.</li>
            <li>When innerPrecise is set to 2, the high precision mode is enabled, and row invalidation correction is performed.</li>
            <li>When innerPrecise is set to 3, the high performance mode is enabled, and row invalidation correction is performed.</li>
            <li>When sparse_mode is set to 0 or 1 and a user-defined mask is passed, you are advised to enable row invalidation correction.</li>
            <li>High precision and high performance are not distinguished for BFLOAT16 and INT8. Row invalidation correction takes effect for FLOAT16, BFLOAT16, and INT8.</li>
            <li>Currently, 0 and 1 are reserved. If all elements in a row are 1s in the mask used for computation, the precision may be reduced. In this case, you can set this parameter to `2` or `3` to enable invalid row correction and improve precision. However, this configuration may degrade performance.</li>
            <li>If the operator can determine that invalid rows exist, invalid row computation is automatically enabled. For example, this parameter is automatically enabled in the Sq > Skv scenario when sparse_mode is set to 3.</li>
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
        <td>Maximum number of tokens in each block for KV storage in paged attention.</td>
        <td>If this parameter is not passed, the value 0 is used.</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>antiquantMode</td>
        <td>Input</td>
        <td>Fake-quantization mode</td>
        <td>Not supported.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>softmaxLseFlag</td>
        <td>Input</td>
        <td>Whether to output <code>softmax_lse</code>.</td>
        <td>
        <ul>
            <li>S-axis extrusion (output increase) is supported.</li>
            <li>If no specific value is required, <code>false</code> is recommended.</li>
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
        <td>Key dequantization mode.</td>
        <td>
        <ul>
            <li>If no specific value is required, <code>0</code> is recommended.</li>
            <li>Except for the scenario where <code>keyAntiquantMode</code> is <code>0</code> and <code>valueAntiquantMode</code> is <code>1</code>, the value must be the same as that of <code>valueAntiquantMode</code>.</li>
            <li>When keyAntiquantMode is 0, it indicates the per-channel mode (including per-tensor).</li>
            <li>When keyAntiquantMode is 1, it indicates the per-token mode.</li>
            <li>When keyAntiquantMode is 2, it indicates the per-tensor and per-head mode.</li>
            <li>When keyAntiquantMode is 3, it indicates the per-token and per-head mode.</li>
            <li>When keyAntiquantMode is 4, it indicates the per-token and per-head mode, with the paged attention mode used to manage the scale/offset mode.</li>
            <li>When keyAntiquantMode is 5, it indicates the per-token and per-head mode, with the paged attention mode used to manage the scale/offset mode.</li>
            <li>When keyAntiquantMode is 6, it indicates the per-token-group mode.</li>
            <li>If a value other than 0 to 6 is passed in, an exception occurs.</li>
            <li>For details about the constraints, see <a href="#constraints">Constraints</a>.</li>
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
        <td>Reverse quantization mode of the value.</td>
        <td>
        <ul>
            <li>The mode ID is the same as that of keyAntiquantMode.</li>
            <li>If no specific value is required, <code>0</code> is recommended.</li>
            <li>For details about the constraints, see <a href="#constraints">Constraints</a>.</li>
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
        <td>Reverse quantization mode of the query.</td>
        <td>
        <ul>
            <li>The mode ID is the same as that of keyAntiquantMode.</li>
            <li>If no specific value is required, <code>0</code> is recommended.</li>
            <li>For details about the constraints, see <a href="#constraints">Constraints</a>.</li>
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
        <td>PSE mode.</td>
        <td>
        <ul>
            <li>The value can be 0, 2, or 3. (The value cannot be 3 when pseType is set to 1.)</li>
            <li>When pseType is set to 0, the external input PSE is used, and the mul operation is performed before the add operation.</li>
            <li>When pseType is set to 2, the internal PSE is generated. The calculation formula is as follows: -alibi_slope * abs(i - j).</li>
            <li>When pseType is set to 3, the internal PSE is generated. The calculation formula is as follows: -alibi_slope * sqrt(abs(i - j)).</li>
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
        <td>Dimension <code>D</code> of this parameter must be the same as that of <code>value</code>, and other dimensions must be the same as the shapes of <code>query</code>.</td>
        <td>-</td>
    </tr>
    <tr>
        <td>softmaxLse</td>
        <td>Output</td>
        <td>The ring attention algorithm first obtains the max value from the result of query multiplied by key to obtain softmax_max. This max value is subtracted from the product before calculating the exponential, which is then summed to yield <code>softmax_sum</code>. Finally, the log of <code>softmax_sum</code> is added back to <code>softmax_max</code> to obtain the final result.</td>
        <td>
        <ul>
            <li>If the user does not specify this parameter, it is recommended that nullptr be passed.</li>
            <li>If the data is inf, it indicates invalid data. When softmaxLseFlag is set to False, if the tensor passed to softmaxLse is not empty, the tensor data is directly returned. If the tensor passed to softmaxLse is nullptr, a tensor with all 0s and shape of {1} is returned.</li>
        </ul>
        </td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>When softmaxLseFlag is set to True, the shape must be [B, N, Q_S, 1] in general. When inputLayout is set to TND/NTD_TND/TND_NTD, the shape must be [T, N, 1].</td>
        <td>-</td>
    </tr>
    <tr>
        <td>workspaceSize</td>
        <td>Output</td>
        <td>Size of the workspace to be allocated on the device.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>executor</td>
        <td>Output</td>
        <td>Operator executor, containing the operator computation process.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
    </tbody></table>

- **Returns**

    `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

    The first-phase API implements input parameter validation. The following error codes may be returned.

    <table style="undefined;table-layout: fixed;width: 1155px"><colgroup>
    <col style="width: 319px">
    <col style="width: 144px">
    <col style="width: 671px">
    </colgroup>
    <thead>
        <th>Return</th>
        <th>Error Code</th>
        <th>Description</th>
    </thead>
    <tbody>
        <tr>
            <td>ACLNN_ERR_PARAM_NULLPTR</td>
            <td>161001</td>
            <td>The passed <code>query</code>, <code>key</code>, <code>value</code>, or <code>attentionOut</code> is a null pointer.</td>
        </tr>
        <tr>
            <td>ACLNN_ERR_PARAM_INVALID</td>
            <td>161002</td>
            <td>The data types and formats of query, key, value, pseShiftOptional, attenMaskOptional, and attentionOut are not supported.</td>
        </tr>
        <tr>
            <td>ACLNN_ERR_RUNTIME_ERROR</td>
            <td>361001</td>
            <td>An exception occurred when the NPU runtime API was called.</td>
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
        <th>Name</th>
        <th>Input/Output</th>
        <th>Description</th>
        </tr></thead>
    <tbody>
        <tr>
        <td>workspace</td>
        <td>Input</td>
        <td>Memory address of the workspace to be allocated on the device.</td>
        </tr>
        <tr>
        <td>workspaceSize</td>
        <td>Input</td>
        <td>Size of the workspace to be allocated on the device, which is obtained by the first-phase API aclnnFusedInferAttentionScoreV5GetWorkspaceSize.</td>
        </tr>
        <tr>
        <td>executor</td>
        <td>Input</td>
        <td>Operator executor, containing the operator computation process.</td>
        </tr>
        <tr>
        <td>stream</td>
        <td>Input</td>
        <td>Stream for executing the task.</td>
        </tr>
    </tbody>
    </table>

- **Returns**

    `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - `aclnnFusedInferAttentionScoreV5` defaults to a deterministic implementation.
- Common constraints
    - Processing when the input parameter is empty:
        - An empty tensor indicates that the shapeSize of the required input and output is 0. In the empty tensor scenario, if attentionOut is empty, the return value is empty. Otherwise, the return value is all 0s. If lse is empty, the return value is empty. If lse is not empty, the return value is all inf. When the tensor is not empty, the input is intercepted normally.
        - The shapeSize of all tensors in query and attentionOut is 0, which is an empty tensor.
        - The shapeSize of all tensors in query and attentionOut is not 0. If lse is not empty and the shapeSize of all tensors in key and value is 0, the tensor is an empty tensor.
        - If both attentionOut and lse are empty, the tensor is an empty tensor.
        - If the tensor is empty, the verification process is skipped. Otherwise, the normal verification process is performed.
    - The restrictions in the BNSD_BSND, BSH_BNSD, BSND_BNSD, BSH_NBSD, BSND_NBSD, and BNSD_NBSD scenarios are as follows:
        - When dimension `d` of `query` is `512`:
          - Only BSH_NBSD, BSND_NBSD, and BNSD_NBSD are supported.
          - Only the decode mla scenario is supported. The queryRope and keyRope cannot be empty, and the d value of queryRope and keyRope is 64.
        - When dimension `d` of `query` is not `512`:
          - Only BNSD_BSND, BSH_BNSD, and BSND_BNSD are supported.
          - The prefill mla or gqa non-quantization scenario is supported. In the prefill mla scenario, either of the following conditions must be met:
            - The d value of the query, key, and value is 128. The queryRope and keyRope cannot be empty, and the d value of queryRope and keyRope is 64.
            - The d value of the query and key is 192, the d value of the value is 128, and the queryRope and keyRope are empty.
          - In the gqa non-quantization scenario, BSH_BNSD and BSND_BNSD support only D = 64 or D = 128. BNSD_BSND supports only D = 16 alignment (32 alignment when the output dtype is int8).
          - In the BSH_BNSD and BSND_BNSD scenarios, left padding, tensorlist, pse, and prefix are not supported.
          - BSH_BNSD and BSND_BNSD do not support fake-quantization. BNSD_BSND supports fake-quantization.
          - In the fake-quantization scenario, BNSD_BSND does not support QS = 1.
    - The restrictions on the query, key, and value inputs in the TND, NTD, TND_NTD, and NTD_TND scenarios are as follows:
        - When dimension `d` of `query` is `512`:
          - Only TND and TND_NTD are supported.
          - Only the decode mla scenario is supported. The queryRope and keyRope cannot be empty, and the d value of queryRope and keyRope is 64.
          - Left padding, tensorlist, pseType=0, prefix, and fake-quantization are not supported.
        - When dimension `d` of `query` is not `512`:
          - Only TND, NTD, and NTD_TND are supported.
          - The prefill mla or gqa non-quantization scenario is supported. In the prefill mla scenario, either of the following conditions must be met:
            - The d value of the query, key, and value is 128. The queryRope and keyRope cannot be empty, and the d value of queryRope and keyRope is 64.
            - The d value of the query and key is 192, the d value of the value is 128, and the queryRope and keyRope are empty.
          - In the GQA non-quantization scenario, NTD and NTD_TND support only D = 64 or D = 128.
          - Left padding, tensorlist, pseType = 0, prefix, and fake-quantization are not supported.
- <a id="public"></a>General scenarios
    <table style="undefined;table-layout: fixed; width: 1000px">
        <colgroup>
            <col style="width: 150px">
            <col style="width: 100px">
            <col style="width: 750px">
        </colgroup>
        <thead>
            <tr>
                <th>Parameter</th>
                <th>Dimension</th>
                <th>Restriction</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td rowspan="4">query/key/value</td>
                <td>B</td>
                <td><ul><li>The B axis must be less than or equal to 65536.</li>
                    <li>In non-contiguous scenarios, <code>batch</code> in the tensor list of <code>key</code> and <code>value</code> must be <code>1</code>. The number of <code>batch</code> elements is equal to <code>B</code> in <code>query</code>. <code>N</code> and <code>D</code> must be the same. Due to the tensorlist restriction, the B value cannot be greater than 256 in the discontinuous scenario.</li></ul>
                </td>
            </tr>
            <tr>
                <td>N</td>
                <td><ul><li>In the GQA non-quantization scenario and prefill MLA non-quantization scenario, the N axis is not restricted.</li>
                    <li>In other scenarios, the N axis must be less than or equal to 256</li></ul>.
                </td>
            </tr>
            <tr>
                <td>S</td>
                <td><ul><li>When Q_S > 1, the S axis supports a value less than or equal to 20971520 (20M). In some long-sequence scenarios, if the computation workload is too heavy, the operator execution may time out (an AI Core error is reported, and the errorStr is "timeout or trap error"). In this case, you are advised to split the S axis. Note that the computation workload is affected by B, S, N, and D. The larger the value, the heavier the computation workload.<br>
                    Typical long-sequence scenarios where timeout occurs (that is, the product of B, S, N, and D is large) include but are not limited to the following:<ul>
                    <li>B=1, Q_N=20, Q_S=2097152, D = 256, KV_N=1, KV_S=2097152;</li>
                    <li>B=1, Q_N=2, Q_S=20971520, D = 256, KV_N=2, KV_S=20971520;</li>
                    <li>B=20, Q_N=1, Q_S=2097152, D = 256, KV_N=1, KV_S=2097152;</li>
                    <li>B=1, Q_N=10, Q_S=2097152, D = 512, KV_N=1, KV_S=2097152</li></ul>
                    </li></ul>
                </td>
            </tr>
            <tr>
                <td>D</td>
                <td><ul>
                    <li>The D axis is less than or equal to 512.</li>
                    <li>In the fake-quantization scenario, the aclnn single-operator calling supports the KV INT4 input or the INT4 concatenated into the INT32 input. (You are advised to use dynamicQuant to generate the INT4 data because dynamicQuant is an INT32 including eight INT4s.) In this case, the D value of KV is 1/8 of the actual value. (The same applies to the prefix.)</li>
                    <li>When the key and value inputs are of type FLOAT4_E2M1/INT4 (INT32), the D axis of the query and the D axis of the key and value must be 64-byte aligned. (For INT32, only the D axis of the key and value must be 8-byte aligned.)</li>
                </ul></td>
            </tr>
            <tr>
                <td colspan="3"><ul>
                    <li>When Q_S is 1, the INT8 input types of query, key, and value are not supported.</li>
                   <li>Generally, the shapes of the tensors in the key and value parameters must be the same. However, in non-quantization scenarios, the head dimension of the query and key parameters can be different from that of the value parameter, and the head dimension of the three parameters must be less than or equal to 128. In this scenario, other advanced features cannot be used together with sparse = 0/2/3 and mask, FD, and row invalid.</li></ul></td>
            </tr>
        </tbody>
    </table>

- <a id="pseShift"></a>PseShift
    <div style="overflow-x: auto;">
    <table style="undefined;table-layout: fixed;  width: 1560px">
        <colgroup>
            <col style="width: 100px">
            <col style="width: 130px">
            <col style="width: 190px">
            <col style="width: 130px">
            <col style="width: 180px">
            <col style="width: 280px">
            <col style="width: 550px">
        </colgroup>
        <thead>
        <tr>
            <th>pseType</th>
            <th colspan="3" style="text-align: center;">Supported Scenario</th>
            <th>pseShiftOptional Data Type Constraint</th>
            <th>Shape Constraint</th>
            <th>Remarks</th>
        </tr>
        </thead>
        <tbody>
            <td rowspan="6">0</td>
            <tr>
                <td rowspan="3">P_S1 (the third dimension of the PSE shape) > 1</td>
                <td rowspan="3">Data type of <code>query</code></td>
                <td>FLOAT16</td>
                <td>FLOAT16</td>
                <td rowspan="3">(B,Q_N,P_S1,P_S2), (1,Q_N,P_S1,P_S2)</td>
                <td rowspan="3">
                <ul>
                <li>When the data type of <code>query</code> is FLOAT16 and <code>pseShift</code> exists, the high-precision mode is forcibly used. The corresponding restrictions are the same as those of the high-precision mode.</li>
                <li>P_S1 must be greater than or equal to the S length of the query, and P_S2 must be greater than or equal to the S length of the key. In the prefix scenario, P_S2 must be greater than or equal to the sum of actualSharedPrefixLen and the S length of the key.</li>
                <li>It is recommended that P_S2 be padded to 32 bytes to improve performance.</li>
                <li>In the fake-quantization scenario, this scenario is not supported when the S length of the query is 1.</li>
                </ul>
                </td>
            </tr>
            <tr>
                <td>BFLOAT16</td>
                <td>BFLOAT16</td>
            </tr>
            <tr> 
                <td>INT8</td> 
                <td>FLOAT16</td> 
            </tr>
            <tr>
                <td rowspan="2">P_S1 (the third dimension of the PSE shape) = 1</td>
                <td rowspan="2">Data type of <code>query</code></td>
                <td>FLOAT16</td>
                <td>FLOAT16</td>
                <td rowspan="2">(B,Q_N,1,P_S2), (1,Q_N,1,P_S2)</td>
                <td rowspan="2">
                <ul>
                <li>P_S2 must be greater than or equal to the length of S in the key. In the prefix scenario, P_S2 must be greater than or equal to the sum of actualSharedPrefixLen and the length of S in the key. </li>
                <li>It is recommended that P_S2 be padded to 32 bytes for alignment to improve performance.</li> 
                </ul>
                </td>
            </tr>
            <tr>
                <td>BFLOAT16</td>
                <td>BFLOAT16</td>
            </tr>
            <tr>
                <td rowspan="1">1</td>
                <td colspan="3">FA inference scenarios are not supported. Only FA training scenarios are supported.</td> 
                <td colspan="1">-</td>
                <td colspan="1">-</td>
                <td colspan="1">-</td>
            </tr>
            <tr> 
                <td rowspan="2">2/3</td>
                <td rowspan="2">-</td>
                <td rowspan="2">Data type of <code>query</code></td>
                <td>FLOAT16</td>
                <td rowspan="2">FLOAT32</td>
                <td rowspan="2">[N]</td>
                <td rowspan="2">
                <ul>                
                <li>N=numHeads, which is used to pass alibi_slope.</li>
                <li>Currently, only the scenario where the length of qs is the same as that of kvs in each batch is supported.</li> 
                <li>MLA and left padding scenarios are not supported.</li>
                <li>If qStartIdxOptional or kvStartIdxOptional is not empty, the first data in the list is used as qStartIdx or kvStartIdx. In addition, the value range of qStartIdx and kvStartIdx must be [–2147483648, 2147483647], and the value range of kvStartIdx – qStartIdx must be [–1048576, 1048576].</li>
                </ul>
                </td>
            </tr>
            <tr>
                <td>BFLOAT16</td>
            </tr>
        </tbody>
    </table></div>

- <a id="Mask"></a>Mask
    <table style="undefined;table-layout: fixed; width: 1480px"><colgroup>
        <col style="width: 100px">
        <col style="width: 740px">
        <col style="width: 280px">
        <col style="width: 360px">
        </colgroup>
        <thead>
            <tr>
                <th>sparseMode</th>
                <th>Description</th>
                <th>Shape Constraint</th>
                <th>Remarks</th>
            </tr>
        </thead>
        <tbody>
        <tr>
            <td>0</td>
            <td>defaultMask</td>
            <td>(B,M_S1,M_S2), (1,M_S1,M_S2), (B,1,M_S1,M_S2), (1,1,M_S1,M_S2)</td>
            <td>
            <ul>
            <li>M_S1 must be greater than or equal to the S length of the query, and M_S2 must be greater than or equal to the S length of the key.</li>
            <li>If attenmask is not passed, the mask operation is not performed. Alternatively, if attenMask is passed in the left padding scenario, preTokens and nextTokens are ignored.</li>
            </ul>
            </td>
        </tr>
        <tr>
            <td>1</td>
            <td><code>allMask</code> mode. A complete <code>attenMask</code> matrix must be passed.</td>
            <td>(B,M_S1,M_S2), (1,M_S1,M_S2), (B,1,M_S1,M_S2), (1,1,M_S1,M_S2)</td>
            <td>
            <ul>
            <li>M_S1 must be greater than or equal to the S length of the query, and M_S2 must be greater than or equal to the S length of the key.</li>
            <li>The input parameters preTokens and nextTokens are ignored and assigned values according to related rules.</li>
            </ul>
            </td>
        </tr>
        <tr>
            <td>2</td>
            <td><code>leftUpCausal</code> mode. An optimized <code>attenMask</code> matrix needs to be passed.</td>
            <td>(S,S), (1,S,S), (1,1,S,S)</td>
            <td rowspan="2">
            <ul>
            <li>The value of S must be fixed to 2048.</li>
            <li>The input parameters preTokens and nextTokens are ignored and assigned values according to related rules.</li>
            <li>The passed attenMask is a lower triangular matrix, and the diagonal is all 0s. If attenMask is nullptr or the passed shape is incorrect, an error is reported.</li>
            </ul>
            </td>
        </tr>
        <tr>
            <td>3</td>
            <td><code>rightDownCausal</code> mode. This corresponds to a lower-triangular matrix partitioned by the top-right vertex. An optimized <code>attenMask</code> matrix needs to be passed.</td>
            <td>(S,S), (1,S,S), (1,1,S,S)</td>
        </tr>
        <tr>
            <td>4</td>
            <td><code>band</code> mode. An optimized <code>attenMask</code> matrix needs to be passed.</td>
            <td>(S,S), (1,S,S), (1,1,S,S)</td>
            <td>
            <ul>
            <li>The value of S must be fixed at 2048.</li>
            <li>The input attenMask is a lower triangular matrix, with all zeros on the diagonal. If attenMask is nullptr or the input shape is incorrect, an error is reported.</li>
            </ul>
            </td>
        </tr>
        <tr>
        <td colspan="4"><ul>
            <li>When the data type of attenMask is INT8 or UINT8, the value in the tensor must be 0 or 1</li>.
            <li>This parameter takes effect when sparseMode Q_S>1 is used in non-<a href="#MLA">MLA scenarios.</a> </li>
        </ul></td>
        </tr>
        </tbody>
    </table>

- <a id="actSeqLen"></a>ActualSeqLen
    <table style="undefined;table-layout: fixed; width: 1148px"><colgroup>
    <col style="width: 195px">
    <col style="width: 156px">
    <col style="width: 608px">
    <col style="width: 189px">
        </colgroup>
        <thead>
            <tr>
                <th>Parameter</th>
                <th>Layout</th>
                <th>Description</th>
                <th>Restriction</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td rowspan="2">actualSeqLengths</td>
                <td>Not TND</td>
                <td>This input parameter is optional. Its length is 1 or greater than or equal to the batch value of the query. The value of this parameter indicates the actual length of each batch. The value must be less than or equal to Q_S</td>.
                <td rowspan="4">The value must be a non-negative number.</td>
            </tr>
            <tr>
                <td>TND</td>
                <td>This input parameter is mandatory. The bth value indicates the accumulated length of the S axis of the first b batches. The values must be in ascending order (greater than or equal to the previous value), and the length of this parameter indicates the total number of batches.</td>
            </tr>
            <tr>
                <td rowspan="2">actualSeqLengthsKv</td>
                <td>Not TND</td>
                <td>This input parameter is optional. Its length is 1 or greater than or equal to the batch value of the key/value. The value of this parameter indicates the actual length of each batch. The value must be less than or equal to KV_S.</td>
            </tr>
            <tr>
                <td>TND</td>
                <td>This input parameter is mandatory.<br>
                    In the non-PA scenario, the bth value indicates the accumulated length of the S axis of the first b batches. The values must be in ascending order (greater than or equal to the previous value), and the length of this parameter indicates the total number of batches.<br>
                    In the PA scenario, the length is equal to the batch value of the key/value, indicating the actual length of each batch. The value is less than or equal to KV_S.</td>
            </tr>
        </tbody>
    </table>

- <a id="AntiQuant"></a>Fake-quantization parameters
    <table style="undefined;table-layout: fixed;  width: 1380px">
        <colgroup>
            <col style="width: 200px">
            <col style="width: 150px">
            <col style="width: 100px">
            <col style="width: 160px">
            <col style="width: 380px">
            <col style="width: 280px">
        </colgroup>
        <thead>
            <tr>
                <th>Quantization mode</th>
                <th>KV data type</th>
                <th>Scenario</th>
                <th>keyAntiquantMode and valueAntiquantMode</th>
                <th>keyAntiquantScale and valueAntiquantScale</th>
                <th>keyAntiquantOffset and valueAntiquantOffset</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td rowspan="2">per-channel (per-tensor)</td>
                <td rowspan="2">The kv_dtype can be INT8, INT4 (INT32), HIFLOAT8, or FLOAT8_E4M3FN.</td>
                <td>Q_S>1</td>
                <td rowspan="2">0</td>
                <td>
                    <ul>
                        <li>Per-channel mode: The shape is (1, N, 1, D), (1, N, D), (1, H), (N, 1, D), (N, D), or (H). The parameter data type is the same as the query data type.</li>
                        <li>Per-tensor mode: The shape is (1). The data type is the same as the query data type. This mode is supported only when the key and value data types are both INT8.</li>
                    </ul>
                </td>
                <td rowspan="10">
                    <ul>
                        <li>Both <code>keyAntiquantOffset </code> and <code>valueAntiquantOffset</code> must be either null or non-null.</li>
                        <li>
                            If neither <code>keyAntiquantOffset</code> nor <code>valueAntiquantOffset</code> is null,
                            their shapes must be the same, except when <code>keyAntiquantMode</code> is <code>0</code> and <code>valueAntiquantMode</code> is <code>1</code>.
                        </li>
                    </ul>
                </td>
            </tr>
            <tr>
                <td>Q_S=1</td>
                <td>
                    <ul>
                        <li>Per-channel mode: The shape is (1, N, 1, D), (1, N, D), or (1, H). The parameter data type is the same as the query data type.</li>
                        <li>Per-tensor mode: The shape is (1). The data type is the same as the query data type. This mode is supported only when the key and value data types are INT8.</li>
                    </ul>
                </td>
            </tr>
            <tr>
                <td rowspan="2">per-token</td>
                <td rowspan="2">kv_dtype can be INT8, INT4 (INT32), or FLOAT8_E4M3FN.</td>
                <td>Q_S>1</td>
                <td rowspan="2">1</td>
                <td>The shape is (1, B, S) or (B, S). The data type is fixed to FLOAT32.</td>
            </tr>
            <tr>
                <td>Q_S=1</td>
                <td>The shape is (1, B, S). The data type is fixed to FLOAT32.</td>
            </tr>
            <tr>
                <td>Per-tensor + per-head</td>
                <td>kv_dtype can be INT8.</td>
                <td>Q_S=1</td>
                <td>2</td>
                <td>The shape is (N). The data type is the same as the query data type.</td>
            </tr>
            <tr>
                <td>Per-token + per-head</td>
                <td>kv_dtype can be INT8 or INT4(INT32).</td>
                <td>Q_S=1</td>
                <td>3</td>
                <td>The shape is (B, N, S), and the data type is fixed to FLOAT32.</td>
            </tr>
            <tr>
                <td rowspan="2">The per-token mode uses paged attention to manage scale/offset.</td>
                <td rowspan="2">kv_dtype can be INT8 or FLOAT8_E4M3FN.</td>
                <td>Q_S>1</td>
                <td rowspan="2">4</td>
                <td rowspan="2">The shape is (blocknum, blocksize), and the data type is fixed to FLOAT32.</td>
            </tr>
            <tr>
                <td>Q_S=1</td>
            </tr>
            <tr>
                <td>The per-token mode is used together with the per-head mode and paged attention is used to manage scale/offset.</td>
                <td>kv_dtype can be INT8.</td>
                <td>Q_S=1</td>
                <td>5</td>
                <td>The shape is (blocknum, N, blocksize), and the data type is fixed to FLOAT32.</td>
            </tr>
            <tr>
                <td rowspan="2">The key supports per-channel and the value supports per-token.</td>
                <td rowspan="2">kv_dtype can be INT8 or INT4(INT32).</td>
                <td>Q_S>1</td>
                <td rowspan="2">keyAntiquantMode is 0 and valueAntiquantMode is 1.</td>
                <td>For the key, per-channel is supported. The shape can be (1, N, 1, D), (1, N, D), (1, H), (N, 1, D), (N, D), or (H). The data type is the same as that of `query`.
                    For the value, per-token is supported. The shape can be (1, B, S) or (B, S) and the data type is fixed to FLOAT32.</td>
            </tr>
            <tr>
                <td>Q_S=1</td>
                <td>For the key, per-channel is supported. The shape can be (1, N, 1, D), (1, N, D), or (1, H). The data type is the same as that of `query`.
                    For the value, per-token is supported. The shape is (1, B, S) and the data type is fixed to FLOAT32</td>.
            </tr>
            <tr>
                <td>per-token-group</td>
                <td>The kv_dtype can be FLOAT4_E2M1</td>.
                <td>-</td>
                <td>6</td>
                <td>The shape is (1, B, N, S, D/32) and the data type is fixed to FLOAT8_E8M0</td>.
            </tr>
            <tr>
                <td colspan="8">
                    <ul>
                        <li>Post-quantization is not supported in the INT4 (INT32) and FLOAT4_E2M1 fake-quantization scenarios.</li>
                        <li>In the FLOAT8_E4M3 fake-quantization scenario, post-quantization is not supported when keyAntiquantMode and valueAntiquantMode are set to 1 or 4.</li>
                        <li>In the INT8 fake-quantization scenario, when keyAntiquantMode is 0 and valueAntiquantMode is 1, query and output support only FP16</li>.
                        <li>In the INT8/INT4 (INT32) fake-quantization scenario, when keyAntiquantMode and valueAntiquantMode are set to 2, 3, 4, or 5, Q_S supports only 1.</li>
                    </ul>
                </td>
            </tr>
        </tbody>
    </table>

- <a id="PagedAttention"></a>PagedAttention
    <table style="undefined;table-layout: fixed; width: 1354px">
        <colgroup>
            <col style="width: 155px">
            <col style="width: 169px">
            <col style="width: 550px">
            <col style="width: 600px">
        </colgroup>
        <thead>
            <tr>
                <th>Scenario/Feature</th>
                <th>Parameter</th>
                <th>Constraint</th>
                <th>Remarks</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td rowspan="2">PagedAttention</td>
                <td>blockSize</td>
                <td>
                    <ul>
                        <li>When PagedAttention is enabled and in the non-quantization scenario, blockSize must be a non-zero value. The restrictions are as follows:
                            In the MLA scenario, blockSize must be 16-byte aligned and cannot exceed 1024.
                            In the GQA scenario, when the head dimension of the query, key, and value is 64 or 128, blockSize must be 16-byte aligned and cannot exceed 1024.
                            In the GQA scenario, when the head dimension of the query, key, and value is not 64 or 128 and Q_S is greater than 1, blockSize must be 128-byte aligned and cannot exceed 1024.
                            In the GQA scenario, when the head dimensions of the query, key, and value are not 64 or 128 and Q_S is 1, the block size must be 16-byte aligned and cannot exceed 512.
                            In the full quantization scenario of MLA, the block size must be 128.</li>
                        <li>When PagedAttention is enabled and full quantization is used, the block size must be a non-zero value and cannot exceed 512.</li>
                        <li>When PagedAttention is enabled, full quantization is used, and Q_S is 1:</li>
                            The key and value inputs must be 16-byte aligned when the input type is FLOAT16 or BFLOAT16.<br>
                            The key and value inputs must be 32-byte aligned when the input type is INT8/HIFLOAT8/FLOAT8_E4M3FN.<br>
                            The key and value inputs must be 64-byte aligned when the input type is FLOAT4_E2M1/INT4 (INT32).<br>
                        <li>When PagedAttention is enabled, full quantization is used, and Q_S is greater than 1:</li>
                            The block size ranges from 128 to 512 and must be a multiple of 128.<br>
                    </ul>
                </td>
                <td><code>BlockSize</code> is a user-defined parameter. Its value affects the paged attention performance. Generally, paged attention can improve the throughput but deteriorate the performance.</td>
            </tr>
            <tr>
                <td>blockTable</td>
                <td>In the paged attention scenario, <code>blockTable</code> must be two-dimensional. The length of the first dimension must be equal to <code>B</code>, and the length of the second dimension must be greater than or equal to <code>maxBlockNumPerSeq</code> (the maximum number of blocks corresponding to <code>actualSeqLengthsKv</code> in each batch).
                </td>
                <td>-</td>
            </tr>
            <tr>
                <td rowspan="2">General scenarios</td>
                <td><code>key</code>/<code>value</code></td>
                <td>
                    <ul>
                        <li>The key and value data types can be FLOAT16/BFLOAT16/INT8/INT4(INT32)/HIFLOAT8/FLOAT8_E4M3FN/FLOAT4_E2M1</li>.
                        <li>In non-quantization scenarios, when the inputLayout of the query is BNSD, TND, BSH, or BSND, the KV cache layout supports three formats: BnBsH (blocknum, blocksize, H), BnNBsD (blocknum, KV_N, blocksize, D), and NZ (blocknum, KV_N, D/16, blocksize, 16).</li>
                        <li>In the MLA full-quantization scenario, when the inputLayout of the query is BNSD or TND, the KV cache layout supports the BnBsH (blocknum, blocksize, H), BnNBsD (blocknum, KV_N,
                          blocksize, D), and NZ (blocknum, KV_N, D/16, blocksize, 16).</li>
                        <li>In the full quantization scenario of MLA, when the input layout of the query is BSH or BSND, the KV cache layout supports only two formats: BnBsH and NZ.</li>
                        <li>In the fake-quantization scenario, when the KV cache is five-dimensional, the KV cache layout is (blocknum, KV_N, D/16, blocksize, 16). In addition, when the key and value data types are INT32, the KV
                            cache layout is (blocknum, KV_N, D/2, blocksize, 2).</li>
                        <li>The PagedAttention</li><li> is not supported in the full quantization scenario of GQA.</li>
                     </ul>
                </td>
                <td>
                <ul>
                    <li>In the PagedAttention scenario, the performance of the BnNBsD KV cache layout is usually better than that of the BnBsH KV cache layout. You are advised to use the BnNBsD format.</li>
                    <li>The value of blocknum cannot be less than the sum of the number of blocks in each batch calculated based on actualSeqLengthsKv and blockSize. The shapes of `key` and `value` must be identical.</li>
                    <li>In the PagedAttention scenario, when the input KV cache layout format is BnBsH (blocknum, blocksize, H) and KV_N x D exceeds 65535, the error will be reported due to hardware instruction restrictions. This problem can be solved by enabling GQA (decreasing <code>KV_N</code>) or adjusting the KV cache layout to BnNBsD <code>(BlockNum, KV_N, BlockSize, D)</code>.</li>
                </ul>
                </td>
            </tr>
            <tr>
                <td>actualSeqLengthsKv</td>
                <td>In the paged attention scenario, <code>actualSeqLengthsKv</code> must be passed.</td>
                <td>-</td>
            </tr>
            <tr>
                <td rowspan="3">Feature intersection scenario</td>
                <td>mask</td>
                <td rowspan="2">In the scenario where paged attention is enabled, the last dimension of the input must be greater than or equal to maxBlockNumPerSeq * blockSize.</td>
                <td rowspan="2">-</td>
            </tr>
            <tr>
                <td>pseShift</td>
            </tr>
            <tr>
                <td>keyAntiquantScale, keyAntiquantOffset, valueAntiquantScale, valueAntiquantOffset
                </td>
                <td>
                    <ul>
                        <li>In the fake-quantization per-token mode or the fake-quantization per-token mode with per-head mode, the last dimension of antiquantScale and antiquantOffset must be greater than or equal to maxBlockNumPerSeq.
                            * blockSize</li>
                        <li>In fake-quantization per-token-group mode, the second-to-last dimension of the input keyAntiquantScale/valueAntiquantScale must be greater than or equal to maxBlockNumPerSeq * blockSize.</li>
                    </ul>
                </td>
                <td>-</td>
            </tr>
            <tr>
                <td colspan="4">
                <ul><li>PagedAttention does not support the tensor list scenario or left padding scenario.</li>
                <li>PagedAttention can be enabled only when the blocktable exists and is valid, and the keys and values are arranged in a continuous memory block according to the indexes in the blocktable. In this scenario, the inputLayout parameter of the keys and values is invalid.</li>
                </ul></td>
            </tr>
        </tbody>
    </table>

- <a id="INT8"></a>Constraints on the number of input parameters and input and output [data formats](../../../docs/en/context/data_format.md) related to INT8/FP8 quantization:
    <table style="undefined;table-layout: fixed;  width: 1190px">
        <colgroup>
            <col style="width: 320px">
            <col style="width: 120px">
            <col style="width: 750px">
        </colgroup>
        <thead>
            <tr>
                <th>Scenario</th>
                <th>Parameter</th>
                <th>Constraint</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td rowspan="9">The input is FLOAT16 or BFLOAT16, and the output is INT8/FP8.</td>
                <td>query</td>
                <td>The type is FLOAT16 or BFLOAT16.</td>
            </tr>
            <tr>
                <td>key</td>
                <td>The type is FLOAT16/BFLOAT16/INT8/FLOAT8_E4M3FN/HIFLOAT8.</td>
            </tr>
            <tr>
                <td>value</td>
                <td>The type is FLOAT16/BFLOAT16/INT8/FLOAT8_E4M3FN/HIFLOAT8.</td>
            </tr>
            <tr>
                <td>deqScale1</td>
                <td>If the input parameter <code>deqScale1</code> exists, an error is reported and returned.</td>
            </tr>
            <tr>
                <td>quantScale1</td>
                <td>If the input parameter <code>quantScale1</code> exists, an error is reported and returned.</td>
            </tr>
            <tr>
                <td>deqScale2</td>
                <td>If the input parameter <code>deqScale2</code> exists, an error is reported and returned.</td>
            </tr>
            <tr>
                <td>quantScale2</td>
                <td>Both the per-tensor and per-channel data formats and the FLOAT32 and BFLOAT16 data types are supported.
                    <ul>
                        <li>When the input is of type BFLOAT16, both FLOAT32 and BFLOAT16 are supported. Otherwise, only FLOAT32 is supported.</li>
                        <li>In per-channel format, quantScale2 is required when the layout is BSH, BSND, BNSD, or BNSD_BSND.
                            The product of all dimensions is equal to N x D(H). For other layouts, the shape must be [N, D].</li>
                        <li>In per-tensor format, only the shape [1] is supported.</li>
                    </ul>
                </td>
            </tr>
            <tr>
                <td>quantOffset2</td>
                <td>If the optional parameter <code>quantOffset2</code> is passed, ensure that its type and shape are consistent with those of <code>quantScale2</code>. If this parameter is not passed, the default value is <code>nullptr</code>, indicating <code>0</code>.
                </td>
            </tr>
            <tr>
                <td>attentionOut</td>
                <td>The type is INT8/FP8(FLOAT8_E4M3FN/HIFLOAT8).</td>
            </tr>
        </tbody>
    </table>

- <a id="leftPadding"></a>Left padding
    <table style="undefined;table-layout: fixed; width: 1000px">
        <colgroup>
            <col style="width: 100px">
            <col style="width: 450px">
            <col style="width: 450px">
        </colgroup>
        <thead>
            <tr>
                <th>Parameter</th>
                <th>Formula</th>
                <th>Remarks</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>queryPaddingSize</td>
                <td>
                    <ul>
                        <li>Start point of the query: Q_S - queryPaddingSize - actualSeqLengths</li>
                        <li>End point of the query: Q_S - queryPaddingSize</li>
                    </ul>
                </td>
                <td>
                    <ul>
                        <li>If the start or end point is less than 0, the returned data result is all 0</li>.
                        <li>If queryPaddingSize is less than 0, it will be set to 0</li>.
                        <li>This parameter must be used together with the actualSeqLengths parameter. Otherwise, the right padding scenario of the query is used by default.</li>
                    </ul>
                </td>
            </tr>
            <tr>
                <td>kvPaddingSize</td>
                <td>
                    <ul>
                        <li>Start point for moving keys and values: KV_S - kvPaddingSize - actualSeqLengthsKv</li>
                        <li>End point for moving keys and values: KV_S - kvPaddingSize</li>
                    </ul>
                </td>
                <td>
                    <ul>
                        <li>If the start or end point is less than 0, the returned data is all 0.</li>
                        <li>If kvPaddingSize is less than 0, it will be set to 0.</li>
                        <li>This parameter must be used together with the actualSeqLengthsKv parameter. Otherwise, the right padding scenario of the kv is used by default.</li>
                    </ul>
                </td>
            </tr>
            <tr>
                <td colspan="3">
                    <ul>
                        <li>PageAttention and tensorlist are not supported. Otherwise, the right padding scenario is used by default.</li>
                        <li>When this parameter is used together with the attenMask parameter, ensure that the meaning of attenMask is correct, that is, invalid data can be correctly hidden. Otherwise, accuracy problems will occur.</li>
                    </ul>
                </td>
            </tr>
        </tbody>
    </table>

- <a id="prefix"></a>Prefix
    <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
    <col style="width: 218px">
    <col style="width: 932px">
        </colgroup>
        <thead>
            <tr>
                <th>Parameter</th>
                <th>Restriction</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>keySharedPrefix, valueSharedPrefix</td>
                <td>
                    <ul>
                        <li>Both of them are either empty or not empty.</li>
                        <li>If neither of them is empty, the dimensions of keySharedPrefix, valueSharedPrefix, key, and value are the same, and the data types are the same.</li>
                        <li>If neither of them is empty, the first dimension (batch) of the shape must be 1. When the layout is BNSD or BSND, the N and D axes must be the same as the key. When the layout is BSH, the H axis must be the same as the key. The S values of keySharedPrefix and valueSharedPrefix must be the same.</li>
                    </ul>
                </td>
            </tr>
            <tr>
                <td>actualSharedPrefixLen</td>
                <td>The shape must be [1], and the value cannot be greater than the S</td> of keySharedPrefix and valueSharedPrefix.
            </tr>
            <tr>
                <td colspan="2">
                    <ul>
                        <li>The result of adding the S value of the public prefix to the S value of the key or value must meet the original S value restrictions of the key or value.</li>
                        <li>The prefix is not supported in the PageAttention, left padding, tensor list, alibi, TND, PFA MLA (including the scenario where the D dimension is not equal and the ROPE is independently input), or IFA MLA scenario.</li>
                        <li>If sparse is set to 0 or 1 and attenmask is passed, S2 must be greater than or equal to the sum of actualSharedPrefixLen and the S value of the key.</li>
                        <li>The scenario where all qkv inputs are INT8/FP8/HiF8 (including full quantization in MLA and GQA) is not supported.</li>
                        <li>Post-training quantization (INT8) is supported.</li>
                        <li>
                            In the fake-quantization key/value fusion scenario, all quantization modes support the prefix. In the fake-quantization key/value separation scenario, only the following quantization modes support the prefix:
                        </li>
                    </ul>
                </td>
            </tr>
            <tr>
                <td colspan="2">
                    <table style="table-layout: fixed; width: 1140px" border="1" cellpadding="6" cellspacing="0">
                        <colgroup>
                            <col style="width: 218px">
                            <col style="width: 700px">
                            <col style="width: 222px">
                        </colgroup>
                        <thead style="font-size: 12px;">
                            <tr>
                                <th>Key/Value Separation Scenario</th>
                                <th>fake-quantization mode</th>
                                <th>Key/Value supports dtype.</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td rowspan="2" style="background-color: #f5f5f5; font-weight: 500; text-align: left;">Q_S&gt;1</td>
                                <td>
                                    <ul>
                                        <li>per-channel (per-tensor)</li>
                                        <li>per-token</li>
                                    </ul>
                                </td>
                                <td>INT8</td>
                            </tr>
                            <tr>
                                <td colspan="2" style="display: none;"></td>
                            </tr>
                            <tr>
                                <td rowspan="3" style="background-color: #f5f5f5; font-weight: 500; text-align: left;">Q_S=1</td>
                                <td>
                                    <ul>
                                        <li>per-tensor</li>
                                        <li>Per-tensor + per-head</li>
                                        <li>Per-token with paged attention mode for scale/offset</li>
                                        <li>Per-token + per-head with paged attention mode for scale/offset</li>
                                    </ul>
                                </td>
                              <td>INT8</td>
                            </tr>
                            <tr>
                                <td>
                                    <ul>
                                        <li>per-channel</li>
                                        <li>per-token</li>
                                        <li>Per-token + per-head</li>
                                        <li>Per-channel for `key` + per-token for `value`</li>
                                    </ul>
                                </td>
                                <td>INT8, INT4(INT32)</td>
                            </tr>
                        </tbody>
                    </table>
                </td>
            </tr>
        </tbody>
    </table>

- <a id="MLA"></a>MLA (<code>queryRope</code> and <code>keyRope</code> not null)
    <table style="undefined;table-layout: fixed; width: 1389px"><colgroup>
        <col style="width: 158px">
        <col style="width: 125px">
        <col style="width: 226px">
        <col style="width: 520px">
        <col style="width: 360px">
        </colgroup>
        <thead>
        <tr>
            <th colspan="2">Scenario</th>
            <th>Parameter</th>
            <th>Supported Configuration</th>
            <th>Remarks</th>
        </tr>
        </thead>
        <tbody>
        <tr>
            <td colspan="2" rowspan="2">Common constraints</td>
            <td>queryRope</td>
            <td>The shape is the same as that of <code>query</code> except that dimension <code>D</code> must be equal to <code>64</code>.</td>
            <td>-</td>
        </tr>
        <tr>
            <td>keyRope</td>
            <td>The shape is the same as that of <code>key</code> except that dimension <code>D</code> must be equal to <code>64</code>.</td>
            <td>-</td>
        </tr>
        <tr>
            <td rowspan="18">query d=512</td>
            <td rowspan="6">General scenarios</td>
            <td>query</td>
            <td>Q_N=[1,2,4,8,16,32,64,128]</td>
            <td>The Ascend 950PR/Ascend 950DT supports Q_S = [1-16]. This restriction will be removed in later versions.</td>
        </tr>
        <tr>
            <td>key</td>
            <td>The dtype is the same as that of the query. K_N=1</td>
            <td>ND input is supported.</td>
        </tr>
        <tr>
            <td>value</td>
            <td>The dtype is the same as that of the query. K_N=1</td>
            <td>ND input is supported.</td>
        </tr>
        <tr>
            <td>attention</td>
            <td>The data type is the same as that of <code>query</code>.</td>
            <td>-</td>
        </tr>
        <tr>
            <td>actualSeqLengths</td>
            <td></td>
            <td>Currently, Ascend 950PR/Ascend 950DT supports actualSeqLengthsQ only in the TND/TND_NTD layout. This restriction will be removed in later versions. actualSeqLengthsKV can be configured in all layouts.</td>
        </tr>
        <tr>
            <td>inputLayout</td>
            <td>Supports BSH, BSND, BNSD, BSH_NBSD, BSND_NBSD, BNSD_NBSD, TND, and TND_NTD.</td>
            <td>-</td>
        </tr>
        <tr>
            <td>MASK</td>
            <td>sparseMode</td>
            <td>Supports sparse0, sparse = 3 with mask input, and sparse = 4 with mask input.</td>
            <td>-</td>
        </tr>
        <tr>
            <td rowspan="10">Full quantization</td>
            <td>query</td>
            <td>FLOAT8_E4M3FN; Q_N=[32,64,128]</td>
            <td>-</td>
        </tr>
        <tr>
            <td>key</td>
            <td>FLOAT8_E4M3FN</td>
            <td>-</td>
        </tr>
        <tr>
            <td>value</td>
            <td>FLOAT8_E4M3FN</td>
            <td>-</td>
        </tr>
        <tr>
            <td>attention</td>
            <td>BFLOAT16</td>
            <td>-</td>
        </tr>
        <tr>
            <td>queryRope</td>
            <td>BFLOAT16</td>
            <td>-</td>
        </tr>
        <tr>
            <td>keyRope</td>
            <td>BFLOAT16</td>
            <td>-</td>
        </tr>
        <tr>
            <td>keyAntiquantScaleOptional</td>
            <td>FLOAT32</td>
            <td><ul><li>This parameter must be used together with dequantScaleQueryOptional and valueAntiquantScaleOptional. keyAntiquantOffsetOptional is not supported.</li>
                   <li>Only the pertensor mode is supported. The value of keyAntiquantMode is 0.</li>
                   <li>The shape is (1).</li></ul></td>
        </tr>
        <tr>
            <td>valueAntiquantScaleOptional</td>
            <td>FLOAT32</td>
            <td><ul><li>This parameter must be used together with dequantScaleQueryOptional and keyAntiquantScaleOptional. valueAntiquantOffsetOptional is not supported.</li>
                    <li>Only the pertensor mode is supported. The value of valueAntiquantMode is 0.</li>
                    <li>The shape is (1)</li></ul></td>.
        </tr>
        <tr>
            <td>dequantScaleQueryOptional</td>
            <td>FLOAT32</td>
            <td><ul><li>This parameter must be used together with keyAntiquantScaleOptional and valueAntiquantScaleOptional.</li>
                    <li>queryQuantMode supports only the per-token + per-head mode. queryQuantMode is set to 3.</li>
                    <li>The shape is the same as that of query except that the dimension D is missing. For example, if inputLayout is set to BSH or BSND, dequantScaleQuery_shape is set to (B,S,N).</li></ul></td>
        </tr>
        <tr>
            <td>inputLayout</td>
            <td>BSH, BSND, BNSD, and TND are  supported.</td>
            <td>-</td>
        </tr>
        <tr>
            <td colspan="4">Left padding, tensor list, PSE, prefix, and fake-quantization are not supported.</td>
        </tr>
        <tr>
            <td rowspan="6">query d=128</td>
            <td>Non-quantization</td>
            <td>inputLayout</td>
            <td>BSH, BSND, TND, NTD, NTD_NTD, BNSD, BNSD_BSND, BSH_BNSD, BSND_BNSD</td>
            <td>-</td>
        </tr>
        <tr>
            <td rowspan="2">MLA</td>
            <td>queryRope</td>
            <td>The dtype is the same as that of query. The b, n, and s in the shape are the same as those in query. The d is 64.</td>
            <td>-</td>
        </tr>
        <tr>
            <td>keyRope</td>
            <td>The dtype is the same as the key. The b, n, and s in the shape are the same as those in the key. The d is 64.</td>
            <td>When kv is a tensor list, the b in the shape of keyRope must be the same as the length of the tensor list, and the n and s must be the same as those of each tensor in the tensor list. The d is 64.</td>
        </tr>
        <tr>
            <td colspan="4">PSE, prefix, fake-quantization, and full quantization are not supported.</td>
        </tr>
        </tbody>
    </table>

## Examples

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

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
      // (Boilerplate) Initialize resources.
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
      // Call aclrtMemcpy to copy the data on the host to the memory on the device. 
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
      // 1. (Fixed writing) Initialize the device and stream. For details, see the AscendCL API manual.
      // Set the device ID in use.
      int32_t deviceId = 0;
      aclrtStream stream;
      auto ret = Init(deviceId, &stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
  
      // 2. Construct the inputs and outputs based on the API definition.
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
  
      // Create a query aclTensor.
      ret = CreateAclTensor(queryHostData, queryShape, &queryDeviceAddr, aclDataType::ACL_FLOAT16, &queryTensor);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create a key aclTensor.
      ret = CreateAclTensor(keyHostData, keyShape, &keyDeviceAddr, aclDataType::ACL_FLOAT16, &keyTensor);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      int kvTensorNum = 1;
      aclTensor *tensorsOfKey[kvTensorNum];
      tensorsOfKey[0] = keyTensor;
      auto tensorKeyList = aclCreateTensorList(tensorsOfKey, kvTensorNum);
      // Create a value aclTensor.
      ret = CreateAclTensor(valueHostData, valueShape, &valueDeviceAddr, aclDataType::ACL_FLOAT16, &valueTensor);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      aclTensor *tensorsOfValue[kvTensorNum];
      tensorsOfValue[0] = valueTensor;
      auto tensorValueList = aclCreateTensorList(tensorsOfValue, kvTensorNum);
      // Create an atten aclTensor.
      ret = CreateAclTensor(attenHostData, attenShape, &attenDeviceAddr, aclDataType::ACL_BOOL, &attenTensor);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create an out aclTensor.
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
      // Allocate device memory based on workspaceSize computed by the first-phase API.
      void* workspaceAddr = nullptr;
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
      }
      // Call the second-phase API.
      ret = aclnnFusedInferAttentionScoreV5(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFusedInferAttentionScoreV5 failed. ERROR: %d\n", ret); return ret);
  
      // 4. (Boilerplate) Wait until the task execution is complete.
      ret = aclrtSynchronizeStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
  
      // 5. Obtain the output value and copy the result from the device memory to the host. Modify the code based on the API definition.
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
