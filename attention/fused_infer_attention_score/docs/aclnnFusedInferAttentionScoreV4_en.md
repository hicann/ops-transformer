
# aclnnFusedInferAttentionScoreV4

[📄 View Source Code](https://gitcode.com/cann/ops-transformer/tree/master/attention/fused_infer_attention_score)

## Supported Products

|Product     | Supported|
|:----------------------------|:-----------:|
|Ascend 950PR/Ascend 950DT|      ×     |
|<term>Atlas A3 training products/Atlas A3 inference products</term>|      √     |
|<term>Atlas A2 training products/Atlas A2 inference products</term>|      √     |
|<term>Atlas 200I/500 A2 inference products</term>|      ×     |
|<term>Atlas inference products</term>|      ×     |
|<term>Atlas training products</term>|      ×     |

## Function

- Adapts to the `FlashAttention` operator in the decode (`IncreFlashAttention`) and prefill (`PromptFlashAttention`) inference scenarios.

    Compared with FusedInferAttentionScoreV3, this API adds the dequantScaleQueryOptional, learnableSinkOptional, and queryQuantMode parameters, and the fullmask capability of alibi.

    **Note:**
    KV cache specific to the decode scenario: KV cache is a common technology for optimizing the inference performance of foundation models. During sampling, the transformer model uses the given prompt/context as the initial input for inference (parallel processing supported), and then generates additional tokens one by one to improve the generated sequence (reflecting the auto-regressive property of the model). The transformer performs the self-attention operation during sampling. Therefore, KV vectors need to be extracted for each item (regardless of the prompt/context or generated token) in the current sequence. These vectors are stored in a matrix called KV cache.
- Formula:

    Self-attention constructs an attention model by leveraging the relationships within the input samples. The principle assumes there is an input sample sequence $x$ of length $n$, where each element of $x$ is a $d$-dimensional vector. Each $d$-dimensional vector can be regarded as a token embedding. Such a sequence is transformed by three weight matrices to obtain three $n × d$ matrices.

    The computation formula for self-attention is generally defined as follows, where $Q$, $K$, and $V$ are key attribute elements of the input sample, obtained through spatial transformation and unified into a single feature space. "Attention" in the formula and operator name is an abbreviation for "self-attention."

    $$
    Attention(Q,K,V)=Score(Q,K)V
    $$

    In this operator, the `Softmax` function is used, instead of the `Score` function. The self-attention computation formula is as follows:

    $$
    Attention(Q,K,V)=Softmax(\frac{QK^T}{\sqrt{d}} + FullMask)V
    $$

    The product of Q and K^T represents the attention of the input x. To avoid the value being too large, the value is usually scaled by dividing the square root of d, and the fullmask of alibi is added. Then, softmax normalization is performed on each row, and the result is multiplied by V to obtain an n x d matrix.

    **Note:**
    <blockquote>The data layout of <code>query</code>, <code>key</code>, and <code>value</code> can be interpreted from multiple dimensions. To be specific, <code>B</code> (<code>Batch</code>) indicates the size of an input sample batch, <code>S</code> (<code>Seq-Length</code>) indicates the length of the input sample sequence, <code>H</code> (<code>Hidden-Size</code>) indicates the size of the hidden layer, <code>N</code> (<code>Head-Num</code>) indicates the number of heads, and <code>D</code> (<code>Head-Dim</code>) indicates the minimum unit size of the hidden layer (<code>D</code> = <code>H</code>/<code>N</code>). <code>T</code> indicates the total length of all input sample sequences.
    <br><code>Q_S</code> indicates <code>S</code> in the shape of <code>query</code>, <code>KV_S</code> indicates <code>S</code> in the shapes of <code>key</code> and <code>value</code>, <code>Q_N</code> indicates <code>num_query_heads</code>, and <code>KV_N</code> indicates <code>num_key_value_heads</code>. <code>P</code> indicates the computation result of Softmax(<span>(QK<sup class="superscript">T</sup>)/<span class="sqrt">d</span></span>).</blockquote>

## Function Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnFusedInferAttentionScoreV4GetWorkspaceSize` is called to obtain the input parameters and compute the required workspace size based on the process. Then, `aclnnFusedInferAttentionScoreV4` is called to perform computation.

```c++
aclnnStatus aclnnFusedInferAttentionScoreV4GetWorkspaceSize(
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
    int64_t              numHeads, 
    double               scaleValue, 
    int64_t              preTokens, 
    int64_t              nextTokens, 
    char                *inputLayout, 
    int64_t              numKeyValueHeads, 
    int64_t              sparseMode, 
    int64_t              innerPrecise, 
    int64_t              blockSize, 
    int64_t              antiquantMode, 
    bool                 softmaxLseFlag, 
    int64_t              keyAntiquantMode, 
    int64_t              valueAntiquantMode, 
    int64_t              queryQuantMode, 
    const aclTensor     *attentionOut, 
    const aclTensor     *softmaxLse, 
    uint64_t            *workspaceSize, 
    aclOpExecutor      **executor)
```

```c++
aclnnStatus aclnnFusedInferAttentionScoreV4(
    void             *workspace, 
    uint64_t          workspaceSize, 
    aclOpExecutor    *executor, 
    const aclrtStream stream)
```

## aclnnFusedInferAttentionScoreV4GetWorkspaceSize

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
        <th>Usage Notes</th>
        <th>Data Type</th>
        <th>Data Format</th>
        <th>Dimension (Shape)</th>
        <th>Non-contiguous Tensor</th>
    </tr></thead>
    <tbody>
    <tr>
        <td>query</td>
        <td>Input</td>
        <td>Input <code>Q</code> in the attention structure.</td>
        <td>-</td>
        <td>FLOAT16, BFLOAT16, INT8</td>
        <td>ND</td>
        <td>See the <code>inputLayout</code> parameter.</td>
        <td>×</td>
    </tr>
    <tr>
        <td>key</td>
        <td>Input</td>
        <td>Input <code>K</code> in the attention structure.</td>
        <td>
        <ul>
            <li>The shapes of the corresponding tensors of <code>key</code> and <code>value</code> must be the same.</li>
            <li>In non-contiguous scenarios, <code>batch</code> in the tensor list of <code>key</code> and <code>value</code> must be <code>1</code>. The number of <code>batch</code> elements is equal to <code>B</code> in <code>query</code>. <code>N</code> and <code>D</code> must be the same.</li>
            <li>Due to the tensor list restrictions, <code>B</code> cannot be greater than <code>256</code> in non-contiguous scenarios.</li>
        </ul>
        </td>
        <td>FLOAT16, BFLOAT16, INT8, INT4(INT32)</td>
        <td>ND</td>
        <td>See the <code>inputLayout</code> parameter.</td>
        <td>×</td>
    </tr>
    <tr>
        <td>value</td>
        <td>Input</td>
        <td>Input <code>V</code> in the attention structure.</td>
        <td>
        <ul>
            <li>The shapes of the corresponding tensors of <code>key</code> and <code>value</code> must be the same.</li>
            <li>In non-contiguous scenarios, <code>batch</code> in the tensor list of <code>key</code> and <code>value</code> must be <code>1</code>. The number of <code>batch</code> elements is equal to <code>B</code> in <code>query</code>. <code>N</code> and <code>D</code> must be the same.</li>
            <li>Due to the tensor list restrictions, <code>B</code> cannot be greater than <code>256</code> in non-contiguous scenarios.</li></ul>
        </td>
        <td>FLOAT16, BFLOAT16, INT8, INT4(INT32)</td>
        <td>ND</td>
        <td>See the <code>inputLayout</code> parameter.</td>
        <td>×</td>
    </tr>
    <tr>
        <td>pseShiftOptional</td>
        <td>Optional input</td>
        <td>Position encoding parameter in the attention structure.</td>
        <td><ul><li>Empty tensors are not supported.</li>
                <li>For details about the constraints, see <a href="#pseShift">pseShift</a>.</li></ul></td>
        <td>FLOAT16, BFLOAT16</td>
        <td>ND</td>
        <td>Recommended shape input: <code>(B, Q_N, Q_S, KV_S)</code> or <code>(1, Q_N, Q_S, KV_S)</code>.</td>
        <td>×</td>
    </tr>
    <tr>
        <td>attenMaskOptional</td>
        <td>Optional input</td>
        <td>Masks the QK result to define the attention visibility between tokens.</td>
        <td>
        <ul>
            <li>Empty tensors are not supported.</li>
            <li>If <code>Q_S</code> and <code>KV_S</code> are not 16- or 32-byte aligned, the value can be rounded up to the aligned <code>S</code>.</li>
            <li>When the data type of <code>attenMask</code> is INT8 or UINT8, the value in the tensor must be <code>0</code> or <code>1</code>.</li>
            <li>If this parameter is not used, pass <code>nullptr</code>.</li>
            <li>For details about other constraints, see <a href="#Mask">Mask</a>.</li>
        </ul>
        </td>
        <td>BOOL, INT8, UINT8</td>
        <td>ND</td>
        <td>
        <ul>
            <li>When sparseMode is set to 0 or 1:
                <ul>
                    <li>The shape can be (B,Q_S,KV_S), (1,Q_S,KV_S), (B,1,Q_S,KV_S), or (1,1,Q_S,KV_S).</li>
                    <li> When the input layout is <code>BSH</code>, <code>BSND</code>, <code>BNSD</code>, or <code>BNSD_BSND</code>, <code>D</code> of <code>query</code> and <code>key</code> is equal to <code>D</code> of <code>value</code>, and <code>queryRope</code> and <code>keyRope</code> are not passed, the shape can be <code>(B, KV_S)</code> if <code>Q_S</code> is equal to <code>1</code> and <code>(Q_S, KV_S)</code> if <code>Q_S</code> is greater than <code>1</code>.</li>
                </ul>
            </li>
            <li>When sparseMode is set to 2, 3, or 4, the shape of attenMaskOptional can be (2048, 2048), (1, 2048, 2048), or (1,1,2048,2048).</li>
        </ul>
        </td>
        <td>×</td>
    </tr>
    <tr>
        <td>actualSeqLengthsOptional</td>
        <td>Optional input</td>
        <td>Valid sequence length of <code>query</code> in different batches. If the layout is <code>TND</code>, the number of this input parameter is used as the batch size.</td>
        <td>
        <ul>
            <li>The value must be a non-negative number.</li>
            <li>The valid sequence length of each batch in this input parameter must be less than or equal to that of the corresponding batch in <code>query</code>.</li>
            <li>If the input length of <code>seqlen</code> is <code>1</code>, all batches use the same <code>seqlen</code>. If the input length is greater than or equal to the batch size, the first *N* elements (where *N* equals the batch size) of <code>seqlen</code> are used. Other lengths are not supported.</li>
            <li>If this parameter is not passed, each batch uses the same <code>seqlen</code>, and the value of <code>seqlen</code> is equal to the value of <code>S</code> in the shape.</li>
            <li>If the layout is <code>TND</code>, this parameter works in the accumulation mode.</li>
            <li>For details about the constraints, see <a href="#constraints">Constraints</a>.</li>
        </ul>
        </td>
        <td>INT64</td>
        <td>-</td>
        <td><code>(1)</code>, <code>(B)</code>, or <code>(>B)</code> </td>
        <td>×</td>
    </tr>
    <tr>
        <td>actualSeqLengthsKvOptional</td>
        <td>Optional input</td>
        <td>Valid sequence length of <code>key</code>/<code>value</code> in different batches. If the layout is <code>TND</code>, the number of this input parameter is used as the batch size.</td>
        <td>
        <ul>
            <li>The value must be a non-negative number.</li>
            <li>The valid sequence length of each batch in this input parameter must be less than or equal to that of the corresponding batch in <code>key</code>/<code>value</code>.</li>
            <li>If the input length of <code>seqlenKv</code> is <code>1</code>, all batches use the same <code>seqlenKv</code>. If the input length is greater than or equal to the batch size, the first *N* elements (where *N* equals the batch size) of <code>seqlenKv</code> are used. Other lengths are not supported.</li>
            <li>If this parameter is not passed, each batch uses the same <code>seqlenKv</code>, and the value of <code>seqlenKv</code> is equal to the value of <code>S</code> in the shape.</li>
            <li>In the PA scenario, this parameter works in the batch mode. In non-PA scenarios, when the layout is <code>TND </code>, this parameter works in the accumulation mode.</li>
            <li>For details about the constraints, see <a href="#constraints">Constraints</a>.</li>
        </ul>
        </td>
        <td>INT64</td>
        <td>-</td>
        <td><code>(1)</code>, <code>(B)</code>, or <code>(>B)</code> </td>
        <td>×</td>
    </tr>
    <tr>
        <td>deqScale1Optional</td>
        <td>Optional input</td>
        <td>Dequantization factor of the QK result.</td>
        <td><ul><li>Empty tensors are not supported.</li>
        <li>Per-tensor is supported.</li></ul></td>
        <td>UINT64, FLOAT32</td>
        <td>ND</td>
        <td>See <a href="#INT8">INT8 quantization</a>.</td>
        <td>×</td>
    </tr>
    <tr>
        <td>quantScale1Optional</td>
        <td>Optional input</td>
        <td>Quantization factor of <code>P</code>.</td>
        <td><ul><li>Empty tensors are not supported.</li>
        <li>Per-tensor is supported.</li></ul></td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>See <a href="#INT8">INT8 quantization</a>.</td>
        <td>×</td>
    </tr>
    <tr>
        <td>deqScale2Optional</td>
        <td>Optional input</td>
        <td>Dequantization factor of the PV result.</td>
        <td><ul><li>Empty tensors are not supported.</li>
        <li>Per-tensor is supported.</li></ul></td>
        <td>UINT64, FLOAT32</td>
        <td>ND</td>
        <td>See <a href="#INT8">INT8 quantization</a>.</td>
        <td>×</td>
    </tr>
    <tr>
        <td>quantScale2Optional</td>
        <td>Optional input</td>
        <td>Quantization factor of the output result.</td>
        <td><ul><li>Empty tensors are not supported.</li>
        <li>Per-tensor and per-channel are supported.</li></ul></td>
        <td>FLOAT32, BFLOAT16</td>
        <td>ND</td>
        <td>See <a href="#INT8">INT8 quantization</a>.</td>
        <td>×</td>
    </tr>
    <tr>
        <td>quantOffset2Optional</td>
        <td>Optional input</td>
        <td>Offset for quantizing the output result. If this parameter is set, asymmetric quantization is performed. Otherwise, symmetric quantization is performed.</td>
        <td><ul><li>Empty tensors are not supported.</li>
        <li>The per-tensor and per-channel types and shapes can be the same as those of <code>quantScale2Optional</code>.</li></ul></td>
        <td>FLOAT32, BFLOAT16</td>
        <td>ND</td>
        <td>Same as <code>quantScale2Optional</code>.</td>
        <td>×</td>
    </tr>
    <tr>
        <td>antiquantScaleOptional</td>
        <td>Optional input</td>
        <td>Fake-quantization factor of <code>key</code>/<code>value</code>.</td>
        <td><ul><li>Empty tensors are not supported.</li>
        <li>Per-tensor, per-channel, and per-token are supported.</li>
            <li>You are advised to use the separate mode of the KV fake-quantization parameters.</li>
            </ul></td>
        <td>When <code>Q_S</code> is equal to <code>1</code>, FLOAT16, BFLOAT16, or FLOAT32 can be used. When <code>Q_S</code> is greater than <code>1</code>, FLOAT16 is supported.</td>
        <td>ND</td>
        <td>See <a href="#AntiQuant">Fake-quantization parameters</a>.</td>
        <td>×</td>
    </tr>
    <tr>
        <td>antiquantOffsetOptional</td>
        <td>Optional input</td>
        <td>Fake-quantization offset for the key/value. If this parameter is set to a non-zero value, asymmetric quantization is performed. Otherwise, symmetric quantization is performed.</td>
        <td><ul><li>Empty tensors are not supported.</li>
        <li>The per-tensor, per-channel, and per-token shapes must be consistent with those of antiquantScaleOptional. </li><li>You are advised to use the separate mode of the KV fake-quantization parameters.</li></ul></td>
        <td>Same as <code>antiquantScaleOptional</code>.</td>
        <td>ND</td>
        <td>Same as <code>antiquantScaleOptional</code>.</td>
        <td>×</td>
    </tr>
    <tr>
        <td>blockTableOptional</td>
        <td>Optional input</td>
        <td>Block mapping table used for KV storage in paged attention.</td>
        <td><ul><li>Empty tensors are not supported.</li>
                <li>For details about the constraints, see <a href="#constraints">constraints</a>.</li></ul></td>
        <td>INT32</td>
        <td>ND</td>
        <td>The length of the first dimension must be equal to <code>B</code>, and the length of the second dimension must be greater than or equal to <code>maxBlockNumPerSeq</code> (the maximum number of blocks corresponding to <code>actualSeqLengthsKv</code> in different batches).</td>
        <td>×</td>
    </tr>
    <tr>
        <td>queryPaddingSizeOptional</td>
        <td>Optional input</td>
        <td>Whether the data in each batch of <code>query</code> is right-aligned and the number of right-aligned elements.</td>
        <td>
        <ul>
            <li>Empty tensors are not supported.</li>
            <li>The transfer start point of <code>query</code> in the left padding scenario is calculated as follows: <code>Q_S</code> – <code>queryPaddingSize</code> – <code>actualSeqLengths</code>. The transfer end point of <code>query</code> is calculated as follows: <code>Q_S</code> – <code>queryPaddingSize</code>. The transfer start point of <code>query</code> cannot be less than <code>0</code>, while the end point cannot be greater than <code>Q_S</code>. Otherwise, the result will not meet the expectation.</li>
            <li>In the left padding scenario of <code>query</code>, if the value of <code>queryPaddingSizeOptional</code> is less than <code>0</code>, it is set to <code>0</code>.</li>
            <li>Left padding of <code>query</code> must be enabled together with the <code>actualSeqLengths</code> parameter. Otherwise, right padding of <code>query</code> is used by default.</li>
            <li>This parameter is valid only when <code>Q_S</code> is greater than <code>1</code>. In other scenarios, it is invalid.</li>
        </ul>
        </td>
        <td>INT64</td>
        <td>ND</td>
        <td><code>(1)</code></td>
        <td>×</td>
    </tr>
    <tr>
        <td>kvPaddingSizeOptional</td>
        <td>Optional input</td>
        <td>Whether the data in each batch of <code>key</code>/<code>value</code> is right-aligned and the number of right-aligned elements.</td>
        <td>
        <ul>
            <li>Empty tensors are not supported.</li>
            <li>The transfer start point of <code>key</code> and <code>value</code> in the left padding scenario of <code>kv</code> is calculated as follows: <code>KV_S</code> – <code>kvPaddingSize</code> – <code>actualSeqLengthsKv</code>. The transfer end point of <code>key</code> and <code>value</code> is calculated as follows: <code>KV_S</code> – <code>kvPaddingSize</code>. The transfer start point of <code>key</code> and <code>value</code> cannot be less than <code>0</code>, while the end point cannot be greater than <code>KV_S</code>. Otherwise, the result will not meet the expectation.</li>
            <li>In the left padding scenario of <code>kv</code>, if the value of <code>kvPaddingSize</code> is less than <code>0</code>, it is set to <code>0</code>.</li>
            <li>Left padding of <code>kv</code> must be enabled together with the <code>actualSeqLengths</code> parameter. Otherwise, right padding of <code>kv</code> is used by default.</li>
            <li>The scenario where Q is BF16/FP16 and KV is INT4 (INT32) is not supported.</li>
        </ul>
        </td>
        <td>INT64</td>
        <td>ND</td>
        <td><code>(1)</code></td>
        <td>×</td>
    </tr>
    <tr>
        <td>keyAntiquantScaleOptional</td>
        <td>Optional input</td>
        <td>Dequantization factor of <code>key</code>.</td>
        <td>
        <ul>
            <li>Empty tensors are not supported.</li>
            <li>Both <code>keyAntiquantScaleOptional</code> and <code>valueAntiquantScaleOptional</code> must be either null or non-null.</li>
            <li>For details about other restrictions, see <a href="#AntiQuant">Restrictions on fake-quantization parameters</a> and <a href="#MLA">Restrictions on quantization parameters in the MLA scenario</a>.</li>
        </ul>
        </td>
        <td>FLOAT16, BFLOAT16, FLOAT32</td>
        <td>ND</td>
        <td>See <a href="#AntiQuant">fake-quantization parameters</a> and <a href="#MLA">quantization parameters in the MLA scenario</a>.</td>
        <td>×</td>
    </tr>
    <tr>
        <td>keyAntiquantOffsetOptional</td>
        <td>Optional input</td>
        <td>Dequantization offset of <code>key</code>. If this parameter is set, asymmetric quantization is used. Otherwise, symmetric quantization is used.</td>
        <td>
        <ul>
            <li>Empty tensors are not supported.</li>
            <li>If this function is used, its data type and shape must be the same as those of <code>keyAntiquantScaleOptional</code>.</li>
            <li>For details about other constraints, see <a href="#AntiQuant">Fake-quantization parameters</a>.</li>
        </ul>
        </td>
        <td>Same as keyAntiquantScaleOptional</td>
        <td>ND</td>
        <td>See <a href="#AntiQuant">Fake-quantization parameters</a>.</td>
        <td>×</td>
    </tr>
    <tr>
        <td>valueAntiquantScaleOptional</td>
        <td>Optional input</td>
        <td>Dequantization factor of <code>value</code>.</td>
        <td>
        <ul>
            <li>Empty tensors are not supported.</li>
            <li>Both <code>keyAntiquantScaleOptional</code> and <code>valueAntiquantScaleOptional</code> must be either null or non-null.</li>
            <li>For details about other restrictions, see <a href="#AntiQuant">Restrictions on fake-quantization parameters</a> and <a href="#MLA">Restrictions on quantization parameters in the MLA scenario</a>.</li>
        </ul>
        </td>
        <td>FLOAT16, BFLOAT16, FLOAT32</td>
        <td>ND</td>
        <td>See <a href="#AntiQuant">fake-quantization parameters</a> and <a href="#MLA">quantization parameters in the MLA scenario</a>.</td>
        <td>×</td>
    </tr>
    <tr>
        <td>valueAntiquantOffsetOptional</td>
        <td>Optional input</td>
        <td>Dequantization offset of <code>value</code>. If this parameter is set, asymmetric quantization is used. Otherwise, symmetric quantization is used.</td>
        <td>
        <ul>
            <li>Empty tensors are not supported.</li>
            <li>If this function is used, its data type and shape must be the same as those of <code>valueAntiquantScaleOptional</code>.</li>
            <li>For details about other constraints, see <a href="#AntiQuant">Fake-quantization parameters</a>.</li>
        </ul>
        </td>
        <td>Same as valueAntiquantScaleOptional</td>
        <td>ND</td>
        <td>See <a href="#AntiQuant">Fake-quantization parameters</a>.</td>
        <td>×</td>
    </tr>
    <tr>
        <td>keySharedPrefixOptional</td>
        <td>Optional input</td>
        <td>System prefix of <code>key</code> in the attention structure.</td>
        <td>
        <ul>
            <li>Empty tensors are not supported.</li>
            <li>Both <code>keySharedPrefix</code> and <code>valueSharedPrefix</code> must be either null or non-null.</li>
            <li>If neither <code>keySharedPrefix</code> nor <code>valueSharedPrefix</code> is null, the dimensions and data types of <code>keySharedPrefix</code>, <code>valueSharedPrefix</code>, <code>key</code>, and <code>value</code> must be the same.</li>
            <li>If neither <code>keySharedPrefix</code> nor <code>valueSharedPrefix</code> is null, the first dimension (batch) of the shape of <code>keySharedPrefix</code> must be <code>1</code>. When the layout is <code>BNSD</code> or <code>BSND</code>, the N and D axes must be the same as those of <code>key</code>. When the layout is <code>BSH</code>, the H axis must be the same as that of <code>key</code>. The same rules apply to <code>valueSharedPrefix</code>. <code>S</code> of <code>keySharedPrefix</code> and <code>valueSharedPrefix</code> must be the same.</li>
            <li>When <code>actualSharedPrefixLen</code> exists, its shape must be <code>[1]</code>, and its value cannot be greater than <code>S</code> of <code>keySharedPrefix</code> and <code>valueSharedPrefix</code>.</li>
            <li>The sum of <code>S</code> of the public prefix and <code>S</code> of <code>key</code>/<code>value</code> must meet the original restriction on <code>S</code> of <code>key</code>/<code>value</code>.</li>
            <li>For details about the constraints, see <a href="#constraints">Constraints</a>.</li>
        </ul>
        </td>
        <td>FLOAT16, BFLOAT16, INT8</td>
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
        <td>Optional input</td>
        <td>System prefix of <code>value</code> in the attention structure.</td>
        <td><ul><li>Empty tensors are not supported.</li>
        <li>The value must be the same as that of <code>keySharedPrefixOptional</code>.</li></ul></td>
        <td>FLOAT16, BFLOAT16, INT8</td>
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
        <td>Optional input</td>
        <td>System prefix parameters of <code>key</code>/<code>value</code>, indicating the valid sequence length of <code>keySharedPrefix</code>/<code>valueSharedPrefix</code>.</td>
        <td>The valid sequence length in this input parameter must be less than or equal to that in <code>keySharedPrefix</code>/<code>valueSharedPrefix</code>.</td>
        <td>INT64</td>
        <td>-</td>
        <td><code>(1)</code></td>
        <td>-</td>
    </tr>
    <tr>
        <td>queryRopeOptional</td>
        <td>Optional input</td>
        <td>Rope information of <code>query</code> in the MLA structure.</td>
        <td>
        <ul>
            <li>Empty tensors are not supported.</li>
            <li>The data type and format of <code>queryRope</code> must be the same as those of <code>query</code>.</li>
            <li>Both <code>queryRope</code> and <code>keyRope</code> are configured, or neither of them is configured. Configuring only one of the parameters is not supported.</li>
        </ul>
        </td>
        <td>FLOAT16, BFLOAT16</td>
        <td>ND</td>
        <td> In <code>queryRope</code>, dimension <code>d</code> of the shape is <code>64</code>, and the values of other dimensions are the same as those of <code>query</code>.</td>
        <td>×</td>
    </tr>
    <tr>
        <td>keyRopeOptional</td>
        <td>Optional input</td>
        <td>Rope information of <code>key</code> in the MLA structure.</td>
        <td>
        <ul>
            <li>Empty tensors are not supported.</li>
            <li>The data type and format of <code>keyRope</code> must be the same as those of <code>key</code>.</li>
            <li>Both <code>queryRope</code> and <code>keyRope</code> are configured, or neither of them is configured. Configuring only one of the parameters is not supported.</li>
        </ul>
        </td>
        <td>FLOAT16, BFLOAT16</td>
        <td>ND</td>
        <td> In <code>keyRope</code>, dimension <code>d</code> of the shape is <code>64</code>, and the values of other dimensions are the same as those of <code>key</code>.</td>
        <td>×</td>
    </tr>
    <tr>
        <td>keyRopeAntiquantScaleOptional</td>
        <td>Optional input</td>
        <td>Dequantization factor of the rope information of <code>key</code> in the MLA structure.</td>
        <td>
        <ul>
            <li>Empty tensors are not supported.</li>
            <li>This parameter is reserved and does not take effect in the current version.</li>
        </ul>
        </td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>dequantScaleQueryOptional</td>
        <td>Optional input</td>
        <td>Dequantization factor of <code>query</code>.</td>
        <td>
        <ul>
            <li>Empty tensors are not supported.</li>
            <li>This parameter is involved in the full quantization scenario. Per-token + per-head mode is supported.</li>
        </ul>
        </td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>See <a href="#AntiQuant">Fake-quantization parameters</a>.</td>
        <td>×</td>
    </tr>
    <tr>
        <td>learnableSinkOptional</td>
        <td>Optional input</td>
        <td>Absorbs attention scores through learnable sink tokens.</td>
        <td>
        <ul>
            <li>Only non-quantization scenarios are supported.</li>
            <li><code>V_D</code> can only be <code>128</code>/<code>64</code>.</li>
        </ul>
        </td>
        <td>BFLOAT16, FLOAT16</td>
        <td>ND</td>
        <td>(Q_N)</td>
        <td>×</td>
    </tr>
    <tr>
        <td>numHeads</td>
        <td>Input</td>
        <td>Number of heads of <code>query</code>.</td>
        <td>The value is equal to <code>Q_N</code> in the shape of <code>query</code>.</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>scaleValue</td>
        <td>Optional input</td>
        <td>Reciprocal of the square root of <code>d</code> in the formula, indicating the scale factor. </td>
        <td>Its data type must be compatible with that of <code>query</code> according to the type deduction rules.</td>
        <td>DOUBLE</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>preTokens</td>
        <td>Optional input</td>
        <td>Number of preceding tokens to associate in attention computation for sparse computation.</td>
        <td>This parameter is invalid when the input layout is <code>BSH</code>, <code>BSND</code>, or <code>BNSD</code>, <code>Q_S</code> is equal to <code>1</code>, <code>QK_D</code> is equal to <code>V_D</code>, and <code>queryRope</code> and <code>keyRope</code> are not passed.</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>nextTokens</td>
        <td>Optional input</td>
        <td>Number of succeeding tokens to associate in attention computation for sparse computation.</td>
        <td>This parameter is invalid when the input layout is <code>BSH</code>, <code>BSND</code>, or <code>BNSD</code>, <code>Q_S</code> is equal to <code>1</code>, <code>QK_D</code> is equal to <code>V_D</code>, and <code>queryRope</code> and <code>keyRope</code> are not passed.</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>inputLayout</td>
        <td>Optional input</td>
        <td>Identifies the data layout of the input query, key, and value. If this field contains an underscore (_), it indicates the input layout and output layout.</td>
        <td>
        <ul>
            <li>The supported input layouts include BSH, BSND, TND, BNSD, NTD, BSH_BNSD, BSND_BNSD, BNSD_BSND, NTD_TND, BSH_NBSD, BSND_NBSD, and BNSD_NBSD.</li>
            <li>When inputLayout is set to BSH_BNSD or BSND_BNSD, Q_D, K_D, and V_D must be set to 64 or 128, or Q_D and K_D must be set to 192 and V_D must be set to 128.<br></li>
        </ul>
        </td>
        <td>STRING</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>numKeyValueHeads</td>
        <td>Optional input</td>
        <td>Number of heads of <code>key</code>/<code>value</code>.</td>
        <td>
        <ul>
            <li><code>numHeads</code> must be exactly divisible by <code>numKeyValueHeads</code>.</li>
            <li>When the layout is <code>BSND</code>, <code>TND</code>, <code>BNSD</code>, <code>NTD</code>, <code>BSND_BNSD</code>, <code>BNSD_BSND</code>, or <code>NTD_TND</code>, the value must be the same as the N-axis value of <code>key</code>/<code>value</code> in the shape. Otherwise, an exception occurs.</li>
        </ul>
        </td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>sparseMode</td>
        <td>Optional input</td>
        <td>Sparse mode.</td>
        <td>
        <ul>
            <li>This parameter is invalid when the input layout is <code>BSH</code>, <code>BSND</code>, or <code>BNSD</code>, <code>Q_S</code> is equal to <code>1</code>, <code>QK_D</code> is equal to <code>V_D</code>, and <code>queryRope</code> and <code>keyRope</code> are not passed.</li>
            <li>For details about the parameters, see <a href="#Mask">Mask</a>.</li>
            <li>When <code>inputLayout</code> is <code>TND</code>, <code>TND_NTD</code>, or <code>NTD_TND</code>, see <a href="#TND">Restrictions on <code>query</code>, <code>key</code>, and <code>value</code> in the <code>TND</code>, <code>TND_NTD</code>, and <code>NTD_TND</code> scenarios</a>.</li>
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
        <td>High precision, high performance, and whether to perform invalid row correction.</td>
        <td>
        For details about the parameters, see <a href="#innerPrecise">innerPrecise</a>.</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>blockSize</td>
        <td>Optional input</td>
        <td>Maximum number of tokens in each block for KV storage in paged attention.</td>
        <td>For details about the constraints, see <a href="#constraints">constraints</a>.</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>antiquantMode</td>
        <td>Optional input</td>
        <td>Fake-quantization mode of <code>key</code>/<code>value</code>. </td>
        <td>
        <ul>
            <li>For details about <code>quantMode</code>, see <a href="#AntiQuant">Fake-quantization parameters</a>. </li>
            <li>For details about the constraints, see <a href="#constraints">Constraints</a>.</li>
        </ul>
        </td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>softmaxLseFlag</td>
        <td>Optional input</td>
        <td>Whether to output <code>softmaxLse</code>. S-axis outer splitting (augmented output) is supported.</td>
        <td>-</td>
        <td>bool</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>keyAntiquantMode</td>
        <td>Optional input</td>
        <td>Dequantization mode of <code>key</code>.</td>
        <td>
        <ul>
            <li>Except for the scenario where <code>keyAntiquantMode</code> is <code>0</code> and <code>valueAntiquantMode</code> is <code>1</code>, the value must be the same as that of <code>valueAntiquantMode</code>.</li>
            <li>For details about <code>quantMode</code>, see <a href="#AntiQuant">Fake-quantization parameters</a>.</li>
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
        <td>Optional input</td>
        <td>Dequantization mode of <code>value</code>.</td>
        <td>
        <ul>
            <li>Except for the scenario where <code>keyAntiquantMode</code> is <code>0</code> and <code>valueAntiquantMode</code> is <code>1</code>, the value must be the same as that of <code>keyAntiquantMode</code>.</li>
            <li>For details about <code>quantMode</code>, see <a href="#AntiQuant">Fake-quantization parameters</a>.</li>
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
        <td>Optional input</td>
        <td>Dequantization mode of <code>query</code>.</td>
        <td>
        <ul>
            <li>In the current version, only <code>3</code> can be passed, indicating mode 3, namely the per-token + per-head mode.</li>
            <li>For details about the constraints, see <a href="#constraints">Constraints</a>.</li>
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
        <td>Output of <code>attention</code> in the formula.</td>
        <td>-</td>
        <td>FLOAT16, BFLOAT16, INT8</td>
        <td>ND</td>
        <td>Dimension <code>D</code> of this parameter must be the same as that of <code>value</code>, and other dimensions must be the same as those in the shape of <code>query</code>.</td>
        <td>-</td>
    </tr>
    <tr>
        <td>softmaxLse</td>
        <td>Output</td>
        <td>
        In the ring attention algorithm, the product of <code>query</code> and <code>key</code> is first processed to obtain <code>softmax_max</code>. This max value is subtracted from the product before calculating the exponential, which is then summed to yield <code>softmax_sum</code>. Finally, the log of <code>softmax_sum</code> is added back to <code>softmax_max</code> to obtain the final result.</td>
        <td>
        <ul>    
            <li>When <code>softmaxLseFlag</code> is <code>True</code>, data that is <code>inf</code> is invalid.</li>
            <li>When <code>softmaxLseFlag</code> is <code>False</code>, if the <code>softmaxLse</code> tensor is not <code>nullptr</code>, the tensor data is returned directly. If <code>softmaxLse</code> is <code>nullptr</code>, a tensor of shape <code>{1}</code> filled with zeros is returned.</li>
        </ul>
        </td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>The shape must be <code>[B, N, Q_S, 1]</code> in general. When <code>inputLayout</code> is <code>TND</code> or <code>NTD_TND</code>, the shape must be <code>[T, N, 1]</code>.</td>
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
    </tbody>
    </table>

- **Returns:**

    `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

    The first-phase API implements input parameter verification. The following errors may be thrown.

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
                <td>The data type or data format of <code>query</code>, <code>key</code>, <code>value</code>, <code>pseShift</code>, <code>attenMaskOptional</code>, or <code>attentionOut</code> is not supported.</td>
            </tr>
            <tr>
                <td>ACLNN_ERR_RUNTIME_ERROR</td>
                <td>361001</td>
                <td>An exception occurred when the NPU Runtime API was called.</td>
            </tr>
        </tbody>
    </table>

## aclnnFusedInferAttentionScoreV4

- **Parameters:**

    <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
    <col style="width: 168px">
    <col style="width: 128px">
    <col style="width: 854px">
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
        <td>Address of the workspace to be allocated on the device.</td>
        </tr>
        <tr>
        <td>workspaceSize</td>
        <td>Input</td>
        <td>Size of the workspace to be allocated on the device, which is obtained by calling the first-phase API <code>aclnnFusedInferAttentionScoreV4GetWorkspaceSize</code>.</td>
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

- **Returns:**

    `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - `aclnnFusedInferAttentionScoreV4` defaults to a deterministic implementation.
- Common constraints
    - Processing when the input parameter is empty:
        - An empty tensor indicates that the shapeSize of the required input and output is 0. In the empty tensor scenario, if attentionOut is empty, the empty tensor is returned. Otherwise, all 0s are returned. If lse is empty, the empty tensor is returned. If lse is not empty, all inf values are returned. When the tensor is not empty, the input is intercepted normally.
        - The shapeSize of all tensors in query and attentionOut is 0, which means that the tensor is empty.
        - The shapeSize of all tensors in query and attentionOut is not 0. If lse is not empty and the shapeSize of all tensors in key and value is 0, the tensor is empty.
        - If both attentionOut and lse are empty, the tensor is empty.
        - If the tensor is empty, the verification process is skipped. Otherwise, the normal verification process is performed.
- Restrictions on the alibi fullmask scenario
  - innerPrecise is 0.
  - pseShiftOptional is not empty, and its shape is [B, N, maxQ, maxKV].
  - attenMaskOptional is empty.
  - sparseMode is 0.

<details>

<summary><a id="Mask"></a>Mask</summary>
    &nbsp;&nbsp;<table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
        <col style="width: 165px">
        <col style="width: 625px">
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
            <td><code>defaultMask</code> mode. If <code>attenMask</code> is not passed, the mask operation is not performed, and <code>preTokens</code> and <code>nextTokens</code> are ignored. If <code>attenMask</code> is passed, a complete <code>attenMask</code> matrix needs to be passed, indicating that the portion between <code>preTokens</code> and <code>nextTokens</code> needs to be calculated.</td>
            <td>-</td>
        </tr>
        <tr>
            <td>1</td>
            <td><code>allMask</code> mode. A complete <code>attenMask</code> matrix must be passed.</td>
            <td>-</td>
        </tr>
        <tr>
            <td>2</td>
            <td><code>leftUpCausal</code> mode. An optimized <code>attenMask</code> matrix needs to be passed.</td>
            <td rowspan="3">The passed <code>attenMask</code> is a lower triangular matrix, with all zeros on the diagonal. If attenMask is nullptr or the passed shape is incorrect, an error is reported.</td>
        </tr>
        <tr>
            <td>3</td>
            <td><code>rightDownCausal</code> mode. This corresponds to a lower-triangular matrix partitioned by the top-right vertex. An optimized <code>attenMask</code> matrix needs to be passed.</td>
        </tr>
        <tr>
            <td>4</td>
            <td><code>band</code> mode. An optimized <code>attenMask</code> matrix needs to be passed.</td>
        </tr>
        <tr>
            <td>5</td>
            <td><code>prefix</code> mode</td>
            <td>Not supported.</td>
        </tr>
        <tr>
            <td>6</td>
            <td><code>global</code> mode</td>
            <td>Not supported.</td>
        </tr>
        <tr>
            <td>7</td>
            <td><code>dilated</code> mode</td>
            <td>Not supported.</td>
        </tr>
        <tr>
            <td>8</td>
            <td><code>block_local</code> mode</td>
            <td>Not supported.</td>
        </tr>
        </tbody>
    </table>

</details>

<details>

<summary><a id="PagedAttention"></a>PagedAttention</summary>

- The prerequisite for enabling paged attention is that `blockTable` exists and is valid, and `key` and `value` are arranged in a continuous memory based on the indexes in `blockTable`. In this scenario, `inputLayout` of `key` and `value` is invalid.

- <term>Atlas A2 training products/Atlas A2 inference products</term>:

    <table style="undefined;table-layout: fixed; width: 1354px"><colgroup>
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
            <td rowspan="2">constraints</td>
            <td>blockSize</td>
            <td>When PagedAttention is enabled and in non-quantization scenarios, blockSize must be set to a non-zero value. The restrictions are as follows: In the MLA scenario, blockSize must be 16-byte aligned and cannot exceed 1024. In the GQA scenario, when the head dimension of the query, key, and value is 64 or 128, blockSize must be 16-byte aligned and cannot exceed 1024. In the GQA scenario, when the head dimension of the query, key, and value is not 64 or 128 and Q_S is greater than 1, blockSize must be 128-byte aligned and cannot exceed 1024. In the GQA scenario, when the head dimension of the query, key, and value is not 64 or 128 and Q_S is equal to 1, blockSize must be 16-byte aligned and cannot exceed 512</td>.
            <td><code>BlockSize</code> is a user-defined parameter. Its value affects the paged attention performance. Generally, paged attention can improve the throughput but deteriorate the performance.</td>
        </tr>
        <tr>
            <td><code>blockTable</code></td>
            <td>In the paged attention scenario, <code>blockTable</code> must be two-dimensional. The length of the first dimension must be equal to <code>B</code>, and the length of the second dimension must be greater than or equal to <code>maxBlockNumPerSeq</code> (the maximum number of blocks corresponding to <code>actualSeqLengthsKv</code> in each batch).</td>
            <td>-</td>
        </tr>
        <tr>
            <td rowspan="2">General scenarios</td>
            <td><code>key</code>/<code>value</code></td>
            <td>
            The data types of <code>key</code> and <code>value</code> can be FLOAT16, BFLOAT16, or INT8.
            In the paged attention scenario, the supported KV cache layouts include BnBsH <code>(BlockNum, BlockSize, H)</code>, BnNBsD <code>(BlockNum, N, BlockSize, D)</code>, and NZ <code>(BlockNum, N, D/16, BlockSize, 16)</code>. The layouts of <code>Q</code> (<code>BSH</code>/<code>BSND</code>, <code>BNSD</code>, <code>TND</code>, and <code>NTD</code>) can be crossed.</td>
            <td>In the paged attention scenario, the performance is generally better when the KV cache layout is BnNBsD than when it is BnBsH. Therefore, BnNBsD is recommended.<br>The value of <code>BlockNum</code> cannot be less than the sum of blocks in each batch calculated based on <code>actualSeqLengthsKv</code> and <code>BlockSize</code>. The shapes of <code>key</code> and <code>value</code> must be the same.<br>In the paged attention scenario, if the input KV cache layout is BnBsH <code>(BlockNum, BlockSize, H)</code> and the product of <code>KV_N</code> multiplied by <code>D</code> exceeds <code>65535</code>, an error will be reported due to hardware instruction constraints. This problem can be solved by enabling GQA (decreasing <code>KV_N</code>) or adjusting the KV cache layout to BnNBsD <code>(BlockNum, KV_N, BlockSize, D)</code>.</td>
        </tr>
        <tr>
            <td><code>actualSeqLengthsKv</code></td>
            <td>In the paged attention scenario, <code>actualSeqLengthsKv</code> must be passed.</td>
            <td>-</td>
        </tr>
        <tr>
            <td colspan="4">constraints does not support the tensor list or left padding.</td>
        </tr>
        </tbody>
    </table>

</details>

<details>

<summary><a id="innerPrecise"></a>innerPrecise</summary>

**NOTE**

<blockquote>
The high-precision and high-performance modes are applicable to both BFLOAT16 and INT8. Invalid row correction takes effect for FLOAT16, BFLOAT16, and INT8.<br>
The values <code>0</code> and <code>1</code> are reserved. If the masks involved in the computation are all 1s, the precision may be affected. In this case, you can set this parameter to <code>2</code> or <code>3</code> to enable invalid row correction to improve the precision. However, this configuration deteriorates the performance.<br>
If the operator can determine that invalid rows exist, invalid row correction is automatically enabled, such as in scenarios where <code>sparseMode</code> is <code>3</code> and <code>Sq</code> is greater than <code>Skv</code>.
</blockquote>

<div style="overflow-x: auto;">
<table style="undefined;table-layout: fixed;  width: 1110px">
    <colgroup>
        <col style="width: 110px">
        <col style="width: 150px">
        <col style="width: 350px">
        <col style="width: 500px">
    </colgroup>
    <thead>
        <tr>
            <th>Scenario</th>
            <th>innerPrecise</th>
            <th>Description</th>
            <th>Remarks</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan="4"><code>Q_S</code> = <code>1</code></td>
            <td>0</td>
            <td> High-precision mode is enabled and invalid rows are not corrected.</td>
            <td>-</td>
        </tr>
        <tr>
            <td>1</td>
            <td> High-performance mode is enabled and invalid rows are not corrected.</td>
            <td>-</td>
        </tr>
        <tr>
            <td>2</td>
            <td> High-precision mode is enabled and invalid rows are corrected.</td>
            <td rowspan="2"> <ul>
            <li>Invalid row correction is supported when <code>D</code> is <code>512</code> and in rope-separated implementations.</li>
            <li>Invalid row correction is supported when dimension <code>d</code> of <code>query</code>/<code>key</code> is <code>128</code>, dimension <code>d</code> of <code>rope</code> is <code>0</code>, and dimension <code>d</code> of <code>value</code> is <code>128</code>.</li>
            <li>Invalid row correction is supported when dimension <code>d</code> of <code>query</code>/<code>key</code> is <code>64</code>, dimension <code>d</code> of <code>rope</code> is <code>0</code>, and dimension <code>d</code> of <code>value</code> is <code>64</code>.</li>
            <li>Invalid row correction is supported when dimension <code>d</code> of <code>query</code>/<code>key</code> is <code>192</code>, dimension <code>d</code> of <code>rope</code> is <code>0</code>, and dimension <code>d</code> of <code>value</code> is <code>128</code>.</li>
            <li>Invalid row correction is supported when dimension <code>d</code> of <code>query</code>/<code>key</code> is <code>128</code>, dimension <code>d</code> of <code>rope</code> is <code>64</code>, and dimension <code>d</code> of <code>value</code> is <code>128</code>.</li>
            </ul></td>
        </tr>
        <tr>
            <td>3</td>
            <td> High-performance mode is enabled and invalid rows are corrected.</td>
            <td>-</td>
        </tr>
        <tr>
            <td rowspan="4"><code>Q_S</code> > <code>1</code></td>
            <td>0</td>
            <td> High-precision mode is enabled and invalid rows are not corrected.</td>
            <td rowspan="4"> If <code>sparseMode</code> is <code>0</code> or <code>1</code> and a user-defined mask is passed, it is recommended that invalid row correction be enabled (by setting <code>innerPrecise</code> to <code>2</code> or <code>3</code>).</td>
        </tr>
        <tr>
            <td>1</td>
            <td> High-performance mode is enabled and invalid rows are not corrected.</td>
        </tr>
        <tr>
            <td>2</td>
            <td> High-precision mode is enabled and invalid rows are corrected.</td>
        </tr>
        <tr>
            <td>3</td>
            <td> High-performance mode is enabled and invalid rows are corrected.</td>
        </tr>
    </tbody>
</table></div>

</details>

<details>

<summary><a id="pseShift"></a>pseShift:</summary>
    <div style="overflow-x: auto;">
    &nbsp;&nbsp;<table style="undefined;table-layout: fixed;  width: 1460px">
        <colgroup>
            <col style="width: 130px">
            <col style="width: 190px">
            <col style="width: 130px">
            <col style="width: 180px">
            <col style="width: 280px">
            <col style="width: 550px">
        </colgroup>
        <thead>
        <tr>
            <th colspan="3" style="text-align: center;">Supported Scenario</th>
            <th>pseShiftOptional Data Type Constraint</th>
            <th>Shape Constraint</th>
            <th>Remarks</th>
        </tr>
        </thead>
        <tbody>
            <tr>
                <td rowspan="3">P_S1 (the third dimension of the PSE shape) > 1</td>
                <td rowspan="3">Data type of <code>query</code></td>
                <td>FLOAT16</td>
                <td>FLOAT16</td>
                <td rowspan="3">(B,Q_N,P_S1,P_S2), (1,Q_N,P_S1,P_S2)</td>
                <td rowspan="3">
                <ul>
                <li>When the data type of <code>query</code> is FLOAT16 and <code>pseShift</code> exists, the high-precision mode is forcibly used. The corresponding restrictions are the same as those of the high-precision mode.</li>
                <li>P_S1 must be greater than or equal to the S length of query, and P_S2 must be greater than or equal to the S length of key. In the prefix scenario, P_S2 must be greater than or equal to the sum of actualSharedPrefixLen and the S length of key.</li>
                <li>It is recommended that P_S2 be padded to 32 bytes for alignment to improve performance.</li>
                <li>In the fake-quantization scenario, this scenario is not supported when the S value of the query is 1.</li>
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
                <li>It is recommended that P_S2 be padded to 32 bytes for alignment to improve performance.</li>
                <li>Only D-axis alignment is supported. That is, the D axis is exactly divisible by 16.</li>
                </ul>
                </td>
            </tr>
            <tr>
                <td>BFLOAT16</td>
                <td>BFLOAT16</td>
            </tr>
        </tbody>
    </table></div>

</details>

<details>

<summary><a id="INT8"></a>INT8 quantization scenario:</summary>

- <term>Atlas A2 training products/Atlas A2 inference products</term>:

    <table style="undefined;table-layout: fixed;  width: 1150px">
        <colgroup>
            <col style="width: 275px">
            <col style="width: 151px">
            <col style="width: 724px">
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
                <td rowspan="9">Input and output of type INT8</td>
                <td>query</td>
                <td>The data type is INT8.</td>
            </tr>
            <tr>
                <td>key</td>
                <td>The data type is INT8.</td>
            </tr>
            <tr>
                <td>value</td>
                <td>The data type is INT8.</td>
            </tr>
            <tr>
                <td>deqScale1</td>
                <td rowspan="3">All of the parameters must exist at the same time.</td>
            </tr>
            <tr>
                <td>quantScale1</td>
            </tr>
            <tr>
                <td>deqScale2</td>
            </tr>
            <tr>
                <td>quantScale2</td>
                <td>The data type can be FLOAT32 or BFLOAT16. The data format can be per-tensor or per-channel.
                </td>
            </tr>
            <tr>
                <td>quantOffset2</td>
                <td>If the optional parameter <code>quantOffset2</code> is passed, ensure that its type and shape are consistent with those of <code>quantScale2</code>. If this parameter is not passed, the default value is <code>nullptr</code>, indicating <code>0</code>.
                </td>
            </tr>
            <tr>
                <td>attentionOut</td>
                <td>The data type is INT8.</td>
            </tr>
            <tr>
                <td rowspan="9">Input of type INT8, output of type FLOAT16</td>
                <td>query</td>
                <td>The data type is INT8.</td>
            </tr>
            <tr>
                <td>key</td>
                <td>The data type is INT8.</td>
            </tr>
            <tr>
                <td>value</td>
                <td>The data type is INT8.</td>
            </tr>
            <tr>
                <td>deqScale1</td>
                <td rowspan="3">All of the parameters must exist at the same time.</td>
            </tr>
            <tr>
                <td>quantScale1</td>
            </tr>
            <tr>
                <td>deqScale2</td>
            </tr>
            <tr>
                <td>quantScale2</td>
                <td>If the input parameter <code>quantScale2</code> exists, an error is reported and returned.
                </td>
            </tr>
            <tr>
                <td>quantOffset2</td>
                <td>If the input parameter <code>quantOffset2</code> exists, an error is reported and returned.</td>
            </tr>
            <tr>
                <td>attentionOut</td>
                <td>The data type is FLOAT16.</td>
            </tr>
            <tr>
                <td rowspan="10">The input is FLOAT16 or BFLOAT16 and the output is INT8.</td>
                <td>query</td>
                <td>The data type is FLOAT16 or BFLOAT16.</td>
            </tr>
            <tr>
                <td>key</td>
                <td>The data type is FLOAT16 or BFLOAT16.</td>
            </tr>
            <tr>
                <td>value</td>
                <td>The data type is FLOAT16 or BFLOAT16.</td>
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
                    <li>Per-tensor format: Only shape [1] is supported.</li>
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
                <td>The data type is INT8.</td>
            </tr>
            <tr>
                <td><code>sparseMode</code></td>
                <td>
                    <ul>
                    <li>When the output is of type INT8, <code>sparse</code> cannot be <code>band</code> and <code>preTokens</code> or <code>nextTokens</code> cannot be negative.</li>
                    <li>When the output is of type INT8, if the input parameter <code>quantOffset2</code> is a non-null pointer and a non-null tensor, and <code>sparseMode</code>, <code>preTokens</code>, and <code>nextTokens</code> meet the following conditions, certain rows of the matrix will not be involved in computation, resulting in a computation result error. In this scenario, the computation will be intercepted. (Solution: To prevent interception, perform post-quantization outside the FIA interface.)</li>
                        <ul>
                        <li>When <code>sparseMode</code> is <code>0</code> and <code>attenMaskOptional</code> is a non-null pointer, interception occurs if for any batch: <code>actualSeqLengths</code> – <code>actualSeqLengthsKV</code> – <code>actualSharedPrefixLen</code> – <code>preTokens</code> > <code>0</code>, or <code>nextTokens</code> < <code>0</code>.</li>
                        <li>When <code>sparseMode</code> is <code>1</code> or <code>2</code>, interception does not occur.</li>
                        <li>When <code>sparseMode</code> is <code>3</code>, interception occurs if for any batch: <code>actualSeqLengthsKV</code> + <code>actualSharedPrefixLen</code> – <code>actualSeqLengths</code> < <code>0</code>.</li>
                        <li>When <code>sparseMode</code> is <code>4</code>, interception occurs if for any batch: <code>preTokens</code> < <code>0</code>, or <code>nextTokens</code> + <code>actualSeqLengthsKV</code> + <code>actualSharedPrefixLen</code> – <code>actualSeqLengths</code> < <code>0</code>.</li>
                        </ul>
                    </ul>
                </td>
            </tr>
        </tbody>
    </table>

</details>

<details>

<summary><a id="AntiQuant"></a>Constraints on fake-quantization parameters:</summary>

- When both fake-quantization parameters and KV separation quantization parameters are passed, the KV separation quantization parameters take effect.

- <term>Atlas A2 training products/Atlas A2 inference products</term>:

    <table style="undefined;table-layout: fixed;  width: 2084px">
        <colgroup>
            <col style="width: 105px">
            <col style="width: 134px">
            <col style="width: 198px">
            <col style="width: 161px">
            <col style="width: 159px">
            <col style="width: 166px">
            <col style="width: 187px">
            <col style="width: 251px">
            <col style="width: 430px">
            <col style="width: 293px">
        </colgroup>
        <thead>
            <tr>
                <th rowspan="2">Scenario </th>
                <th rowspan="2">quantMode</th>
                <th rowspan="2">Quantization Mode </th>
                <th colspan="3">No KV Separation </th>
                <th colspan="4">KV Separation </th>
            </tr>
            <tr>
                <th>AntiquantMode</th>
                <th>antiquantScale</th>
                <th>AntiquantOffset</th>
                <th>keyAntiquantMode and valueAntiquantMode</th>
                <th colspan="2">keyAntiquantScaleOptional and valueAntiquantScaleOptional</th>
                <th>keyAntiquantOffset and valueAntiquantOffset</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td rowspan="2"><code>Q_S</code> > <code>1</code></td>
                <td>0</td>
                <td>Per-channel (including per-tensor)</td>
                <td rowspan="2">Invalid parameter.
                </td>
                <td rowspan="2">
                    The data type can only be FLOAT16.
                </td>
                <td rowspan="2">
                    The data type can only be FLOAT16.
                </td>
                <td rowspan="2">
                    <ul>
                    <li>Only the values <code>0</code> and <code>1</code> are supported. Other values will result in an execution error.</li>
                    <li><code>keyAntiquantMode</code> and <code>valueAntiquantMode</code> must be the same.</li>
                    </ul>
                </td>
                <td rowspan="2">
                    If neither <code>keyAntiquantScale</code> nor <code>valueAntiquantScaleOptional</code> is null:
                        <ul>
                        <li>Their shapes must be the same.</li>
                        <li>The value of <code>S</code> in <code>query</code> must be less than or equal to <code>16</code>.</li>
                        <li>The data type of <code>query</code> must be BFLOAT16, the data types of <code>key</code> and <code>value</code> must be INT8, and the output data type must be BFLOAT16.</li>
                        <li>The tensor list, left padding, and paged attention are not supported.</li>
                    </ul>
                </td>
                <td>
                In per-channel mode, the shapes of the two parameters must be <code>(N, D)</code>, <code>(N, 1, D)</code>, or <code>(H)</code>, and the data type is fixed at BF16.
                </td>
                <td rowspan="2">
                    <ul>
                    <li>Both <code>keyAntiquantOffset </code> and <code>valueAntiquantOffset</code> must be either null or non-null.</li>
                    <li>If neither <code>keyAntiquantOffset</code> nor <code>valueAntiquantOffset</code> is null, their shapes must be the same.
                    </li>
                    </ul>
                </td>
            </tr>
            <tr>
                <td>1</td>
                <td>Per-token</td>
                <td>
                    In per-token mode, the shapes of the two parameters must be <code>(B, S)</code>, and the data type is fixed at FLOAT32.
                </td>
            </tr>
            <tr>
                <td rowspan="7"><code>Q_S</code> = <code>1</code></td>
                <td>0</td>
                <td>Per-channel (including per-tensor)</td>
                <td rowspan="7">If a value other than <code>0</code> or <code>1</code> is passed, an exception occurs.</td>
                <td rowspan="7">The data type can be FLOAT16, BFLOAT16, or FLOAT32.</td>
                <td rowspan="7">The data type can be FLOAT16, BFLOAT16, or FLOAT32.</td>
                <td rowspan="7">Except when <code>keyAntiquantMode</code> is <code>0</code> and <code>valueAntiquantMode</code> is <code>1</code>, the values of <code>keyAntiquantMode</code> and <code>valueAntiquantMode</code> must be the same.</td>
                <td rowspan="7">
                If neither <code>keyAntiquantScaleOptional</code> nor <code>valueAntiquantScaleOptional</code> is null:
                    <ul>
                    <li>Their shapes must be the same, except when <code>keyAntiquantMode</code> is <code>0</code> and <code>valueAntiquantMode</code> is <code>1</code>.</li>
                    </ul>
                </td>
                <td>
                    <ul>
                    <li>Per-channel mode: The shapes of the two parameters can be <code>(1, N, 1, D)</code>, <code>(1, N, D)</code>, or <code>(1, H)</code>, their data type is the same as that of <code>query</code>, and the data types of <code>key</code> and <code>value</code> are INT8 or INT4 (INT32).</li>
                    <li>Per-tensor mode: The shapes of the two parameters are both <code>(1)</code>, their data type is the same as that of <code>query</code>, and the data types of <code>key</code> and <code>value</code> are INT8.</li>
                    </ul>
                </td>
                <td rowspan="7">
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
                <td>1</td>
                <td>Per-token</td>
                <td>The shapes of the two parameters are both <code>(1, B, S)</code>, their data type is fixed at FLOAT32, and the data types of <code>key</code> and <code>value</code> are INT8 or INT4 (INT32).</td>
            </tr>
            <tr>
                <td>2</td>
                <td>Per-tensor + per-head</td>
                <td>The shapes of the two parameters are both <code>(N)</code>, their data type is the same as that of <code>query</code>, and the data types of <code>key</code> and <code>value</code> are INT8.</td>
            </tr>
            <tr>
                <td>3</td>
                <td>Per-channel + per-token for <code>value</code></td>
                <td>Per-channel for <code>key</code> + per-token for <code>value</code>: In per-channel for <code>key</code>, the shapes of the two parameters can be <code>(1, N, 1, D)</code>, <code>(1, N, D)</code>, or <code>(1, H)</code>, and their data type is the same as that of <code>query</code>. In per-token for <code>value</code>, the shapes of the two parameters are both <code>(1, B, S)</code>, and their data type is fixed at FLOAT32. The data types of <code>key</code> and <code>value</code> are INT8 or INT4 (INT32). When the data types of <code>key</code> and <code>value</code> are INT8, only the data types of <code>query</code> and <code>attentionOut</code> can be FLOAT16.</td>
            </tr>
            <tr>
                <td>4</td>
                <td>Per-token + paged attention, used to manage scale/offset</td>
                <td>-</td>
            </tr>
            <tr>
                <td>5</td>
                <td>Per-token + per-head + paged attention, used to manage scale/offset</td>
                <td>-</td>
            </tr>
            <tr>
                <td>6</td>
                <td>Per-token-group</td>
                <td>-</td>
            </tr>
        </tbody>
    </table>

</details>

<details>

<summary><a id="TND"></a>Constraints on the query, key, and value inputs in the TND, TND_NTD, and NTD_TND scenarios:</summary>

- Both `actualSeqLengths` and `actualSeqLengthsKv` must be passed.

- <term>Atlas A2 training products/Atlas A2 inference products</term>:
    <div style="overflow-x: auto;">
    <table style="undefined;table-layout: fixed; width: 1390px"><colgroup>
        <col style="width: 210px">
        <col style="width: 410px">
        <col style="width: 250px">
        <col style="width: 520px">
    </colgroup>
    <thead>
    <tr>
        <th colspan="2">Scenario </th>
        <th>Parameter/Feature</th>
        <th>Constraint</th>
    </tr>
    </thead>
    <tbody>
    <tr>
        <td rowspan="8"><code>d</code> of <code>query</code> = <code>512</code></td>
        <td rowspan="4">General scenarios</td>
        <td><code>inputLayout</code></td>
        <td><code>TND</code> and <code>TND_NTD</code> are supported.</td>
    </tr>
    <tr>
        <td><code>numHeads</code></td>
        <td><code>1</code>, <code>2</code>, <code>4</code>, <code>8</code>, <code>16</code>, <code>32</code>, <code>64</code>, and <code>128</code> are supported.</td>
    </tr>
    <tr>
        <td><code>numKeyValueHeads</code></td>
        <td><code>1</code></td>
    </tr>
    <tr>
        <td><code>sparseMode</code></td>
        <td><code>0</code>, <code>3</code>, and <code>4</code> are supported.</td>
    </tr>
    <tr>
        <td rowspan="2">constraints</td>
        <td><code>blockTable</code></td>
        <td>Not <code>nullptr</code>.</td>
    </tr>
    <tr>
        <td><code>actualSeqLengthsKv</code></td>
        <td><code>actualSeqLengthsKv</code> is equal to the batch size of <code>key</code>/<code>value</code>, indicating the actual length of each batch. The value must be less than or equal to <code>KV_S</code>.</td>
    </tr>
    <tr>
        <td>MLA (<code>queryRope</code> and <code>keyRope</code> not null)</td>
        <td><code>queryRopeOptional</code>/<code>keyRopeOptional</code></td>
        <td>Dimension <code>d</code> of <code>queryRopeOptional</code> and <code>keyRopeOptional</code> is <code>64</code>.</td>
    </tr>
    <tr>
        <td colspan="3"><code>softmaxLse</code>, left padding, tensor list, PSE, prefix, fake-quantization, full quantization, and post-quantization are not supported.</td>
    </tr>
    <tr>
        <td rowspan="7"><code>d</code> of <code>query</code> ≠ <code>512</code></td>
        <td rowspan="1">General scenario</td>
        <td>inputLayout</td>
        <td><code>TND</code>, <code>NTD</code>, and <code>NTD_TND</code> are supported.</td>
    </tr>
    <tr>
        <td>constraints</td>
        <td><code>blockSize</code></td>
        <td>Only 16-byte aligned values less than or equal to <code>1024</code> are supported.</td>
    </tr>
    <tr>
        <td>MLA (<code>queryRope</code> and <code>keyRope</code> not null)</td>
        <td><code>Q_D</code>/<code>K_D</code>/<code>V_D</code></td>
        <td><code>Q_D</code>, <code>K_D</code>, and <code>V_D</code> must be equal to <code>128</code>.</td>
    </tr>
    <tr>
        <td>GQA/MHA/MQA (<code>queryRope</code> and <code>keyRope</code> are null)</td>
        <td><code>Q_D</code>/<code>K_D</code>/<code>V_D</code></td>
        <td>
            Q_D, K_D, and V_D are all equal to 64.<br>Or, Q_D, K_D, and V_D are all equal to 128.<br>Or, when Q_D and K_D are equal to 192, V_D is equal to 128.<br>Or in the TND, GQA/MQA, and innerPrecise=0 scenario, Q_D, K_D, and V_D can be set to the same value and the value must be less than or equal to 256.<br><br>
            <ul>
            <li><strong>GQA/MQA scenario</strong> (numHeads is an integer multiple of numKeyValueHeads and they are not equal):
                <ul>
                <li>Data type: FLOAT16 or BFLOAT16</li>
                <li>Sparse mode:
                    <ul>
                    <li>sparse = 0 (attenMask is nullptr)</li>
                    <li>sparse = 3 (The optimized attenMask is passed.)</li>
                    <li>sparse = 4 (The optimized attenMask is passed. The following conditions must be met: Q_D = K_D = V_D ≤ 256 or Q_D = K_D = 192 and V_D = 128/192. In addition, preTokens ≥ –actualSeqLengths, nextTokens ≥ –actualSeqLengthsKv, and preTokens + nextTokens ≥ 0.)</li>
                    </ul>
                </li>
                <li>innerPrecise: Only 0 is supported (high-precision mode without invalid rows).</li>
                <li>constraints: The BnBsH format is supported (H ≤ 65535, blockSize<=128 16 aligned).</li>
                </ul>
            </li>
            <li><strong>MHA scenario</strong>:
                <ul>
                <li>Data type: FLOAT16 or BFLOAT16</li>
                <li>Sparse mode:
                    <ul>
                    <li>sparse = 0 (attenMask is nullptr)</li>
                    <li>sparse = 3 or 4 (The optimized attenMask is passed.)</li>
                    </ul>
                </li>
                <li>innerPrecise:
                    <ul>
                    <li>FLOAT16: 0 and 1 are supported.</li>
                    <li>BFLOAT16: Only 0 is supported.</li>
                    </ul>
                </li>
                <li>constraints: The BnBsH format is supported (H ≤ 65535, blockSize ≤ 128, 16-byte aligned).</li>
                </ul>
            </li>
            </ul>
        </td>
    </tr>
    <tr>
        <td colspan="3">Left padding, tensor list, PSE, prefix, fake-quantization, full quantization, and post-quantization are not supported.</td>
    </tr>
    </tbody>
    </table></div>

</details>

<details>

<summary><a id="MLA"></a>MLA scenario (when the queryRope and keyRope inputs are not empty)</summary>
    &nbsp;&nbsp;<table style="undefined;table-layout: fixed; width: 1389px"><colgroup>
        <col style="width: 158px">
        <col style="width: 125px">
        <col style="width: 226px">
        <col style="width: 520px">
        <col style="width: 360px">
        </colgroup>
        <thead>
        <tr>
            <th colspan="2">Scenario </th>
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
            <td rowspan="24">query d=512</td>
            <td rowspan="6">General scenarios</td>
            <td>query</td>
            <td>The data type is FLOAT16 or BFLOAT16. <code>Q_N</code> is <code>[1, 2, 4, 8, 16, 32, 64, 128]</code>.</td>
            <td>-</td>
        </tr>
        <tr>
            <td>key</td>
            <td>The data type is the same as that of <code>query</code>. <code>K_N</code> is <code>1</code>. The shape can be five-dimensional.</td>
            <td>When the shape is five-dimensional, the constraints on each dimension are <code>[blockNum, N, D/16, blockSize, 16]</code>.</td>
        </tr>
        <tr>
            <td>value</td>
            <td>The data type is the same as that of <code>query</code>. <code>K_N</code> is <code>1</code>. The shape can be five-dimensional.</td>
            <td>When the shape is five-dimensional, the constraints on each dimension are <code>[blockNum, N, D/16, blockSize, 16]</code>.</td>
        </tr>
        <tr>
            <td>attention</td>
            <td>The data type is the same as that of <code>query</code>.</td>
            <td>-</td>
        </tr>
        <tr>
            <td>actualSeqLengths</td>
            <td>Supported only when the layout is <code>TND</code> and <code>Q_S</code> is greater than <code>1</code>.</td>
            <td>-</td>
        </tr>
        <tr>
            <td>inputLayout</td>
            <td><code>BSH</code>, <code>BSND</code>, <code>BNSD</code>, <code>BNSD_NBSD</code>, <code>BSND_NBSD</code>, <code>BSH_NBSD</code>, <code>TND</code>, and <code>TND_NTD</code> are supported.</td>
            <td>-</td>
        </tr>
        <tr>
            <td>Mask</td>
            <td>sparseMode</td>
            <td><code>0</code>, <code>3</code>, and <code>4</code> are supported.</td>
            <td>-</td>
        </tr>
        <tr>
            <td rowspan="12">Full quantization</td>
            <td>query</td>
            <td>INT8 and the qs range is 1-16.</td>
            <td>-</td>
        </tr>
        <tr>
            <td>key</td>
            <td>INT8. The shape can only be five-dimensional.</td>
            <td>When the shape is five-dimensional, the constraints on each dimension are <code>[blockNum, N, D/32, blockSize, 32]</code>.</td>
        </tr>
        <tr>
            <td>value</td>
            <td>INT8. The shape can only be five-dimensional.</td>
            <td>When the shape is five-dimensional, the constraints on each dimension are <code>[blockNum, N, D/32, blockSize, 32]</code>.</td>
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
            <td>BFLOAT16. The shape can only be five-dimensional.</td>
            <td>When the shape is five-dimensional, the constraints on each dimension are <code>[blockNum, N, D/16, blockSize, 16]</code>.</td>
        </tr>
        <tr>
            <td>dequantScaleQueryOptional</td>
            <td>FLOAT32. It must be used together with keyAntiquantScaleOptional and valueAntiquantScaleOptional.</td>
            <td>There is no dimension D. Other dimensions must be the same as the shape of the input parameter query.</td>
        </tr>
        <tr>
            <td>keyAntiquantScaleOptional</td>
            <td>FLOAT32. It must be used together with dequantScaleQueryOptional and valueAntiquantScaleOptional. keyAntiquantOffsetOptional and valueAntiquantOffsetOptional cannot be passed. Only the per-tensor mode is supported.</td>
            <td>shape is (1)</td>
        </tr>
        <tr>
            <td>valueAntiquantScaleOptional</td>
            <td>FLOAT32. It must be used together with dequantScaleQueryOptional and keyAntiquantScaleOptional. keyAntiquantOffsetOptional and valueAntiquantOffsetOptional cannot be passed. Only the per-tensor mode is supported.</td>
            <td>shape is (1)</td>
        </tr>
        <tr>
            <td>sparseMode</td>
            <td>In full quantization scenarios, sparseMode can only be set to 0 or 3.</td>
            <td>When qs = 1, only sparseMode = 0 is supported, and attenMask is nullptr. When qs > 1, only sparseMode = 3 is supported, and the shape of attenMask is [2048,2048].</td>
        </tr>
        <tr>
            <td>blockSize</td>
            <td>Only 128 is supported.</td>
            <td>-</td>
        </tr>
        <tr>
            <td>inputLayout</td>
            <td>BSH, BSH_NBSD, BSND, BSND_NBSD, TND, TND_NTD</td>
            <td>-</td>
        </tr>
        <tr>
            <td rowspan="2">constraints</td>
            <td><code>blockTable</code></td>
            <td>Not <code>nullptr</code>.</td>
            <td>constraints must be enabled.</td>
        </tr>
        <tr>
            <td>blockSize</td>
            <td>blockSize must be 16-byte aligned and <= 1024</td>.
            <td>-</td>
        </tr>
        <tr>
            <td rowspan="2">MLA</td>
            <td>queryRope</td>
            <td>The data type is the same as that of <code>query</code>.</td>
            <td>-</td>
        </tr>
        <tr>
            <td>keyRope</td>
            <td>The data type is the same as that of <code>key</code>.</td>
            <td>-</td>
        </tr>
        <tr>
            <td colspan="4">Left padding, tensor list, PSE, prefix, fake-quantization, and post-quantization are not supported.</td>
        </tr>
        <tr>
            <td rowspan="5"><code>d</code> of <code>query</code> = <code>128</code></td>
            <td>Non-quantization</td>
            <td>inputLayout</td>
            <td><code>BSH</code>, <code>BSND</code>, <code>TND</code>, <code>BNSD</code>, <code>NTD</code>, <code>BSH_BNSD</code>, <code>BSND_BNSD</code>, <code>BNSD_BSND</code>, and <code>NTD_TND</code> are supported.</td>
            <td>-</td>
        </tr>
        <tr>
            <td rowspan="2">MLA</td>
            <td>queryRope</td>
            <td>The data type is the same as that of <code>query</code>.</td>
            <td>-</td>
        </tr>
        <tr>
            <td>keyRope</td>
            <td>The data type is the same as that of <code>key</code>.</td>
            <td>-</td>
        </tr>
        <tr>
            <td colspan="4">Other constraints are the same as those when the layout is <code>TND</code> or <code>NTD_TND</code>.</td>
        </tr>
        <tr>
            <td colspan="4">Left padding, tensor list, PSE, prefix, fake-quantization, full quantization, and post-quantization are not supported.</td>
        </tr>
        </tbody>
    </table>

</details>

<details>

<summary><a id="GQA"></a>In the GQA/MHA/MQA fake-quantization scenario, when the key/value shape is five-dimensional, the restrictions are as follows:</summary>
&nbsp;&nbsp;<table style="undefined;table-layout: fixed; width: 983px"><colgroup>
    <col style="width: 96px">
    <col style="width: 104px">
    <col style="width: 179px">
    <col style="width: 312px">
    <col style="width: 340px">
    </colgroup>
    <thead>
        <tr>
            <th colspan="2">Scenario/Feature</th>
            <th>Parameter</th>
            <th>Constraint</th>
            <th>Remarks</th>
        </tr>
    </thead>
    <tbody>
    <tr>
        <td colspan="2" rowspan="5">General scenarios</td>
        <td>query</td>
        <td>The data type is BFLOAT16. <code>Q_S</code> is <code>[1, 16]</code>. <code>Q_D</code> is equal to <code>128</code>.</td>
        <td>-</td>
    </tr>
    <tr>
        <td>key</td>
        <td>INT8. The shape can only be five-dimensional and <code>K_D</code> is equal to <code>128</code>.</td>
        <td>The shape is <code>[blockNum, N, D/32, blockSize, 32]</code>.</td>
    </tr>
    <tr>
        <td>value</td>
        <td>INT8. The shape can only be five-dimensional and <code>V_D</code> is equal to <code>128</code>.</td>
        <td>The shape is <code>[blockNum, N, D/32, blockSize, 32]</code>.</td>
    </tr>
    <tr>
        <td>inputLayout</td>
        <td>BSH, BSND, BNSD, TND</td>
        <td>-</td>
    </tr>
    <tr>
        <td>innerPrecise</td>
        <td><code>1</code></td>
        <td>Only the high-performance mode is supported.</td>
    </tr>
    <tr>
        <td colspan="2">Mask</td>
        <td colspan="3">When MTP is 0, sparseMode=0 and attenMask is nullptr are supported. When MTP is greater than 0 and less than 16, sparseMode=3 and the optimized attenMask matrix are supported. The shape of the attenMask matrix must be (2048 x 2048).</td>
    </tr>
    <tr>
        <td rowspan="9">Fake-quantization</td>
        <td colspan="4">Only KV separation is supported. Only the high-performance mode is supported. Only fake-quantization with q being BF16 and kv being INT8 is supported. queryRope and keyRope cannot be configured. Asymmetric quantization (antiquantOffset, keyAntiquantOffset, and valueAntiquantOffset) is not supported. When inputLayout is TND, only per-channel quantization is supported.</td>
    </tr>
    <tr>
        <td rowspan="4">Per-channel</td>
        <td>keyAntiquantMode</td>
        <td><code>0</code></td>
        <td>-</td>
    </tr>
    <tr>
        <td>valueAntiquantMode</td>
        <td><code>0</code></td>
        <td>-</td>
    </tr>
    <tr>
        <td>keyAntiquantScale</td>
        <td>When <code>inputLayout</code> is <code>BSH</code>: <code>[H]</code><br>When <code>inputLayout</code> is <code>BNSD</code>: <code>[N, 1, D]</code><br>When inputLayout is BSND or TND: [N,D]</td>
        <td>-</td>
    </tr>
    <tr>
        <td>valueAntiquantScale</td>
        <td>Same as <code>keyAntiquantScale</code>.</td>
        <td>-</td>
    </tr>
    <tr>
        <td rowspan="4">Per-token</td>
        <td>keyAntiquantMode</td>
        <td><code>1</code></td>
        <td>-</td>
    </tr>
    <tr>
        <td>valueAntiquantMode</td>
        <td><code>1</code></td>
        <td>-</td>
    </tr>
    <tr>
        <td>keyAntiquantScale</td>
        <td><code>[B, S]</code></td>
        <td><code>S</code> must be greater than or equal to <code>blockTable</code> second dimension × <code>BlockSize</code>.</td>
    </tr>
    <tr>
        <td>valueAntiquantScale</td>
        <td>Same as <code>keyAntiquantScale</code>.</td>
        <td>-</td>
    </tr>
    <tr>
        <td colspan="2" rowspan="2">constraints</td>
        <td><code>blockTable</code></td>
        <td>Not <code>nullptr</code>.</td>
        <td rowspan="2">constraints must be enabled.</td>
    </tr>
    <tr>
        <td>blockSize</td>
        <td><code>128</code> or <code>512</code></td>
    </tr>
    <tr>
        <td colspan="5">Left padding, tensorlist, PSE, prefix, and post-quantization are not supported.</td>
    </tr>
    </tbody>
</table>

</details>

<details>

<summary>When Q_S is greater than 1:</summary>

- <term>Atlas A2 training products/Atlas A2 inference products</term>:

    <table style="undefined;table-layout: fixed; width: 1080px"><colgroup>
    <col style="width: 180px">
    <col style="width: 150px">
    <col style="width: 750px"></colgroup>
        <thead>
            <tr>
                <th>Scenario</th>
                <th>Parameter</th>
                <th>Constraint</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td rowspan="4">General scenarios</td>
                <td><code>query</code>/<code>key</code>/<code>value</code></td>
                <td>
                    <li><code>B</code>: The B-axis value can be less than or equal to <code>65536</code>.
                        <ul>
                        <li>If the input type is INT8 and the D-axis value is not 32-byte aligned, the maximum B-axis value is <code>128</code>. If the input type is FLOAT16 or BFLOAT16 and the D-axis value is not 16-byte aligned, the B-axis value can only be <code>128</code>.</li>
                        </ul>
                    </li>
                    <li><code>N</code> and <code>D</code>: The N-axis value can be less than or equal to <code>256</code>, and the D-axis value can be less than or equal to <code>512</code>. If <code>inputLayout</code> is <code>BSH</code> or <code>BSND</code>, it is recommended that N × D be less than <code>65535</code>.</li>
                    <li><code>S</code>: The S-axis value can be less than or equal to <code>20971520</code> (20M). In some long sequence scenarios, if the computation load is too large, the PFA operator execution may time out (an AI Core error is reported, and <code>errorStr</code> is <code>timeout or trap error</code>). In this case, S axis splitting is recommended. Note: The computation load is affected by parameters such as <code>B</code>, <code>S</code>, <code>N</code>, and <code>D</code>. Larger values indicate larger computation loads. The following lists some typical scenarios with long sequences (that is, the product of <code>B</code>, <code>S</code>, <code>N</code>, and <code>D</code> is large):
                        <ul>
                        <li><code>B</code> = <code>1</code>, <code>Q_N</code> = <code>20</code>, <code>Q_S</code> = <code>2097152</code>, <code>D</code> = <code>256</code>, <code>KV_N</code> = <code>1</code>, <code>KV_S</code> = <code>2097152</code>;</li>
                        <li><code>B</code> = <code>1</code>, <code>Q_N</code> = <code>2</code>, <code>Q_S</code> = <code>20971520</code>, <code>D</code> = <code>256</code>, <code>KV_N</code> = <code>2</code>, <code>KV_S</code> = <code>20971520</code>;</li>
                        <li><code>B</code> = <code>20</code>, <code>Q_N</code> = <code>1</code>, <code>Q_S</code> = <code>2097152</code>, <code>D</code> = <code>256</code>, <code>KV_N</code> = <code>1</code>, <code>KV_S</code> = <code>2097152</code>;</li>
                        <li><code>B</code> = <code>1</code>, <code>Q_N</code> = <code>10</code>, <code>Q_S</code> = <code>2097152</code>, <code>D</code> = <code>512</code>, <code>KV_N</code> = <code>1</code>, <code>KV_S</code> = <code>2097152</code>.</li></ul></li>
                    <li><code>D</code>:
                        <ul>
                        <li>If the data type of <code>query</code>, <code>key</code>, <code>value</code>, or <code>attentionOut</code> includes INT8, the D-axis value must be 32-byte aligned;</li>
                        <li>If the data type of <code>query</code>, <code>key</code>, <code>value</code>, or <code>attentionOut</code> includes INT4, the D-axis value must be 64-byte aligned;</li>
                        <li>If the data type is FLOAT16 or BFLOAT16, the D-axis value must be 16-byte aligned.</li>
                        </ul>
                    </li>
                </td>
            </tr>
            <tr>
                <td><code>actualSeqLengths</code></td>
                <td>
                <ul>
                <li>When <code>inputLayout</code> of <code>query</code> is <code>TND</code> or <code>NTD_TND</code>, see <a href="#TND">Restrictions on <code>query</code>, <code>key</code>, and <code>value</code> in the <code>TND</code>, <code>TND_NTD</code>, and <code>NTD_TND</code> scenarios</a>.</li>
                </ul>
                </td>
            </tr>
            <tr>
                <td><code>actualSeqLengthsKv</code></td>
                <td>
                <ul>
                    <li>When <code>inputLayout</code> of <code>key</code>/<code>value</code> is <code>TND</code> or <code>NTD_TND</code>, see <a href="#TND">Restrictions on <code>query</code>, <code>key</code>, and <code>value</code> in the <code>TND</code>, <code>TND_NTD</code>, and <code>NTD_TND</code> scenarios</a>.</li>
                </ul>
                </td>
            </tr>
            <tr>
                <td><code>innerPrecise</code></td>
                <td>
                <ul>
                <li>When <code>sparseMode</code> is <code>0</code> or <code>1</code> and a user-defined mask is passed, it is recommended that invalid row correction be enabled.</li>
                </ul>
                </td>
            </tr>
            <tr>
                <td rowspan="2">Prefix</td>
                <td><code>keySharedPrefix</code>/<code>valueSharedPrefix</code></td>
                <td>
                <ul>
                <li>When <code>sparseMode</code> is <code>0</code> or <code>1</code> and <code>attenMaskOptional</code> is passed, <code>KV_S</code> must be greater than or equal to the sum of <code>actualSharedPrefixLen</code> and <code>S</code> of <code>key</code>.</li>
                </ul>
                </td>
            </tr>
            <tr>
                <td colspan="2">
                <ul>
                <li>The input types of <code>query</code>, <code>key</code>, and <code>value</code> cannot be all INT8.</li>
                <li>constraints, left padding, and tensor list are not supported.</li>
                </ul>
                </td>
            </tr>
            <tr>
                <td rowspan="2">Mask</td>
                <td><code>attenMaskOptional</code></td>
                <td>
                <ul>
                <li>The recommended input shapes are <code>(Q_S, KV_S)</code>, <code>(B, Q_S, KV_S)</code>, <code>(1, Q_S, KV_S)</code>, <code>(B, 1, Q_S, KV_S)</code>, and <code>(1, 1, Q_S, KV_S)</code>.</li>
                </ul>
                </td>
            </tr>
            <tr>
                <td><code>sparseMode</code></td>
                <td>
                <ul>
                <li>When <code>sparseMode</code> is <code>0</code>: If <code>attenMaskOptional</code> is a null pointer or is passed in the left padding scenario, the input parameters <code>preTokens</code> and <code>nextTokens</code> are ignored.</li>
                <li>When sparseMode is set to 2, 3, or 4, the shape of attenMaskOptional must be (2048, 2048), (1, 2048, 2048), or (1, 1, 2048, 2048). In addition, you need to ensure that the passed attenMaskOptional is a lower triangle. If attenMaskOptional is nullptr or the passed shape is incorrect, an error is reported.</li>
                <li>When <code>sparseMode</code> is <code>1</code>, <code>2</code>, or <code>3</code>: The input parameters <code>preTokens</code> and <code>nextTokens</code> are ignored, and their values are assigned based on related rules.</li>
                <li>When <code>sparseMode</code> is set to other values: An error is reported.</li>
                </ul>
                </td>
            </tr>
            <tr>
                <td rowspan="3">constraints</td>
                <td><code>query</code>/<code>key</code>/<code>value</code></td>
                <td>
                <ul>
                    <li>In the scenario of paged attention fake-quantization, <code>query</code> can be of type FLOAT16 or BFLOAT16, and <code>key</code>/<code>value</code> can be of type INT8.</li>
                    <li>In the scenario of paged attention full quantization, the data type of <code>query</code> cannot be INT8.</li>
                <li>When a mask is passed and <code>sparseMode</code> is not <code>2</code>, <code>3</code>, or <code>4</code>, the last dimension of the mask must be greater than or equal to <code>maxBlockNumPerSeq</code> × <code>blockSize</code>.</li>
                <li>When <code>pseShift</code> is passed, the last dimension of <code>pseShift</code> must be greater than or equal to <code>maxBlockNumPerSeq</code> × <code>blockSize</code>.</li>
                </ul>
                </td>
            </tr>
            <tr>
                <td><code>blockTable</code></td>
                <td>
                <ul>
                <li><code>blockTable</code> is filled with block IDs. Currently, the validity of block IDs is not verified. You need to ensure the validity of block IDs.</li>
                </ul>
                </td>
            </tr>
            <tr>
                <td colspan="2">
                For details about other constraints, see <a href="#constraints">constraints.</a> 
                </td>
            </tr>
            <tr>
                <td rowspan="2">Left padding for <code>query</code></td>
                <td><code>query</code>/<code>key</code>/<code>value</code></td>
                <td>
                    The scenario where <code>Q</code> is of type BF16/FP16 and <code>KV</code> of type INT4 is not supported.
                </td>
            </tr>
            <tr>
                <td colspan="2">Left padding for <code>query</code> does not support paged attention and cannot be enabled together with the <code>blockTable</code> parameter.</td>
            </tr>
            <tr>
                <td rowspan="1">Left padding for <code>kv</code></td>
                <td colspan="2">Left padding for <code>kv</code> does not support paged attention and cannot be enabled together with the <code>blockTable</code> parameter.</td>
            </tr>
            <tr>
                <td rowspan="3">INT8 quantization</td>
                <td><code>quantScale2</code>/<code>quantOffset2</code></td>
                <td>In the synthesis parameter scenario of KV cache dequantization, only when <code>query</code> is of type FLOAT16 can <code>key</code> and <code>value</code> of type INT8 be dequantized to FLOAT16. If the product of the data ranges of the input parameters <code>key</code> and <code>value</code> and the data range of the input parameter <code>antiquantScale</code> must be within the range of (–1, 1), the high-performance mode can ensure precision. Otherwise, the high-precision mode needs to be enabled to ensure precision.
                    <ul>
                    <li>When the output is of type INT8 and <code>quantScale2</code> and <code>quantOffset2</code> are per-channel, left padding, ring attention, or non-32-byte alignment of the D axis is not supported.</li>
                    </ul>
                </td>
            </tr>
        </tbody>
    </table>

</details>

<details>

<summary>When Q_S is equal to 1 (in the IFA non-MTP scenario):</summary>

- <term>Atlas A2 training products/Atlas A2 inference products</term>:

    <table style="undefined;table-layout: fixed; width: 1080px"><colgroup>
    <col style="width: 180px">
    <col style="width: 150px">
    <col style="width: 750px"></colgroup>
        <thead>
            <tr>
                <th>Scenario</th>
                <th>Parameter</th>
                <th>Constraint</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td rowspan="3">Basic parameters</td>
                <td><code>query</code>/<code>key</code>/<code>value</code></td>
                <td>
                <ul>
                <li>The B-axis value can be less than or equal to <code>65536</code>, the N-axis value can be less than or equal to <code>256</code>, and the D-axis value can be less than or equal to <code>512</code>.</li>
                <li>The input types of <code>query</code>, <code>key</code>, and <code>value</code> cannot be all INT8.</li>
                <li>In the INT4 (INT32) fake-quantization scenario, the aclnn single-operator API supports the scenario where the KV inputs are of type int4 or the concatenated int4 inputs are of type int32. (You are advised to use dynamicQuant to generate int4 data because dynamicQuant is an int32 that contains eight int4s.)</li>
                <li>In the INT4 (INT32) fake-quantization scenario, if the KV INT4 is concatenated into an INT32 input, the N, D, or H of the KV is one eighth of the actual value. (The same applies to the prefix.)</li>
                <li>The key and value have restrictions on the D axis under specific data types.
                    <ul>
                    <li>When the key and value are of the INT4 (INT32) type, the D axis must be 64-aligned. (INT32 supports only 8-aligned D axis.)</li>
                    </ul>
                </li>
                </ul>
                </td>
            </tr>
            <tr>
                <td><code>actualSeqLengths</code></td>
                <td>
                <ul>
                    <li>
                    This parameter is invalid when <code>inputLayout</code> of <code>query</code> is not <code>TND</code>. When <code>inputLayout</code> of <code>query</code> is <code>TND</code> or <code>TND_NTD</code>, see <a href="#TND">Restrictions on <code>query</code>, <code>key</code>, and <code>value</code> in the <code>TND</code>, <code>TND_NTD</code>, and <code>NTD_TND</code> scenarios</a>.</li>
                </ul>
                </td>
            </tr>
            <tr>
                <td><code>actualSeqLengthsKv</code></td>
                <td>
                <ul>
                    <li>When <code>inputLayout</code> of <code>key</code>/<code>value</code> is <code>TND</code> or <code>TND_NTD</code>, see <a href="#TND">Restrictions on <code>query</code>, <code>key</code>, and <code>value</code> in the <code>TND</code>, <code>TND_NTD</code>, and <code>NTD_TND</code> scenarios</a>.
                    </li>
                </ul>
                </td>
            </tr>
            <tr>
                <td rowspan="4">constraints</td>
                <td><code>blockSize</code></td>
                <td>
                <ul>
                <li>If the input type of <code>key</code> and <code>value</code> is FLOAT16 or BFLOAT16, 16-byte alignment is required. If the input type of <code>key</code> and <code>value</code> is INT8, 32-byte alignment is required, but 128-byte alignment is recommended.</li>
                </ul>
                </td>
            </tr>
            <tr>
                <td><code>query</code>/<code>key</code>/<code>value</code></td>
                <td>The scenario where Q is BF16/FP16 and KV is INT4 (INT32) is not supported.</td>
            </tr>
            <tr>
                <td><code>blockTable</code></td>
                <td>
                <ul>
                    <li>Attention mask: When <code>sparseMode</code> is not <code>2</code>, <code>3</code>, or <code>4</code>, the last dimension of the passed mask must be greater than or equal to <code>blockTable</code> second dimension × <code>blockSize</code>.</li>
                    <li><code>pseShift</code>: The last dimension of the passed <code>pseShift</code> must be greater than or equal to <code>blockTable</code> second dimension × <code>blockSize</code>.</li>
                    <li>Fake-quantization per-token: The last dimension of <code>antiquantScale</code> and <code>antiquantOffset</code> must be greater than or equal to <code>blockTable</code> second dimension × <code>blockSize</code>.</li>
                    <li>Per-token + per-head: The last dimension of <code>antiquantScale</code> and <code>antiquantOffset</code> must be greater than or equal to <code>blockTable</code> second dimension × <code>blockSize</code>. The data type is fixed at FLOAT32. The data types of <code>key</code> and <code>value</code> are INT8 or INT4 (INT32).</li>
                </ul>
                </td>
            </tr>
            <tr>
                <td colspan="2">
                For details about other constraints, see <a href="#constraints">constraints.</a> 
                </td>
            </tr>
            <tr>
                <td rowspan="2">Left padding for <code>kv</code></td>
                <td><code>attenMaskOptional</code></td>
                <td>When left padding for <code>kv</code> is enabled together with the <code>attenMaskOptional</code> parameter, ensure that the meaning of <code>attenMaskOptional</code> is correct, that is, it can correctly hide invalid data. Otherwise, precision issues may occur.</td>
            </tr>
            <tr>
                <td colspan="2">constraints and tensor list are not supported. Otherwise, right padding of <code>kv</code> is used by default.</td>
            </tr>
            <tr>
                <td>Mask</td>
                <td><code>attenMaskOptional</code></td>
                <td>The recommended input shapes are <code>(B, KV_S)</code>, <code>(B, 1, KV_S)</code>, and <code>(B, 1, 1, KV_S)</code>.</td>
            </tr>
            <tr>
                <td><code>innerPrecise</code></td>
                <td><code>innerPrecise</code></td>
                <td>Only <code>0</code> and <code>1</code> are supported.</td>
            </tr>
            <tr>
                <td rowspan="1">INT8 quantization</td>
                <td colspan="2">The input types of <code>query</code>, <code>key</code>, and <code>value</code> cannot be all INT8.</td>
            </tr>
        </tbody>
    </table>

</details>

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```c++
#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include "acl/acl.h"
#include "aclnn/opdev/fp16_t.h"
#include "aclnnop/aclnn_fused_infer_attention_score_v4.h"
#include "securec.h"

using namespace std;

namespace {

#define CHECK_RET(cond) ((cond) ? true :(false))

#define LOG_PRINT(message, ...)                                                                                        \
    do {                                                                                                               \
        printf(message, ##__VA_ARGS__);                                                                                \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t> &shape) {
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream *stream) {
    // Fixed writing method, AscendCL initialization.
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
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor) {
    auto size = GetShapeSize(shape) * sizeof(T);
    // Call aclrtMalloc to request device side memory.
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    if (!CHECK_RET(ret == ACL_SUCCESS)) { 
        LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); 
        return ret;
    }
    // Call aclrtMemcpy to copy host side data to device side memory.
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); 
        return ret;
    }

    // Calculate the strides of continuous tensors.
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // Call the aclCreateTensor interface to create aclTensor.
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

} // namespace

int main() {
    // 1. (Fixed writing method)  device/stream initialization. Refer to AscendCL's list of external interfaces.
    // Fill in the deviceId based on your actual device.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("Init acl failed. ERROR: %d\n", ret); 
        return ret;
    }

    // 2. To construct input and output, it is necessary to customize the construction according to the API interface.
    int32_t batchSize = 1;
    int32_t numHeads = 2;
    int32_t sequenceLengthQ = 1;
    int32_t headDims = 16;
    int32_t numKeyValueHeads = 2;
    int32_t sequenceLengthKV = 16;
    std::vector<int64_t> queryShape = {batchSize, numHeads, sequenceLengthQ, headDims};           // BNSD
    std::vector<int64_t> keyShape = {batchSize, numKeyValueHeads, sequenceLengthKV, headDims};    // BNSD
    std::vector<int64_t> valueShape = {batchSize, numKeyValueHeads, sequenceLengthKV, headDims};  // BNSD
    std::vector<int64_t> attenMaskShape = {batchSize, 1, sequenceLengthQ, sequenceLengthKV};      // B 1 S1 S2
    std::vector<int64_t> outShape = {batchSize, numHeads, sequenceLengthQ, headDims};             // BNSD
    void *queryDeviceAddr = nullptr;
    void *keyDeviceAddr = nullptr;
    void *valueDeviceAddr = nullptr;
    void *attenMaskDeviceAddr = nullptr;
    void *outDeviceAddr = nullptr;
    aclTensor *queryTensor = nullptr;
    aclTensor *keyTensor = nullptr;
    aclTensor *valueTensor = nullptr;
    aclTensor *attenMaskTensor = nullptr;
    aclTensor *outTensor = nullptr;
    int64_t queryShapeSize = GetShapeSize(queryShape);          // BNSD
    int64_t keyShapeSize = GetShapeSize(keyShape);              // BNSD
    int64_t valueShapeSize = GetShapeSize(valueShape);          // BNSD
    int64_t attenMaskShapeSize = GetShapeSize(attenMaskShape);  // B 1 S1 S2
    int64_t outShapeSize = GetShapeSize(outShape);              // BNSD
    std::vector<op::fp16_t> queryHostData(queryShapeSize, 1);
    std::vector<op::fp16_t> keyHostData(keyShapeSize, 1);
    std::vector<op::fp16_t> valueHostData(valueShapeSize, 1);
    std::vector<int8_t> attenMaskHostData(attenMaskShapeSize, 1);
    std::vector<op::fp16_t> outHostData(outShapeSize, 1);

    // Create query aclTensor.
    ret = CreateAclTensor(queryHostData, queryShape, &queryDeviceAddr, aclDataType::ACL_FLOAT16, &queryTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }
    // Create key aclTensor.
    ret = CreateAclTensor(keyHostData, keyShape, &keyDeviceAddr, aclDataType::ACL_FLOAT16, &keyTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }
    int kvTensorNum = 1;
    aclTensor *tensorsOfKey[kvTensorNum];
    tensorsOfKey[0] = keyTensor;
    auto tensorKeyList = aclCreateTensorList(tensorsOfKey, kvTensorNum);
    // Create value aclTensor.
    ret = CreateAclTensor(valueHostData, valueShape, &valueDeviceAddr, aclDataType::ACL_FLOAT16, &valueTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }
    aclTensor *tensorsOfValue[kvTensorNum];
    tensorsOfValue[0] = valueTensor;
    auto tensorValueList = aclCreateTensorList(tensorsOfValue, kvTensorNum);
    // Create attenMask aclTensor.
    ret = CreateAclTensor(attenMaskHostData, attenMaskShape, &attenMaskDeviceAddr, aclDataType::ACL_BOOL, &attenMaskTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }
    // Create out aclTensor.
    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT16, &outTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }

    std::vector<int64_t> actualSeqlenVector = {2};
    auto actualSeqLengths = aclCreateIntArray(actualSeqlenVector.data(), actualSeqlenVector.size());
    
    double scaleValue = 1 / sqrt(2); // 1/sqrt(d)
    int64_t preTokens = 2147483647;
    int64_t nextTokens = 2147483647;
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
    // 3. Call CANN operator library API.
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor;
    // Call the first interface.
    ret = aclnnFusedInferAttentionScoreV4GetWorkspaceSize(
        queryTensor, tensorKeyList, tensorValueList, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, numHeads, scaleValue, preTokens, nextTokens, layerOut,
        numKeyValueHeads, sparseMode, innerPrecise, blockSize, antiquantMode, softmaxLseFlag, keyAntiquantMode,
        valueAntiquantMode, 0, outTensor, nullptr, &workspaceSize, &executor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclnnFusedInferAttentionScoreV4GetWorkspaceSize failed. ERROR: %d\n", ret);
        return ret;
    }
    // Apply for device memory based on the workspaceSize calculated from the first interface paragraph.
    void *workspaceAddr = nullptr;
    if (workspaceSize > 0U) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (!CHECK_RET(ret == ACL_SUCCESS)) {
            LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); 
            return ret;
        }
    }
    // Call the second interface.
    ret = aclnnFusedInferAttentionScoreV4(workspaceAddr, workspaceSize, executor, stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclnnFusedInferAttentionScoreV4 failed. ERROR: %d\n", ret); 
        return ret;
    }

    // 4. (Fixed writing method) Synchronize and wait for task execution to end.
    ret = aclrtSynchronizeStream(stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); 
        return ret;
    }

    // 5. Retrieve the output value, copy the result from the device side memory to the host side, and modify it
    // according to the specific API interface definition.
    auto size = GetShapeSize(outShape);
    std::vector<op::fp16_t> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                      size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    if (!CHECK_RET(ret == ACL_SUCCESS)) { 
        LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); 
        return ret;
    }
    for (int64_t i = 0; i < size; i++) {
        std::cout << "index: " << i << ": " << static_cast<float>(resultData[i]) << std::endl;
    }
    // 6. Release resources.
    aclDestroyTensor(queryTensor);
    aclDestroyTensor(keyTensor);
    aclDestroyTensor(valueTensor);
    aclDestroyTensor(attenMaskTensor);
    aclDestroyTensor(outTensor);
    aclDestroyIntArray(actualSeqLengths);
    aclrtFree(queryDeviceAddr);
    aclrtFree(keyDeviceAddr);
    aclrtFree(valueDeviceAddr);
    aclrtFree(attenMaskDeviceAddr);
    aclrtFree(outDeviceAddr);
    if (workspaceSize > 0U) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
