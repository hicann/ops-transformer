# Ascendir_FlashAttentionScore

## Product Support

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 training products</term>: supported
<!-- end id2 -->
<!-- npu="A3" id3 -->
- <term>Atlas A3 inference products</term>: not supported
<!-- end id3 -->
<!-- npu="910b" id4 -->
- <term>Atlas A2 training products</term>: supported
<!-- end id4 -->
<!-- npu="910b" id5 -->
- <term>Atlas A2 inference products</term>: not supported
<!-- end id5 -->
<!-- npu="310b" id6 -->
- <term>Atlas 200I/500 A2 inference products</term>: not supported
<!-- end id6 -->
<!-- npu="310p" id7 -->
- <term>Atlas inference products</term>: not supported
<!-- end id7 -->
<!-- npu="910" id8 -->
- <term>Atlas training products</term>: not supported
<!-- end id8 -->

## Function Description

- Operator function: In the training scenario, the FlashAttention algorithm is used to implement self-attention computation:

    - When pseType=1, add must be performed before mul.
    - When pseType≠1, mul must be performed before add.

- Computation formula:

  The forward computation formula of attention is as follows:

    - When pseType=1, the formula is as follows:
      $$
      attention\_out=Dropout(Softmax(Mask(scale*(pse+(query*d\_scale\_q)*(key*d\_scale\_k)^T), atten\_mask)), keep\_prob)*(value*d\_scale\_v)
      $$

    - When pseType≠1, the formula is as follows:
      $$
      attention\_out=Dropout(Softmax(Mask(scale*((query*d\_scale\_q)*(key*d\_scale\_k)^T) + pse),atten\_mask),keep\_prob)*(value*d\_scale\_v)
      $$

## Ascend IR Definition

The header file containing the Ascend IR definition is located at [op_graph/flash_attention_score_proto.h](../op_graph/flash_attention_score_proto.h).

```cpp
REG_OP(FlashAttentionScore)
    .INPUT(query, TensorType({DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .INPUT(key, TensorType({DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .INPUT(value, TensorType({DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .OPTIONAL_INPUT(real_shift, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .OPTIONAL_INPUT(drop_mask, TensorType({DT_UINT8}))
    .OPTIONAL_INPUT(padding_mask, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .OPTIONAL_INPUT(atten_mask, TensorType({DT_BOOL, DT_UINT8}))
    .OPTIONAL_INPUT(prefix, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(actual_seq_qlen, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(actual_seq_kvlen, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(q_start_idx, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(kv_start_idx, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(d_scale_q, TensorType({DT_FLOAT32}))
    .OPTIONAL_INPUT(d_scale_k, TensorType({DT_FLOAT32}))
    .OPTIONAL_INPUT(d_scale_v, TensorType({DT_FLOAT32}))
    .OPTIONAL_INPUT(query_rope, TensorType({DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .OPTIONAL_INPUT(key_rope, TensorType({DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .OPTIONAL_INPUT(sink, TensorType({DT_FLOAT32}))
    .OPTIONAL_INPUT(p_scale, TensorType({DT_FLOAT32}))
    .OUTPUT(softmax_max, TensorType({DT_FLOAT32}))
    .OUTPUT(softmax_sum, TensorType({DT_FLOAT32}))
    .OUTPUT(softmax_out, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .OUTPUT(attention_out, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .ATTR(scale_value, Float, 1.0)
    .ATTR(keep_prob, Float, 1.0)
    .ATTR(pre_tockens, Int, 2147483647)
    .ATTR(next_tockens, Int, 2147483647)
    .REQUIRED_ATTR(head_num, Int)
    .REQUIRED_ATTR(input_layout, String)
    .ATTR(inner_precise, Int, 0)
    .ATTR(sparse_mode, Int, 0)
    .ATTR(pse_type, Int, 1)
    .ATTR(seed, Int, 0)
    .ATTR(offset, Int, 0)
    .ATTR(out_dtype, Int, 0)
    .ATTR(softmax_out_layout, String, "")
    .OP_END_FACTORY_REG(FlashAttentionScore)
```

### Parameter Description

<table style="undefined;table-layout: fixed; width: 1340px"><colgroup>
  <col style="width: 150px">
  <col style="width: 120px">
  <col style="width: 260px">
  <col style="width: 280px">
  <col style="width: 210px">
  <col style="width: 80px">
  <col style="width: 240px">
  </colgroup>
  <thead>
    <tr>
      <th>Parameter</th>
      <th>Input/Output/Attribute</th>
      <th>Description</th>
      <th>Usage Instructions</th>
      <th>Data Type</th>
      <th>Data Format</th>
      <th>Dimension (shape)</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>query (tensor)</td>
      <td>Input</td>
      <td>Input query in the formula.</td>
      <td>The data type is consistent with that of key/value.</td>
      <td>float8_e5m2, float8_e4m3fn, float16, bf16, float32</td>
      <td>ND</td>
      <td>[BNSD], [BSND], [BSH], [SBH], [TND]</td>
    </tr>
    <tr>
      <td>key (tensor)</td>
      <td>Input</td>
      <td>Input key in the formula.</td>
      <td>The data type is consistent with that of query/value.</td>
      <td>float8_e5m2, float8_e4m3fn, float16, bf16, float32</td>
      <td>ND</td>
      <td>[BNSD], [BSND], [BSH], [SBH], [TND]</td>
    </tr>
    <tr>
      <td>value (tensor)</td>
      <td>Input</td>
      <td>Input value in the formula.</td>
      <td>The data type is consistent with that of query/key.</td>
      <td>float8_e5m2, float8_e4m3fn, float16, bf16, float32</td>
      <td>ND</td>
      <td>[BNSD], [BSND], [BSH], [SBH], [TND]</td>
    </tr>
    <tr>
      <td>real_shift (tensor)</td>
      <td>Optional input</td>
      <td>pse in the formula, indicating positional encoding.</td>
      <td>The data type is consistent with that of query. This parameter must be used together with pse_type.</td>
      <td>float16, bf16, float32</td>
      <td>ND</td>
      <td>[B,N,Sq,Skv], [B,N,1,Skv], [1,N,Sq,Skv], [B,N,1024,Skv], [1,N,1024,Skv], [B,N], [N]</td>
    </tr>
    <tr>
      <td>drop_mask (tensor)</td>
      <td>Optional input</td>
      <td>Dropout in the formula, indicating the dropout mask. A value of 1 indicates that the data is retained, and a value of 0 indicates that the data is discarded.</td>
      <td>-</td>
      <td>uint8</td>
      <td>ND</td>
      <td>0, 1</td>
    </tr>
    <tr>
      <td>padding_mask (tensor)</td>
      <td>Optional input</td>
      <td>Reserved parameter, currently unused.</td>
      <td>-</td>
      <td>float16, bf16, float32</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>atten_mask (tensor)</td>
      <td>Optional input</td>
      <td>atten_mask in the formula, indicating the attention mask.</td>
      <td>A value of 1 indicates that the position does not participate in computation (does not take effect), and a value of 0 indicates that the position participates in computation.</td>
      <td>bool, uint8</td>
      <td>ND</td>
      <td>[B,N,Sq,Skv], [B,1,Sq,Skv], [1,1,Sq,Skv], [Sq,Skv]</td>
    </tr>
    <tr>
      <td>prefix (tensor)</td>
      <td>Optional input</td>
      <td>N value of each batch in the prefix sparse computation scenario.</td>
      <td>-</td>
      <td>int64</td>
      <td>ND</td>
      <td>0, 1</td>
    </tr>
    <tr>
      <td>actual_seq_qlen (tensor)</td>
      <td>Optional input</td>
      <td>Actual Q sequence length.</td>
      <td>input_layout must be set to TND when this parameter is used. For example, if the seqlen of q is [2,2,2,2,2], this parameter must be set to [2,4,6,8,10].</td>
      <td>int64</td>
      <td>ND</td>
      <td>1</td>
    </tr>
    <tr>
      <td>actual_seq_kvlen (tensor)</td>
      <td>Optional input</td>
      <td>Actual KV sequence length.</td>
      <td>input_layout must be set to TND when this parameter is used. For example, if the seqlen of kv is [2,2,2,2,2], this parameter must be set to [2,4,6,8,10].</td>
      <td>int64</td>
      <td>ND</td>
      <td>1</td>
    </tr>
    <tr>
      <td>q_start_idx (tensor)</td>
      <td>Optional input</td>
      <td>Start index of Q.</td>
      <td>-</td>
      <td>int64</td>
      <td>ND</td>
      <td>0, 1</td>
    </tr>
    <tr>
      <td>kv_start_idx (tensor)</td>
      <td>Optional input</td>
      <td>Start index of KV.</td>
      <td>-</td>
      <td>int64</td>
      <td>ND</td>
      <td>0, 1</td>
    </tr>
    <tr>
      <td>d_scale_q (tensor)</td>
      <td>Optional input</td>
      <td>d_scale_q in the formula, the per-tensor quantization parameter of query in FP8 scenarios.</td>
      <td>-</td>
      <td>float32</td>
      <td>ND</td>
      <td>-</td>
    </tr>
    <tr>
      <td>d_scale_k (tensor)</td>
      <td>Optional input</td>
      <td>d_scale_k in the formula, the per-tensor quantization parameter of key in FP8 scenarios.</td>
      <td>-</td>
      <td>float32</td>
      <td>ND</td>
      <td>-</td>
    </tr>
    <tr>
      <td>d_scale_v (tensor)</td>
      <td>Optional input</td>
      <td>d_scale_v in the formula, the per-tensor quantization parameter of value in FP8 scenarios.</td>
      <td>-</td>
      <td>float32</td>
      <td>ND</td>
      <td>-</td>
    </tr>
    <tr>
      <td>query_rope (tensor)</td>
      <td>Optional input</td>
      <td>RoPE rotary position encoding input of Q.</td>
      <td>-</td>
      <td>float8_e5m2, float8_e4m3fn, float16, bf16, float32</td>
      <td>ND</td>
      <td>-</td>
    </tr>
    <tr>
      <td>key_rope (tensor)</td>
      <td>Optional input</td>
      <td>RoPE rotary position encoding input of K.</td>
      <td>-</td>
      <td>float8_e5m2, float8_e4m3fn, float16, bf16, float32</td>
      <td>ND</td>
      <td>-</td>
    </tr>
    <tr>
      <td>sink (tensor)</td>
      <td>Optional input</td>
      <td>Sinking parameter.</td>
      <td>-</td>
      <td>float32</td>
      <td>ND</td>
      <td>-</td>
    </tr>
    <tr>
      <td>p_scale (tensor)</td>
      <td>Optional input</td>
      <td>Scale factor of P.</td>
      <td>-</td>
      <td>float32</td>
      <td>ND</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scale_value (float)</td>
      <td>Optional attribute</td>
      <td>
        <ul>
          <li>scale in the formula, indicating the scale factor. It serves as the scalar value of Muls in the computation flow.</li>
          <li>Default value: 1.0.</li>
        </ul>
      </td>
      <td>-</td>
      <td>float</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>keep_prob (float)</td>
      <td>Optional attribute</td>
      <td>
        <ul>
          <li>keep_prob in the formula, indicating the probability that data is retained.</li>
          <li>Default value: 1.0.</li>
        </ul>
      </td>
      <td>The value range is (0, 1].</td>
      <td>float</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>pre_tockens (int)</td>
      <td>Optional attribute</td>
      <td>
        <ul>
          <li>Used for sparse computation, indicating the left boundary of the sliding window.</li>
          <li>Default value: 2147483647.</li>
        </ul>
      </td>
      <td>-</td>
      <td>int</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>next_tockens (int)</td>
      <td>Optional attribute</td>
      <td>
        <ul>
          <li>Used for sparse computation, indicating the right boundary of the sliding window.</li>
          <li>Default value: 2147483647.</li>
        </ul>
      </td>
      <td>-</td>
      <td>int</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>head_num (int)</td>
      <td>Required attribute</td>
      <td>Number of heads on a single device, that is, the length of the N axis of the input query.</td>
      <td>-</td>
      <td>int</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>input_layout (string)</td>
      <td>Required attribute</td>
      <td>
        <ul>
          <li>Data layout format of the input query, key, and value.</li>
          <li>Default value: empty.</li>
        </ul>
      </td>
      <td>Supported values: BSH, SBH, BNSD, BSND, and TND.</td>
      <td>string</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>inner_precise (int)</td>
      <td>Optional attribute</td>
      <td>
        <ul>
          <li>Used to improve precision.</li>
          <li>Default value: 0.</li>
        </ul>
      </td>
      <td>0 and 1 are reserved values; 2 supports invalid-row computation. Use the default value 0.</td>
      <td>int</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>sparse_mode (int)</td>
      <td>Optional attribute</td>
      <td>
        <ul>
          <li>Indicates the sparse mode. 0 indicates defaultMask mode, 1 indicates allMask, 2 indicates leftUpCausal, 3 indicates rightDownCausal, 4 indicates band, 5 indicates prefix, 6 indicates prefixCompress, 7 indicates rightDownCausalBand, and 8 indicates bandLeftUpCausal.</li>
          <li>Default value: 0.</li>
        </ul>
      </td>
      <td>Supported values: 0 to 8.</td>
      <td>int</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>pse_type (int)</td>
      <td>Optional attribute</td>
      <td>
        <ul>
          <li>Controls the execution order of add and mul.</li>
          <li>Default value: 1.</li>
        </ul>
      </td>
      <td>Supported values: 0, 1, 2, and 3.</td>
      <td>int</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>seed (int)</td>
      <td>Optional attribute</td>
      <td>
        <ul>
          <li>Random seed for Dropout.</li>
          <li>Default value: 0.</li>
        </ul>
      </td>
      <td>-</td>
      <td>int</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>offset (int)</td>
      <td>Optional attribute</td>
      <td>
        <ul>
          <li>Dropout offset.</li>
          <li>Default value: 0.</li>
        </ul>
      </td>
      <td>-</td>
      <td>int</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out_dtype (int)</td>
      <td>Optional attribute</td>
      <td>
        <ul>
          <li>Output precision control.</li>
          <li>Default value: 0.</li>
        </ul>
      </td>
      <td>-</td>
      <td>int</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>softmax_out_layout (string)</td>
      <td>Optional attribute</td>
      <td>
        <ul>
          <li>Data layout format of the Softmax output.</li>
          <li>Default value: empty.</li>
        </ul>
      </td>
      <td>Supported values: "" (empty) and "same_as_input".</td>
      <td>string</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>softmax_max (tensor)</td>
      <td>Output</td>
      <td>Max intermediate result of the Softmax computation, used for backward computation.</td>
      <td>-</td>
      <td>float32</td>
      <td>ND</td>
      <td>[B,N,Sq,8]</td>
    </tr>
    <tr>
      <td>softmax_sum (tensor)</td>
      <td>Output</td>
      <td>Sum intermediate result of the Softmax computation, used for backward computation.</td>
      <td>-</td>
      <td>float32</td>
      <td>ND</td>
      <td>[B,N,Sq,8]</td>
    </tr>
    <tr>
      <td>softmax_out (tensor)</td>
      <td>Output</td>
      <td>Reserved parameter, currently unused.</td>
      <td>-</td>
      <td>float16, bf16, float32</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>attention_out (tensor)</td>
      <td>Output</td>
      <td>attention_out in the formula.</td>
      <td>The data type and shape are consistent with those of query.</td>
      <td>float16, bf16, float32</td>
      <td>ND</td>
      <td>[BNSD], [BSND], [BSH], [SBH], [TND]</td>
    </tr>
  </tbody>
</table>

## Constraints

- Deterministic computation:
  - Ascendir_FlashAttentionScore uses a deterministic implementation by default.
- Constraints on the input query, key, and value:
  - B: The batchsize must be equal.
  - D: Head-Dim must satisfy (qD == kD && kD >= vD).
  - input_layout must be consistent.
- The input shape of query_rope differs from that of query only in the D dimension; other shape parameters must be the same.
- The input shape of key_rope differs from that of key only in the D dimension; other shape parameters must be the same.
- MLA concat scenario: only dSize=128 and dSizeRope=64 are supported.
- Constraints on the data shape are described as follows, taking TND, BSND, and BNSD of input_layout as an example (H=N\*D for BSH and SBH):
    - T (B*S): The value range is 1\~1M. In TND format, the maximum length supported by actual_seq_qlen is 20000.
    - B: The value range is 1\~2M. When prefix is used, the maximum supported value of B is 2K.
    - N: The value range is 1\~256.
    - S: The value range is 1\~1M.
    - D: The value range is 1\~768.
- Empty tensor scenarios:
    - If any of B, N1 (head_num), and S1 (sequence length of query) is 0, the operator does not perform any computation, directly returns success, and the workspaceSize is 0.
    - If any of N2 (head count of key/value), S2 (sequence length of key/value), and D (Head-Dim) is 0, query/key/value are empty tensors (shapeSize is 0), but the output shapes of softmax_max and softmax_sum may be non-empty (their shapes do not depend on the D dimension). The operator executes the empty-input processing flow and initializes the non-empty outputs.
    - In the preceding empty tensor scenarios, the operator returns normally without reporting an error.
- The data layout formats of query, key, and value can be interpreted from multiple dimensions: B (Batch) indicates the batch size of the input sample, S (Seq-Length) indicates the sequence length of the input sample, H (Head-Size) indicates the size of the hidden layer, N (Head-Num) indicates the number of heads, and D (Head-Dim) indicates the smallest unit size of the hidden layer, satisfying D=H/N.
- inner_precise: Currently, 0 and 1 are reserved configuration values, and 2 enables invalid-row computation, which avoids precision loss caused by a full row mask during computation. However, this configuration degrades performance. If the operator can determine that an invalid-row scenario exists, for example, sparse_mode is 3 and Sq > Skv, it automatically enables invalid-row computation.
- Meaning of each pse_type value

    | pse_type    | Description                              |      Remarks   |
    | ----------- | --------------------------------- | ----------|
    | 0           | When pse is passed in during operator invocation, mul is performed before add              | - |
    | 1           | When pse is passed in during operator invocation, add is performed before mul              | Consistent with the FlashAttentionScore implementation. |
    | 2           | The operator automatically generates pse, and mul is performed before add              | - |
    | 3           | The operator automatically generates pse, and mul is performed before add, followed by sqrt         | - |

- When pse_type is 2 or 3, only Sq equal to Skv is supported currently.
- When sink is not None, input tensors such as query and key support only float16 and bfloat16.
- The sparse_mode constraints are as follows:
  - When the shapes of all atten_mask tensors are smaller than 2048 and identical, the default mode is recommended to reduce memory usage.
  - When this parameter is set to 1, 2, 3, or 5, the user-configured pre_tockens and next_tockens do not take effect.
  - When this parameter is set to 0 or 4, ensure that atten_mask is consistent with the ranges specified by pre_tockens and next_tockens.
  - If you have no special requirement, pass 0.
  - For details about the different sparse modes, see [Sparse Mode Description](../../../docs/zh/context/sparse_mode_introduction_en.md).
- In some scenarios, an excessive amount of computation may cause operator execution to time out (with an aicore error, where errorStr is: timeout or trap error). In this case, axis splitting is recommended. Note: The amount of computation is affected by parameters such as B, S, N, and D; the larger the values, the greater the amount of computation.
- In the band scenario, pre_tockens and next_tockens must have an intersection.
- prefix sparse computation scenarios include sparse_mode=5 and sparse_mode=6 when sequence lengths are equal, and sparse_mode=6 when sequence lengths are not equal. In both scenarios, when Sq > Skv, the N value of prefix ranges from \[0, Skv\]; when Sq <= Skv, the N value of prefix ranges from \[Skv-Sq, Skv\]. When sparse_mode=5 and the N value of prefix is greater than Skv or prefix is not passed, full computation is performed. sparse_mode=6 requires prefix to be passed.
- real_shift: If Sq is greater than 1024 and BNHS or 1NHS is configured, Sq must be equal to Skv.
- The actual_seq_qlen input supports an S length of 0 for a batch. In this case, the optional input real_shift is not supported.
- In the TND scenario, the atten_mask input does not support padding; that is, atten_mask must not contain a row of all 1s.
- An S length of 0 for a batch in actual_seq_qlen is supported; if an S value of 0 exists, pse input is not supported.
  If the actual S lengths are \[2,2,0,2,2\], pass \[2,4,4,6,8\] as actual_seq_qlen.
- <term>Ascend 950PR&950DT products</term>:
    - seed and offset take effect only when keep_prob is less than 1.0; otherwise, they do not take effect.
    - When keep_prob is less than 1.0, the input drop_mask is used if drop_mask is not nullptr; otherwise, the drop_mask generated by seed and offset is used.
- In TND format, some trailing batches can be excluded from computation. In this case, pass the corresponding number of 0s at the end of actual_seq_qlen and actual_seq_kvlen. If the actual S lengths are [2, 3, 4, 5, 6] and you want the last two batches to be excluded from computation, pass [2, 3, 4, 0, 0] as actual_seq_qlen. In this case, if prefix needs to be passed, the same number of 0s must also be passed at its end, for example, [1, 1, 1, 0, 0].
- softmax_out_layout supports: an empty string and "same_as_input".
- real_shift: If Sq is greater than 1024, Sq is equal to Skv for each batch, and sparse_mode is 0, 2, or 3 (lower-triangular mask scenarios), you can enable alibi position encoding compression. In this case, you only need to input the last 1024 rows of the original PSE to optimize memory usage, that is, alibi_compress = ori_pse[:, :, -1024:, :]. Details are as follows:
  - If the parameter differs across batches, the shape is BNHSkv (H=1024).
  - If the parameter is the same across batches, the shape is 1NHSkv (H=1024).
  - If pse_type is 2 or 3, the data type must be FLOAT32, and the supported shapes are [B,N] or [N].
  - If this feature is not enabled, pass nullptr for real_shift and pass 1 for pse_type.

## Calling Example

The following sample code is for reference only. For the compilation and execution process in GE graph mode, see [Operator Invocation](../../../docs/zh/invocation/quick_op_invocation.md#ge图模式). For the complete code, see [test_geir_flash_attention_score](../examples/test_geir_flash_attention_score.cpp).

```c++
/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_geir_flash_attention_score.cpp
 * \brief GE graph construction sample for FlashAttentionScore (SBH layout, fp32).
 */

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "array_ops.h"
#include "ge_api.h"
#include "ge_api_types.h"
#include "ge_error_codes.h"
#include "graph.h"
#include "tensor.h"
#include "types.h"

#include "../op_graph/flash_attention_score_proto.h"

#define FAILED (-1)
#define SUCCESS 0

#define CHECK_RET(cond, return_expr) \
    do { \
        if (!(cond)) { \
            return_expr; \
        } \
    } while (0)

#define LOG_PRINT(message, ...) \
    do { \
        std::printf(message, ##__VA_ARGS__); \
    } while (0)

using namespace ge;
using std::map;
using std::string;
using std::vector;

namespace {
constexpr int64_t kBatch = 1;
constexpr int64_t kQHeadNum = 1;
constexpr int64_t kKvHeadNum = 1;
constexpr int64_t kQSeqLen = 256;
constexpr int64_t kKvSeqLen = 256;
constexpr int64_t kHeadDim = 128;
constexpr int64_t kH1 = kQHeadNum * kHeadDim;
constexpr int64_t kH2 = kKvHeadNum * kHeadDim;
constexpr double kScale = 0.08838834764831845; // 1.0 / sqrt(128)

std::string GetGeError()
{
    const AscendString errorMessage = GEGetErrorMsgV2();
    return errorMessage.GetString() == nullptr ? "" : errorMessage.GetString();
}

int64_t GetShapeSize(const vector<int64_t>& shape)
{
    int64_t size = 1;
    for (const int64_t dim : shape) {
        size *= dim;
    }
    return size;
}

// Deterministic pseudo-random value in [-0.5, 0.5), same value reused by the CPU reference.
float GenValue(uint64_t seed)
{
    seed ^= seed >> 33;
    seed *= 0xff51afd7ed558ccdULL;
    seed ^= seed >> 33;
    return (static_cast<float>(seed & 0xFFFFU) / 65536.0f) - 0.5f;
}

template <typename T>
int32_t AddDataInput(Graph& graph, vector<Tensor>& inputTensors, vector<Operator>& inputOps, const string& name,
                     int32_t index, const vector<int64_t>& shape, DataType dtype, const vector<T>& hostData)
{
    auto dataOp = op::Data(name).set_attr_index(index - 1);
    TensorDesc desc(Shape(shape), FORMAT_ND, dtype);
    desc.SetPlacement(kPlacementHost);
    desc.SetRealDimCnt(shape.size());
    const int64_t elementCount = GetShapeSize(shape);
    CHECK_RET(static_cast<int64_t>(hostData.size()) == elementCount,
              LOG_PRINT("[ERROR] %s data size mismatch\n", name.c_str());
              return FAILED);
    Tensor tensor;
    auto ret = tensor.SetTensorDesc(desc);
    CHECK_RET(ret == GRAPH_SUCCESS,
              LOG_PRINT("[ERROR] Tensor::SetTensorDesc failed, status=%u, error=%s\n", ret, GetGeError().c_str());
              return FAILED);
    ret = tensor.SetData(reinterpret_cast<const uint8_t*>(hostData.data()), hostData.size() * sizeof(T));
    CHECK_RET(ret == GRAPH_SUCCESS,
              LOG_PRINT("[ERROR] Tensor::SetData failed, status=%u, error=%s\n", ret, GetGeError().c_str());
              return FAILED);
    ret = dataOp.update_input_desc_x(desc);
    CHECK_RET(ret == GRAPH_SUCCESS,
              LOG_PRINT("[ERROR] Data::update_input_desc_x failed, status=%u, error=%s\n", ret, GetGeError().c_str());
              return FAILED);
    ret = dataOp.update_output_desc_y(desc);
    CHECK_RET(ret == GRAPH_SUCCESS,
              LOG_PRINT("[ERROR] Data::update_output_desc_y failed, status=%u, error=%s\n", ret, GetGeError().c_str());
              return FAILED);
    ret = graph.AddOp(dataOp);
    CHECK_RET(ret == GRAPH_SUCCESS,
              LOG_PRINT("[ERROR] Graph::AddOp failed, status=%u, error=%s\n", ret, GetGeError().c_str());
              return FAILED);
    inputTensors.push_back(tensor);
    inputOps.push_back(dataOp);
    return SUCCESS;
}

int32_t BuildGraph(Graph& graph, vector<Tensor>& inputTensors, vector<Operator>& inputOps, vector<Operator>& outputOps,
                   vector<float>& qData, vector<float>& kData, vector<float>& vData)
{
    auto node = op::FlashAttentionScore("flash_attention_score");
    const vector<int64_t> qShape = {kQSeqLen, kBatch, kH1};
    const vector<int64_t> kvShape = {kKvSeqLen, kBatch, kH2};
    const vector<int64_t> maskShape = {kQSeqLen, kKvSeqLen};
    const vector<int64_t> idxShape = {1};

    const int64_t qSize = GetShapeSize(qShape);
    const int64_t kvSize = GetShapeSize(kvShape);
    qData.resize(qSize);
    kData.resize(kvSize);
    vData.resize(kvSize);
    for (int64_t i = 0; i < qSize; ++i) {
        qData[i] = GenValue(static_cast<uint64_t>(i) * 7 + 1);
    }
    for (int64_t i = 0; i < kvSize; ++i) {
        kData[i] = GenValue(static_cast<uint64_t>(i) * 13 + 5);
        vData[i] = GenValue(static_cast<uint64_t>(i) * 17 + 9);
    }
    const vector<uint8_t> maskData(kQSeqLen * kKvSeqLen, 0); // 0: the position attends
    const vector<int64_t> prefixData = {0};
    const vector<int64_t> qStartIdxData = {0};
    const vector<int64_t> kvStartIdxData = {0};

    CHECK_RET(AddDataInput(graph, inputTensors, inputOps, "query", 1, qShape, DT_FLOAT, qData) == SUCCESS,
              return FAILED);
    CHECK_RET(AddDataInput(graph, inputTensors, inputOps, "key", 2, kvShape, DT_FLOAT, kData) == SUCCESS,
              return FAILED);
    CHECK_RET(AddDataInput(graph, inputTensors, inputOps, "value", 3, kvShape, DT_FLOAT, vData) == SUCCESS,
              return FAILED);
    CHECK_RET(AddDataInput(graph, inputTensors, inputOps, "atten_mask", 4, maskShape, DT_UINT8, maskData) == SUCCESS,
              return FAILED);
    CHECK_RET(AddDataInput(graph, inputTensors, inputOps, "prefix", 5, idxShape, DT_INT64, prefixData) == SUCCESS,
              return FAILED);
    CHECK_RET(
        AddDataInput(graph, inputTensors, inputOps, "q_start_idx", 6, idxShape, DT_INT64, qStartIdxData) == SUCCESS,
        return FAILED);
    CHECK_RET(
        AddDataInput(graph, inputTensors, inputOps, "kv_start_idx", 7, idxShape, DT_INT64, kvStartIdxData) == SUCCESS,
        return FAILED);

    node.set_input_query(inputOps[0]);
    node.set_input_key(inputOps[1]);
    node.set_input_value(inputOps[2]);
    node.set_input_atten_mask(inputOps[3]);
    node.set_input_prefix(inputOps[4]);
    node.set_input_q_start_idx(inputOps[5]);
    node.set_input_kv_start_idx(inputOps[6]);
    node.set_attr_scale_value(kScale);
    node.set_attr_keep_prob(1.0);
    node.set_attr_pre_tockens(2147483647);
    node.set_attr_next_tockens(2147483647);
    node.set_attr_head_num(kQHeadNum);
    node.set_attr_input_layout("SBH");
    node.set_attr_inner_precise(0);
    node.set_attr_sparse_mode(0);
    node.set_attr_pse_type(1);
    node.set_attr_out_dtype(0);

    const vector<int64_t> statShape = {kBatch, kQHeadNum, kQSeqLen, 8};
    TensorDesc softmaxMaxDesc(Shape(statShape), FORMAT_ND, DT_FLOAT);
    TensorDesc softmaxSumDesc(Shape(statShape), FORMAT_ND, DT_FLOAT);
    TensorDesc softmaxOutDesc(Shape(qShape), FORMAT_ND, DT_FLOAT);
    TensorDesc attentionOutDesc(Shape(qShape), FORMAT_ND, DT_FLOAT);
    CHECK_RET(node.update_output_desc_softmax_max(softmaxMaxDesc) == GRAPH_SUCCESS, return FAILED);
    CHECK_RET(node.update_output_desc_softmax_sum(softmaxSumDesc) == GRAPH_SUCCESS, return FAILED);
    CHECK_RET(node.update_output_desc_softmax_out(softmaxOutDesc) == GRAPH_SUCCESS, return FAILED);
    CHECK_RET(node.update_output_desc_attention_out(attentionOutDesc) == GRAPH_SUCCESS, return FAILED);
    CHECK_RET(graph.AddOp(node) == GRAPH_SUCCESS, return FAILED);
    outputOps.push_back(node);
    return SUCCESS;
}

// CPU reference: softmax(scale * Q @ K^T) @ V with SBH layout, N1 == N2 == 1, B == 1.
void ComputeReference(const vector<float>& qData, const vector<float>& kData, const vector<float>& vData,
                      vector<float>& softmaxMax, vector<float>& softmaxSum, vector<float>& attentionOut)
{
    const int64_t s1 = kQSeqLen;
    const int64_t s2 = kKvSeqLen;
    const int64_t d = kHeadDim;
    vector<float> score(s2);
    softmaxMax.assign(s1, 0.0f);
    softmaxSum.assign(s1, 0.0f);
    attentionOut.assign(s1 * d, 0.0f);
    for (int64_t i = 0; i < s1; ++i) {
        for (int64_t j = 0; j < s2; ++j) {
            float dot = 0.0f;
            for (int64_t t = 0; t < d; ++t) {
                dot += qData[i * d + t] * kData[j * d + t];
            }
            score[j] = static_cast<float>(kScale) * dot;
        }
        float maxScore = score[0];
        for (int64_t j = 1; j < s2; ++j) {
            maxScore = std::fmax(maxScore, score[j]);
        }
        float sum = 0.0f;
        for (int64_t j = 0; j < s2; ++j) {
            score[j] = std::exp(score[j] - maxScore);
            sum += score[j];
        }
        softmaxMax[i] = maxScore;
        softmaxSum[i] = sum;
        for (int64_t j = 0; j < s2; ++j) {
            const float prob = score[j] / sum;
            for (int64_t t = 0; t < d; ++t) {
                attentionOut[i * d + t] += prob * vData[j * d + t];
            }
        }
    }
}

bool CheckOutput(const Tensor& tensor, const string& name, const vector<int64_t>& shape, const vector<float>& expected,
                 float rtol, float atol)
{
    const auto desc = tensor.GetTensorDesc();
    size_t count = 1;
    for (const int64_t dim : shape) {
        count *= static_cast<size_t>(dim);
    }
    CHECK_RET(desc.GetShape().GetDims() == shape && desc.GetDataType() == DT_FLOAT &&
                  tensor.GetSize() == count * sizeof(float),
              LOG_PRINT("[CHECK] %s FAIL: unexpected shape, dtype, or data size\n", name.c_str());
              return false);
    vector<float> values(count);
    std::memcpy(values.data(), tensor.GetData(), count * sizeof(float));
    size_t mismatches = 0;
    float maxAbsErr = 0.0f;
    for (size_t i = 0; i < count; ++i) {
        const float tolerance = atol + rtol * std::fabs(expected[i]);
        const float err = std::fabs(values[i] - expected[i]);
        maxAbsErr = std::fmax(maxAbsErr, err);
        if (err > tolerance) {
            if (mismatches < 4) {
                LOG_PRINT("[CHECK] %s[%zu]=%.7f expected=%.7f tolerance=%.7f\n", name.c_str(), i, values[i],
                          expected[i], tolerance);
            }
            ++mismatches;
        }
    }
    LOG_PRINT("[CHECK] %s: count=%zu, maxAbsErr=%.7g, mismatches=%zu: %s\n", name.c_str(), count, maxAbsErr, mismatches,
              mismatches == 0 ? "PASS" : "FAIL");
    return mismatches == 0;
}

int32_t ValidateOutputs(const vector<Tensor>& outputs, const vector<float>& qData, const vector<float>& kData,
                        const vector<float>& vData)
{
    CHECK_RET(outputs.size() == 4, LOG_PRINT("[CHECK] FAIL: expected 4 outputs, got %zu\n", outputs.size());
              return FAILED);
    vector<float> softmaxMax;
    vector<float> softmaxSum;
    vector<float> attentionOut;
    ComputeReference(qData, kData, vData, softmaxMax, softmaxSum, attentionOut);
    // softmax_max/softmax_sum are stored as [B, N, S, 8]; all 8 lanes hold the same per-row statistic.
    const int64_t s1 = kQSeqLen;
    vector<float> maxExpanded;
    vector<float> sumExpanded;
    maxExpanded.reserve(softmaxMax.size() * 8);
    sumExpanded.reserve(softmaxSum.size() * 8);
    for (size_t i = 0; i < softmaxMax.size(); ++i) {
        for (int64_t lane = 0; lane < 8; ++lane) {
            maxExpanded.push_back(softmaxMax[i]);
            sumExpanded.push_back(softmaxSum[i]);
        }
    }
    bool ok = true;
    ok = CheckOutput(outputs[0], "softmax_max", {kBatch, kQHeadNum, kQSeqLen, 8}, maxExpanded, 1e-3f, 1e-3f) && ok;
    ok = CheckOutput(outputs[1], "softmax_sum", {kBatch, kQHeadNum, kQSeqLen, 8}, sumExpanded, 1e-3f, 1e-3f) && ok;
    ok = CheckOutput(outputs[3], "attention_out", {kQSeqLen, kBatch, kH1}, attentionOut, 1e-3f, 1e-3f) && ok;
    // softmax_out is a reserved output that the kernel does not write; its content is not validated.
    LOG_PRINT("[CHECK] total: %s\n", ok ? "PASS" : "FAIL");
    return ok ? SUCCESS : FAILED;
}
} // namespace

int main()
{
    std::map<AscendString, AscendString> globalOptions = {{"ge.exec.deviceId", "0"}, {"ge.graphRunMode", "1"}};
    auto status = GEInitialize(globalOptions);
    CHECK_RET(status == GRAPH_SUCCESS,
              LOG_PRINT("[ERROR] GEInitialize failed, status=%u, error=%s\n", status, GetGeError().c_str());
              return FAILED);

    Graph graph("flash_attention_score_graph");
    vector<Tensor> inputTensors;
    vector<Operator> inputOps;
    vector<Operator> outputOps;
    vector<float> qData;
    vector<float> kData;
    vector<float> vData;
    int32_t ret = BuildGraph(graph, inputTensors, inputOps, outputOps, qData, kData, vData);
    if (ret == SUCCESS) {
        graph.SetInputs(inputOps).SetOutputs(outputOps);
        std::map<AscendString, AscendString> buildOptions;
        Session* session = new (std::nothrow) Session(buildOptions);
        CHECK_RET(session != nullptr, LOG_PRINT("[ERROR] Session allocation failed.\n"); return FAILED);
        constexpr uint32_t graphId = 0;
        std::map<AscendString, AscendString> graphOptions;
        auto addRet = session->AddGraph(graphId, graph, graphOptions);
        CHECK_RET(addRet == GRAPH_SUCCESS,
                  LOG_PRINT("[ERROR] Session::AddGraph failed, status=%u, error=%s\n", addRet, GetGeError().c_str());
                  delete session; return FAILED);
        vector<Tensor> outputTensors;
        auto runRet = session->RunGraph(graphId, inputTensors, outputTensors);
        CHECK_RET(runRet == GRAPH_SUCCESS,
                  LOG_PRINT("[ERROR] Session::RunGraph failed, status=%u, error=%s\n", runRet, GetGeError().c_str());
                  delete session; return FAILED);
        LOG_PRINT("FlashAttentionScore graph run success, output count: %zu\n", outputTensors.size());
        ret = ValidateOutputs(outputTensors, qData, kData, vData);
        delete session;
    }

    status = GEFinalize();
    CHECK_RET(status == GRAPH_SUCCESS,
              LOG_PRINT("[ERROR] GEFinalize failed, status=%u, error=%s\n", status, GetGeError().c_str());
              return FAILED);
    return ret;
}
```
