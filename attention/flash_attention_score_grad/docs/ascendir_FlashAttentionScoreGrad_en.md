# Ascendir_FlashAttentionScoreGrad

## Product Support

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id2 -->
<!-- npu="A3" id3 -->
- <term>Atlas A3 inference products</term>: Not supported
<!-- end id3 -->
<!-- npu="910b" id4 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id4 -->
<!-- npu="910b" id5 -->
- <term>Atlas A2 inference products</term>: Not supported
<!-- end id5 -->
<!-- npu="310b" id6 -->
- <term>Atlas 200I/500 A2 inference products</term>: Not supported
<!-- end id6 -->
<!-- npu="310p" id7 -->
- <term>Atlas inference products</term>: Not supported
<!-- end id7 -->
<!-- npu="910" id8 -->
- <term>Atlas training products</term>: Not supported
<!-- end id8 -->

## Description

- Function: Computes the backward output of attention in training scenarios, that is, the backward computation of FlashAttentionScore:

  - When pseType=1, add first and then multiply.
  - When pseType is not 1, multiply first and then add.

- Computation formula:

  The forward computation formula of attention is as follows:

    - When pseType=1, the formula is as follows:
      $$
      Y=Dropout(Softmax(Mask(\frac{QK^T+pse}{\sqrt{d}}),atten\_mask),keep\_prob)V
      $$

    - When pseType is not 1, the formula is as follows:
      $$
      Y=Dropout(Softmax(Mask(\frac{QK^T}{\sqrt{d}}+pse),atten\_mask),keep\_prob)V
      $$

  For ease of expression, variables $S$ and $P$ are used to represent the computation formula:

  $$
  S=Mask(\frac{QK^T}{\sqrt{d}}+pse),atten\_mask
  $$

  $$
  P=Dropout(Softmax(S),keep\_prob)
  $$

  $$
  Y=PV
  $$

  The backward computation formula of attention is as follows:

  $$
  dV=P^TdY
  $$

  $$
  dQ=\frac{((dS)*K)}{\sqrt{d}}
  $$

  $$
  dK=\frac{((dS)^T*Q)}{\sqrt{d}}
  $$

  After sink is added, the computation logic is as follows. The main change is in the softmax_max and softmax_sum computation logic:

  $$
  m = max(sink, max(S))
  $$

  $$
  Attention = \frac{e^{S - m} @ V}{\sum e^{S-m} + e^{sink - m}}
  $$

  $$
  dSink = reduce(-Softmax(S) * dP * SimpleSoftmax(sink, x\_max, x\_sum))
  $$

## Ascend IR Definition

The header file that contains the Ascend IR definition is located at [op_graph/flash_attention_score_grad_proto.h](../op_graph/flash_attention_score_grad_proto.h).

```cpp
REG_OP(FlashAttentionScoreGrad)
    .INPUT(query, TensorType({DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .INPUT(key, TensorType({DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .INPUT(value, TensorType({DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .INPUT(dy, TensorType({DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .OPTIONAL_INPUT(pse_shift, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .OPTIONAL_INPUT(drop_mask, TensorType({DT_UINT8}))
    .OPTIONAL_INPUT(padding_mask, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .OPTIONAL_INPUT(atten_mask, TensorType({DT_BOOL, DT_UINT8}))
    .OPTIONAL_INPUT(softmax_max, TensorType({DT_FLOAT32}))
    .OPTIONAL_INPUT(softmax_sum, TensorType({DT_FLOAT32}))
    .OPTIONAL_INPUT(softmax_in, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .OPTIONAL_INPUT(attention_in, TensorType({DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .OPTIONAL_INPUT(prefix, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(actual_seq_qlen, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(actual_seq_kvlen, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(q_start_idx, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(kv_start_idx, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(d_scale_q, TensorType({DT_FLOAT32}))
    .OPTIONAL_INPUT(d_scale_k, TensorType({DT_FLOAT32}))
    .OPTIONAL_INPUT(d_scale_v, TensorType({DT_FLOAT32}))
    .OPTIONAL_INPUT(d_scale_dy, TensorType({DT_FLOAT32}))
    .OPTIONAL_INPUT(d_scale_o, TensorType({DT_FLOAT32}))
    .OPTIONAL_INPUT(query_rope, TensorType({DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .OPTIONAL_INPUT(key_rope, TensorType({DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .OPTIONAL_INPUT(sink, TensorType({DT_FLOAT32}))
    .OPTIONAL_INPUT(ds_scale, TensorType({DT_FLOAT32}))
    .OPTIONAL_INPUT(p_scale, TensorType({DT_FLOAT32}))
    .OUTPUT(dq, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .OUTPUT(dk, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .OUTPUT(dv, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .OUTPUT(dpse, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .OUTPUT(dq_rope, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .OUTPUT(dk_rope, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .OUTPUT(dsink, TensorType({DT_FLOAT32}))
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
    .ATTR(softmax_in_layout, String, "")
    .OP_END_FACTORY_REG(FlashAttentionScoreGrad)
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
      <th>Parameter Name</th>
      <th>Input/Output/Attribute</th>
      <th>Description</th>
      <th>Usage Instructions</th>
      <th>Data Type</th>
      <th>Data Format</th>
      <th>Dimensions (shape)</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>query (tensor)</td>
      <td>Input</td>
      <td>Input query in the formula.</td>
      <td>The data type is the same as that of key/value.</td>
      <td>float8_e5m2, float8_e4m3fn, float16, bf16, float32</td>
      <td>ND</td>
      <td>[BNSD], [BSND], [BSH], [SBH], [TND]</td>
    </tr>
    <tr>
      <td>key (tensor)</td>
      <td>Input</td>
      <td>Input key in the formula.</td>
      <td>The data type is the same as that of query/value.</td>
      <td>float8_e5m2, float8_e4m3fn, float16, bf16, float32</td>
      <td>ND</td>
      <td>[BNSD], [BSND], [BSH], [SBH], [TND]</td>
    </tr>
    <tr>
      <td>value (tensor)</td>
      <td>Input</td>
      <td>Input value in the formula.</td>
      <td>The data type is the same as that of query/key.</td>
      <td>float8_e5m2, float8_e4m3fn, float16, bf16, float32</td>
      <td>ND</td>
      <td>[BNSD], [BSND], [BSH], [SBH], [TND]</td>
    </tr>
    <tr>
      <td>dy (tensor)</td>
      <td>Input</td>
      <td>dY in the formula, which indicates the gradient of the output.</td>
      <td>-</td>
      <td>float8_e5m2, float8_e4m3fn, float16, bf16, float32</td>
      <td>ND</td>
      <td>[BNSD], [BSND], [BSH], [SBH], [TND]</td>
    </tr>
    <tr>
      <td>pse_shift (tensor)</td>
      <td>Optional input</td>
      <td>pse in the formula, which indicates the position encoding.</td>
      <td>The data type is the same as that of query/key/value. This parameter must be used together with pse_type.</td>
      <td>float16, bf16, float32</td>
      <td>ND</td>
      <td>[B,N,Sq,Skv], [B,N,1,Skv], [1,N,Sq,Skv], [B,N,1024,Skv], [1,N,1024,Skv]</td>
    </tr>
    <tr>
      <td>drop_mask (tensor)</td>
      <td>Optional input</td>
      <td>Dropout in the formula, which indicates the dropout mask. A value of 1 indicates that the data is retained, and a value of 0 indicates that the data is dropped.</td>
      <td>[B,N,S,S] is supported.</td>
      <td>uint8</td>
      <td>ND</td>
      <td>0, 1</td>
    </tr>
    <tr>
      <td>padding_mask (tensor)</td>
      <td>Optional input</td>
      <td>Reserved parameter, not used currently.</td>
      <td>-</td>
      <td>float16, bf16, float32</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>atten_mask (tensor)</td>
      <td>Optional input</td>
      <td>atten_mask in the formula, which indicates the attention mask.</td>
      <td>A value of 1 indicates that the position does not participate in computation (does not take effect), and a value of 0 indicates that the position participates in computation.</td>
      <td>bool, uint8</td>
      <td>ND</td>
      <td>[B,N,Sq,Skv], [B,1,Sq,Skv], [1,1,Sq,Skv], [Sq,Skv]</td>
    </tr>
    <tr>
      <td>softmax_max (tensor)</td>
      <td>Optional input</td>
      <td>Max intermediate result of the forward attention computation.</td>
      <td>-</td>
      <td>float32</td>
      <td>ND</td>
      <td>[B,N,Sq,8], [N,T,8], [T,N,8]</td>
    </tr>
    <tr>
      <td>softmax_sum (tensor)</td>
      <td>Optional input</td>
      <td>Sum intermediate result of the forward attention computation.</td>
      <td>-</td>
      <td>float32</td>
      <td>ND</td>
      <td>[B,N,Sq,8], [N,T,8], [T,N,8]</td>
    </tr>
    <tr>
      <td>softmax_in (tensor)</td>
      <td>Optional input</td>
      <td>Intermediate output of the forward attention computation. Reserved parameter, not used currently.</td>
      <td>-</td>
      <td>float16, bf16, float32</td>
      <td>ND</td>
      <td>[B,N,Sq,8]</td>
    </tr>
    <tr>
      <td>attention_in (tensor)</td>
      <td>Optional input</td>
      <td>Final output of the forward attention.</td>
      <td>The data type and shape are the same as those of dy.</td>
      <td>float8_e5m2, float8_e4m3fn, float16, bf16, float32</td>
      <td>ND</td>
      <td>[BNSD], [BSND], [BSH], [SBH], [TND]</td>
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
      <td>Query sequence length of each batch.</td>
      <td>When this parameter is used, input_layout must be set to TND. For example, if the seqlen of q is [2,2,2,2,2], set this parameter to [2,4,6,8,10].</td>
      <td>int64</td>
      <td>ND</td>
      <td>0, 1</td>
    </tr>
    <tr>
      <td>actual_seq_kvlen (tensor)</td>
      <td>Optional input</td>
      <td>kv sequence length of each batch.</td>
      <td>When this parameter is used, input_layout must be set to TND. For example, if the seqlen of kv is [2,2,2,2,2], set this parameter to [2,4,6,8,10].</td>
      <td>int64</td>
      <td>ND</td>
      <td>0, 1</td>
    </tr>
    <tr>
      <td>q_start_idx (tensor)</td>
      <td>Optional input</td>
      <td>In the outer splitting scenario, the start index of the sequence of the current Q chunk in the global sequence.</td>
      <td>-</td>
      <td>int64</td>
      <td>ND</td>
      <td>0, 1</td>
    </tr>
    <tr>
      <td>kv_start_idx (tensor)</td>
      <td>Optional input</td>
      <td>In the outer splitting scenario, the start index of the sequence of the current KV chunk in the global sequence.</td>
      <td>-</td>
      <td>int64</td>
      <td>ND</td>
      <td>0, 1</td>
    </tr>
    <tr>
      <td>d_scale_q (tensor)</td>
      <td>Optional input</td>
      <td>Dequantization parameter of the query input.</td>
      <td>Not used currently.</td>
      <td>float32</td>
      <td>ND</td>
      <td>0, 1</td>
    </tr>
    <tr>
      <td>d_scale_k (tensor)</td>
      <td>Optional input</td>
      <td>Dequantization parameter of the key input.</td>
      <td>Not used currently.</td>
      <td>float32</td>
      <td>ND</td>
      <td>0, 1</td>
    </tr>
    <tr>
      <td>d_scale_v (tensor)</td>
      <td>Optional input</td>
      <td>Dequantization parameter of the value input.</td>
      <td>Not used currently.</td>
      <td>float32</td>
      <td>ND</td>
      <td>0, 1</td>
    </tr>
    <tr>
      <td>d_scale_dy (tensor)</td>
      <td>Optional input</td>
      <td>Dequantization parameter of the dy input.</td>
      <td>Not used currently.</td>
      <td>float32</td>
      <td>ND</td>
      <td>0, 1</td>
    </tr>
    <tr>
      <td>d_scale_o (tensor)</td>
      <td>Optional input</td>
      <td>Dequantization parameter of the attention_in input.</td>
      <td>Not used currently.</td>
      <td>float32</td>
      <td>ND</td>
      <td>0, 1</td>
    </tr>
    <tr>
      <td>query_rope (tensor)</td>
      <td>Optional input</td>
      <td>RoPE rotary position encoding input of Q.</td>
      <td>The data type is the same as that of query.</td>
      <td>float8_e5m2, float8_e4m3fn, float16, bf16, float32</td>
      <td>ND</td>
      <td>[BNSD], [BSND], [BSH], [SBH], [TND]</td>
    </tr>
    <tr>
      <td>key_rope (tensor)</td>
      <td>Optional input</td>
      <td>RoPE rotary position encoding input of K.</td>
      <td>The data type is the same as that of key.</td>
      <td>float8_e5m2, float8_e4m3fn, float16, bf16, float32</td>
      <td>ND</td>
      <td>[BNSD], [BSND], [BSH], [SBH], [TND]</td>
    </tr>
    <tr>
      <td>sink (tensor)</td>
      <td>Optional input</td>
      <td>Sinking parameter.</td>
      <td>It has 1 dimension, and its length is the same as head_num of query.</td>
      <td>float32</td>
      <td>ND</td>
      <td>[headNum]</td>
    </tr>
    <tr>
      <td>ds_scale (tensor)</td>
      <td>Optional input</td>
      <td>Reserved parameter, not used currently.</td>
      <td>-</td>
      <td>float32</td>
      <td>ND</td>
      <td>-</td>
    </tr>
    <tr>
      <td>p_scale (tensor)</td>
      <td>Optional input</td>
      <td>Reserved parameter, not used currently.</td>
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
          <li>scale in the formula, which indicates the scaling coefficient. It is used as the scalar value of Muls in the computation flow.</li>
          <li>The default value is 1.0.</li>
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
          <li>keep_prob in the formula, which indicates the probability that data is retained.</li>
          <li>The default value is 1.0.</li>
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
          <li>Used for sparse computation. It indicates the left boundary of the sliding window.</li>
          <li>The default value is 2147483647.</li>
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
          <li>Used for sparse computation. It indicates the right boundary of the sliding window.</li>
          <li>The default value is 2147483647.</li>
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
      <td>Number of heads on a single device, that is, the length of the N dimension of input query.</td>
      <td>The value must be the same as the N value of the passed query.</td>
      <td>int</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>input_layout (string)</td>
      <td>Required attribute</td>
      <td>
        <ul>
          <li>Data layout format of input query, key, and value.</li>
          <li>The default value is empty.</li>
        </ul>
      </td>
      <td>BSH, SBH, BNSD, BSND, and TND are supported.</td>
      <td>string</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>inner_precise (int)</td>
      <td>Optional attribute</td>
      <td>
        <ul>
          <li>Reserved parameter, not used currently.</li>
          <li>The default value is 0.</li>
        </ul>
      </td>
      <td>-</td>
      <td>int</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>sparse_mode (int)</td>
      <td>Optional attribute</td>
      <td>
        <ul>
          <li>Indicates the sparse mode. 0 indicates the defaultMask mode, 1 indicates allMask, 2 indicates leftUpCausal, 3 indicates rightDownCausal, 4 indicates band, 5 indicates prefix, 6 indicates prefixCompress, 7 indicates rightDownCausalBand, and 8 indicates bandLeftUpCausal.</li>
          <li>The default value is 0.</li>
        </ul>
      </td>
      <td>Values 0 to 8 are supported.</td>
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
          <li>The default value is 1.</li>
        </ul>
      </td>
      <td>Values 0, 1, 2, and 3 are supported.</td>
      <td>int</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>seed (int)</td>
      <td>Optional attribute</td>
      <td>
        <ul>
          <li>Random seed of Dropout. When keep_prob is less than 1.0 and no drop_mask is passed in externally, the DropoutMask is generated based on seed and offset.</li>
          <li>The default value is 0.</li>
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
          <li>Offset of Dropout.</li>
          <li>The default value is 0.</li>
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
          <li>Output precision control. A value of 0 indicates that outputs such as dq are FLOAT16, and a value of 1 indicates that they are BFLOAT16.</li>
          <li>The default value is 0.</li>
        </ul>
      </td>
      <td>-</td>
      <td>int</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>softmax_in_layout (string)</td>
      <td>Optional attribute</td>
      <td>
        <ul>
          <li>Controls the actual data layout of softmax_max and softmax_sum.</li>
          <li>The default value is empty.</li>
        </ul>
      </td>
      <td>Pass "same_as_input" when the shape is TND but the actual data layout is NTD. Pass "" when the actual data layout is TND.</td>
      <td>string</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dq (tensor)</td>
      <td>Output</td>
      <td>dQ in the formula, which indicates the gradient of query.</td>
      <td>The data type is the same as that of query.</td>
      <td>float16, bf16, float32</td>
      <td>ND</td>
      <td>[BNSD], [BSND], [BSH], [SBH], [TND]</td>
    </tr>
    <tr>
      <td>dk (tensor)</td>
      <td>Output</td>
      <td>dK in the formula, which indicates the gradient of key.</td>
      <td>The data type is the same as that of key.</td>
      <td>float16, bf16, float32</td>
      <td>ND</td>
      <td>[BNSD], [BSND], [BSH], [SBH], [TND]</td>
    </tr>
    <tr>
      <td>dv (tensor)</td>
      <td>Output</td>
      <td>dV in the formula, which indicates the gradient of value.</td>
      <td>The data type is the same as that of value.</td>
      <td>float16, bf16, float32</td>
      <td>ND</td>
      <td>[BNSD], [BSND], [BSH], [SBH], [TND]</td>
    </tr>
    <tr>
      <td>dpse (tensor)</td>
      <td>Output</td>
      <td>Indicates the gradient of pse.</td>
      <td>Not used currently. The passed shape must be the same as pse_shape.</td>
      <td>float16, bf16, float32</td>
      <td>ND</td>
      <td>0, 4</td>
    </tr>
    <tr>
      <td>dq_rope (tensor)</td>
      <td>Output</td>
      <td>Indicates the gradient of query_rope.</td>
      <td>The data type is the same as that of dq.</td>
      <td>float16, bf16, float32</td>
      <td>ND</td>
      <td>[BNSD], [BSND], [BSH], [SBH], [TND]</td>
    </tr>
    <tr>
      <td>dk_rope (tensor)</td>
      <td>Output</td>
      <td>Indicates the gradient of key_rope.</td>
      <td>The data type is the same as that of dk.</td>
      <td>float16, bf16, float32</td>
      <td>ND</td>
      <td>[BNSD], [BSND], [BSH], [SBH], [TND]</td>
    </tr>
    <tr>
      <td>dsink (tensor)</td>
      <td>Output</td>
      <td>dSink in the formula, which indicates the gradient of sink.</td>
      <td>-</td>
      <td>float32</td>
      <td>ND</td>
      <td>1</td>
    </tr>
  </tbody>
</table>

## Constraints

- Deterministic computation:
  - By default, Ascendir_FlashAttentionScoreGrad provides a non-deterministic implementation. You can enable determinism through aclrtCtxSetSysParamOpt.
- Constraints on input query, key, value, and dy:
  - B: The batch size must be the same.
  - input_layout must be the same.
  - D: Head-Dim must meet the following requirements: the D of query is equal to the D of key, the D of value is equal to the D of dy, and the D of query and key is greater than or equal to the D of value and dy.
- The N of input query/dy can be different from the N of key/value, but they must be proportional. That is, Nq/Nkv must be a non-zero integer, and Nq ranges from 1 to 256.
- The input shape of query_rope differs from that of query only in the D dimension, and other shape parameters must be the same. The input shape of key_rope differs from that of key only in the D dimension, and other shape parameters must be the same.
- For constraints on the data shape, take TND of input_layout as an example:
    - B: The value ranges from 1\~2K. When prefix is used, B supports 1K at most.
    - N: The value ranges from 1\~256.
    - S: The value ranges from 1\~1M.
    - D: The value ranges from 1\~768.
    - KeepProb: The value range is (0, 1].
- The data layout formats of query, key, and value can be interpreted in multiple dimensions, where B (Batch) indicates the batch size of input samples, S (Seq-Length) indicates the sequence length of input samples, H (Head-Size) indicates the size of the hidden layer, N (Head-Num) indicates the number of heads, D (Head-Dim) indicates the smallest unit size of the hidden layer, and D=H/N.
- Constraints on the softmax_max and softmax_sum parameters: The input format is fixed to [B, N, S, 8], except for the TND input format, where the format is [N, T, 8]. Note: T=B*S.
- When the data layouts of softmax_max and softmax_sum are TND, set softmax_in_layout to "same_as_input".
- The value of head_num must be the same as the N value of the passed query.
- prefix sparse computation supports only the compression scenario, that is, sparse_mode=6. When Sq > Skv, the N value of prefix ranges from \[0, Skv\]. When Sq <= Skv, the N value of prefix ranges from \[Skv-Sq, Skv\]. Full computation is performed when sparse_mode=5, the N value of prefix is greater than Skv, or prefix is not passed. sparse_mode=6 requires prefix to be passed.
- When sparse_mode=7, the optional input pse_shift is not supported.
- When sparse_mode=8, the optional input pse_shift is supported when q and kv of each sequence have the same length, and pse is generated globally. Outer splitting in the q direction is supported. Before outer splitting, q and kv of each sequence must have the same length. After outer splitting, the passed actual_seq_qlen[0] - actual_seq_kvlen[0] + q_start_idx - kv_start_idx must be 0. (This function is experimental.)
- The input actual_seq_qlen supports an S length of 0 for a batch. In this case, the optional input pse_shift is not supported.
- In TND format, some batches at the tail can be excluded from computation. In this case, pass the corresponding number of 0s at the end of actual_seq_qlen and actual_seq_kvlen. Assume that the actual S lengths are [2, 3, 4, 5, 6] and you want the last two batches to be excluded from computation. In this case, the passed actual_seq_qlen is [2, 3, 4, 0, 0]. If prefix also needs to be passed, pass the same number of 0s at its tail, for example, [1, 1, 1, 0, 0].
- When sink is not None, input tensors such as query and key support only float16 and bfloat16.
- When pse_type is 2 or 3, only the case where Sq and Skv have the same length is supported currently.
- The constraints on sparse_mode are as follows:
  - When all atten_mask shapes are less than 2048 and are the same, it is recommended to use the default mode to reduce memory usage.
  - When sparse_mode is set to 1, 2, 3, or 5, the user-configured pre_tockens and next_tockens do not take effect.
  - When sparse_mode is set to 0 or 4, ensure that atten_mask is consistent with the ranges specified by pre_tockens and next_tockens.
  - If you do not specify a value, it is recommended to pass 0.
  - For details about different sparse modes, see [Sparse Mode Description](../../../docs/zh/context/sparse_mode_introduction_en.md).
- In some scenarios, an excessive amount of computation may cause the operator to time out (an aicore error is reported, with errorStr set to timeout or trap error). In this case, axis splitting is recommended. Note: The amount of computation is affected by parameters such as B, S, N, and D. The larger the values, the greater the amount of computation.
- <term>Ascend 950PR&950DT products</term>:
    - When query_rope and key_rope are not empty, the per-head D of query and key must be 128, the per-head D of value and dy must be 128, and the per-head D of query_rope and key_rope must be 64.
    - seed and offset take effect only when keep_prob is less than 1.0. Otherwise, they do not take effect.
    - When keep_prob is less than 1.0, the input drop_mask is used if drop_mask is not nullptr; otherwise, the drop_mask generated by seed and offset is used.

## Calling Example

The following is the example code for reference only. For details about the compilation and execution process in GE graph mode, see [Operator Invocation](../../../docs/zh/invocation/quick_op_invocation.md#ge图模式). For the complete code, see [test_geir_flash_attention_score_grad](../examples/test_geir_flash_attention_score_grad.cpp).

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
 * \file test_geir_flash_attention_score_grad.cpp
 * \brief GE graph construction sample for FlashAttentionScoreGrad (SBH layout, fp16).
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

#include "../op_graph/flash_attention_score_grad_proto.h"

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
constexpr float kRtol = 3e-3f;
constexpr float kAtol = 3e-3f;

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

// Deterministic value in [-0.45, 0.45]; the magnitude stays in the normal fp16 range.
float GenValue(uint64_t seed)
{
    seed ^= seed >> 33;
    seed *= 0xff51afd7ed558ccdULL;
    seed ^= seed >> 33;
    return ((static_cast<float>(seed & 0xFFU) / 255.0f) - 0.5f) * 0.9f;
}

uint16_t FloatToHalf(float value)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint16_t sign = static_cast<uint16_t>((bits >> 16) & 0x8000U);
    const int32_t exponent = static_cast<int32_t>((bits >> 23) & 0xFFU) - 127 + 15;
    const uint32_t mantissa = bits & 0x7FFFFFU;
    if (exponent <= 0) {
        return sign; // flush tiny magnitudes to zero
    }
    if (exponent >= 31) {
        return static_cast<uint16_t>(sign | 0x7C00U); // clamp to infinity
    }
    return static_cast<uint16_t>(sign | (static_cast<uint16_t>(exponent) << 10) | (mantissa >> 13));
}

float HalfToFloat(uint16_t value)
{
    const uint32_t sign = static_cast<uint32_t>(value & 0x8000U) << 16;
    const uint32_t exponent = (value & 0x7C00U) >> 10;
    const uint32_t mantissa = value & 0x03FFU;
    if (exponent == 0) {
        return std::ldexp(static_cast<float>(mantissa), -24) * ((sign != 0) ? -1.0f : 1.0f);
    }
    const uint32_t bits = sign | ((exponent - 15 + 127) << 23) | (mantissa << 13);
    float result = 0.0f;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

template <typename T>
int32_t AddInput(Graph& graph, vector<Tensor>& inputTensors, vector<Operator>& inputOps, const string& name,
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
    CHECK_RET(ret == GRAPH_SUCCESS, LOG_PRINT("[ERROR] %s SetTensorDesc failed\n", name.c_str()); return FAILED);
    ret = tensor.SetData(reinterpret_cast<const uint8_t*>(hostData.data()), hostData.size() * sizeof(T));
    CHECK_RET(ret == GRAPH_SUCCESS, LOG_PRINT("[ERROR] %s SetData failed\n", name.c_str()); return FAILED);
    ret = dataOp.update_input_desc_x(desc);
    CHECK_RET(ret == GRAPH_SUCCESS, LOG_PRINT("[ERROR] %s update_input_desc_x failed\n", name.c_str()); return FAILED);
    ret = dataOp.update_output_desc_y(desc);
    CHECK_RET(ret == GRAPH_SUCCESS, LOG_PRINT("[ERROR] %s update_output_desc_y failed\n", name.c_str()); return FAILED);
    ret = graph.AddOp(dataOp);
    CHECK_RET(ret == GRAPH_SUCCESS, LOG_PRINT("[ERROR] Graph::AddOp failed for %s\n", name.c_str()); return FAILED);
    inputTensors.push_back(tensor);
    inputOps.push_back(dataOp);
    return SUCCESS;
}

struct ForwardStats {
    vector<float> softmaxMax;  // [S1]
    vector<float> softmaxSum;  // [S1]
    vector<float> attentionIn; // [S1, D] fp32 exact reference of the forward output
};

void ComputeForward(const vector<float>& q, const vector<float>& k, const vector<float>& v, ForwardStats& stats)
{
    const int64_t s1 = kQSeqLen;
    const int64_t s2 = kKvSeqLen;
    const int64_t d = kHeadDim;
    vector<float> score(s2);
    stats.softmaxMax.assign(s1, 0.0f);
    stats.softmaxSum.assign(s1, 0.0f);
    stats.attentionIn.assign(s1 * d, 0.0f);
    for (int64_t i = 0; i < s1; ++i) {
        for (int64_t j = 0; j < s2; ++j) {
            float dot = 0.0f;
            for (int64_t t = 0; t < d; ++t) {
                dot += q[i * d + t] * k[j * d + t];
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
        stats.softmaxMax[i] = maxScore;
        stats.softmaxSum[i] = sum;
        for (int64_t t = 0; t < d; ++t) {
            float y = 0.0f;
            for (int64_t j = 0; j < s2; ++j) {
                y += (score[j] / sum) * v[j * d + t];
            }
            stats.attentionIn[i * d + t] = y;
        }
    }
}

int32_t BuildGraph(Graph& graph, vector<Tensor>& inputTensors, vector<Operator>& inputOps, vector<Operator>& outputOps,
                   vector<float>& q, vector<float>& k, vector<float>& v, vector<float>& dy, vector<float>& statsMax,
                   vector<float>& statsSum, vector<float>& attentionIn)
{
    const int64_t qSize = kQSeqLen * kH1;
    const int64_t kvSize = kKvSeqLen * kH2;
    q.resize(qSize);
    k.resize(kvSize);
    v.resize(kvSize);
    dy.resize(qSize);
    for (int64_t i = 0; i < qSize; ++i) {
        q[i] = GenValue(static_cast<uint64_t>(i) * 7 + 1);
        dy[i] = GenValue(static_cast<uint64_t>(i) * 23 + 3);
    }
    for (int64_t i = 0; i < kvSize; ++i) {
        k[i] = GenValue(static_cast<uint64_t>(i) * 13 + 5);
        v[i] = GenValue(static_cast<uint64_t>(i) * 17 + 9);
    }
    ForwardStats stats;
    ComputeForward(q, k, v, stats);
    statsMax = stats.softmaxMax;
    statsSum = stats.softmaxSum;
    attentionIn = stats.attentionIn;

    const vector<int64_t> qShape = {kQSeqLen, kBatch, kH1};
    const vector<int64_t> kvShape = {kKvSeqLen, kBatch, kH2};
    const vector<int64_t> maskShape = {kQSeqLen, kKvSeqLen};
    const vector<int64_t> statShape = {kBatch, kQHeadNum, kQSeqLen, 8};

    vector<uint16_t> qHalf(q.size());
    vector<uint16_t> kHalf(k.size());
    vector<uint16_t> vHalf(v.size());
    vector<uint16_t> dyHalf(dy.size());
    vector<uint16_t> outHalf(attentionIn.size());
    for (size_t i = 0; i < q.size(); ++i) {
        qHalf[i] = FloatToHalf(q[i]);
        dyHalf[i] = FloatToHalf(dy[i]);
    }
    for (size_t i = 0; i < k.size(); ++i) {
        kHalf[i] = FloatToHalf(k[i]);
        vHalf[i] = FloatToHalf(v[i]);
    }
    for (size_t i = 0; i < attentionIn.size(); ++i) {
        outHalf[i] = FloatToHalf(attentionIn[i]);
    }
    const vector<uint8_t> maskData(kQSeqLen * kKvSeqLen, 0);
    // softmax statistics are [B, N, S, 8]; every lane holds the same per-row value.
    vector<float> maxLanes;
    vector<float> sumLanes;
    maxLanes.reserve(statsMax.size() * 8);
    sumLanes.reserve(statsSum.size() * 8);
    for (size_t i = 0; i < statsMax.size(); ++i) {
        for (int64_t lane = 0; lane < 8; ++lane) {
            maxLanes.push_back(statsMax[i]);
            sumLanes.push_back(statsSum[i]);
        }
    }

    // Input order follows the IR definition: query, key, value, dy, atten_mask, softmax_max, softmax_sum,
    // attention_in, prefix, q_start_idx, kv_start_idx.
    CHECK_RET(AddInput(graph, inputTensors, inputOps, "query", 1, qShape, DT_FLOAT16, qHalf) == SUCCESS, return FAILED);
    CHECK_RET(AddInput(graph, inputTensors, inputOps, "key", 2, kvShape, DT_FLOAT16, kHalf) == SUCCESS, return FAILED);
    CHECK_RET(AddInput(graph, inputTensors, inputOps, "value", 3, kvShape, DT_FLOAT16, vHalf) == SUCCESS,
              return FAILED);
    CHECK_RET(AddInput(graph, inputTensors, inputOps, "dy", 4, qShape, DT_FLOAT16, dyHalf) == SUCCESS, return FAILED);
    CHECK_RET(AddInput(graph, inputTensors, inputOps, "atten_mask", 5, maskShape, DT_UINT8, maskData) == SUCCESS,
              return FAILED);
    CHECK_RET(AddInput(graph, inputTensors, inputOps, "softmax_max", 6, statShape, DT_FLOAT, maxLanes) == SUCCESS,
              return FAILED);
    CHECK_RET(AddInput(graph, inputTensors, inputOps, "softmax_sum", 7, statShape, DT_FLOAT, sumLanes) == SUCCESS,
              return FAILED);
    CHECK_RET(AddInput(graph, inputTensors, inputOps, "attention_in", 8, qShape, DT_FLOAT16, outHalf) == SUCCESS,
              return FAILED);
    CHECK_RET(AddInput(graph, inputTensors, inputOps, "prefix", 9, {1}, DT_INT64, vector<int64_t>{0}) == SUCCESS,
              return FAILED);
    CHECK_RET(AddInput(graph, inputTensors, inputOps, "q_start_idx", 10, {1}, DT_INT64, vector<int64_t>{0}) == SUCCESS,
              return FAILED);
    CHECK_RET(AddInput(graph, inputTensors, inputOps, "kv_start_idx", 11, {1}, DT_INT64, vector<int64_t>{0}) == SUCCESS,
              return FAILED);

    auto node = op::FlashAttentionScoreGrad("flash_attention_score_grad");
    node.set_input_query(inputOps[0]);
    node.set_input_key(inputOps[1]);
    node.set_input_value(inputOps[2]);
    node.set_input_dy(inputOps[3]);
    node.set_input_atten_mask(inputOps[4]);
    node.set_input_softmax_max(inputOps[5]);
    node.set_input_softmax_sum(inputOps[6]);
    node.set_input_attention_in(inputOps[7]);
    node.set_input_prefix(inputOps[8]);
    node.set_input_q_start_idx(inputOps[9]);
    node.set_input_kv_start_idx(inputOps[10]);

    node.set_attr_scale_value(kScale);
    node.set_attr_keep_prob(1.0);
    node.set_attr_pre_tockens(65536);
    node.set_attr_next_tockens(65536);
    node.set_attr_head_num(kQHeadNum);
    node.set_attr_input_layout("SBH");
    node.set_attr_inner_precise(0);
    node.set_attr_sparse_mode(0);
    node.set_attr_pse_type(1);
    node.set_attr_out_dtype(1);
    node.set_attr_seed(0);
    node.set_attr_offset(0);

    TensorDesc dqDesc(Shape(qShape), FORMAT_ND, DT_FLOAT16);
    TensorDesc dkDesc(Shape(kvShape), FORMAT_ND, DT_FLOAT16);
    TensorDesc dvDesc(Shape(kvShape), FORMAT_ND, DT_FLOAT16);
    TensorDesc emptyFp16Desc(Shape({0}), FORMAT_ND, DT_FLOAT16);
    TensorDesc emptyFp32Desc(Shape({0}), FORMAT_ND, DT_FLOAT);
    CHECK_RET(node.update_output_desc_dq(dqDesc) == GRAPH_SUCCESS, return FAILED);
    CHECK_RET(node.update_output_desc_dk(dkDesc) == GRAPH_SUCCESS, return FAILED);
    CHECK_RET(node.update_output_desc_dv(dvDesc) == GRAPH_SUCCESS, return FAILED);
    CHECK_RET(node.update_output_desc_dpse(emptyFp16Desc) == GRAPH_SUCCESS, return FAILED);
    CHECK_RET(node.update_output_desc_dq_rope(emptyFp16Desc) == GRAPH_SUCCESS, return FAILED);
    CHECK_RET(node.update_output_desc_dk_rope(emptyFp16Desc) == GRAPH_SUCCESS, return FAILED);
    CHECK_RET(node.update_output_desc_dsink(emptyFp32Desc) == GRAPH_SUCCESS, return FAILED);
    CHECK_RET(graph.AddOp(node) == GRAPH_SUCCESS, return FAILED);
    outputOps.push_back(node);
    return SUCCESS;
}

void ComputeBackward(const vector<float>& q, const vector<float>& k, const vector<float>& v, const vector<float>& dy,
                     const ForwardStats& stats, vector<float>& dq, vector<float>& dk, vector<float>& dv)
{
    const int64_t s1 = kQSeqLen;
    const int64_t s2 = kKvSeqLen;
    const int64_t d = kHeadDim;
    dq.assign(s1 * d, 0.0f);
    dk.assign(s2 * d, 0.0f);
    dv.assign(s2 * d, 0.0f);
    for (int64_t i = 0; i < s1; ++i) {
        vector<float> prob(s2);
        for (int64_t j = 0; j < s2; ++j) {
            float dot = 0.0f;
            for (int64_t t = 0; t < d; ++t) {
                dot += q[i * d + t] * k[j * d + t];
            }
            const float score = static_cast<float>(kScale) * dot;
            prob[j] = std::exp(score - stats.softmaxMax[i]) / stats.softmaxSum[i];
        }
        for (int64_t j = 0; j < s2; ++j) {
            float dP = 0.0f;
            for (int64_t t = 0; t < d; ++t) {
                dP += dy[i * d + t] * v[j * d + t];
                dv[j * d + t] += prob[j] * dy[i * d + t];
            }
            float dYdotY = 0.0f;
            for (int64_t t = 0; t < d; ++t) {
                dYdotY += dy[i * d + t] * stats.attentionIn[i * d + t];
            }
            const float dS = prob[j] * (dP - dYdotY);
            for (int64_t t = 0; t < d; ++t) {
                dq[i * d + t] += static_cast<float>(kScale) * dS * k[j * d + t];
                dk[j * d + t] += static_cast<float>(kScale) * dS * q[i * d + t];
            }
        }
    }
}

bool CheckOutput(const Tensor& tensor, const string& name, const vector<int64_t>& shape, const vector<float>& expected)
{
    const auto desc = tensor.GetTensorDesc();
    size_t count = 1;
    for (const int64_t dim : shape) {
        count *= static_cast<size_t>(dim);
    }
    CHECK_RET(desc.GetShape().GetDims() == shape && desc.GetDataType() == DT_FLOAT16 &&
                  tensor.GetSize() == count * sizeof(uint16_t),
              LOG_PRINT("[CHECK] %s FAIL: unexpected shape, dtype, or data size\n", name.c_str());
              return false);
    vector<uint16_t> halfValues(count);
    std::memcpy(halfValues.data(), tensor.GetData(), count * sizeof(uint16_t));
    size_t mismatches = 0;
    float maxAbsErr = 0.0f;
    for (size_t i = 0; i < count; ++i) {
        const float value = HalfToFloat(halfValues[i]);
        const float tolerance = kAtol + kRtol * std::fabs(expected[i]);
        const float err = std::fabs(value - expected[i]);
        maxAbsErr = std::fmax(maxAbsErr, err);
        if (err > tolerance) {
            if (mismatches < 4) {
                LOG_PRINT("[CHECK] %s[%zu]=%.7f expected=%.7f tolerance=%.7f\n", name.c_str(), i, value, expected[i],
                          tolerance);
            }
            ++mismatches;
        }
    }
    LOG_PRINT("[CHECK] %s: count=%zu, maxAbsErr=%.7g, mismatches=%zu: %s\n", name.c_str(), count, maxAbsErr, mismatches,
              mismatches == 0 ? "PASS" : "FAIL");
    return mismatches == 0;
}

int32_t ValidateOutputs(const vector<Tensor>& outputs, const vector<float>& q, const vector<float>& k,
                        const vector<float>& v, const vector<float>& dy, const ForwardStats& stats)
{
    CHECK_RET(outputs.size() == 7, LOG_PRINT("[CHECK] FAIL: expected 7 outputs, got %zu\n", outputs.size());
              return FAILED);
    vector<float> dq;
    vector<float> dk;
    vector<float> dv;
    ComputeBackward(q, k, v, dy, stats, dq, dk, dv);
    const vector<int64_t> qShape = {kQSeqLen, kBatch, kH1};
    const vector<int64_t> kvShape = {kKvSeqLen, kBatch, kH2};
    bool ok = true;
    ok = CheckOutput(outputs[0], "dq", qShape, dq) && ok;
    ok = CheckOutput(outputs[1], "dk", kvShape, dk) && ok;
    ok = CheckOutput(outputs[2], "dv", kvShape, dv) && ok;
    // dpse/dq_rope/dk_rope/dsink are optional outputs and stay empty because the matching inputs are not bound;
    // their content is not validated.
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

    Graph graph("flash_attention_score_grad_graph");
    vector<Tensor> inputTensors;
    vector<Operator> inputOps;
    vector<Operator> outputOps;
    vector<float> q;
    vector<float> k;
    vector<float> v;
    vector<float> dy;
    vector<float> statsMax;
    vector<float> statsSum;
    vector<float> attentionIn;
    int32_t ret = BuildGraph(graph, inputTensors, inputOps, outputOps, q, k, v, dy, statsMax, statsSum, attentionIn);
    if (ret == SUCCESS) {
        ForwardStats stats;
        stats.softmaxMax = statsMax;
        stats.softmaxSum = statsSum;
        stats.attentionIn = attentionIn;
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
        LOG_PRINT("FlashAttentionScoreGrad graph run success, output count: %zu\n", outputTensors.size());
        ret = ValidateOutputs(outputTensors, q, k, v, dy, stats);
        delete session;
    }

    status = GEFinalize();
    CHECK_RET(status == GRAPH_SUCCESS,
              LOG_PRINT("[ERROR] GEFinalize failed, status=%u, error=%s\n", status, GetGeError().c_str());
              return FAILED);
    return ret;
}
```
