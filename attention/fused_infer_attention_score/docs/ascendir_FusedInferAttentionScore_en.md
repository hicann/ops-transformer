# Ascendir_FusedInferAttentionScore

## Product Support

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 products</term>: supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 products</term>: supported
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2 inference products</term>: not supported
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas inference products</term>: not supported
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas training products</term>: not supported
<!-- end id6 -->

## Function Description

- Operator function: This is a FlashAttention operator adapted to the incremental (decode) and full (prefill) inference scenarios. It supports both the full computation scenario ([PromptFlashAttention](../../prompt_flash_attention/README_en.md)) and the incremental computation scenario ([IncreFlashAttention](../../incre_flash_attention/README_en.md)). It supports features including KV Cache, PageAttention, GQA/MLA, and pseudo-quantization/full quantization/post-quantization.

- Computation formula:

  self-attention builds an attention model by using the relationships within the input sample itself. The principle is as follows: assume an input sample sequence $x$ of length $n$, where each element of $x$ is a $d$-dimensional vector. Each $d$-dimensional vector can be viewed as a token embedding, and the sequence is transformed by three weight matrices to obtain three matrices of dimension $n*d$.

  $$
  Attention(Q,K,V)=Score(Q,K)V
  $$

  In this operator, the Score function uses the Softmax function. The self-attention computation formula is:

  $$
  Attention(Q,K,V)=Softmax(\frac{QK^T}{\sqrt{d}})V
  $$

  where the product of $Q$ and $K^T$ represents the attention of the input $x$. To prevent this value from becoming too large, it is usually scaled by dividing by the square root of $d$, and softmax normalization is performed on each row. The result is multiplied by $V$ to obtain a matrix of $n*d$.

## Ascend IR Definition

The header file containing the Ascend IR definition is located at [op_graph/fused_infer_attention_score_proto.h](../op_graph/fused_infer_attention_score_proto.h).

```cpp
REG_OP(FusedInferAttentionScore)
    .INPUT(query, TensorType({DT_INT8, DT_FLOAT16, DT_BF16, DT_HIFLOAT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN}))
    .DYNAMIC_INPUT(key, TensorType({DT_INT8, DT_FLOAT16, DT_BF16, DT_HIFLOAT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN,
                                    DT_FLOAT4_E2M1, DT_FLOAT4_E1M2, DT_INT4}))
    .DYNAMIC_INPUT(value, TensorType({DT_INT8, DT_FLOAT16, DT_BF16, DT_HIFLOAT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN,
                                      DT_FLOAT4_E2M1, DT_FLOAT4_E1M2, DT_INT4}))
    .OPTIONAL_INPUT(pse_shift, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(atten_mask, TensorType({DT_FLOAT16, DT_BOOL, DT_UINT8, DT_INT8}))
    .OPTIONAL_INPUT(actual_seq_lengths, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(actual_seq_lengths_kv, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(dequant_scale1, TensorType({DT_UINT64, DT_FLOAT}))
    .OPTIONAL_INPUT(quant_scale1, TensorType({DT_FLOAT32}))
    .OPTIONAL_INPUT(dequant_scale2, TensorType({DT_UINT64, DT_FLOAT}))
    .OPTIONAL_INPUT(quant_scale2, TensorType({DT_FLOAT32, DT_BF16}))
    .OPTIONAL_INPUT(quant_offset2, TensorType({DT_FLOAT32, DT_BF16}))
    .OPTIONAL_INPUT(antiquant_scale, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .OPTIONAL_INPUT(antiquant_offset, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .OPTIONAL_INPUT(block_table, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(query_padding_size, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(kv_padding_size, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(key_antiquant_scale, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .OPTIONAL_INPUT(key_antiquant_offset, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .OPTIONAL_INPUT(value_antiquant_scale, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .OPTIONAL_INPUT(value_antiquant_offset, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .OPTIONAL_INPUT(key_shared_prefix, TensorType({DT_INT8, DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(value_shared_prefix, TensorType({DT_INT8, DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(actual_shared_prefix_len, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(query_rope, TensorType({DT_INT8, DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(key_rope, TensorType({DT_INT8, DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(key_rope_antiquant_scale, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(dequant_scale_query, TensorType({DT_FLOAT32}))
    .OPTIONAL_INPUT(learnable_sink, TensorType({DT_BF16}))
    .OPTIONAL_INPUT(q_start_idx, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(kv_start_idx, TensorType({DT_INT64}))
    .OUTPUT(attention_out, TensorType({DT_FLOAT16, DT_INT8, DT_BF16}))
    .OUTPUT(softmax_lse, TensorType({DT_FLOAT32}))
    .REQUIRED_ATTR(num_heads, Int)
    .ATTR(scale, Float, 1.0)
    .ATTR(pre_tokens, Int, 2147483647)
    .ATTR(next_tokens, Int, 2147483647)
    .ATTR(input_layout, String, "BSH")
    .ATTR(num_key_value_heads, Int, 0)
    .ATTR(sparse_mode, Int, 0)
    .ATTR(inner_precise, Int, 1)
    .ATTR(block_size, Int, 0)
    .ATTR(antiquant_mode, Int, 0)
    .ATTR(softmax_lse_flag, Bool, false)
    .ATTR(key_antiquant_mode, Int, 0)
    .ATTR(value_antiquant_mode, Int, 0)
    .ATTR(query_quant_mode, Int, 0)
    .ATTR(pse_type, Int, 0)
    .ATTR(out_dtype, Int, 0)
    .OP_END_FACTORY_REG(FusedInferAttentionScore)
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
      <td>The input query in the formula.</td>
      <td>Empty tensors are not supported.</td>
      <td>int8, float16, bf16, hifloat8, float8_e5m2, float8_e4m3fn</td>
      <td>ND</td>
      <td>See input_layout</td>
    </tr>
    <tr>
      <td>key (tensor)</td>
      <td>Dynamic input</td>
      <td>The input key in the formula. tensorlist is supported.</td>
      <td>In the non-contiguous scenario, the batch in the tensorlist can only be 1, the number of tensors must equal the B of query, and N and D must be equal.</td>
      <td>int8, float16, bf16, hifloat8, float8_e5m2, float8_e4m3fn, float4_e2m1, float4_e1m2, int4</td>
      <td>ND</td>
      <td>See input_layout</td>
    </tr>
    <tr>
      <td>value (tensor)</td>
      <td>Dynamic input</td>
      <td>The input value in the formula. tensorlist is supported.</td>
      <td>The shape must be exactly the same as the shape of key.</td>
      <td>int8, float16, bf16, hifloat8, float8_e5m2, float8_e4m3fn, float4_e2m1, float4_e1m2, int4</td>
      <td>ND</td>
      <td>See input_layout</td>
    </tr>
    <tr>
      <td>pse_shift (tensor)</td>
      <td>Optional input</td>
      <td>The pse in the formula, indicating position encoding.</td>
      <td>Pass in nullptr when this function is not used. This parameter is not supported in the MLA scenario, the scenario where D values are not equal, or the GQA full quantization scenario.</td>
      <td>float16, bf16</td>
      <td>ND</td>
      <td>[B,Q_N,Q_S,KV_S], [1,Q_N,Q_S,KV_S]</td>
    </tr>
    <tr>
      <td>atten_mask (tensor)</td>
      <td>Optional input</td>
      <td>The atten_mask in the formula, indicating the attention mask. A value of 1 indicates that the bit does not participate in the calculation (takes no effect), and a value of 0 indicates that the bit participates in the calculation.</td>
      <td>Pass in nullptr when this function is not used.</td>
      <td>float16, bool, uint8, int8</td>
      <td>ND</td>
      <td>[Q_S,KV_S], [B,Q_S,KV_S], [B,N,Q_S,KV_S]</td>
    </tr>
    <tr>
      <td>actual_seq_lengths (tensor)</td>
      <td>Optional input</td>
      <td>The valid sequence length of query in each batch.</td>
      <td>Pass in nullptr when the sequence length is not specified, indicating that the length is the same as the S of the query shape.</td>
      <td>int64</td>
      <td>ND</td>
      <td>1, B</td>
    </tr>
    <tr>
      <td>actual_seq_lengths_kv (tensor)</td>
      <td>Optional input</td>
      <td>The valid sequence length of key/value in each batch.</td>
      <td>Pass in nullptr when the sequence length is not specified, indicating that the length is the same as the S of the key/value shape.</td>
      <td>int64</td>
      <td>ND</td>
      <td>1, B</td>
    </tr>
    <tr>
      <td>dequant_scale1 (tensor)</td>
      <td>Optional input</td>
      <td>The dequantization factor after BMM1.</td>
      <td>per-tensor is supported. Pass in nullptr when this function is not used.</td>
      <td>uint64, float32</td>
      <td>ND</td>
      <td>-</td>
    </tr>
    <tr>
      <td>quant_scale1 (tensor)</td>
      <td>Optional input</td>
      <td>The quantization factor before BMM2.</td>
      <td>per-tensor and MxFP8 are supported. Pass in nullptr when this function is not used.</td>
      <td>float32</td>
      <td>ND</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dequant_scale2 (tensor)</td>
      <td>Optional input</td>
      <td>The dequantization factor after BMM2.</td>
      <td>per-tensor is supported. Pass in nullptr when this function is not used.</td>
      <td>uint64, float32</td>
      <td>ND</td>
      <td>-</td>
    </tr>
    <tr>
      <td>quant_scale2 (tensor)</td>
      <td>Optional input</td>
      <td>The quantization factor of the output.</td>
      <td>per-tensor and per-channel are supported. Pass in nullptr when this function is not used.</td>
      <td>float32, bf16</td>
      <td>ND</td>
      <td>-</td>
    </tr>
    <tr>
      <td>quant_offset2 (tensor)</td>
      <td>Optional input</td>
      <td>The quantization offset of the output.</td>
      <td>The shape must be the same as that of quant_scale2. Pass in nullptr when this function is not used.</td>
      <td>float32, bf16</td>
      <td>ND</td>
      <td>-</td>
    </tr>
    <tr>
      <td>antiquant_scale (tensor)</td>
      <td>Optional input</td>
      <td>The pseudo-quantization factor.</td>
      <td>This parameter is reserved and is not used currently.</td>
      <td>float16, bf16, float32</td>
      <td>ND</td>
      <td>-</td>
    </tr>
    <tr>
      <td>antiquant_offset (tensor)</td>
      <td>Optional input</td>
      <td>The pseudo-quantization offset.</td>
      <td>This parameter is reserved and is not used currently.</td>
      <td>float16, bf16, float32</td>
      <td>ND</td>
      <td>-</td>
    </tr>
    <tr>
      <td>block_table (tensor)</td>
      <td>Optional input</td>
      <td>The block mapping table used for KV storage in PageAttention.</td>
      <td>The length of the first dimension must equal B, and the length of the second dimension must not be less than maxBlockNumPerSeq.</td>
      <td>int32</td>
      <td>ND</td>
      <td>[B, KV_S_max/block_size]</td>
    </tr>
    <tr>
      <td>query_padding_size (tensor)</td>
      <td>Optional input</td>
      <td>Indicates whether the data of each batch in query is right-aligned and the number of right-aligned elements.</td>
      <td>Takes effect only in the scenario where Q_S is greater than 1. Pass in nullptr when this function is not used.</td>
      <td>int64</td>
      <td>ND</td>
      <td>1</td>
    </tr>
    <tr>
      <td>kv_padding_size (tensor)</td>
      <td>Optional input</td>
      <td>Indicates whether the data of each batch in key/value is right-aligned and the number of right-aligned elements.</td>
      <td>Pass in nullptr when this function is not used.</td>
      <td>int64</td>
      <td>ND</td>
      <td>1</td>
    </tr>
    <tr>
      <td>key_antiquant_scale (tensor)</td>
      <td>Optional input</td>
      <td>The dequantization factor of key in the scenario where KV pseudo-quantization parameters are separated.</td>
      <td>Pass in nullptr when this function is not used. Supported modes include per-tensor, per-channel, per-token, per-token-group, per-tensor combined with per-head, and per-token combined with per-head.</td>
      <td>float16, bf16, float32, float8_e8m0</td>
      <td>ND</td>
      <td>-</td>
    </tr>
    <tr>
      <td>key_antiquant_offset (tensor)</td>
      <td>Optional input</td>
      <td>The dequantization offset of key in the scenario where KV pseudo-quantization parameters are separated.</td>
      <td>When this parameter is used, the shape must be the same as that of key_antiquant_scale. Pass in nullptr when this function is not used.</td>
      <td>float16, bf16, float32</td>
      <td>ND</td>
      <td>-</td>
    </tr>
    <tr>
      <td>value_antiquant_scale (tensor)</td>
      <td>Optional input</td>
      <td>The dequantization factor of value in the scenario where KV pseudo-quantization parameters are separated.</td>
      <td>Pass in nullptr when this function is not used. The mode number is the same as that of key_antiquant_scale.</td>
      <td>float16, bf16, float32, float8_e8m0</td>
      <td>ND</td>
      <td>-</td>
    </tr>
    <tr>
      <td>value_antiquant_offset (tensor)</td>
      <td>Optional input</td>
      <td>The dequantization offset of value in the scenario where KV pseudo-quantization parameters are separated.</td>
      <td>When this parameter is used, the shape must be the same as that of value_antiquant_scale. Pass in nullptr when this function is not used.</td>
      <td>float16, bf16, float32</td>
      <td>ND</td>
      <td>-</td>
    </tr>
    <tr>
      <td>key_shared_prefix (tensor)</td>
      <td>Optional input</td>
      <td>The input of the system prefix part of key in the attention structure.</td>
      <td>Pass in nullptr when this function is not used.</td>
      <td>int8, float16, bf16, int4</td>
      <td>ND</td>
      <td>[1,prefix_S,H], [1,prefix_S,KV_N,KV_D], [1,KV_N,prefix_S,KV_D]</td>
    </tr>
    <tr>
      <td>value_shared_prefix (tensor)</td>
      <td>Optional input</td>
      <td>The input of the system prefix part of value in the attention structure.</td>
      <td>The shape must be the same as that of key_shared_prefix. Pass in nullptr when this function is not used.</td>
      <td>int8, float16, bf16, int4</td>
      <td>ND</td>
      <td>[1,prefix_S,H], [1,prefix_S,KV_N,KV_D], [1,KV_N,prefix_S,KV_D]</td>
    </tr>
    <tr>
      <td>actual_shared_prefix_len (tensor)</td>
      <td>Optional input</td>
      <td>The valid sequence length of key_shared_prefix/value_shared_prefix.</td>
      <td>Pass in nullptr when this function is not used, indicating that the length is the same as the S of the system prefix.</td>
      <td>int64</td>
      <td>ND</td>
      <td>1</td>
    </tr>
    <tr>
      <td>query_rope (tensor)</td>
      <td>Optional input</td>
      <td>The rope information of query in the MLA structure.</td>
      <td>D in the shape is 64, and the other dimensions are the same as those of query.</td>
      <td>int8, float16, bf16</td>
      <td>ND</td>
      <td>-</td>
    </tr>
    <tr>
      <td>key_rope (tensor)</td>
      <td>Optional input</td>
      <td>The rope information of key in the MLA structure.</td>
      <td>D in the shape is 64, and the other dimensions are the same as those of key.</td>
      <td>int8, float16, bf16</td>
      <td>ND</td>
      <td>-</td>
    </tr>
    <tr>
      <td>key_rope_antiquant_scale (tensor)</td>
      <td>Optional input</td>
      <td>The dequantization factor of the rope information of key in the MLA structure.</td>
      <td>This parameter is reserved and does not take effect in the current version. Pass in nullptr.</td>
      <td>float16, bf16</td>
      <td>ND</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dequant_scale_query (tensor)</td>
      <td>Optional input</td>
      <td>The factor used to dequantize query.</td>
      <td>This parameter is involved in the full quantization scenario. Supported modes are per-token-group and per-token combined with per-head. Pass in nullptr when this function is not used.</td>
      <td>float32</td>
      <td>ND</td>
      <td>-</td>
    </tr>
    <tr>
      <td>learnable_sink (tensor)</td>
      <td>Optional input</td>
      <td>Indicates that the learnable "Sink Token" is used to absorb the Attention Score.</td>
      <td>Only the non-quantization scenario is supported. Pass in nullptr when this function is not used.</td>
      <td>bf16</td>
      <td>ND</td>
      <td>[Q_N]</td>
    </tr>
    <tr>
      <td>q_start_idx (tensor)</td>
      <td>Optional input</td>
      <td>In the external splitting scenario, the global start index of the sequence of the current query chunk.</td>
      <td>You can pass in nullptr.</td>
      <td>int64</td>
      <td>ND</td>
      <td>-</td>
    </tr>
    <tr>
      <td>kv_start_idx (tensor)</td>
      <td>Optional input</td>
      <td>In the external splitting scenario, the global start index of the sequence of the current key and value chunks.</td>
      <td>You can pass in nullptr.</td>
      <td>int64</td>
      <td>ND</td>
      <td>-</td>
    </tr>
    <tr>
      <td>num_heads (int)</td>
      <td>Required attribute</td>
      <td>The number of heads of query.</td>
      <td>In scenarios such as BNSD, BSND, TND, and NTD, this value must be the same as the N-axis shape value of query in the shape.</td>
      <td>int</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scale (float)</td>
      <td>Optional attribute</td>
      <td>
        <ul>
          <li>The reciprocal of the square root of d in the formula, indicating the scaling coefficient.</li>
          <li>The default value is 1.0.</li>
        </ul>
      </td>
      <td>If you do not specifically set this parameter, pass in 1.0.</td>
      <td>float</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>pre_tokens (int)</td>
      <td>Optional attribute</td>
      <td>
        <ul>
          <li>Used for sparse calculation, indicating that attention needs to be associated with the previous Tokens in the calculation.</li>
          <li>The default value is 2147483647.</li>
        </ul>
      </td>
      <td>If you do not specifically set this parameter, pass in 2147483647.</td>
      <td>int</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>next_tokens (int)</td>
      <td>Optional attribute</td>
      <td>
        <ul>
          <li>Used for sparse calculation, indicating that attention needs to be associated with the following Tokens in the calculation.</li>
          <li>The default value is 2147483647.</li>
        </ul>
      </td>
      <td>If you do not specifically set this parameter, pass in 2147483647.</td>
      <td>int</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>input_layout (string)</td>
      <td>Optional attribute</td>
      <td>
        <ul>
          <li>Represents the data layout format of the input query, key, and value.</li>
          <li>The default value is "BSH".</li>
        </ul>
      </td>
      <td>Supports BSH, BSND, BNSD, BNSD_BSND, BSND_BNSD, BSH_BNSD, BNSD_NBSD, BSND_NBSD, BSH_NBSD, TND, NTD, NTD_TND, and TND_NTD. If the layout format contains an underscore, the part on the left of the underscore indicates the layout of the input query, and the part on the right indicates the format of the output attention_out.</td>
      <td>string</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>num_key_value_heads (int)</td>
      <td>Optional attribute</td>
      <td>
        <ul>
          <li>The number of heads in key and value.</li>
          <li>The default value is 0.</li>
        </ul>
      </td>
      <td>The value 0 indicates that key/value and query have the same number of heads. num_heads must be divisible by num_key_value_heads.</td>
      <td>int</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>sparse_mode (int)</td>
      <td>Optional attribute</td>
      <td>
        <ul>
          <li>Indicates the sparse mode. 0 indicates the defaultMask mode, 1 indicates allMask, 2 indicates leftUpCausal, 3 indicates rightDownCausal, and 4 indicates band.</li>
          <li>The default value is 0.</li>
        </ul>
      </td>
      <td>Supported configuration values are 0~4. Values 5~8 are not supported currently.</td>
      <td>int</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>inner_precise (int)</td>
      <td>Optional attribute</td>
      <td>
        <ul>
          <li>Indicates the high-precision or high-performance option. 0 indicates the high-precision mode without row-invalid correction, 1 indicates the high-performance mode without row-invalid correction, 2 indicates the high-precision mode with row-invalid correction, and 3 indicates the high-performance mode with row-invalid correction.</li>
          <li>The default value is 1.</li>
        </ul>
      </td>
      <td>If the "mask part that participates in the calculation" contains a row that is entirely 1 during the calculation, it is recommended that you set this parameter to 2 or 3 to enable row-invalid correction and improve precision. This configuration degrades performance.</td>
      <td>int</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>block_size (int)</td>
      <td>Optional attribute</td>
      <td>
        <ul>
          <li>The maximum number of tokens in each block of the KV storage in PageAttention.</li>
          <li>The default value is 0.</li>
        </ul>
      </td>
      <td>If this parameter is not passed in, it is processed as 0.</td>
      <td>int</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>antiquant_mode (int)</td>
      <td>Optional attribute</td>
      <td>
        <ul>
          <li>The pseudo-quantization mode.</li>
          <li>The default value is 0.</li>
        </ul>
      </td>
      <td>This parameter is reserved and is not used currently.</td>
      <td>int</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>softmax_lse_flag (bool)</td>
      <td>Optional attribute</td>
      <td>
        <ul>
          <li>Whether to output softmax_lse.</li>
          <li>The default value is false.</li>
        </ul>
      </td>
      <td>Supports external splitting along the S axis (with an additional output).</td>
      <td>bool</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>key_antiquant_mode (int)</td>
      <td>Optional attribute</td>
      <td>
        <ul>
          <li>The dequantization mode of key. 0 indicates per-channel (including per-tensor), 1 indicates per-token, 2 indicates per-tensor combined with per-head, 3 indicates per-token combined with per-head, 4 indicates per-token combined with managing scale/offset in page attention mode, 5 indicates per-token combined with per-head and managing scale/offset in page attention mode, and 6 indicates per-token-group.</li>
          <li>The default value is 0.</li>
        </ul>
      </td>
      <td>This parameter must be the same as value_antiquant_mode, except in the scenario where key_antiquant_mode is 0 and value_antiquant_mode is 1.</td>
      <td>int</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>value_antiquant_mode (int)</td>
      <td>Optional attribute</td>
      <td>
        <ul>
          <li>The dequantization mode of value. Mode numbers 0~6 are the same as those of key_antiquant_mode, and 8 indicates per-channel-group.</li>
          <li>The default value is 0.</li>
        </ul>
      </td>
      <td>-</td>
      <td>int</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>query_quant_mode (int)</td>
      <td>Optional attribute</td>
      <td>
        <ul>
          <li>The dequantization mode of query. The mode number is the same as that of key_antiquant_mode.</li>
          <li>The default value is 0.</li>
        </ul>
      </td>
      <td>Only mode 3 (per-token combined with per-head) is supported.</td>
      <td>int</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>pse_type (int)</td>
      <td>Optional attribute</td>
      <td>
        <ul>
          <li>The pse mode. 0 indicates that pse is passed in externally and that mul is performed before add.</li>
          <li>The default value is 0.</li>
        </ul>
      </td>
      <td>Only the value 0 is supported (pseType=1 is not supported in the inference scenario).</td>
      <td>int</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out_dtype (int)</td>
      <td>Optional attribute</td>
      <td>
        <ul>
          <li>The data type of attention_out. 5 indicates fp16, 15 indicates bf16, 23 indicates fp8_e5m2, 24 indicates fp8_e4m3, and 290 indicates hifp8.</li>
          <li>The default value is 0.</li>
        </ul>
      </td>
      <td>Supported only in the PTA graph mode.</td>
      <td>int</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>attention_out (tensor)</td>
      <td>Output</td>
      <td>The attention_out in the formula.</td>
      <td>The D dimension is the same as the D of value, and the other dimensions must be the same as the shape of query.</td>
      <td>float16, int8, bf16</td>
      <td>ND</td>
      <td>See input_layout</td>
    </tr>
    <tr>
      <td>softmax_lse (tensor)</td>
      <td>Output</td>
      <td>The ring attention algorithm takes the max of the result of query multiplied by key to obtain softmax_max, then subtracts softmax_max, computes exp, and sums the values to obtain softmax_sum, and finally takes the log of softmax_sum and adds softmax_max to obtain the result.</td>
      <td>Takes effect when softmax_lse_flag is True. A value of inf indicates invalid data.</td>
      <td>float32</td>
      <td>ND</td>
      <td>[B,N,Q_S,1], or [T,N,1] in TND-related scenarios</td>
    </tr>
  </tbody>
</table>

## Constraints

- When this interface is used together with PyTorch, ensure that the versions of CANN-related packages match those of PyTorch-related packages.
- Handling of empty input parameters: The operator internally checks whether the parameter query is empty. If query is empty, the operator returns directly. If query is not an empty tensor while key and value are empty tensors (that is, KV_S is 0), attention_out is filled with all zeros. When attention_out is an empty tensor, the AscendCLNN framework handles it.
- The shapes of the corresponding tensors in key and value must be exactly the same. In the non-contiguous scenario, the batch in the tensorlist of key and value can only be 1, the number of tensors must equal the B of query, and N and D must be equal. Due to the tensorlist limitation, B cannot be greater than 256 in the non-contiguous scenario.
- When Q_S is greater than 1, the function usage restrictions for the query, key, and value inputs are as follows:
    - The B axis supports a value less than or equal to 65536.
    - If the input type is INT8 and the D axis is not 32-byte aligned, the maximum supported value of the B axis is 128. If the input type is FLOAT16 or BFLOAT16 and the D axis is not 16-byte aligned, the B axis also supports only up to 128.
    - The N axis supports a value less than or equal to 256, and the D axis supports a value less than or equal to 512. If input_layout is BSH or BSND, it is recommended that N*D be less than 65535.
    - S supports a value less than or equal to 20971520 (20M). In some long-sequence scenarios, an excessively large computation load may cause the operator execution to time out (an aicore error is reported, with errorStr: timeout or trap error). In this scenario, S splitting is recommended. Note: The computation load is affected by B, S, N, D, and other factors. The larger the values, the larger the computation load.
    - D axis restrictions: If the types of query, key, value, or attention_out include INT8, the D axis must be 32-aligned. If the types include INT4, the D axis must be 64-aligned. If all types are FLOAT16 or BFLOAT16, the D axis must be 16-aligned.
- When Q_S is 1, the function usage restrictions for the query, key, and value inputs are as follows:
    - The B axis supports a value less than or equal to 65536, the N axis supports a value less than or equal to 256, and the D axis supports a value less than or equal to 512.
    - The scenario where the input types of query, key, and value are all INT8 is not supported currently.
    - In the INT4 pseudo-quantization scenario, the aclnn single-operator invocation supports KV INT4 input or INT4 concatenated into INT32 input (generating INT4-format data through dynamicQuant is recommended).
    - In the INT4 pseudo-quantization scenario, if KV INT4 is concatenated into INT32 input, the N, D, or H of KV is one eighth of the actual value (the same applies to prefix).
    - If the input types of key and value are INT4 (INT32), the D axis must be 64-aligned (INT32 supports only D 8-alignment).
- The data layout formats of query, key, and value support interpretation from multiple dimensions, where B (Batch) indicates the batch size of the input samples, S (Seq-Length) indicates the sequence length of the input samples, H (Head-Size) indicates the size of the hidden layer, N (Head-Num) indicates the number of heads, and D (Head-Dim) indicates the size of the smallest unit of the hidden layer, satisfying D=H/N. T indicates the accumulated sum of the sequence lengths of the input samples of all batches.
- FA scenario constraints:
    - GQA: In the non-quantization scenario, when the Head-Dim values of the three groups query/key/value are all less than or equal to 128 and the layout is not NTD/NTD_TND/BSH_BNSD/BSND_BNSD/BNSD_BSND, the Head-Dim of query/key can be different from the Head-Dim of value. When the Head-Dim of query/key is different from the Head-Dim of value, no parameter group except atten_mask is supported.
    - Prefill MLA: The scenario where the Head-Dim of query/key is 192 and the Head-Dim of value is 128 is supported. In other scenarios, the Head-Dim of query/key/value/attention_out must be the same. Full quantization and pseudo-quantization are not supported.
    - Decode MLA: The Head-Dim of query/key/value/attention_out must be the same. tensorlist, left padding, pseudo-quantization, and prefix are not supported.
    - In the MLA scenario, pse is not supported and pse_shift cannot be passed in.
    - If the layout of query is TND/NTD, pse_shift and tensorlist are not supported.
    - In the PagedAttention scenario, if input_layout is BNSD, TND, BSH, or BSND, the key/value layout supports three formats: BnBsH (blockNum, blockSize, H), BnNBsD (blockNum, KV_N, blockSize, D), and NZ (blockNum, KV_N, D/16, blockSize, 16).
- pse_shift constraints:
    - When pse_type is 0, if the data type of query is FLOAT16 or INT8, the data type of pse_shift must be FLOAT16. If the data type of query is BFLOAT16, the data type of pse_shift must be BFLOAT16.
    - The shape of pse_shift must have 4 dimensions. The second dimension must equal Q_N, the third dimension must be greater than or equal to Q_S, and the fourth dimension must be greater than or equal to KV_S (KV_S + actual_shared_prefix_len in the prefix scenario).
    - In the PagedAttention scenario, the last dimension of the pse_shift shape must be greater than or equal to maxBlockNumPerBatch * block_size.
    - In the alibi scenario, Q_S must equal KV_S.
- atten_mask constraints:
    - Only 2/3/4 input dimensions are supported. If Q_S and KV_S in the input atten_mask shape are not 32B-aligned, they can be rounded up to the aligned Q_S and KV_S.
    - If sparse_mode is 2, 3, or 4, pass in the optimized atten_mask matrix (2048*2048).
    - If sparse_mode is 0 or 4, ensure that atten_mask is consistent with the ranges of pre_tokens and next_tokens.
- num_heads must be divisible by num_key_value_heads. In the full quantization scenario, if the types of query/key/value are FLOAT8_E4M3FN and input_layout is TND, the ratio of num_heads to num_key_value_heads supports only 1, 2, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, and 128. In other full quantization scenarios, the ratio of num_heads to num_key_value_heads supports only 1, 2, 4, 8, 16, 32, 64, and 128.
- input_layout=BSH_BNSD, BSND_BNSD, NTD, and NTD_TND support only Q_D=K_D=V_D all equal to 64 or 128, or Q_D=K_D equal to 192 and V_D equal to 128; input_layout=BNSD_BSND supports only Q_D=K_D=V_D all 16-aligned (32-aligned when the data type of attention_out is INT8), or Q_D=K_D equal to 192 and V_D equal to 128.
- In the GQA full quantization scenario (with query/key/value of the FLOAT8_E4M3FN type), Head-Dim supports only 128, and pse is not supported.
- In the MxFP8 scenario, Head-Dim supports only 64 or 128.
- <term>Ascend 950PR&950DT products</term>:
    - antiquant_scale/antiquant_offset (unified pseudo-quantization) is not supported. Use the pseudo-quantization parameters separated for key/value.
- <term>Atlas A3 products</term> and <term>Atlas A2 products</term>:
    - In the Prefill MLA scenario, tensorlist and left padding are not supported.
    - In the MLA scenario, post-quantization is not supported.

## Calling Example

The following sample code is for reference only. For the compilation and execution process in GE graph mode, see [Operator Invocation](../../../docs/zh/invocation/quick_op_invocation.md#ge图模式). For the complete code, see [test_geir_fused_infer_attention_score](../examples/test_geir_fused_infer_attention_score.cpp).

> **Note**: key/value are dynamic inputs. Use `create_dynamic_input_byindex_key/value` to register the input instances at the positions (1/2) defined by the IR. Do not use `create_dynamic_input_key/value` (its push_back semantics place the dynamic inputs after the static inputs, causing the input slots to misalign with the IR definition).

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
 * \file test_geir_fused_infer_attention_score.cpp
 * \brief GE graph construction sample for FusedInferAttentionScore (BNSD layout, fp16, no-quant path).
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

#include "../op_graph/fused_infer_attention_score_proto.h"

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
constexpr int64_t kQSeqLen = 2;
constexpr int64_t kKvSeqLen = 2;
constexpr int64_t kNumHeads = 2;
constexpr int64_t kNumKvHeads = 2;
constexpr int64_t kHeadDim = 16;
constexpr double kScale = 0.25; // 1.0 / sqrt(16)
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
        return sign;
    }
    if (exponent >= 31) {
        return static_cast<uint16_t>(sign | 0x7C00U);
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

int32_t BuildGraph(Graph& graph, vector<Tensor>& inputTensors, vector<Operator>& inputOps, vector<Operator>& outputOps,
                   vector<float>& qData, vector<float>& kData, vector<float>& vData)
{
    // BNSD layout: [B, N, S, D].
    const vector<int64_t> qShape = {kBatch, kNumHeads, kQSeqLen, kHeadDim};
    const vector<int64_t> kvShape = {kBatch, kNumKvHeads, kKvSeqLen, kHeadDim};
    const vector<int64_t> pseShape = {kBatch, kNumHeads, kQSeqLen, kKvSeqLen};
    const vector<int64_t> maskShape = {kBatch, 1, kQSeqLen, kKvSeqLen};

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
    vector<uint16_t> qHalf(qData.size());
    vector<uint16_t> kHalf(kData.size());
    vector<uint16_t> vHalf(vData.size());
    for (size_t i = 0; i < qData.size(); ++i) {
        qHalf[i] = FloatToHalf(qData[i]);
    }
    for (size_t i = 0; i < kData.size(); ++i) {
        kHalf[i] = FloatToHalf(kData[i]);
        vHalf[i] = FloatToHalf(vData[i]);
    }
    // pse_shift is bound but filled with zeros so the reference stays a plain softmax attention.
    const vector<uint16_t> pseHalf(pseShape[0] * pseShape[1] * pseShape[2] * pseShape[3], 0);
    // bool mask, 0 means the position attends.
    const vector<uint8_t> maskData(maskShape[0] * maskShape[1] * maskShape[2] * maskShape[3], 0);
    // actual_seq_lengths: int64, effective sequence length of each batch.
    const vector<int64_t> seqLenData = {kQSeqLen};

    CHECK_RET(AddInput(graph, inputTensors, inputOps, "query", 1, qShape, DT_FLOAT16, qHalf) == SUCCESS, return FAILED);
    CHECK_RET(AddInput(graph, inputTensors, inputOps, "key", 2, kvShape, DT_FLOAT16, kHalf) == SUCCESS, return FAILED);
    CHECK_RET(AddInput(graph, inputTensors, inputOps, "value", 3, kvShape, DT_FLOAT16, vHalf) == SUCCESS,
              return FAILED);
    CHECK_RET(AddInput(graph, inputTensors, inputOps, "pse_shift", 4, pseShape, DT_FLOAT16, pseHalf) == SUCCESS,
              return FAILED);
    CHECK_RET(AddInput(graph, inputTensors, inputOps, "atten_mask", 5, maskShape, DT_BOOL, maskData) == SUCCESS,
              return FAILED);
    CHECK_RET(
        AddInput(graph, inputTensors, inputOps, "actual_seq_lengths", 6, {kBatch}, DT_INT64, seqLenData) == SUCCESS,
        return FAILED);
    CHECK_RET(AddInput(graph, inputTensors, inputOps, "actual_seq_lengths_kv", 7, {kBatch}, DT_INT64,
                       vector<int64_t>{kKvSeqLen}) == SUCCESS,
              return FAILED);
    CHECK_RET(AddInput(graph, inputTensors, inputOps, "q_start_idx", 8, {1}, DT_INT64, vector<int64_t>{0}) == SUCCESS,
              return FAILED);
    CHECK_RET(AddInput(graph, inputTensors, inputOps, "kv_start_idx", 9, {1}, DT_INT64, vector<int64_t>{0}) == SUCCESS,
              return FAILED);

    auto node = op::FusedInferAttentionScore("fused_infer_attention_score");
    node.set_input_query(inputOps[0]);
    // key/value are dynamic inputs. create_dynamic_input_byindex_* registers one instance for each AT the position
    // declared by the IR definition (key at index 1, value at index 2); a plain create_dynamic_input_* would append
    // them after the static inputs and misalign the slot order. The input descs come from the connected Data ops.
    node.create_dynamic_input_byindex_key(1, 1);
    node.create_dynamic_input_byindex_value(1, 2);
    node.set_dynamic_input_key(0, inputOps[1]);
    node.set_dynamic_input_value(0, inputOps[2]);
    node.set_input_pse_shift(inputOps[3]);
    node.set_input_atten_mask(inputOps[4]);
    node.set_input_actual_seq_lengths(inputOps[5]);
    node.set_input_actual_seq_lengths_kv(inputOps[6]);
    node.set_input_q_start_idx(inputOps[7]);
    node.set_input_kv_start_idx(inputOps[8]);

    node.set_attr_num_heads(kNumHeads);
    node.set_attr_scale(kScale);
    node.set_attr_pre_tokens(2147483647);
    node.set_attr_next_tokens(2147483647);
    node.set_attr_input_layout("BNSD");
    node.set_attr_num_key_value_heads(kNumKvHeads);
    node.set_attr_sparse_mode(0);
    node.set_attr_inner_precise(1);
    node.set_attr_block_size(0);
    node.set_attr_antiquant_mode(0);
    node.set_attr_softmax_lse_flag(false);
    node.set_attr_key_antiquant_mode(0);
    node.set_attr_value_antiquant_mode(0);
    node.set_attr_query_quant_mode(0);
    node.set_attr_pse_type(0);
    node.set_attr_out_dtype(0);

    TensorDesc attentionOutDesc(Shape(qShape), FORMAT_ND, DT_FLOAT16);
    TensorDesc softmaxLseDesc(Shape({0}), FORMAT_ND, DT_FLOAT);
    CHECK_RET(node.update_output_desc_attention_out(attentionOutDesc) == GRAPH_SUCCESS, return FAILED);
    CHECK_RET(node.update_output_desc_softmax_lse(softmaxLseDesc) == GRAPH_SUCCESS, return FAILED);
    CHECK_RET(graph.AddOp(node) == GRAPH_SUCCESS, return FAILED);
    outputOps.push_back(node);
    return SUCCESS;
}

// CPU reference: softmax(scale * Q @ K^T + pse) @ V with BNSD layout and GQA support (pse is zero here).
void ComputeReference(const vector<float>& qData, const vector<float>& kData, const vector<float>& vData,
                      vector<float>& attentionOut)
{
    const int64_t s1 = kQSeqLen;
    const int64_t s2 = kKvSeqLen;
    const int64_t d = kHeadDim;
    const int64_t group = kNumHeads / kNumKvHeads;
    attentionOut.assign(GetShapeSize({kBatch, kNumHeads, kQSeqLen, kHeadDim}), 0.0f);
    for (int64_t b = 0; b < kBatch; ++b) {
        for (int64_t n = 0; n < kNumHeads; ++n) {
            const int64_t kvN = n / group;
            vector<float> score(s2);
            for (int64_t i = 0; i < s1; ++i) {
                for (int64_t j = 0; j < s2; ++j) {
                    float dot = 0.0f;
                    for (int64_t t = 0; t < d; ++t) {
                        const int64_t qIdx = ((b * kNumHeads + n) * s1 + i) * d + t;
                        const int64_t kIdx = ((b * kNumKvHeads + kvN) * s2 + j) * d + t;
                        dot += qData[qIdx] * kData[kIdx];
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
                for (int64_t t = 0; t < d; ++t) {
                    float y = 0.0f;
                    for (int64_t j = 0; j < s2; ++j) {
                        const int64_t vIdx = ((b * kNumKvHeads + kvN) * s2 + j) * d + t;
                        y += (score[j] / sum) * vData[vIdx];
                    }
                    const int64_t oIdx = ((b * kNumHeads + n) * s1 + i) * d + t;
                    attentionOut[oIdx] = y;
                }
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

int32_t ValidateOutputs(const vector<Tensor>& outputs, const vector<float>& qData, const vector<float>& kData,
                        const vector<float>& vData)
{
    CHECK_RET(outputs.size() == 2, LOG_PRINT("[CHECK] FAIL: expected 2 outputs, got %zu\n", outputs.size());
              return FAILED);
    vector<float> attentionOut;
    ComputeReference(qData, kData, vData, attentionOut);
    const vector<int64_t> qShape = {kBatch, kNumHeads, kQSeqLen, kHeadDim};
    const bool ok = CheckOutput(outputs[0], "attention_out", qShape, attentionOut);
    CHECK_RET(outputs[1].GetSize() == 0, LOG_PRINT("[CHECK] softmax_lse expected empty\n"); return FAILED);
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

    Graph graph("fused_infer_attention_score_graph");
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
        LOG_PRINT("FusedInferAttentionScore graph run success, output count: %zu\n", outputTensors.size());
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
