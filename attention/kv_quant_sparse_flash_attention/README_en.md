# KvQuantSparseFlashAttention

English | [简体中文](./README.md)

## Product Support

|Product      | Supported |
|:----------------------------|:-----------:|
|<term>Ascend 950PR&950DT products</term>|      √     |
|<term>Atlas A3 products</term>|      √     |
|<term>Atlas A2 products</term>|      √     |
|<term>Atlas 200I/500 A2 inference products</term>|      ×     |
|<term>Atlas inference products</term>|      ×     |
|<term>Atlas training products</term>|      ×     |

## Description

- API function: `kv_quant_sparse_flash_attention` supports [Per-Token-Head-Tile-128 quantization] input on the basis of `sparse_flash_attention`. As the context length of large models increases, Sparse Attention becomes increasingly important. This technique greatly reduces the computation load by computing only the critical parts. However, it introduces a large number of scattered memory accesses, which increases the data movement time and affects the overall performance.

- Computation formula:

    $$
    Attention=\text{softmax}(\frac{Q @ \text{Dequant}({\tilde{K}^{INT8}},{Scale_K})^T}{\sqrt{d_k}})@\text{Dequant}(\tilde{V}^{INT8},{Scale_V}),
    $$

    where $\tilde{K},\tilde{V}$ are the keys and values with high importance selected by a selection algorithm (such as `LightningIndexer`). They are generally sparse or block-wise sparse. $d_k$ is the dimension of each head of $Q,\tilde{K}$, and $\text{Dequant}(\cdot,\cdot)$ is the dequantization function.

### INT4 (TQ4) Quantization Type

This operator adds the TurboQuant 4-bit (TQ4) quantization type. When `key_quant_mode=3` and `value_quant_mode=3`, key/value uses the TQ4 packed KV cache. Both quantization modes must be set to 3 at the same time.

Each TQ4 KV token uses a 386-byte slot with the following layout:

- `[0, 256)`: 256 bytes of packed INT4. Each byte contains two 4-bit codes, corresponding to 512-dimension latent/nope data.
- `[256, 384)`: 128 bytes of BF16 RoPE data, corresponding to 64 RoPE elements.
- `[384, 386)`: 2 bytes of FP16 token scale. This scale is an independent dequantization coefficient for each KV token and has a different meaning from the attribute `scale_value`.

TQ4 uses 16 codebook centroids for dequantization. Inside the operator, a byte-LUT decodes each packed byte into two centroids, which then participate in the key/value computation together with the token scale. TQ4 does not require the additional `key_dequant_scale` and `value_dequant_scale` inputs. The quantization parameters are stored together with the KV cache slots.

TQ4 currently targets the MLA-absorb scenario on Ascend 910B. query supports FLOAT16/BFLOAT16, `attention_mode=2`, and `rope_head_dim=64`. The TQ4 cases in the existing single-operator parameter set use `layout_kv=PA_BSND`.
The released `kv_quant_sparse_flash_attention` is a new operator designed for Sparse Attention. It provides careful optimizations, including instruction reduction and data movement aggregation, for scattered memory accesses.

## Parameter Description

> **Description:**<br>
> Parameter dimension meanings: B indicates Batch Size; Q_S and KV_S indicate the Sequence Length of query and key/value, respectively; Q_N and KV_N indicate the Head Num of query and key/value, respectively; Q_D and KV_D indicate the Head Dim of query and key/value, respectively; Q_T and KV_T indicate the Total Tokens of query and key/value, respectively; sparse_size indicates the number of blocks selected in one discrete selection; block_num and block_size indicate the total number of blocks and the number of tokens per block in the PageAttention scenario, respectively.

<table style="undefined;table-layout: fixed; width: 1080px"><colgroup>
  <col style="width: 200px">
  <col style="width: 150px">
  <col style="width: 280px">
  <col style="width: 330px">
  <col style="width: 120px">
  </colgroup>
  <thead>
    <tr>
      <th>Parameter Name</th>
      <th>Input/Output/Attribute</th>
      <th>Description</th>
      <th>Data Type</th>
      <th>Data Format</th>
    </tr></thead>
      <tbody>
      <tr>
          <td>query</td>
          <td>Input</td>
          <td>Q input of the attention structure. Non-contiguous input is not supported. query is obtained by concatenating q_nope and q_rope of the same data type along the D dimension. When layout_query is "BSND", the shape is [B, Q_S, Q_N, Q_D]. When layout_query is "TND", the shape is [Q_T, Q_N, Q_D]. Q_D supports only 576, that is, q_nope+q_rope=512+64. Q_N supports 1/2/4/8/16/32/48/64/128.</td>
          <td>FLOAT16, BFLOAT16</td>
          <td>ND</td>
      </tr>
      <tr>
          <td>key</td>
          <td>Input</td>
          <td>K input of the attention structure. key is obtained by concatenating k_nope, k_rope of the same data type as query, and the float32 quantization parameters along the D dimension. When layout_kv is "BSND", the shape is [B, KV_S, KV_N, KV_D]. When layout_kv is "TND", the shape is [KV_T, KV_N, KV_D]. When layout_kv is "PA_BSND", the shape is [block_num, block_size, KV_N, KV_D], where block_num is the total number of blocks in the PageAttention scenario and block_size is the number of tokens per block. block_size is a multiple of 16 and supports up to 1024. KV_N supports only 1. KV_D supports only 656, that is, nope+rope*2+dequant_scale*4=512+64*2+4*4. In TQ4 mode (key_quant_mode=3), the 386-byte packed slot is used instead.</td>
          <td>FLOAT8_E4M3, INT8, HIFLOAT8</td>
          <td>ND</td>
      </tr>
      <tr>
          <td>value</td>
          <td>Input</td>
          <td>V input of the attention structure. In TQ4 mode, the same 386-byte packed slot as key is used.</td>
          <td>FLOAT8_E4M3, INT8, HIFLOAT8</td>
          <td>ND</td>
      </tr>
      <tr>
          <td>sparse_indices</td>
          <td>Input</td>
          <td>Represents the indices for discretely selecting kvCache. Non-contiguous input is not supported. When layout_query is "BSND", the shape is [B, Q_S, KV_N, sparse_size]. When layout_query is "TND", the shape is [Q_T, KV_N, sparse_size]. sparse_size is the number of blocks selected in one discrete selection. The valid values in each row must be in the first half, and the invalid values must be in the second half. sparse_size must be greater than 0. When the data types of key and value are hifloat8, sparse_size supports only 2048.</td>
          <td>INT32</td>
          <td>ND</td>
      </tr>
      <tr>
          <td>scale_value</td>
          <td>Attribute</td>
          <td>Reciprocal of the square root of d<sub>k</sub> in the formula. It represents the scaling factor and serves as the scalar value of Muls after the matrix multiplication of query and key.</td>
          <td>FLOAT</td>
          <td>-</td>
      </tr>
      <tr>
          <td>key_quant_mode</td>
          <td>Attribute</td>
          <td>Represents the quantization mode of key. 2 indicates the per_tile quantization mode. The new value 3 indicates TQ4 INT4 quantization, and value_quant_mode must be set to 3 at the same time.</td>
          <td>INT64</td>
          <td>-</td>
      </tr>
      <tr>
          <td>value_quant_mode</td>
          <td>Attribute</td>
          <td>Represents the quantization mode of value. 2 indicates the per_tile quantization mode. The new value 3 indicates TQ4 INT4 quantization, and key_quant_mode must be set to 3 at the same time.</td>
          <td>INT64</td>
          <td>-</td>
      </tr>
      <tr>
          <td>key_dequant_scale</td>
          <td>Optional input</td>
          <td>Reserved parameter. In TQ4 mode, the token scale has been stored together with the KV cache slots, so this input is not required. The default value is None.</td>
          <td>-</td>
          <td>-</td>
      </tr>
      <tr>
          <td>value_dequant_scale</td>
          <td>Optional input</td>
          <td>Reserved parameter. In TQ4 mode, the token scale has been stored together with the KV cache slots, so this input is not required. The default value is None.</td>
          <td>-</td>
          <td>-</td>
      </tr>
      <tr>
          <td>block_table</td>
          <td>Optional input</td>
          <td>Represents the block mapping table used for kvCache storage in PageAttention. The shape is [B, KV_S_max/block_size]. The length of the first dimension is B. The length of the second dimension is not less than the number of blocks corresponding to the largest KV_S among all batches, that is, KV_S_max / block_size rounded up. The default value is None.</td>
          <td>INT32</td>
          <td>ND</td>
      </tr>
      <tr>
          <td>actual_seq_lengths_query</td>
          <td>Optional input</td>
          <td>Represents the number of valid tokens of query in different batches. If seqlen is not specified, None can be passed, indicating that the value is the same as the Q_S length of the query shape. The shape is [B,]. The number of valid tokens in each batch must not exceed the Q_S size of query and must not be less than 0. When layout_query is "TND", this parameter must be passed, and the number of elements in this parameter is used as the value of B. Each element in this parameter indicates the total number of tokens in the current batch and all previous batches, that is, the prefix sum. Therefore, the value of each element must be greater than or equal to that of the previous element. The default value is None.</td>
          <td>INT32</td>
          <td>ND</td>
      </tr>
      <tr>
          <td>actual_seq_lengths_kv</td>
          <td>Optional input</td>
          <td>Represents the number of valid tokens of key and value in different batches. If not specified, None indicates that the value is the same as the KV_S length of the key shape. The shape is [B,]. The number of valid tokens in each batch must not exceed the KV_S size of key/value and must not be less than 0. When layout_kv is "TND" or "PA_BSND", this parameter must be passed. When layout_kv is "TND", each element in this parameter indicates the total number of tokens in the current batch and all previous batches, that is, the prefix sum. Therefore, the value of each element must be greater than or equal to that of the previous element. The default value is None.</td>
          <td>INT32</td>
          <td>ND</td>
      </tr>
      <tr>
          <td>sparse_block_size</td>
          <td>Optional attribute</td>
          <td>Represents the block size in the sparse phase. When sparse_block_size is 1, it is a Token-wise sparsification scenario. When sparse_block_size is greater than 1 and less than or equal to 128, it is a Block-wise sparsification scenario, where tokens within a block share the same sparsification decision. The default value is 1.</td>
          <td>INT64</td>
          <td>-</td>
      </tr>
      <tr>
          <td>layout_query</td>
          <td>Optional attribute</td>
          <td>Identifies the data layout format of the input query. The default value is "BSND". BSND and TND are supported.</td>
          <td>STRING</td>
          <td>-</td>
      </tr>
      <tr>
          <td>layout_kv</td>
          <td>Optional attribute</td>
          <td>Identifies the data layout format of the input key. The default value is "BSND". BSND, TND, and PA_BSND are supported. PA_BSND is used when PageAttention is enabled.</td>
          <td>STRING</td>
          <td>-</td>
      </tr>
      <tr>
          <td>sparse_mode</td>
          <td>Optional attribute</td>
          <td>Represents the sparse mode. When sparse_mode is 0, all computations are performed. When sparse_mode is 3, the mask of the rightDownCausal mode is used, corresponding to the lower triangular scenario divided by the line from the lower-right vertex to the upper-left. The default value is 3.</td>
          <td>INT64</td>
          <td>-</td>
      </tr>
      <tr>
          <td>pre_tokens</td>
          <td>Optional attribute</td>
          <td>Used for sparse computation. It indicates the number of previous tokens that attention needs to compute correlations with. Only 2^63-1 is supported. The default value is 2^63-1.</td>
          <td>INT64</td>
          <td>-</td>
      </tr>
      <tr>
          <td>next_tokens</td>
          <td>Optional attribute</td>
          <td>Used for sparse computation. It indicates the number of following tokens that attention needs to compute correlations with. Only 2^63-1 is supported. The default value is 2^63-1.</td>
          <td>INT64</td>
          <td>-</td>
      </tr>
      <tr>
          <td>attention_mode</td>
          <td>Optional attribute</td>
          <td>Represents the attention mode. Only 2 is supported, indicating the MLA-absorb mode, that is, the D of QK contains both rope and nope parts, and KV shares the same copy.</td>
          <td>INT64</td>
          <td>-</td>
      </tr>
      <tr>
          <td>quant_scale_repo_mode</td>
          <td>Optional attribute</td>
          <td>Represents the storage mode of quantization parameters. Only 1 is supported, indicating the combine mode, that is, the quantization parameters and data are stored together. The default value is 1.</td>
          <td>INT64</td>
          <td>-</td>
      </tr>
      <tr>
          <td>tile_size</td>
          <td>Optional attribute</td>
          <td>Represents the data block size corresponding to each parameter in per_tile mode. It is valid only in per_tile mode and supports only 128. The default value is 128.</td>
          <td>INT64</td>
          <td>-</td>
      </tr>
      <tr>
          <td>rope_head_dim</td>
          <td>Optional attribute</td>
          <td>Represents the size of rope_head_dim in the MLA architecture. It is valid only when attention_mode is 2. The default value is 64. In the INT8 per-tile mode of Atlas A2/A3 products, 0 and 64 are supported. In other modes, only 64 is supported.</td>
          <td>INT64</td>
          <td>-</td>
      </tr>
      <tr>
          <td>attention_out</td>
          <td>Output</td>
          <td>Represents the output Attention in the formula. The output shape is the same as the shape of the input query. When layout_query is "BSND", the shape is [B, Q_S, Q_N, Q_out_D]. When layout_query is "TND", the shape is [Q_T, Q_N, Q_out_D]. Q_out_D = Q_D - rope_head_dim.</td>
          <td>FLOAT16, BFLOAT16</td>
          <td>ND</td>
      </tr>
      </tbody>
  </table>

## Constraints

- This API supports graph mode.
- In non-PageAttention scenarios, layout\_query and layout\_kv must be set to the same value.
- TQ4 INT4 mode is enabled only when `key_quant_mode=value_quant_mode=3`, and the KV_D of key/value must be 386. Mode 3 cannot be used together with mode 2.
- The head dimension of TQ4 query is 576 (nope 512 + rope 64), `attention_mode=2`, and `rope_head_dim=64`. KV_N supports only 1.
- The TQ4 token scale is an FP16 value in the KV cache slot. The attribute `scale_value` is still used for overall scaling after the query-key matrix multiplication. Do not confuse the two.
- TQ4 currently supports the MLA-absorb scenario on Ascend 910B, and the single-operator test cases use `layout_kv=PA_BSND`.
- <term>Ascend 950PR&950DT products</term>:
  - The data types of key and value support only float8_e4m3, int8, and hifloat8.
  - sparse\_block\_size supports only 1.
  - Only when layout_kv is PA_BSND does key support non-contiguous input along axis 0.
  - TQ4 INT4 mode (key\_quant\_mode=3, value\_quant\_mode=3) is not supported. If 3 is passed in, the tiling check rejects it.
- <term>Atlas A3 products</term> and <term>Atlas A2 products</term>:
  - Q_N of query does not support 48.
  - The data types of key and value support only int8.
  - sparse\_block\_size supports [1,16] and must be a power of two. In the PageAttention scenario, sparse\_block\_size must divide block\_size exactly.
  - Non-contiguous key is not supported.
  - When rope\_head\_dim is 0, only key\_quant\_mode=value\_quant\_mode=2, quant\_scale\_repo\_mode=1, and tile\_size=128 are supported. The Q_D of query is 512, and the KV_D of key/value is 528. Each slot consists of 512 INT8 values and 4 FLOAT32 scales.
  - TQ4 INT4 mode (key\_quant\_mode=3, value\_quant\_mode=3) is supported.
