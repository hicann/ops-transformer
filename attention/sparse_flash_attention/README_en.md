# SparseFlashAttention

English | [简体中文](./README.md)

## Product Support

| Product                                                     | Supported |
| :------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                   |    √    |
| <term>Atlas A3 products</term> |    √    |
| <term>Atlas A2 products</term> |    √    |
| <term>Atlas 200I/500 A2 inference products</term>                  |    ×    |
| <term>Atlas inference products</term>                          |    ×    |
| <term>Atlas training products</term>                          |    ×    |

## Function Description

- API function: sparse_flash_attention (SFA) is an efficient attention computation module for inference scenarios with large sequence lengths. The module greatly reduces the computation load by "computing only the key parts", but it introduces a large number of discrete memory accesses, which increases the data movement time and affects the overall performance.

- Computation formula:

    $$
    \text{softmax}(\frac{Q@\tilde{K}^T}{\sqrt{d_k}})@\tilde{V}
    $$

    Here $\tilde{K},\tilde{V}$ are the Key and Value with higher importance obtained by using a selection algorithm (such as `lightning_indexer`). They are generally sparse or block-sparse. $d_k$ is the dimension of each head of $Q,\tilde{K}$.
    The released `sparse_flash_attention` is a new operator for Sparse Attention. It performs fine-grained optimization for discrete memory accesses through instruction reduction and data movement aggregation.

## Parameter Description

> **Description:**<br>
> Parameter dimension meanings: B indicates the batch size, Q_S and KV_S indicate the sequence lengths of query and key/value respectively, Q_N and KV_N indicate the head numbers of query and key/value respectively, Q_D and KV_D indicate the head dimensions of query and key/value respectively, Dr indicates the head dimension of rope, Q_T and KV_T indicate the total tokens of query and key/value respectively, sparse_size indicates the number of blocks selected at a time, and block_num and block_size indicate the total number of blocks and the number of tokens per block in the PageAttention scenario respectively.

  <table style="undefined;table-layout: fixed; width: 1080px"><colgroup>
  <col style="width: 200px">
  <col style="width: 150px">
  <col style="width: 480px">
  <col style="width: 150px">
  <col style="width: 100px">
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
      <td>Query input of the attention structure. Empty tensors and non-contiguous tensors are not supported. When layout_query is "BSND", the shape is [B, Q_S, Q_N, Q_D]. When layout_query is "TND", the shape is [Q_T, Q_N, Q_D].</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>key</td>
      <td>Input</td>
      <td>Key input of the attention structure. Empty tensors are not supported. When layout_kv is "BSND", the shape is [B, KV_S, KV_N, KV_D]. When layout_kv is "TND", the shape is [KV_T, KV_N, KV_D]. When layout_kv is "PA_BSND", the shape is [block_num, block_size, KV_N, KV_D], where block_num is the total number of blocks in the PageAttention scenario.</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>value</td>
      <td>Input</td>
      <td>Value input of the attention structure. It can be omitted (pass None for torch, and nullptr for aclnn). When it is omitted, key is used as value. Empty tensors are not supported. The shape is the same as that of key.</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>sparse_indices</td>
      <td>Input</td>
      <td>Indices for discretely accessing the kvCache. Empty tensors and non-contiguous tensors are not supported. sparse_size is the number of blocks selected at a time. Ensure that valid values in each row are all in the first half and invalid values are all in the second half, and that sparse_size is greater than 0. When layout_query is "BSND", the shape is [B, Q_S, KV_N, sparse_size]. When layout_query is "TND", the shape is [Q_T, KV_N, sparse_size].</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>block_table</td>
      <td>Optional input</td>
      <td>Indicates the block mapping table used for kvCache storage in PageAttention. Empty tensors and non-contiguous tensors are not supported. The length of the second dimension is not less than the number of blocks corresponding to the maximum KV_S in all batches, that is, KV_S_max / block_size rounded up. The shape is [B, KV_S_max/block_size]. The default value is None.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>actual_seq_lengths_query</td>
      <td>Optional input</td>
      <td>Indicates the number of valid tokens of query in different batches. Empty tensors and non-contiguous tensors are not supported. None can be passed in, indicating that the value is the same as the Q_S length of query. The shape is [B,]. The number of valid tokens in each batch is not greater than the Q_S size of query and not less than 0. When layout_query is "TND", this parameter is required, and the number of elements is used as the value of B. Each element indicates the total number of tokens of the current batch and all preceding batches (prefix sum). The default value is None.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>actual_seq_lengths_kv</td>
      <td>Optional input</td>
      <td>Indicates the number of valid tokens of key and value in different batches. Empty tensors and non-contiguous tensors are not supported. None can be passed in, indicating that the value is the same as the KV_S length of key. The shape is [B,]. The number of valid tokens in each batch is not greater than the KV_S size of key/value and not less than 0. When layout_kv is "TND" or "PA_BSND", this parameter is required. When layout_kv is "TND", each element indicates the total number of tokens of the current batch and all preceding batches, that is, the prefix sum. Therefore, the value of the next element must be greater than or equal to that of the preceding element. The default value is None.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>query_rope</td>
      <td>Optional input</td>
      <td>Indicates the rope information of query in the MLA structure. Empty tensors and non-contiguous tensors are not supported. When layout_query is "BSND", the shape is [B, Q_S, Q_N, Dr]. When layout_query is "TND", the shape is [Q_T, Q_N, Dr]. The default value is None.</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>key_rope</td>
      <td>Optional input</td>
      <td>Indicates the rope information of key in the MLA structure. Empty tensors are not supported. When layout_kv is "BSND", the shape is [B, KV_S, KV_N, Dr]. When layout_kv is "TND", the shape is [KV_T, KV_N, Dr]. When layout_kv is "PA_BSND", the shape is [block_num, block_size, KV_N, Dr]. The default value is None.</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>sinks</td>
      <td>Optional input</td>
      <td>Learnable sinks information in the attention structure. The default value is None.</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scale_value</td>
      <td>Attribute</td>
      <td>Represents the scaling factor.</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>sparse_block_size</td>
      <td>Optional attribute</td>
      <td>Represents the block size in the sparse phase. When sparse_block_size is 1, the scenario is Token-wise sparsification. When sparse_block_size is greater than 1 and less than or equal to 128, the scenario is Block-wise sparsification, where tokens in a block share the same sparsification decision. The default value is 1.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layout_query</td>
      <td>Optional attribute</td>
      <td>Identifies the data layout format of the input query. The default value is "BSND". BSND and TND can be passed in.</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layout_kv</td>
      <td>Optional attribute</td>
      <td>Identifies the data layout format of the input key. The default value is "BSND". TND, BSND, and PA_BSND can be passed in. PA_BSND is used when PageAttention is enabled.</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>sparse_mode</td>
      <td>Optional attribute</td>
      <td>Indicates the sparse mode. When sparse_mode is 0, all data is computed. When sparse_mode is 3, the mask of the rightDownCausal mode is used, corresponding to the lower triangular scenario with the division line from the bottom-right vertex to the top-left. The default value is 3.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>pre_tokens</td>
      <td>Optional attribute</td>
      <td>Used for sparse computation. It indicates the number of preceding tokens with which attention needs to compute associations. Only the default value 2^63-1 is supported.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>next_tokens</td>
      <td>Optional attribute</td>
      <td>Used for sparse computation. It indicates the number of succeeding tokens with which attention needs to compute associations. Only the default value 2^63-1 is supported.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>attention_mode</td>
      <td>Optional attribute</td>
      <td>Indicates the attention mode. Only 2 can be passed in, indicating the MLA-absorb mode.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>return_softmax_lse</td>
      <td>Optional attribute</td>
      <td>Indicates whether to return softmax_max and softmax_sum. True indicates that they are returned, and False indicates that they are not returned. The default value is False.</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>attention_out</td>
      <td>Output</td>
      <td>Output in the formula. Empty tensors and non-contiguous tensors are not supported. When layout_query is "BSND", the shape is [B, Q_S, Q_N, Q_D]. When layout_query is "TND", the shape is [Q_T, Q_N, Q_D].</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>softmax_max</td>
      <td>Output</td>
      <td>In the attention algorithm, softmax_max is obtained by taking the max of the result of query multiplied by key. Empty tensors and non-contiguous tensors are not supported. When layout_query is "BSND", the shape is [B, KV_N, Q_S, Q_N/KV_N]. When layout_query is "TND", the shape is [KV_N, Q_T, Q_N/KV_N].</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>softmax_sum</td>
      <td>Output</td>
      <td>In the attention algorithm, softmax_sum is obtained by subtracting softmax_max from the result of query multiplied by key, taking the exp, and calculating the sum. Empty tensors and non-contiguous tensors are not supported. When layout_query is "BSND", the shape is [B, KV_N, Q_S, Q_N/KV_N]. When layout_query is "TND", the shape is [KV_N, Q_T, Q_N/KV_N].</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
  </tbody>
  </table>

## Constraints

- This API is supported in inference scenarios.
- This API supports graph mode.
- KV_N supports only 1.
- block_size is the number of tokens in a block. block_size is a multiple of 16, and the maximum value is 1024.
- Q_D of query and KV_D of key and value must both be 512, and Dr of query_rope and Dr of key_rope must both be 64.
- The data types of query, key, and value must be the same.
- Currently, only non-empty query_rope and key_rope are supported. Empty rope inputs are not supported.
- sinks is supported only on Ascend 950PR&950DT products.
- <term>Ascend 950PR&950DT products</term>:
  - Q_N supports 1 to 128.
  - sparse_block_size supports only 1.
  - Only when layout_kv is PA_BSND, key, value, and key_rope support a non-contiguous axis 0.
  - When return_softmax_lse is True, layout_kv supports PA_BSND.
- <term>Atlas A3 products</term> and <term>Atlas A2 products</term>:
  - Q_N supports 1/2/4/8/16/32/64/128.
  - sparse_block_size supports [1,128] and must be a power of 2. In the PageAttention scenario, sparse_block_size must be a divisor of block_size.
  - sinks is not supported.
  - key, value, and key_rope do not support non-contiguous tensors.
  - When return_softmax_lse is True, layout_kv does not support PA_BSND.

## Calling Example

<table class="tg"><thead>
  <tr>
    <th class="tg-0pky">Calling Method</th>
    <th class="tg-0pky">Sample Code</th>
    <th class="tg-0pky">Description</th>
  </tr></thead>
<tbody>
  <tr>
    <td class="tg-9wq8" rowspan="6">aclnn API</td>
    <td class="tg-0pky">
    <a href="./examples//test_aclnn_sparse_flash_attention.cpp">test_aclnn_sparse_flash_attention
    </a>
    </td>
    <td class="tg-lboi" rowspan="6">
    Calls the operator through the
    <a href="./docs/aclnnSparseFlashAttention_en.md">aclnnSparseFlashAttention
    </a>
    API
    </td>
  </tr>
</tbody>
<tbody>
  <tr>
    <td class="tg-9wq8" rowspan="6">aclnn API</td>
    <td class="tg-0pky">
    <a href="./examples//test_aclnn_sparse_flash_attention_v2.cpp">test_aclnn_sparse_flash_attention_v2
    </a>
    </td>
    <td class="tg-lboi" rowspan="6">
    Calls the operator through the
    <a href="./docs/aclnnSparseFlashAttentionV2_en.md">aclnnSparseFlashAttentionV2
    </a>
    API
    </td>
  </tr>
</tbody>
<tbody>
  <tr>
    <td class="tg-9wq8">Graph mode</td>
    <td class="tg-0pky">
    <a href="./examples/test_geir_sparse_flash_attention.cpp">test_geir_sparse_flash_attention
    </a>
    </td>
    <td class="tg-lboi">
    Calls the SparseFlashAttention operator through graph construction with
    <a href="./docs/ascendir_SparseFlashAttention_en.md">Ascend IR
    </a>
    </td>
  </tr>
</tbody>
</table>
