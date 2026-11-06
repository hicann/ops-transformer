# LightningIndexer

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

- Operator function: `lightning_indexer` obtains the Top-$k$ positions corresponding to each token based on a series of operations.

- Computation formula:

     $$
     Indices=\text{Top-}k\left\{[1]_{1\times g}@\left[(W@[1]_{1\times S_{k}})\odot\text{ReLU}\left(Q_{index}@K_{index}^T\right)\right]\right\}
     $$

     For the Index Query $Q_{index}\in\R^{g\times d}$ corresponding to a token, given the context Index Key $K_{index}\in\R^{S_{k}\times d},W\in\R^{g\times 1}$, where $g$ is the group size corresponding to GQA, $d$ is the dimension of each head, and $S_{k}$ is the context length.

## Parameter Description

> **Description:**<br>
> Parameter dimension meanings: B indicates Batch Size; Q_S and K_S indicate the Sequence Length of query and key, respectively; Q_N and K_N indicate the Head Num of query and key, respectively; D indicates Head Dim (Q_D and K_D are equal, both 128); Q_T and K_T indicate the Total Tokens of query and key, respectively; sparse_count indicates the number of indices finally selected (topK); block_num and block_size indicate the total number of blocks and the number of tokens per block in the PageAttention scenario, respectively. K_N supports only 1.

  <table style="undefined;table-layout: fixed; width: 1080px"><colgroup>
  <col style="width: 200px">
  <col style="width: 150px">
  <col style="width: 480px">
  <col style="width: 200px">
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
      <td>
          <ul>
                <li>Input Q in the formula.</li>
                <li>Empty tensors and non-contiguous tensors are not supported.</li>
                <li>When layout_query is BSND, the shape is [B, Q_S, Q_N, D]. When layout_query is TND, the shape is [Q_T, Q_N, D].</li>
          </ul>
      </td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>key</td>
      <td>Input</td>
      <td>
          <ul>
                <li>Input K in the formula.</li>
                <li>Empty tensors are not supported.</li>
                <li>When layout_key is PA_BSND, non-contiguous input along axis 0 is supported.</li>
                <li>When layout_key is PA_BSND, the shape is [block_num, block_size, K_N, D], where block_num is the total number of blocks in the PageAttention scenario and block_size is the number of tokens per block. When layout_key is BSND, the shape is [B, K_S, K_N, D]. When layout_key is TND, the shape is [K_T, K_N, D].</li>
          </ul>
      </td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>weights</td>
      <td>Input</td>
      <td>
          <ul>
                <li>Input W in the formula.</li>
                <li>Empty tensors and non-contiguous tensors are not supported.</li>
                <li>When layout_query is BSND, the shape is [B, Q_S, Q_N]. When layout_query is TND, the shape is [Q_T, Q_N].</li>
          </ul>
      </td>
      <td>FLOAT16, BFLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>actual_seq_lengths_query</td>
      <td>Optional input</td>
      <td>
          <ul>
                <li>Number of valid tokens of Query in each batch.</li>
                <li>Empty tensors and non-contiguous tensors are not supported.</li>
                <li>None can be passed to indicate that the value is the same as the Q_S length of query.</li>
                <li>A 1D tensor with length B is supported. The number of valid tokens in each batch must not exceed the Q_S size of query and must not be less than 0. When layout_query is TND, this parameter must be passed, and the number of elements is used as the value of B.</li>
                <li>Each element indicates the total number of tokens in the current batch and all previous batches, that is, the prefix sum. Therefore, the value of each element must be greater than or equal to that of the previous element.</li>
                <li>The default value is None.</li>
          </ul>
      </td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>actual_seq_lengths_key</td>
      <td>Optional input</td>
      <td>
          <ul>
                <li>Number of valid tokens of Key in each batch.</li>
                <li>Empty tensors and non-contiguous tensors are not supported.</li>
                <li>None can be passed to indicate that the value is the same as the K_S length of key.</li>
                <li>A 1D tensor with length B is supported. The number of valid tokens in each batch must not exceed the K_S size of key and must not be less than 0.</li>
                <li>This parameter must be passed when layout_key is TND or PA_BSND. When layout_key is TND, each element indicates the total number of tokens in the current batch and all previous batches, that is, the prefix sum. Therefore, the value of each element must be greater than or equal to that of the previous element.</li>
                <li>The default value is None.</li>
          </ul>
      </td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>block_table</td>
      <td>Optional input</td>
      <td>
          <ul>
                <li>Represents the block mapping table used for KV storage in PageAttention.</li>
                <li>Empty tensors and non-contiguous tensors are not supported.</li>
                <li>In the PageAttention scenario, block_table must be 2D. The length of the first dimension must equal B, and the length of the second dimension must not be less than maxBlockNumPerSeq (the number of blocks corresponding to the largest K_S in each batch, that is, max_K_S/block_size).</li>
                <li>The shape [B, maxBlockNumPerSeq] is supported.</li>
                <li>The default value is None.</li>
          </ul>
      </td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>layout_query</td>
      <td>Optional attribute</td>
      <td>
          <ul>
                <li>Identifies the data layout format of the input Query.</li>
                <li>The default value is "BSND". Currently, BSND and TND are supported.</li>
          </ul>
      </td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layout_key</td>
      <td>Optional attribute</td>
      <td>
          <ul>
                <li>Identifies the data layout format of the input Key.</li>
                <li>The default value is "BSND". Currently, PA_BSND, BSND, and TND are supported.</li>
          </ul>
      </td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>sparse_count</td>
      <td>Optional attribute</td>
      <td>
          <ul>
                <li>Number of blocks to be retained in the topK phase.</li>
                <li>Values in [1, 2048] are supported, as well as 3072, 4096, 5120, 6144, 7168, and 8192.</li>
                <li>The default value is 2048.</li>
          </ul>
      </td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>sparse_mode</td>
      <td>Optional attribute</td>
      <td>
          <ul>
                <li>Represents the sparse mode.</li>
                <li>When sparse_mode is 0, the defaultMask mode is used.</li>
                <li>When sparse_mode is 3, the mask of the rightDownCausal mode is used, corresponding to the lower triangular scenario divided by the right vertex.</li>
                <li>The default value is 3.</li>
          </ul>
      </td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>pre_tokens</td>
      <td>Optional attribute</td>
      <td>Used for sparse computation. It indicates the number of previous tokens that attention needs to compute correlations with. Only the default value 2^63-1 is supported.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>next_tokens</td>
      <td>Optional attribute</td>
      <td>Used for sparse computation. It indicates the number of following tokens that attention needs to compute correlations with. Only the default value 2^63-1 is supported.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>return_values</td>
      <td>Optional attribute</td>
      <td>
          <ul>
                <li>Specifies whether to output sparse_values.</li>
                <li>True indicates output and False indicates no output. The default value is False. This parameter is supported only in training scenarios where layout_key is not PA_BSND.</li>
          </ul>
      </td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>sparse_indices</td>
      <td>Output</td>
      <td>
          <ul>
                <li>Indices output in the formula.</li>
                <li>Empty tensors and non-contiguous tensors are not supported.</li>
                <li>When layout_query is BSND, the output shape is [B, Q_S, K_N, sparse_count]. When layout_query is TND, the output shape is [Q_T, K_N, sparse_count].</li>
          </ul>
      </td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>sparse_values</td>
      <td>Output</td>
      <td>
          <ul>
                <li>Value corresponding to the Indices output in the formula.</li>
                <li>Empty tensors and non-contiguous tensors are not supported.</li>
                <li>The shape is the same as that of sparse_indices.</li>
          </ul>
      </td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody>
  </table>

## Constraints

- This API supports graph mode.
- K_N in key supports 1.
- headdim supports 128.
- block_size is a multiple of 16 and supports up to 1024.
- The data types of query and key should be consistent.
- When weights is not `float32`, the data types of query, key, and weights should be consistent.
- Ascend 950PR&950DT products:
  - Q_N of query supports only 8, 16, 24, 32, and 64.
  - weights does not support the `float32` type.
  - Only when layout_key is PA_BSND does key support non-contiguous input along axis 0.
- <term>Atlas A3 products</term> and <term>Atlas A2 products</term>:
  - Q_N of query supports values less than or equal to 64.
  - Non-contiguous key is not supported.
  - In the BSND layout, the Q_S/K_S sequence length must not exceed 8M (8388608). If the length exceeds this value, the operator reports an error indicating that the input is not supported. In the TND/PA layouts, the sequence length is invisible on the host side, so this check is not performed.

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
    <a href="./examples//test_aclnn_lightning_indexer.cpp">test_aclnn_lightning_indexer
    </a>
    </td>
    <td class="tg-lboi" rowspan="6">
    Invokes the operator through the
    <a href="./docs/aclnnLightningIndexer.md">aclnnLightningIndexer
    </a>
    API
    </td>
  </tr>
</tbody></table>
