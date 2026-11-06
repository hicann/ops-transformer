# QuantLightningIndexer

English | [简体中文](./README.md)

## Product Support Status

|Product      | Support |
|:----------------------------|:-----------:|
|<term>Ascend 950PR&950DT products</term>|      √     |
|<term>Atlas A3 products</term>|      √     |
|<term>Atlas A2 products</term>|      √     |
|<term>Atlas 200I/500 A2 inference products</term>|      ×     |
|<term>Atlas inference products</term>|      ×     |
|<term>Atlas training products</term>|      ×     |

## Function Description

- Operator function: In inference scenarios, QuantLightningIndexer performs the preprocessing computation of SparseFlashAttention (SFA). It selects key sparse tokens and quantizes the input query and key to implement 8-bit storage and 8-bit computation, achieving maximum benefit.

- Computation formula:
    $$out = \text{Top-}k\left\{[1]_{1\times g}@\left[(W@[1]_{1\times S_{k}})\odot\text{ReLU}\left(\left(Scale_Q@Scale_K^T\right)\odot\left(Q_{index}^{Quant}@{\left(K_{index}^{Quant}\right)}^T\right)\right)\right]\right\}$$
    The main computation process is:
    1. Multiply the input parameter `query` of a token ($Q_{index}^{Quant}\in\R^{g\times d}$) by the given context `key` ($K_{index}^{Quant}\in\R^{S_{k}\times d}$) to obtain the relevance.
    2. Multiply the relevance result by the dequantization coefficients `query_dequant_scale` ($Scale_Q$) and `key_dequant_scale` ($Scale_K^T$) corresponding to `query` and `key`, filter invalid negative relevance signals through the activation function $ReLU$, and obtain the relevance score vector between the current token and all preceding tokens.
    3. After multiplying it by the weight coefficient `weights` ($W$), select the first $Top-k$ index values along the g direction to obtain the output $out$, which serves as the input of SparseFlashAttention.

## Parameter Description

> **Description:**<br>
> Parameter dimension meanings: B denotes the Batch Size, Q_S and K_S denote the Sequence Length of query and key respectively, Q_N and K_N denote the Head Num of query and key respectively, D denotes the Head Dim (Q_D and K_D are equal, both 128), Q_T and K_T denote the Total Tokens of query and key respectively, sparse_count denotes the number of selected indices (topK), and block_num and block_size denote the total number of blocks and the number of tokens per block in the PageAttention scenario. K_N supports only 1.
<table style="undefined;table-layout: fixed; width: 1576px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 300px">
  <col style="width: 200px">
  <col style="width: 170px">
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
                <li>Non-contiguous is not supported.</li>
                <li>When layout_query is BSND, the shape is [B, Q_S, Q_N, D]. When layout_query is TND, the shape is [Q_T, Q_N, D].</li>
                <li>Q_N supports [1, 64].</li>
          </ul>
      </td>
      <td>INT8, FLOAT8_E4M3, HIFLOAT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>key</td>
      <td>Input</td>
      <td>
          <ul>
                <li>Input K in the formula.</li>
                <li>When layout_key is PA_BSND, non-contiguity on axis 0 is supported.</li>
                <li>When layout_key is PA_BSND, the shape is [block_num, block_size, K_N, D]. When layout_key is BSND, the shape is [B, K_S, K_N, D]. When layout_key is TND, the shape is [K_T, K_N, D].</li>
                <li>block_num is the total number of blocks in PageAttention, and block_size is the number of tokens in one block.</li>
                <li>K_N supports only 1.</li>
          </ul>
      </td>
      <td>INT8, FLOAT8_E4M3, HIFLOAT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>weights</td>
      <td>Input</td>
      <td>
          <ul>
                <li>Input W in the formula.</li>
                <li>Non-contiguous is not supported.</li>
                <li>When layout_query is BSND, the shape is [B, Q_S, Q_N]. When layout_query is TND, the shape is [Q_T, Q_N].</li>
          </ul>
      </td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>query_dequant_scale</td>
      <td>Input</td>
      <td>
          <ul>
                <li>Dequantization coefficient Scale_Q of Query in the formula.</li>
                <li>Non-contiguous is not supported.</li>
                <li>When layout_query is BSND, the shape is [B, Q_S, Q_N]. When layout_query is TND, the shape is [Q_T, Q_N].</li>
          </ul>
      </td>
      <td>FLOAT, FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>key_dequant_scale</td>
      <td>Input</td>
      <td>
          <ul>
                <li>Dequantization coefficient Scale_K of Key in the formula.</li>
                <li>When layout_key is PA_BSND, non-contiguity on axis 0 is supported.</li>
                <li>When layout_key is BSND, the shape is [B, K_S, K_N]. When layout_key is TND, the shape is [K_T, K_N].</li>
                <li>When layout_key is PA_BSND, the shape is [block_num, block_size, K_N].</li>
                <li>block_num is the total number of blocks in PageAttention, and block_size is the number of tokens in one block.</li>
          </ul>
      </td>
      <td>FLOAT, FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>actual_seq_lengths_query</td>
      <td>Optional input</td>
      <td>
          <ul>
                <li>Number of valid tokens of Query in each batch.</li>
                <li>Non-contiguous is not supported.</li>
                <li>The shape is [B,]</li>
                <li>If seqlen is not specified, None can be passed in, indicating the same length as Q_S of the query shape.</li>
                <li>The number of valid tokens of each batch in this input does not exceed the Q_S size of query and is not less than 0. A 1-dimensional tensor of length B is supported.</li>
                <li>When layout_query is TND, this input must be passed in, the number of elements of this input is used as the value of B, and the value of each element in this input indicates the total number of tokens of the current batch and all preceding batches, that is, a prefix sum. Therefore, the value of each element must be greater than or equal to that of the previous element.</li>
                <li>Negative values cannot occur.</li>
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
                <li>Non-contiguous is not supported.</li>
                <li>The shape is [B,]</li>
                <li>If seqlen is not specified, None can be passed in, indicating the same length as K_S of the key shape.</li>
                <li>The number of valid tokens of each batch in this parameter does not exceed the K_S size of key and is not less than 0. A 1-dimensional tensor of length B is supported.</li>
                <li>When layout_key is TND or PA_BSND, this input must be passed in. When layout_key is TND, the value of each element in this parameter indicates the total number of tokens of the current batch and all preceding batches, that is, a prefix sum. Therefore, the value of each element must be greater than or equal to that of the previous element.</li>
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
                <li>Non-contiguous is not supported.</li>
                <li>The shape supports [B, K_S_max/block_size]</li>
                <li>In the PageAttention scenario, block_table must be two-dimensional. The length of the first dimension must equal B, and the length of the second dimension cannot be less than maxBlockNumPerSeq (maxBlockNumPerSeq is the number of blocks corresponding to the maximum actual_seq_lengths_key in each batch)</li>
                <li>block_size is a multiple of 16 and supports up to 1024.</li>
                <li>The default value is None.</li>
          </ul>
      </td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>query_quant_mode</td>
      <td>Attribute</td>
      <td>
          <ul>
                <li>Identifies the quantization mode of input Query.</li>
                <li>Currently, the Per-Token-Head quantization mode is supported.</li>
                <li>Currently, only 0 can be passed in.</li>
          </ul>
      </td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>key_quant_mode</td>
      <td>Attribute</td>
      <td>
          <ul>
                <li>Identifies the quantization mode of input Key.</li>
                <li>Currently, the Per-Token-Head quantization mode is supported.</li>
                <li>Currently, only 0 can be passed in.</li>
          </ul>
      </td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layout_query</td>
      <td>Optional attribute</td>
      <td>
          <ul>
                <li>Identifies the data layout format of input Query.</li>
                <li>Currently, BSND and TND are supported.</li>
                <li>The default value is BSND.</li>
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
                <li>Identifies the data layout format of input Key.</li>
                <li>Currently, PA_BSND, BSND, and TND are supported.</li>
                <li>In non-PageAttention scenarios, layout_key must be consistent with layout_query.</li>
                <li>The default value is BSND.</li>
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
                <li>Number of blocks to retain in the topK stage. Supports [1, 2048].</li>
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
                <li>When sparse_mode is 0, it represents the defaultMask mode.</li>
                <li>When sparse_mode is 3, it represents the mask of the rightDownCausal mode, corresponding to the lower-triangular scenario partitioned by the right vertex.</li>
                <li>The default value is 3.</li>
          </ul>
      </td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>pre_tokens</td>
      <td>Optional attribute</td>
      <td>Used for sparse computation; indicates the number of preceding tokens that attention needs to compute correlation with. Only the default value 2^63-1 is supported.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>next_tokens</td>
      <td>Optional attribute</td>
      <td>Used for sparse computation; indicates the number of following tokens that attention needs to compute correlation with. Only the default value 2^63-1 is supported.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>key_stride0</td>
      <td>Optional attribute</td>
      <td>
          <ul>
                <li>Indicates that key obtains the stride information of dimension 0.</li>
                <li>The default value is -1.</li>
          </ul>
      </td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>key_dequant_scale_stride0</td>
      <td>Optional attribute</td>
      <td>
          <ul>
                <li>Indicates that key_dequant_scale obtains the stride information of dimension 0.</li>
                <li>The default value is -1.</li>
          </ul>
      </td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>sparse_indices</td>
      <td>Output</td>
      <td>
          <ul>
                <li>Indices output in the formula.</li>
                <li>When layout_query is BSND, the output shape is [B, Q_S, K_N, sparse_count]. When layout_query is TND, the output shape is [Q_T, K_N, sparse_count].</li>
          </ul>
      </td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
  </tbody>

  </table>

Atlas A3 training products/Atlas A3 inference products:

  - The data types of query and key support `INT8`.
  - Only data types of `FLOAT16, FLOAT16, FLOAT16` are supported for weights, query_dequant_scale, and key_dequant_scale.
  - When layout_key is PA_BSND, key and key_dequant_scale support non-contiguity on axis 0, and the other axes must be contiguous.

Ascend 950PR&950DT products:

  - query Q_N supports only 8, 16, 24, 32, and 64.
  - The data types of query and key support `FLOAT8_E4M3, HIFLOAT8, INT8`.
  - When the data types of query and key are `FLOAT8_E4M3`, data types of `BFLOAT16, FLOAT, FLOAT` or `FLOAT16, FLOAT16, FLOAT16` are supported for weights, query_dequant_scale, and key_dequant_scale;
  - When the data types of query and key are `HIFLOAT8`, only data types of `BFLOAT16, FLOAT, FLOAT` are supported for weights, query_dequant_scale, and key_dequant_scale;
  - When the data types of query and key are `INT8`, only data types of `FLOAT16, FLOAT16, FLOAT16` are supported for weights, query_dequant_scale, and key_dequant_scale.

## Constraint Description

- This interface supports graph mode.
- This interface requires the result of $W \odot Scale_Q$ to be within the representation range of `float16`.
- Sorting NAN in the TopK process of this interface is undefined behavior.

## Invocation Example

<table class="tg"><thead>
  <tr>
    <th class="tg-0pky">Invocation Method</th>
    <th class="tg-0pky">Sample Code</th>
    <th class="tg-0pky">Description</th>
  </tr></thead>
<tbody>
  <tr>
    <td class="tg-9wq8" rowspan="6">Graph mode</td>
    <td class="tg-0pky">
    <a href="./examples/test_npu_quant_lightning_indexer.py">test_npu_quant_lightning_indexer
    </a>
    </td>
    <td class="tg-lboi" rowspan="6">
    Call the npu_quant_lightning_indexer operator by building the graph through the operator IR
    </td>
  </tr>
</tbody>
<tbody>
  <tr>
    <td class="tg-9wq8" rowspan="6">aclnn interface</td>
    <td class="tg-0pky">
    <a href="./examples/test_aclnn_quant_lightning_indexer.cpp">test_aclnn_quant_lightning_indexer
    </a>
    </td>
    <td class="tg-lboi" rowspan="6">
    Call the operator through the
    <a href="./docs/aclnnQuantLightningIndexer.md">aclnnQuantLightningIndexer
    </a>
    interface
    </td>
  </tr>
</tbody></table>
