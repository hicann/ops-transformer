# LightningIndexerV2

English | [简体中文](./README.md)

## Product Support

| Product                                                         | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                     |     √    |
| <term>Atlas A3 products</term>     |    √     |
| <term>Atlas A2 products</term>    |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                              |    ×     |
| <term>Atlas training products</term>                              |    ×     |

## Function Description

- Operator function: LightningIndexerV2 obtains the top-k positions corresponding to each token through a series of operations.

- Computation formula:

  $$
  Top-k \left\{  \left[ 1 \left] \mathop{{}}\nolimits_{{1 \times \text{ }g}}\text{@} \left[  \left( W\text{@} \left[ 1 \left] \mathop{{}}\nolimits_{{1\text{ } \times \text{ }S\mathop{{}}\nolimits_{{k}}}} \left) \text{ } \odot \text{ }ReLU \left( Q\mathop{{}}\nolimits_{{index}}\text{@}K\mathop{{}}\nolimits_{{T}}^{{index}} \left)  \left]  \right\} \right. \right. \right. \right. \right. \right. \right. \right. \right. \right.
  $$

- The main computation process is as follows:

  1. Multiply the input parameter `q` ($Q_{index}\in\R^{g\times d}$) corresponding to a token by the given context `k` ($K_{index}\in\R^{S_{k}\times d}$) to obtain the relevance.
  2. After filtering invalid negative relevance signals with the $ReLU$ activation function, obtain the relevance score vector between the current token and all preceding tokens.
  3. After multiplying it by the weight coefficient `w` ($W$), select the first $Top-k$ index values along the g dimension to obtain the output $sparse\_indices$, and output the corresponding $sparse\_values$ as the input of Attention.

- Interface layering: The operator supports the candidate (two-level TopK candidate source) mode. At the aclnn layer, two interfaces are provided: [aclnnLightningIndexerV2](docs/aclnnLightningIndexerV2_en.md) (candidate not supported) and [aclnnLightningIndexerV3](./docs/aclnnLightningIndexerV3.md) (candidate extension interface, which supports candidate-related attributes and outputs). At the torch layer, `lightning_indexer` provides unified access: when the optional parameter `candidate_topk_blocks` is passed, a 4-tuple is returned (V3 is used internally); when it is not passed, a 2-tuple is returned for compatibility (V2 is used internally).

## Parameter Description

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
    <td>q</td>
    <td>Input</td>
    <td>Input Q in the formula.</td>
    <td>BFLOAT16, FLOAT16</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>k</td>
    <td>Input</td>
    <td>Input K in the formula.</td>
    <td>BFLOAT16, FLOAT16</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>w</td>
    <td>Input</td>
    <td>Input W in the formula.</td>
    <td>FLOAT</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>cu_seqlens_q</td>
    <td>Optional Input</td>
    <td>When Q is TND, the value is the cumulative sum of the numbers of valid tokens in each batch; when Q is BSND, the value is [0, valid Q length of each sample], and each length does not exceed S1. The default value is None.</td>
    <td>INT32</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>cu_seqlens_k</td>
    <td>Optional Input</td>
    <td>When K is TND, the value is the cumulative sum of the numbers of valid tokens in each batch; when K is BSND, this parameter cannot be passed. The default value is None.</td>
    <td>INT32</td>
    <td>ND</td>
    </tr>
  <tr>
    <td>seqused_q</td>
    <td>Optional Input</td>
    <td>Actual used length of q in each batch. The default value is None.</td>
    <td>INT32</td>
    <td>ND</td>
    </tr>
  <tr>
    <td>seqused_k</td>
    <td>Optional Input</td>
    <td>Actual used length of k in each batch. The default value is None.</td>
    <td>INT32</td>
    <td>ND</td>
    </tr>
  <tr>
    <td>cmp_residual_k</td>
    <td>Optional Input</td>
    <td>Remainder of the number of tokens of k before compression divided by cmp_ratio. The default value is None.</td>
    <td>INT32</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>block_table</td>
    <td>Optional Input</td>
    <td>Block mapping table used for KV storage in PageAttention. The default value is None.</td>
    <td>INT32</td>
    <td>ND</td>
    </tr>
  <tr>
    <td>output_idx_offset</td>
    <td>Optional Input</td>
    <td>Offset to be added to the output indexes of the topK result. The default value is None.</td>
    <td>INT32</td>
    <td>ND</td>
    </tr>
  <tr>
  <td>metadata</td>
  <td>Optional Input</td>
  <td>Core allocation information passed by the LightningIndexerV2Metadata operator, including the number of cores used, the block size, and the start point of the data processed by each core. The default value is None.</td>
  <td>INT32</td>
  <td>ND</td>
  </tr>
  <tr>
  <td>topk</td>
  <td>Input</td>
  <td>Number of key token indexes to be retained in the topK stage.</td>
  <td>INT64</td>
  <td>-</td>
  </tr>
  <tr>
  <td>max_seqlen_q</td>
  <td>Optional Attribute</td>
  <td>Maximum sequence length of q. The default value is -1.</td>
  <td>INT64</td>
  <td>-</td>
  </tr>
  <tr>
  <td>layout_q</td>
  <td>Optional Attribute</td>
  <td>Identifies the data layout format of input q. The default value is "BSND".</td>
  <td>STRING</td>
  <td>-</td>
  </tr>
  <tr>
  <td>layout_k</td>
  <td>Optional Attribute</td>
  <td>Identifies the data layout format of input k. The default value is "BSND".</td>
  <td>STRING</td>
  <td>-</td>
  </tr>
  <tr>
  <td>mask_mode</td>
  <td>Optional Attribute</td>
  <td>Specifies the mask mode. The default value is 0.</td>
  <td>INT64</td>
  <td>-</td>
  </tr>
  <tr>
  <td>cmp_ratio</td>
  <td>Optional Attribute</td>
  <td>Used for sparse computation. It specifies the compression ratio of k. The default value is 1.</td>
  <td>INT64</td>
  <td>-</td>
  </tr>
  <tr>
  <td>return_value</td>
  <td>Optional Attribute</td>
  <td>Specifies whether to return the values corresponding to Indices. The default value is 0.</td>
  <td>INT64</td>
  <td>-</td>
  </tr>
  <tr>
  <td>candidate_topk_blocks</td>
  <td>Attribute</td>
  <td>Switch and output width of the two-level TopK for candidate blocks: -1 (default) disables the mode, and the output is exactly the same as that of single-level TopK; when the value is in (0, 2048] and is a multiple of 64, the candidate source mode is enabled, and the first candidate_topk_blocks candidate block indexes with the highest scores are output for each query row.</td>
  <td>INT64</td>
  <td>-</td>
  </tr>
  <tr>
  <td>candidate_block_size</td>
  <td>Attribute</td>
  <td>Candidate block size (number of positions), in [2, 64] and a power of 2, with a default value of 8. The value must be the same on the source and consumer sides (sparse_lightning_indexer).</td>
  <td>INT64</td>
  <td>-</td>
  </tr>
  <tr>
  <td>sparse_indices</td>
  <td>Output</td>
  <td>Indices output in the formula.</td>
  <td>INT32</td>
  <td>ND</td>
  </tr>
  <tr>
  <td>sparse_values</td>
  <td>Output</td>
  <td>Values output corresponding to Indices in the formula.</td>
  <td>FLOAT</td>
  <td>ND</td>
  </tr>
  <tr>
  <td>candidate_block_indices</td>
  <td>Output</td>
  <td>Candidate block indexes output in candidate source mode: [B, S1, N2, candidateTopkBlocks] for BSND and [T, N2, candidateTopkBlocks] for TND; the value is the relative block number in the post-compression position domain within a batch (block number × candidateBlockSize is the start position) or -1 (invalid slot), and slots are unordered; outputIdxOffset is not added. When candidateTopkBlocks = -1, an empty tensor (0,) is output.</td>
  <td>INT32</td>
  <td>ND</td>
  </tr>
  <tr>
  <td>candidate_block_length</td>
  <td>Output</td>
  <td>Reserved interface. Currently, an empty tensor (0,) is always output.</td>
  <td>INT32</td>
  <td>ND</td>
  </tr>
  </tbody>
   </table>

  - Dimension meanings of parameters q, k, and w: B (Batch Size) indicates the batch size of input samples; S (Sequence Length) indicates the sequence length of input samples; H (Head Size) indicates the size of the hidden layer; N (Head Num) indicates the number of heads; D (Head Dim) indicates the size of the smallest unit of the hidden layer and satisfies D = H/N; and T indicates the cumulative sum of the sequence lengths of input samples in all batches.
  - S1 and S2 denote the sequence lengths of the input samples of q and k respectively, N1 and N2 denote the numbers of heads corresponding to q and k respectively, and k denotes the number of indexes finally selected. D in parameter q and D in parameter k are equal, both being 128. T1 and T2 denote the cumulative sums of the sequence lengths of the input samples of q and k respectively.

## Constraints

- N of parameter q supports 1~64, and N of k supports 1.
- headdim supports 128.
- pa_kv_cache supports non-contiguity on axis 0; pa_block_size supports [16, 1024] and must be a multiple of 16.
- The data types of parameters q and k must be the same.
- The invalid part of sparse_indices is filled with -1; the invalid part of sparse_values is filled with -inf.
- candidate (two-level TopK candidate source. Only Atlas A2 products and Atlas A3 products support enabling it; enabling it on Ascend 950PR&950DT products is rejected by host-side tiling validation and returns an error. The operator itself is still supported on 950, and the candidate capability is provided by the aclnnLightningIndexerV3 interface):
  - When candidate_topk_blocks != -1 (source mode enabled), topk must be <= 2048; return_value is now unrestricted (value range ∈ {0,1}). When rv=1, sparse_values outputs the real position-level topk scores (the shape is the same as that of sparse_indices, and the invalid part is filled with -inf); when rv=0, sparse_values is always an empty tensor (0,).
  - The value of candidate_topk_blocks is in (0, 2048] and is a multiple of 64; the invalid part of candidate_block_indices is filled with -1, and the value is a relative block number (within a batch, in the post-compression position domain).
  - candidate_block_size must be a power of 2 within [2, 64] (this value range is also validated when candidate is disabled); it must be the same as candidate_block_size on the consumer side (sparse_lightning_indexer). This consistency is a caller constraint, and the framework layer cannot validate it across operators (the hosts on both sides validate only the value ranges; the behavior of a mismatch is undefined).
  - candidate_block_length is a reserved interface and is always an empty tensor.
- The value of each element in the passed cmp_residual_k must be less than the passed compression ratio cmp_ratio.
- Deterministic computation and batch consistency:
  - A deterministic implementation is used by default, and multiple calls with the same input produce the same result.
  - <term>Atlas A3 products</term> and <term>Atlas A2 products</term>: Batch consistency: set ACL_OPT_DETERMINISTIC to 3 by calling aclrtSetSysParamOpt() to enable batch consistency. After it is enabled, the computation result is independent of the batch size and position. You can also set it by calling `torch_npu.npu.set_deterministic_level(3)`.
- <term>Atlas A3 products</term> and <term>Atlas A2 products</term>:
  - Currently, the value range of topk only supports [1, 2048], as well as 3072, 4096, 5120, 6144, 7168, and 8192.
  - Currently, the seqused_q, output_idx_offset, and max_seqlen_q functions are not supported. Passing these parameters is not recommended.
  - When Q is BSND, cu_seqlens_q is optional. If it is not passed, S1 is used. If it is passed, the format is [0, valid Q length of each sample...], and each length does not exceed S1.
  - When Q is TND, cu_seqlens_q must be passed. Cumulative lengths are used, and the difference between adjacent values is the sample length.
  - When K is BSND, do not pass cu_seqlens_k. seqused_k is optional. If it is not passed, S2 is used. If it is passed, the valid K length of each sample is used.
  - When K is TND, cu_seqlens_k must be passed. Cumulative lengths are used, and the difference between adjacent values is the sample length.
  - When the passed parameter layout_k is PA_BBND, seqused_k must be passed.
- <term>Ascend 950PR&950DT products</term>:
  - When the passed parameter layout_q is BSND, cu_seqlens_q cannot be passed; when layout_k is BSND or PA_BBND, cu_seqlens_k cannot be passed.
  - When output_idx_offset is passed, an index offset value greater than or equal to 0 is supported; the following constraint must be satisfied: after the passed index offset value is added, the obtained sparse_indices value does not exceed the maximum value of INT32.
  - When the passed parameter layout_q is TND, cu_seqlens_q must be passed. If seqused_q is also passed, ensure that the q length of each batch passed by seqused_q does not exceed the q sequence length of each batch calculated based on cu_seqlens_q. When the q sequence length seqlen1 of a batch passed by seqused_q is less than the q length seqlen2 calculated based on cu_seqlens_q, the TND Padding function is enabled, and the sparse_indices and sparse_values outputs of q for the difference of seqlen2 - seqlen1 of the batch are all set to invalid values. In some long sequence scenarios, if too much invalid data needs to be padded, the aicore execution may time out due to hardware limitations. You can calculate the amount of data to be padded by using (seqlen2 - seqlen1) * topk. It is recommended to keep this data volume within 400 million.
  - When the passed cmp_ratio > 1 and mask_mode = 3, cmp_residual_k must be passed; in other cases, it is not passed.
  - When layout_k is PA_BBND, seqused_k must be passed.
  - The parameter metadata must be passed, and its shape is fixed to [2048].

## Invocation Description

| Invocation Method | Invocation Sample                                                                   | Description                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn API | [test_aclnn_lightning_indexer_v2](./examples/test_aclnn_lightning_indexer_v2.cpp) | Call the LightningIndexerV2 operator through the [aclnnLightningIndexerV2](docs/aclnnLightningIndexerV2_en.md) interface. |
| PyTorch API | [torchapi_lightning_indexer](docs/torchapi_lightning_indexer_en.md) | Call the LightningIndexerV2 operator through the [cann_ops_transformer.lightning_indexer](docs/torchapi_lightning_indexer_en.md) interface (including lightning_indexer_metadata; pass the optional parameter candidate_topk_blocks to enable the candidate mode).|
| aclnn API (candidate) | [test_aclnn_lightning_indexer_v3](./examples/test_aclnn_lightning_indexer_v3.cpp) | Call the candidate (two-level TopK source) mode of the LightningIndexerV2 operator through the [aclnnLightningIndexerV3](./docs/aclnnLightningIndexerV3.md) interface.|
