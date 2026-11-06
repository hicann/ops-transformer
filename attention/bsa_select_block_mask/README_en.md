# BSASelectBlockMask

English | [简体中文](./README.md)

## Product Support

| Product                                       | Supported |
| :----------------------------------------- | :------:|
| <term>Ascend 950PR&950DT products</term>     |    √    |
| <term>Atlas A3 training products</term>           |    √    |
| <term>Atlas A3 inference products</term>           |    √    |
| <term>Atlas A2 training products</term>           |    √    |
| <term>Atlas A2 inference products</term>           |    √    |
| <term>Atlas 200I/500 A2 inference products</term>      |    ×    |
| <term>Atlas inference products</term>              |    ×    |
| <term>Atlas training products</term>              |    ×    |

## Function Description

- Operator function: aclnnBSASelectBlockMask is a preceding operator of BSA (BlockSparseAttention). It dynamically generates blockSparseMask based on the content of query and key, changing the BSA call chain from "manually providing the mask" to "adaptively selecting the sparsity pattern based on the Q/K content".
- Computation formula:

  Assume blockShape = [blockShapeX, blockShapeY], Sq is the maximum sequence length of query, and Skv is the maximum sequence length of key. The number of blocks after compression is:

  $$
  Xblocks = \lceil Sq / blockShapeX \rceil,\quad Yblocks = \lceil Skv / blockShapeY \rceil
  $$

  **Step1: Mean Pooling Compression**

  When actualBlockLenQuery / actualBlockLenKey is null (full compression):

  $$
  q\_compressed[b, n, x, d] = \frac{1}{blockShapeX} \sum_{i=0}^{blockShapeX-1} query[b, n, x \cdot blockShapeX + i, d]
  $$

  $$
  k\_compressed[b, n, y, d] = \frac{1}{blockShapeY} \sum_{j=0}^{blockShapeY-1} key[b, n, y \cdot blockShapeY + j, d]
  $$

  When actualBlockLenQuery / actualBlockLenKey is not null (partial compression), the mean is computed only over the first actualBlockLen tokens in each block:

  $$
  q\_compressed[b, n, x, d] = \frac{1}{actualBlockLenQ[b,x]} \sum_{i=0}^{actualBlockLenQ[b,x]-1} query[b, n, x \cdot blockShapeX + i, d]
  $$

  $$
  k\_compressed[b, n, y, d] = \frac{1}{actualBlockLenK[b,y]} \sum_{j=0}^{actualBlockLenK[b,y]-1} key[b, n, y \cdot blockShapeY + j, d]
  $$

  **Step2a: QK Matmul**

  $$
  score[b, n, x, y] = scale \cdot \sum_{d=0}^{D-1} q\_compressed[b, n, x, d] \cdot k\_compressed[b, n, y, d]
  $$

  **Step2b: Softmax**

  $$
  attn\_score[b, n, x, y] = softmax(score[b, n, x, :]) = \frac{\exp(score[b, n, x, y] - m_{final})}{l_{final}}
  $$

  **Step2c: Post-Softmax Mean Pooling**

  When postBlockShape is not null, a second round of mean pooling is performed on attn_score to generate the coarse-grained pooled_score:

  $$
  postXBlocks = \lceil Xblocks / postBlockShapeX \rceil,\quad postYBlocks = \lceil Yblocks / postBlockShapeY \rceil
  $$

  $$
  pooled\_score[b, n, px, py] = \frac{1}{|R_{px,py}|} \sum_{x \in R_x(px)} \sum_{y \in R_y(py)} attn\_score[b, n, x, y]
  $$

  When postBlockShape is null, this step is skipped, and pooled_score = attn_score.

  **Step3: TopK Selection to Generate Indices**

  $$
  topk\_value = \text{round}(sparsity \times postXBlocks \times postYBlocks)
  $$

  $$
  \mathcal{indices}= \text{TopK}\left(pooled\_score[b, n, px, py],\; topK\_value\right)
  $$

  When postBlockShape is null, postXBlocks=Xblocks, postYBlocks=Yblocks, and pooled_score=attn_score, which is equivalent to directly performing TopK on attn_score.

  Here indices is the set of indices corresponding to the topk\_value largest values in pooled\_score[b, n, px, py].

  **Step4: Generate BlockSparseMask**

  When postBlockShape is not null, the output is directly the **coarse-grained** mask (the shape is [B, N, postXBlocks, postYBlocks]; the second-round pooling splitting semantics apply, and no expansion is needed):

  $$
  blockSparseMaskOut[b, n, px, py] =
  \begin{cases}
  1 & (px, py) \in \mathcal{indices} \\
  0 & (px, py) \notin \mathcal{indices}
  \end{cases}
  $$

  When postBlockShape is null, the output is the fine-grained mask (the shape is [B, N, Xblocks, Yblocks]), which is generated element by element directly:

  $$
  blockSparseMaskOut[b, n, x, y] =
  \begin{cases}
  1 & (b, n, x, y) \in \mathcal{indices} \\
  0 & (b, n, x, y) \notin \mathcal{indices}
  \end{cases}
  $$

- Data layout format:

  BSASelectBlockMask supports multiple interpretations of the data layout of the input query and key, which can be passed in through qInputLayout and kvInputLayout. To facilitate understanding of the specific layout formats supported later (such as BNSD and TND), this section first provides a unified description of the dimension meaning of each abbreviation letter in the layout formats:

  - B: Batch size of the input samples (Batch)
  - T: Length when B and S are tightly arranged on the same axis (Total tokens)
  - S: Sequence length of the input samples (Seq-Length)
  - H: Hidden layer size (Head-Size)
  - N: Number of heads (Head-Num)
  - D: Size of the smallest unit of the hidden layer, which must satisfy D = H / N (Head-Dim)

- Currently supported layouts:

  - qInputLayout: "TND" "BNSD"
  - kvInputLayout: "TND" "BNSD"

## Parameter Description

| Parameter | Input/Output/Attribute | Description | Data Type | Data Format |
|-----|-----------|------|----------|----------|
| query | Input | Query matrix in the attention computation, that is, `query` in the formula. | FLOAT16, BFLOAT16 | ND |
| key | Input | Key matrix in the attention computation, that is, `key` in the formula. | The data type is the same as that of query | ND |
| block_shape | Input | Array of sparse block shapes. It specifies the two-dimensional size (number of rows and columns) of each sparse block, that is, `blockShape` in the formula. It must contain at least two elements `[blockShapeX, blockShapeY]`: blockShapeX is the block size along the Q direction, and blockShapeY is the block size along the KV direction. If this parameter is not configured (nullptr is passed in), the operator uses blockShapeX = 128 and blockShapeY = 128 by default. | INT64 | - |
| post_block_shape | Input | Array of second-round pooling block shapes. It specifies the block grouping size of the second-round pooling after Softmax (in blocks), that is, `postBlockShape` in the formula. Optional input: when `[postBlockShapeX, postBlockShapeY]` is passed in, the second-round pooling is enabled, TopK selects on the coarse-grained pooled_score, and the output mask is the coarse-grained [B, N, postXBlocks, postYBlocks] (the second-round pooling splitting semantics apply; the selection result is output directly without expansion). When nullptr is passed in, the second-round pooling is disabled (backward compatibility), TopK selects directly on attn_score, and the output mask is the fine-grained [B, N, Xblocks, Yblocks]. postBlockShapeX/postBlockShapeY are multiples of 8, indicating the number of block groups. | INT64 | - |
| actual_seq_lengths | Input | Actual sequence length of query for each batch, that is, the actual valid value of Sq in the formula for each batch. It describes the number of actually valid query tokens in each batch in variable-length sequence scenarios (that is, scenarios containing Padding data). | INT64 | - |
| actual_seq_lengths_kv | Input | Actual sequence length of key, that is, the actual valid value of Skv in the formula for each batch. It describes the number of actually valid key tokens in each batch in variable-length sequence scenarios (that is, scenarios containing Padding data). | INT64 | - |
| actual_block_len_query | Input | Actual valid sequence length compressed within each query block, that is, `actualBlockLenQuery` in the formula. It is used for partial compression scenarios (such as incomplete blocks at the end or compressing only valid tokens). The mean is computed only over the first actual_block_len_query tokens in each block. If this parameter is not configured (nullptr is passed in), query is fully compressed. | INT64 | - |
| actual_block_len_key | Input | Actual valid sequence length compressed within each key block, that is, `actualBlockLenKey` in the formula. It is used for partial compression scenarios (such as incomplete blocks at the end or compressing only valid tokens). The mean is computed only over the first actual_block_len_key tokens in each block. If this parameter is not configured (nullptr is passed in), key is fully compressed. | INT64 | - |
| q_input_layout | Input | Data layout format of query. It indicates the specific arrangement of the input tensors in memory. Currently only "TND" and "BNSD" are supported, and it must be consistent with kv_input_layout. | String | - |
| kv_input_layout | Input | Data layout format of key. It indicates the specific arrangement of the input tensors in memory. Currently only "TND" and "BNSD" are supported, and it must be consistent with q_input_layout. | String | - |
| num_key_value_heads | Input | Number of attention heads of key, that is, the number of heads N of the key tensor in the formula. Currently only MHA is supported, and it must be consistent with the number of heads of query. | Int | - |
| scale_value | Input | Scaling factor, that is, `scale` in the formula. It is used to normalize the attention scores and is generally set to 1 / sqrt(D). | Float | - |
| sparsity | Input | Sparsity retention ratio, that is, `sparsity` in the formula. It specifies the proportion of block positions to be retained in attn_score out of all block positions. The value range is (0.0, 1.0). | Float | - |
| block_sparse_mask_out | Output | Block sparse mask output, that is, `blockSparseMaskOut` in the formula. It indicates the sparsity pattern adaptively generated based on the Q/K content, and can be directly used as the blockSparseMask input of the BSA operator. The shape changes with post_block_shape: when post_block_shape is not null, the shape is [B, N, postXBlocks, postYBlocks] (coarse-grained), where postXBlocks = ceilDiv(Xblocks, postBlockShapeX) and postYBlocks = ceilDiv(Yblocks, postBlockShapeY); when post_block_shape is null, the shape is [B, N, Xblocks, Yblocks] (fine-grained). The value 1 indicates that the block participates in the attention computation, and the value 0 indicates that it does not. | INT8 | ND |

## Constraints

- actual_seq_lengths is required when q_input_layout is "TND"; actual_seq_lengths_kv is required when kv_input_layout is "TND".
- The size of the head dimension in the query tensor shape is denoted as N1, and the size of the head dimension in the key tensor shape is denoted as N2. N1 = N2 must be satisfied (only MHA is supported).
- headDim = 128.
- blockShapeX and blockShapeY in block_shape must be multiples of 8.
- After query and key are compressed, the corresponding Xblocks and Yblocks of query and key must satisfy Xblocks * Yblocks > 1. When post_block_shape is not null, the BNSD scenario further requires postXBlocks * postYBlocks > 1 (when the coarse-grained grid collapses to 1×1, there is no TopK selection semantics, and the host side rejects the input; the TND variable-length scenario allows it).
- The data types of query and key must be consistent, and only FLOAT16 and BFLOAT16 are supported.
- The data type of block_sparse_mask_out is INT8 (binary: 0 or 1).
- post_block_shape is an optional input. When nullptr is passed in, the second-round pooling is disabled (backward compatibility). When `[postBlockShapeX, postBlockShapeY]` is passed in, the second-round pooling is enabled, and postBlockShapeX/postBlockShapeY are multiples of 8, indicating the number of block groups.
- If actual_block_len_query / actual_block_len_key is not null, the value range of each element is [0, blockShapeX] / [0, blockShapeY]. If it is null, full compression is performed.

## Calling Instructions

| Calling Method | Calling Sample | Description |
|---------|---------|------|
| aclnn API | [test_aclnn_bsa_select_block_mask](examples/test_aclnn_bsa_select_block_mask.cpp) | Only MHA and scenarios with the TND\BNSD format are supported. The BSASelectBlockMask operator is called through the [aclnnBSASelectBlockMask](docs/aclnnBSASelectBlockMask_en.md) API. |
