# pool\_key\_indexer

## Applicable Products

<!-- npu="950" id1 -->

- <term>Ascend 950PR&950DT products</term>: Supported

<!-- end id1 -->

<!-- npu="A3" id2 -->

- <term>Atlas A3 products</term>: Supported

<!-- end id2 -->

<!-- npu="910b" id3 -->

- <term>Atlas A2 products</term>: Supported

<!-- end id3 -->

<!-- npu="310b" id4 -->

- <term>Atlas 200I/500 A2 inference products</term>: Not Supported

<!-- end id4 -->

<!-- npu="310p" id5 -->

- <term>Atlas inference products</term>: Not Supported

<!-- end id5 -->

<!-- npu="910" id6 -->

- <term>Atlas training products</term>: Not Supported

<!-- end id6 -->

## Function

- **Description**:

  The `pool_key_indexer` API obtains the top-k positions for each token based on a series of operations. Unlike `lightning_indexer`, `pool_key_indexer` introduces a **Pooling** mechanism on the Key side, where the original Key sequence is aggregated in groups of `pool_size` tokens (each pool acts as one token). It first selects TopK at the pool level, then expands the selected pools back to original token indices.

  The main computation process is:

  1. Multiply the input parameter `query` ($Q_{index}\in\R^{g\times d}$) corresponding to a token by the pooled context `pool_key` ($K_{index}^{Pool}\in\R^{S_{k}^{pool}\times d}$), where $S_{k}^{pool} = \lfloor S_{k} / pool\_size \rfloor$, and then multiply by the scaling factor $\frac{1}{\sqrt{d}}$ (to prevent the dot product from becoming too large) to obtain the relevance score.
  2. After filtering out invalid negative correlation signals through the activation function $ReLU$, obtain the pool-level relevance score vector between the current token and all preceding tokens.
  3. Multiply it by the weight coefficient `weights` ($W$) and aggregate along the g direction.
  4. Select the top $Top\text{-}k / pool\_size$ pool indices in the pool space, and expand them to $Top\text{-}k$ token indices.
  5. Append valid tokens from the tail pool that are not fully covered by `pool_size` (specified by `pool_tail_k`), and fill invalid positions with -1.
- **Output layout (each row of sparse_indices, compact layout)**: `[0, expandTotal)` contains the valid expanded token indices (expandTotal = min(number of valid pools, topk/pool_size)×pool_size), `[expandTotal, expandTotal+visibleTailK)` contains the appended tail tokens (immediately following the valid expansion, without -1 gaps), and all remaining invalid positions -1 are located at the end of the row. For long sequences (number of valid pools >= topk/pool_size), expandTotal = topk, and the tail tokens start exactly at the offset topk.
- **Formula**:

  $$
  Top\text{-}k \leftarrow Expand\left(Top\text{-}\frac{k}{p}\left\{[1]_{1\times g}@\left[\left(W@[1]_{1\times S_{k}^{pool}}\right)\odot ReLU\left(\left(Q_{index}@{K_{index}^{Pool}}^T\right)\odot\frac{1}{\sqrt{d}}\right)\right]\right\}\right) + AppendTail
  $$

  where $p = pool\_size$, $d$ is the attention head dimension (headDim), and the scaling factor $\frac{1}{\sqrt{d}}$ prevents the dot product from becoming too large. $Expand$ expands each pool index to $p$ consecutive token indices, and $AppendTail$ appends `pool_tail_k` valid tokens from the tail pool.

## Prototype

```python
cann_ops_transformer.pool_key_indexer(
  query,
  pool_key,
  weights,
  pool_tail_k,
  *,
  actual_seq_q=None,
  actual_seq_k=None,
  block_table=None,
  q_descale=None,
  k_descale=None,
  layout_q="BSND",
  layout_k="BSND",
  topk=128,
  pool_size=1,
  mask_mode=0,
  quant_mode=-1,
  return_value=False
) -> (Tensor, Tensor)
```

## Parameters

> **Note:**
>
> - Dimension meanings of query, pool_key, and weights parameters: B (Batch Size) indicates the input sample batch size, S1 indicates the query input sample sequence length, S2 indicates the pool_key input sample sequence length (pool level), N1 indicates the number of query heads, N2 indicates the number of pool_key heads, D (Head Dim) indicates the attention head dimension, T1 indicates the cumulative sum of query input sample sequence lengths, and T2 indicates the cumulative sum of pool_key input sample sequence lengths. The D value in the query parameter and the D value in the pool_key parameter are equal, and currently only 128 is supported. N2 currently only supports 1.

### pool_key_indexer

| Parameter Name | Parameter Type | Optional/Required | Description | Data Type | Shape |
| -------------- | -------------- | ----------------- | ----------- | --------- | ----- |
| query | Tensor | Required | Input Q in the formula. Empty tensor not supported. Data format is ND. | bfloat16, float16, int8 (int8 is supported only when quant_mode is 2) | When layout_q is BSND, shape is (B,S1,N1,D); when layout_q is TND, shape is (T1,N1,D) |
| pool_key | Tensor | Required | Input K after pooling in the formula. Supports the "0-pool" empty tensor (pure tail-block mode: BSND S2=0, TND T2=0, PA block_num=0; in this case, all valid keys of each batch are expressed by pool_tail_k). Data format is ND. Supports non-contiguous tensors (only axis 0 supports non-contiguity in the PA_BBND scenario). | bfloat16, float16, int8 (int8 is supported only when quant_mode is 2) | When layout_k is BSND, shape is (B,S2,N2,D); when layout_k is TND, shape is (T2,N2,D); when layout_k is PA_BBND, shape is (block_num,block_size,N2,D) |
| weights | Tensor | Required | Input W in the formula. Empty tensor not supported. Data format is ND. | bfloat16, float16 (only float16 when quant_mode is 2) | When layout_q is BSND, shape is (B,S1,N1); when layout_q is TND, shape is (T1,N1) |
| pool_tail_k | Tensor | Required | Number of valid tokens in the last pool of each batch when it is not full of pool_size. Value range is [0, pool_size-1]. Data format is ND. | int64 | (B,) |
| actual_seq_q | Tensor | Optional | Valid Sequence Length of Query in different batches. Required only when layout_q is TND. The first value is fixed to 0. Data format is ND. Supports non-contiguous tensors. | int64 | (B,) or (B+1,) |
| actual_seq_k | Tensor | Optional | Valid Sequence Length of pool_key in different batches. Required only when layout_k is TND. The first value is fixed to 0. Data format is ND. Supports non-contiguous tensors. | int64 | (B,) or (B+1,) |
| block_table | Tensor | Optional | Block mapping table used for KV storage in PageAttention. Required only when layout_k is PA_BBND. Data format is ND. | int32 | (B, max_block_num_per_seq) |
| q_descale | Tensor | Optional | Dequantization coefficient of Query. Valid only when quant_mode >= 0. Data format is ND. When quant_mode is 2, the data type is float16, the shape is consistent with that when quant_mode is 0 (query with the D axis removed), each dimension is consistent with the corresponding dimension of query, and elements must be positive numbers. | float, float8_e8m0, float16 (float16 is supported only when quant_mode is 2) | When layout_q is BSND, shape is (B,S1,N1); when layout_q is TND, shape is (T1,N1) |
| k_descale | Tensor | Optional | Dequantization coefficient of pool_key. Valid only when quant_mode >= 0. Data format is ND. When quant_mode is 2, the data type is float16, the shape is consistent with that when quant_mode is 0 (pool_key with the D axis removed; in the PA_BBND scenario, (block_num,block_size,N2), addressed by physical block), each dimension is consistent with the corresponding dimension of pool_key, and elements must be positive numbers. | float, float8_e8m0, float16 (float16 is supported only when quant_mode is 2) | When layout_k is BSND, shape is (B,S2,N2); when layout_k is TND, shape is (T2,N2); when layout_k is PA_BBND, shape is (block_num,block_size,N2) |
| layout_q | str | Optional | Layout format of Query. Supports BSND and TND. Default value is BSND. | string | - |
| layout_k | str | Optional | Layout format of pool_key. Supports BSND, TND, and PA_BBND. Default value is BSND. | string | - |
| topk | int | Optional | Number of key sparse tokens to select from Query. Value range is [1, 8192] and must be divisible by pool_size. Default value is 128. | int32 | - |
| pool_size | int | Optional | Key-side pooling window size. Value range is [1, 128]. Default value is 1, meaning no pooling (degrades to lightning_indexer behavior). | int32 | - |
| mask_mode | int | Optional | Mask mode. 0 indicates No mask, 3 indicates rightDownCausal mode. Default value is 0. | int32 | - |
| quant_mode | int | Optional | Quantization mode. -1 indicates no quantization, 0 indicates FP8 per-token-head quantization, 1 indicates mxFP8 quantization, and 2 indicates int8 quantization (query/pool_key is INT8, and weights and the dequantization coefficients q_descale/k_descale are FLOAT16). Default value is -1. | int32 | - |
| return_value | bool | Optional | Whether to return the Values corresponding to Indices. False means no return, True means return values. Default value is False. | bool | - |

## Return Value Description

| Parameter Name | Parameter Type | Optional/Required | Description | Data Type | Shape |
| -------------- | -------------- | ----------------- | ----------- | --------- | ----- |
| sparse_indices | Tensor | Required | Indices output in the formula. Empty tensor not supported. Invalid parts are filled with -1. Data format is ND. | int32 | When layout_q is BSND, shape is (B,S1,topk+pool_size-1); when layout_q is TND, shape is (T1,topk+pool_size-1) |
| sparse_values | Tensor | Conditional Output | Values output corresponding to Indices in the formula. When return_value is True, outputs the corresponding values. When return_value is False, outputs an empty tensor with shape [0]. Invalid parts are filled with -inf. Data format is ND. | float | When layout_q is BSND, shape is (B,S1,topk//pool_size); when layout_q is TND, shape is (T1,topk//pool_size); when return_value is False, shape is (0,) |

## Constraints

- This API supports inference scenarios.
- This API supports single operator mode, TorchAir (GE graph) mode, and TorchAir (aclgraph) graph mode.
- B (Batch) indicates the input sample batch size.
- The data types of the query and pool_key parameters must be consistent.
- topk must be divisible by pool_size.
- Pure tail-block mode: when a batch has no complete pool (0 pools), all its valid tokens are expressed by pool_tail_k (∈[1, pool_size-1]), and the tail tokens are written to the output based on causal visibility (using the same formula as the tail-block channel of batches with pools); the pool selection is empty and sparse_values is all -inf. Mixed batches (some batches with 0 pools) are also supported.
- The value range of pool_tail_k is [0, pool_size-1]. Value validation is performed on the host side only for CPU inputs. For NPU inputs passed via aclnn Tensor variant directly to device data, no value validation is performed on the host side (consistent with TorchAir graph mode). The caller must ensure the value is valid; otherwise, the kernel behavior is undefined.
- pool_tail_k, actual_seq_q, and actual_seq_k support CPU or NPU inputs. When all are CPU inputs, the host value retrieval path is used (after aclgraph capture, values become graph constants; modification requires recapture). When any is an NPU input, the device input path is automatically used (non-blocking D2H; after aclgraph capture, modifying buffer content takes effect automatically during replay).
- When pool_size=1 and pool_tail_k=0, the behavior degrades to lightning_indexer, and the output dimension is topk.
- Invalid parts of sparse_indices are filled with -1. Invalid parts of sparse_values are filled with -inf.
- The actual_seq_q and actual_seq_k parameters require their values to be the cumulative sum of valid token counts of the current batch and preceding batches (TND scenario). The value of a later element must be greater than or equal to the value of the preceding element.
- pool_key supports non-contiguous axis 0 in the PA_BBND scenario. All other axes must be contiguous.
- For a detailed introduction to the mask modes represented by mask_mode, see [Sparse Mode Introduction](../../../docs/en/context/sparse_mode_introduction.md).
- When layout_k is PA_BBND, block_table and actual_seq_k must be passed. When layout_k is not PA_BBND, block_table is not supported.
- When quant_mode=-1, q_descale and k_descale are not supported.

<!-- npu="A3,910b" id7 -->

- <term>Atlas A3 products</term>, <term>Atlas A2 products</term>:
  - topk value range currently only supports [1, 2048], as well as 3072, 4096, 5120, 6144, 7168, and 8192.
  - When layout_k is PA_BBND, actual_seq_k must be passed.
  - When layout_q is BSND, actual_seq_q is not supported. When layout_k is BSND, actual_seq_k is not supported.
  - The FLOAT8_E4M3FN and FLOAT8_E8M0 data types are not supported, and the FP8 quantization scenario (quant_mode 0/1) is not supported.
  - The int8 quantization scenario is supported (when quant_mode is 2, query and pool_key are INT8, and weights and the dequantization coefficients are FLOAT16).

<!-- end id7 -->

<!-- npu="950" id8 -->

- <term>Ascend 950PR&950DT products</term>:
  - When layout_q is BSND, actual_seq_q is not supported. When layout_k is BSND or PA_BBND, actual_seq_k is not supported.
  - When layout_q is TND, actual_seq_q must be passed.
  - quant_mode supports -1 (no quantization), 0 (FP8 per-token-head quantization), 1 (mxFP8 quantization), and 2 (int8 quantization).

<!-- end id8 -->

## Deterministic Computation

Deterministic computation is supported by default.

## Example

- Single operator mode call (BSND + causal mask + pooling):

  ```python
  import torch
  import torch_npu
  from cann_ops_transformer import pool_key_indexer

  B = 2
  S1 = 64
  S2 = 256  # Pool-level Key sequence length
  N1 = 8
  N2 = 1
  D = 128
  topk = 128
  pool_size = 16
  mask_mode = 3  # Causal mask
  return_value = True

  # Construct inputs
  query = torch.randn(B, S1, N1, D, dtype=torch.float16).npu()
  pool_key = torch.randn(B, S2, N2, D, dtype=torch.float16).npu()
  weights = torch.randn(B, S1, N1, dtype=torch.float16).npu()
  pool_tail_k = torch.tensor([0, 8], dtype=torch.int64).npu()  # batch0 tail block is full, batch1 tail block has 8 tokens

  # Execute pool_key_indexer
  sparse_indices, sparse_values = pool_key_indexer(
      query, pool_key, weights, pool_tail_k,
      layout_q="BSND",
      layout_k="BSND",
      topk=topk,
      pool_size=pool_size,
      mask_mode=mask_mode,
      quant_mode=-1,
      return_value=return_value
  )
  print(f"sparse_indices shape: {sparse_indices.shape}")  # (2, 64, 143)
  print(f"sparse_values shape: {sparse_values.shape}")    # (2, 64, 8)
  ```

- Single operator mode call (TND + no mask + pool_size=1 degradation scenario):

  ```python
  import torch
  import torch_npu
  from cann_ops_transformer import pool_key_indexer

  q_lens = [32, 48]
  k_lens = [128, 192]
  N1 = 8
  N2 = 1
  D = 128
  topk = 64
  pool_size = 1
  mask_mode = 0

  total_q = sum(q_lens)
  total_k = sum(k_lens)

  query = torch.randn(total_q, N1, D, dtype=torch.float16).npu()
  pool_key = torch.randn(total_k, N2, D, dtype=torch.float16).npu()
  weights = torch.randn(total_q, N1, dtype=torch.float16).npu()
  pool_tail_k = torch.tensor([0, 0], dtype=torch.int64).npu()
  # TND prefix sum
  actual_seq_q = torch.tensor([q_lens[0], q_lens[0] + q_lens[1]], dtype=torch.int64).npu()
  actual_seq_k = torch.tensor([k_lens[0], k_lens[0] + k_lens[1]], dtype=torch.int64).npu()

  sparse_indices, sparse_values = pool_key_indexer(
      query, pool_key, weights, pool_tail_k,
      actual_seq_q=actual_seq_q,
      actual_seq_k=actual_seq_k,
      layout_q="TND",
      layout_k="TND",
      topk=topk,
      pool_size=pool_size,
      mask_mode=mask_mode,
      quant_mode=-1,
      return_value=False
  )
  print(f"sparse_indices shape: {sparse_indices.shape}")  # (80, 64)
  # sparse_values is an empty tensor when return_value=False
  ```

- Single operator mode call (int8 quantization + TND/PA_BBND + 128K decode scenario, quant_mode=2):

  ```python
  import torch
  import torch_npu
  from cann_ops_transformer import pool_key_indexer

  # 128K decode scenario: single-token decode (T1=1), pool_key stored in PA_BBND physical blocks,
  # 1024 logical blocks = 32768 pools = 131072 tokens (128K)
  N1, D = 32, 128
  BLOCK_NUM, BLOCK_SIZE, N2 = 10308, 32, 1
  SEQ_K = 32768                      # Number of valid pools = 1024 logical blocks × block_size 32
  LOGICAL_BLOCKS = SEQ_K // BLOCK_SIZE
  BT_COLS = 1041                     # Number of block_table columns (1024 logical blocks + redundant tail columns)
  TOPK, POOL_SIZE, MASK_MODE = 2048, 4, 3

  g = torch.Generator().manual_seed(20261015)

  # block_table: 1024 logical blocks mapped out of order to 10308 physical blocks, tail columns padded with 0
  block_table = torch.zeros(1, BT_COLS, dtype=torch.int32)
  block_table[0, :LOGICAL_BLOCKS] = torch.randperm(BLOCK_NUM, generator=g)[:LOGICAL_BLOCKS].to(torch.int32)

  # int8 quantized inputs: query/pool_key is INT8, weights and dequantization coefficients are FLOAT16
  query = torch.randint(-127, 128, (1, N1, D), dtype=torch.int8, generator=g).npu()          # TND (T1,N1,D)
  pool_key = torch.randint(-127, 128, (BLOCK_NUM, BLOCK_SIZE, N2, D), dtype=torch.int8, generator=g).npu()  # PA_BBND
  weights = (torch.rand((1, N1), generator=g) * 50.0 - 25.0).half().npu()                    # (T1,N1)
  pool_tail_k = torch.tensor([0], dtype=torch.int64).npu()                                   # No residual token in the tail pool
  actual_seq_q = torch.tensor([1], dtype=torch.int64).npu()                                  # TND prefix sum
  actual_seq_k = torch.tensor([SEQ_K], dtype=torch.int64).npu()                              # In the PA scenario, the number of pools of this batch
  block_table = block_table.npu()
  q_descale = (torch.rand((1, N1), generator=g) * 0.099 + 0.001).half().npu()                # (T1,N1) per-token-head
  k_descale = (torch.rand((BLOCK_NUM, BLOCK_SIZE, N2), generator=g) * 0.099 + 0.001).half().npu()  # Addressed by physical block

  sparse_indices, sparse_values = pool_key_indexer(
      query, pool_key, weights, pool_tail_k,
      actual_seq_q=actual_seq_q,
      actual_seq_k=actual_seq_k,
      block_table=block_table,
      q_descale=q_descale,
      k_descale=k_descale,
      layout_q="TND",
      layout_k="PA_BBND",
      topk=TOPK,
      pool_size=POOL_SIZE,
      mask_mode=MASK_MODE,
      quant_mode=2,
      return_value=False,
  )
  torch.npu.synchronize()
  print(f"sparse_indices shape: {sparse_indices.shape}")  # (1, 2051), int32, invalid positions -1
  print(f"sparse_values shape: {sparse_values.shape}")    # (0,), an empty tensor when return_value=False
  ```

- TorchAir (GE graph) mode call:

  ```python
  import torch
  import torch_npu
  import torchair
  from cann_ops_transformer import pool_key_indexer

  B = 1
  S1 = 64
  S2 = 256
  N1 = 8
  N2 = 1
  D = 128

  query = torch.randn(B, S1, N1, D, dtype=torch.float16).npu()
  pool_key = torch.randn(B, S2, N2, D, dtype=torch.float16).npu()
  weights = torch.randn(B, S1, N1, dtype=torch.float16).npu()
  pool_tail_k = torch.tensor([0], dtype=torch.int64).npu()

  class PoolKeyIndexerNetwork(torch.nn.Module):
      def forward(self, query, pool_key, weights, pool_tail_k):
          return torch.ops.cann_ops_transformer.pool_key_indexer(
              query, pool_key, weights, pool_tail_k,
              layout_q="BSND",
              layout_k="BSND",
              topk=128,
              pool_size=16,
              mask_mode=3,
              quant_mode=-1,
              return_value=True
          )

  from torchair.configs.compiler_config import CompilerConfig
  config = CompilerConfig()
  config.mode = "max-autotune"
  npu_backend = torchair.get_npu_backend(compiler_config=config)
  torch._dynamo.reset()
  npu_mode = torch.compile(PoolKeyIndexerNetwork(), fullgraph=True, backend=npu_backend, dynamic=False)
  sparse_indices, sparse_values = npu_mode(query, pool_key, weights, pool_tail_k)
  print(f"sparse_indices shape: {sparse_indices.shape}")
  ```

- TorchAir (aclgraph) graph mode call:

  ```python
  import torch
  import torch_npu
  import torchair
  from cann_ops_transformer import pool_key_indexer

  B = 1
  S1 = 64
  S2 = 256
  N1 = 8
  N2 = 1
  D = 128

  query = torch.randn(B, S1, N1, D, dtype=torch.float16).npu()
  pool_key = torch.randn(B, S2, N2, D, dtype=torch.float16).npu()
  weights = torch.randn(B, S1, N1, dtype=torch.float16).npu()
  pool_tail_k = torch.tensor([0], dtype=torch.int64).npu()

  class PoolKeyIndexerNetwork(torch.nn.Module):
      def forward(self, query, pool_key, weights, pool_tail_k):
          return torch.ops.cann_ops_transformer.pool_key_indexer(
              query, pool_key, weights, pool_tail_k,
              layout_q="BSND",
              layout_k="BSND",
              topk=128,
              pool_size=16,
              mask_mode=3,
              quant_mode=-1,
              return_value=True
          )

  from torchair.configs.compiler_config import CompilerConfig
  config = CompilerConfig()
  # npugraph_ex is the aclgraph capture/replay path (equivalent to the deprecated reduce-overhead)
  config.mode = "npugraph_ex"
  npu_backend = torchair.get_npu_backend(compiler_config=config)
  torch._dynamo.reset()
  npu_mode = torch.compile(PoolKeyIndexerNetwork(), fullgraph=True, backend=npu_backend, dynamic=False)
  # The first call triggers capture, subsequent calls use replay.
  # When value inputs are NPU tensors, they are graph inputs.
  # During replay, the current buffer content is read; modification does not require recapture.
  npu_mode(query, pool_key, weights, pool_tail_k)
  sparse_indices, sparse_values = npu_mode(query, pool_key, weights, pool_tail_k)
  print(f"sparse_indices shape: {sparse_indices.shape}")
  ```
