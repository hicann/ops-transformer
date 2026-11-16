# fused_causal_conv1d

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-21T09:27:59.380Z -->

## Applicable Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 products</term>: Not supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 products</term>: Not supported
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2 inference products</term>: Not supported
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas inference products</term>: Not supported
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas training products</term>: Not supported
<!-- end id6 -->

## Function

- **Description**:

    Performs causal 1D convolution on sequences. Along the sequence dimension, cache data (with a length of the convolution kernel width minus 1) is used to pad the head of each sequence, ensuring that the output depends on the current and historical inputs. After the convolution is complete, part of the current sequence data is updated to the cache. Based on the output of the causal 1D convolution, the original input is added to the output to implement a residual connection. Features such as APC (Automatic Prefix Caching), MTP (speculative decoding), and residual connection are supported. Compared with the standard causal_conv1d operator, this operator adds functions such as APC cache reuse, mixed prefill-decode deployment, and optional residual connection. Compared with fused_causal_conv1d_, this API does not support in-place update of the input x.<br>

- **Formula**:

    K is the convolution kernel width (fixed to 3), L is the original sequence length, and dim is the feature dimension.
    1. Cache read

        Cache line index:

        $$
        readCacheLine = \begin{cases}
        cacheIndices[batchId, \; initialStateIdx[batchId]], & \text{APC mode} \\
        cacheIndices[batchId], & \text{ not APC and cacheIndices exists} \\
        batchId, & \text{ otherwise}
        \end{cases}
        $$

        Case 1: First computation (numComputedTokens[batchId] == 0)

        $$
        cachedState[i, dim] = 0, \quad 0 \leq i < K-1
        $$

        $$
        offset = 0
        $$

        Case 2: Speculative decoding mode (numAcceptedTokens exists)

        $$
        offset = numAcceptedTokens[batchId] - 1
        $$

        $$
        cachedState[i, dim] = convStates[readCacheLine][i, dim], \quad 0 \leq i <   offset + K - 1
        $$

        Case 3: Default mode

        $$
        offset = C - (K - 1)
        $$

        $$
        cachedState[i, dim] = convStates[readCacheLine][i, dim], \quad 0 \leq i < offset + K - 1
        $$

    2. Cache concatenation

        $$
        paddedInput[i, dim] =
        \begin{cases}
        cachedState[i, dim], & 0 \leq i < offset + K - 1 \\
        x[i - (offset + K - 1), dim], & offset + K - 1 \leq i < offset + K - 1 + L
        \end{cases}
        $$

    3. Cache update

        $$
        Len = offset + K - 1 + L
        $$

        $$
        M = \min(C, \; Len)
        $$

        $$
        writeCacheLine = \begin{cases}
        cacheIndices[batchId, \; idxLast], & \text{APC mode} \\
        cacheIndices[batchId], & \text{ non-APC and cacheIndices exists} \\
        batchId, & \text{ otherwise}
        \end{cases}
        $$

        $$
        convStates[writeCacheLine][C - M + i, dim] = paddedInput[Len - M + i, dim], \quad i = 0, 1, \dots, M-1
        $$

    4. Offset cropping

        $$
        x'[i, dim] = paddedInput[i + offset, dim], \quad 0 \leq i < K - 1 + L
        $$

    5. APC cache fill (optional, in APC mode)

        $$
        seqCompletedOffsetToken = numComputedTokens[batchId] \mod B
        $$

        $$
        seqCompletedOffset = B - seqCompletedOffsetToken
        $$

        $$
        seqEndOffset = (L - seqCompletedOffset) \mod B
        $$

        $$
        lastFullBlockTokenIndex = \begin{cases}
        L - seqEndOffset - B, & seqEndOffset = 0 \\
        L - seqEndOffset, & \text{otherwise}
        \end{cases}
        $$

        $$
        nBlockToFill = idxLast - idxFirst
        $$

        For each chunk = 0, 1, ..., nBlockToFill - 1:

        $$
        boundaryIdx = lastFullBlockTokenIndex - (nBlockToFill - chunk - 1) \times B
        $$

        $$
        convStates[cacheIndices[batchId, \; idxFirst + chunk]][C-(K-1)+j, \; dim] = x'[boundaryIdx + j, \; dim], \quad j = 0, \dots, K-2
        $$

    6. Causal 1D convolution

        $$
        y[i, dim] = \sum_{k=0}^{K-1} w[k, dim] \cdot x'[i + k, dim], \quad i = 0, 1, \dots, L-1
        $$

    7. Zero-padding reset (optional, when convMode == 1 and numComputedTokens is not empty)

        $$
        resetIdx = \min\!\Big(\max\!\big(K - 1 - numComputedTokens[batchId], \; 0\big), \; L\Big)
        $$

        $$
        y[i, dim] = 0, \quad 0 \leq i < resetIdx
        $$

    8. Residual connection (optional)

        $$
        y[i, dim] = x[i, dim] + y[i, dim]
        $$

## Function Prototype

```python
cann_ops_transformer.fused_causal_conv1d(
    x, weight, conv_states, *, query_start_loc=None, cache_indices=None, initial_state_mode=None, bias=None, num_accepted_tokens=None, num_computed_tokens=None, block_idx_first_scheduled_token=None, block_idx_last_scheduled_token=None, initial_state_idx=None, activation="None", pad_slot_id=-1, max_query_len=-1, residual_connection=1, block_size=128, conv_mode=1, max_draft_tokens=7
) -> Tensor
```

## Parameters

| Name | Type | Required/Optional | Description | Data Type | Dimension (Shape) |
|--------|----------|-----------|------|----------|-------------|
| x | Tensor | Required | Input sequence. | float16, bfloat16 | [batch, seq_len, dim] (fixed batch) or [cu_seq_len, dim] (variable length) |
| weight | Tensor | Required | Convolution weight. | Same as x | [kW, dim], kW∈{2, 3, 4} |
| conv_states | Tensor | Required | Convolution state cache, updated in place after computation. | Same as x | [num_cache_lines, state_len, dim], state_len ≥ kW-1 |
| query_start_loc | Tensor | Optional | Start position index of variable-length sequences. This parameter must be provided in variable-length scenarios; an empty Tensor can be passed in fixed-batch scenarios. | int32 | [batch+1] |
| cache_indices | Tensor | Optional | Cache index. Specifies the index of the cache state corresponding to each sequence in conv_states. If None, the identity mapping is used by default. | int32 | [batch] |
| initial_state_mode | Tensor | Optional | Initial state flag. 1 = use cached history, 0 = zero initialization (cold start). If None, all states are zero-initialized by default. | int32 | [batch] |
| bias | Tensor | Required | Convolution bias. None indicates it is not used. | Same as x | [dim] |
| num_accepted_tokens | Tensor | Optional | Number of randomly speculated tokens in the current batch. | int32 | [batch] |
| num_computed_tokens | Tensor | Optional | Total number of tokens already processed in the current batch, used to determine the initial state. | int32 | [batch] |
| block_idx_first_scheduled_token | Tensor | Optional | Block index corresponding to the start position of the current batch. | int32 | [batch] |
| block_idx_last_scheduled_token | Tensor | Optional | Block index corresponding to seq_len - 1 of the current batch. | int32 | [batch] |
| initial_state_idx | Tensor | Optional | Index of the initial index block. | int32 | [batch] |
| activation | str | Optional | Activation function type. This field is currently not supported. Default value is "None". | - | - |
| pad_slot_id | int | Optional | Padding slot ID. Default value is -1. | - | - |
| max_query_len | int | Optional | Maximum seq_len among all batches. Setting it to -1 is supported only in the decode scenario (fixed batch). | - | - |
| residual_connection | int | Optional | Whether to perform a residual connection. 1 = perform the residual connection, 0 = do not perform the residual connection. | - | - |
| block_size | int | Optional | Block size. The value must be greater than or equal to 2. Typical values are 128 and 256. | - | - |
| conv_mode | int | Optional | Convolution mode. 0 and 1 are supported. 0 = Qwen3-Next community implementation, 1 = Pangu V2 implementation. | - | - |
| max_draft_tokens | int | Optional | Maximum number of speculative tokens. Supported range: [0, 16]. Default value is 7. | - | - |

## Return Values

Returns the convolution output Tensor y, with the same shape and dtype as x.

## Constraints

- Input shape constraints:
  - prefill use case:
    - x supports 2D [cu_seq_len, dim].
    - weight must be 2D [K, dim], where K is fixed to 3.
    - conv_states must be 3D [..., K-1, dim]. Its dimension 0 size is not fixed and must be greater than or equal to the number of batches participating in the computation (that is, the number of batches where cache_indices is not equal to pad_slot_id).
    - query_start_loc must exist.
    - cache_indices is 1D [batch, ] or 2D [batch, max_num_blocks], where 1D indicates that APC is not enabled, and 2D indicates that APC is enabled.
    - cu_seq_len is in the range [batch, 1024 * 1024], dim is in the range [64, 16384] and is a multiple of 16, and their product must be in the range [64 * batch, 4G]. batch is in the range [1, 256].
    - max_num_blocks >= ceil(max_query_len, block_size).
    - max_query_len > max_draft_tokens + 1.
  - Mixed prefill and decode scenario:
    - x supports 2D [cu_seq_len, dim].
    - weight must be 2D [K, dim], where K is fixed to 3.
    - conv_states must be 3D [..., K-1+m, dim], where the dimension size of dim 0 is not fixed and is greater than or equal to the number of batches involved in the computation (that is, the number of batches whose cache_indices is not equal to pad_slot_id).
    - query_start_loc must exist.
    - cache_indices is 1D [batch, ] or 2D [batch, max_num_blocks], where 1D indicates that APC is not enabled and 2D indicates that APC is enabled.
    - cu_seq_len is in the range [batch, 1024 * 1024], dim is in the range [64, 16384] and is a multiple of 16, their product must be in the range [64 * batch, 4G], and batch is in the range [1, 256].
    - max_num_blocks >= ceil(max_query_len, block_size).
    - max_query_len > max_draft_tokens + 1.
  - decode use case (variable-length sequences):
    - x supports 2D [cu_seq_len, dim].
    - weight must be 2D [K, dim], where K is fixed to 3.
    - conv_states must be 3D [..., k-1+m, dim]. The size of dimension 0 is not fixed and must be greater than or equal to the number of batches involved in the computation (that is, the number of batches whose cache_indices is not equal to pad_slot_id).
    - query_start_loc must exist.
    - cache_indices is 1D [batch, ] or 2D [batch, max_num_blocks], where 1D indicates that APC is not enabled and 2D indicates that APC is enabled.
    - The range of cu_seq_len is [batch, batch * (max_draft_tokens + 1)], and the range of seq_len of each batch is [1, max_draft_tokens + 1]. The range of dim is [64, 16384], and dim must be a multiple of 16. The range of batch is [1, 256].
    - max_num_blocks >= ceil(max_query_len, block_size).
    - The range of max_query_len is [1, max_draft_tokens + 1].
  - decode use case (fixed batch):
    - x supports 3D [batch, m+1, dim].
    - weight must be 2D [K, dim], where K is fixed to 3.
    - conv_states must be 3D [..., K-1+m, dim], where the size of dimension 0 is not fixed and is greater than or equal to the number of batches participating in the computation (that is, the number of batches whose cache_indices is not equal to pad_slot_id).
    - cache_indices is 1D [batch, ] or 2D [batch, max_num_blocks], where the 1D dimension indicates that APC is not enabled, and the 2D dimension indicates that APC is enabled.
    - m is in the range [0, max_draft_tokens], dim is in the range [64, 16384] and is a multiple of 16, and batch is in the range [1, 256].
    - max_num_blocks >= ceil(max_query_len, block_size).
    - max_query_len is in the range [1, max_draft_tokens + 1] and can be -1.

- Input value range restrictions:
  - query_start_loc is the cumulative offset. The value range is [0, cu_seq_len] and the length is batch+1. query_start_loc[i] indicates the start offset of the i-th sequence, and query_start_loc[batch+1] indicates the end position of the last sequence.
  - block_size is 0 or greater than or equal to 2, and is not 0 when APC is enabled.
  - When block_idx_first_scheduled_token, block_idx_last_scheduled_token, initial_state_idx, num_computed_tokens, and cache_indices all exist, APC is enabled, and the following conditions must be met (where i is the batch index):
    - cache_indices is 2-dimensional
    - initial_state_idx[i] <= block_idx_first_scheduled_token[i] + 1
    - initial_state_idx[i] <= block_idx_last_scheduled_token[i]
    - block_idx_first_scheduled_token[i] <= block_idx_last_scheduled_token[i]
    - block_idx_last_scheduled_token[i] < max_num_blocks
  - num_accepted_tokens can be None or non-None. If it is non-None, its length is batch, elements corresponding to prefill are 0, and elements corresponding to decode are greater than 0 and less than or equal to seq_len-1 of the current batch.
  - Each element in num_computed_tokens is greater than or equal to 0.
  - The value range of cache_indices is [0, conv_states.dim[0]-1], and the values must be different from each other (unless equal to pad_slot_id).
  - max_query_len = the maximum seq_len in the batch.
  - The value range of max_draft_tokens is [0, 16], and the default value is 7.
  - In Pangu V2 mode (conv_mode = 1), num_computed_tokens cannot be None.
  - The operator input parameters and intermediate computation results do not exceed the value range of the corresponding data types (float16/bfloat16).
  - Operator inputs containing ±inf or nan are not supported.

## Deterministic Computation

Defaults to a deterministic implementation.

## Example

- Single-operator mode call

    ```python
    import torch
    import torch_npu
    from cann_ops_transformer.ops import fused_causal_conv1d

    torch_npu.npu.set_device(0)

    B = 2
    S = 16
    D = 512
    kW = 3

    x = torch.randn(B, S, D, device="npu", dtype=torch.float16)
    weight = torch.randn(kW, D, device="npu", dtype=torch.float16)
    conv_states = torch.zeros(B, kW - 1 + S, D, device="npu", dtype=torch.float16)
    query_start_loc = None
    cache_indices = None
    initial_state_mode = None
    bias = None
    num_accepted_tokens = None
    num_computed_tokens = None
    block_idx_first_scheduled_token = None
    block_idx_last_scheduled_token = None
    initial_state_idx = None
    activation = "None"
    pad_slot_id = -1
    max_query_len = 16
    residual_connection = 1
    block_size = 128
    conv_mode = 0
    max_draft_tokens = 7

    y = fused_causal_conv1d(
        x, weight, conv_states, query_start_loc=query_start_loc, cache_indices=cache_indices, initial_state_mode=initial_state_mode, bias=bias, num_accepted_tokens=num_accepted_tokens, num_computed_tokens=num_computed_tokens, block_idx_first_scheduled_token=block_idx_first_scheduled_token, block_idx_last_scheduled_token=block_idx_last_scheduled_token, initial_state_idx=initial_state_idx, activation=activation, pad_slot_id=pad_slot_id, max_query_len=max_query_len, residual_connection=residual_connection, block_size=block_size, conv_mode=conv_mode, max_draft_tokens=max_draft_tokens
    )
    ```

- Graph mode call

    The operator is automatically converted to a GE graph operator through `torch.compile(backend="atc")` or `torchair`. No extra configuration is required.

    ```python
    import torch
    import torch_npu
    import torchair
    from cann_ops_transformer.ops import fused_causal_conv1d

    torch_npu.npu.set_device(0)

    B = 2
    S = 16
    D = 512
    kW = 3

    class FusedCausalConv1dModel(torch.nn.Module):
        def forward(self, x, weight, conv_states, query_start_loc, cache_indices, initial_state_mode, bias,
                    num_accepted_tokens, num_computed_tokens, block_idx_first_scheduled_token, block_idx_last_scheduled_token, initial_state_idx, activation, pad_slot_id, max_query_len, residual_connection, block_size, conv_mode, max_draft_tokens):
            y = fused_causal_conv1d(
                x, weight, conv_states, query_start_loc=query_start_loc, cache_indices=cache_indices, initial_state_mode=initial_state_mode, bias=bias, num_accepted_tokens=num_accepted_tokens, num_computed_tokens=num_computed_tokens, block_idx_first_scheduled_token=block_idx_first_scheduled_token, block_idx_last_scheduled_token=block_idx_last_scheduled_token, initial_state_idx=initial_state_idx, activation=activation, pad_slot_id=pad_slot_id, max_query_len=max_query_len, residual_connection=residual_connection, block_size=block_size, conv_mode=conv_mode, max_draft_tokens=max_draft_tokens
            )
            return y

    model = FusedCausalConv1dModel().npu()
    npu_backend = torchair.get_npu_backend()
    model = torch.compile(model, backend=npu_backend, dynamic=False)

    x = torch.randn(B, S, D, device="npu", dtype=torch.float16)
    weight = torch.randn(kW, D, device="npu", dtype=torch.float16)
    conv_states = torch.zeros(B, kW - 1 + S, D, device="npu", dtype=torch.float16)
    query_start_loc = None
    cache_indices = None
    initial_state_mode = None
    bias = None
    num_accepted_tokens = None
    num_computed_tokens = None
    block_idx_first_scheduled_token = None
    block_idx_last_scheduled_token = None
    initial_state_idx = None
    activation = "None"
    pad_slot_id = -1
    max_query_len = S
    residual_connection = 1
    block_size = 128
    conv_mode = 0
    max_draft_tokens = 7

    output = model(x, weight, conv_states, query_start_loc, cache_indices, initial_state_mode, bias,
                    num_accepted_tokens, num_computed_tokens, block_idx_first_scheduled_token, block_idx_last_scheduled_token, initial_state_idx, activation, pad_slot_id, max_query_len,residual_connection, block_size, conv_mode, max_draft_tokens)
    print(f"Finish")
    ```
