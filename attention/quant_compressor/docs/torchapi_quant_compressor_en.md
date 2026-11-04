# quant_compressor

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-21T12:34:00.146Z -->

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

- **Description**: QuantCompressor is a preprocessing operator for SMLA and QLI in inference use cases. It is the quantized version of [compressor](../../compressor/docs/torchapi_compressor_en.md). It compresses the KV cache of every 4 or 128 tokens into one, and each token then performs DSA computation with these compressed KV caches. For long sequences, QuantCompressor can effectively reduce computation overhead. The difference from compressor is that the inputs $x$, $W^{KV}$, and $W^{Gate}$ are HIFLOAT8 quantized data. They directly participate in Matmul computation in HIFLOAT8 format (natively supported by hardware). The FLOAT32 Matmul output is then multiplied by the merged scaling factor to complete dequantization. This reduces memory usage and data transfer overhead. The main computation process is as follows:
    1. Matmul and dequantization: Matmul is performed between the HIFLOAT8 quantized input $X$ and $W^{KV}$ to obtain a FLOAT32 result. The result is multiplied by the merged scaling factor $x\_descale \cdot wkv\_descale$ to complete dequantization and obtain $kv\_state$. Matmul is performed between $X$ and $W^{Gate}$ to obtain a FLOAT32 result. The result is multiplied by the merged scaling factor $x\_descale \cdot wgate\_descale$ to complete dequantization and obtain $score\_state$, where x_descale is per-tensor scaling (a single scalar) and wkv_descale and wgate_descale are per-channel scaling (the number of channels is coff\*D). After merging, the scaling remains per-channel. $kv\_state$ and $score\_state$ are updated based on the input start_pos and cu_seqlens.
    2. When coff is 2, perform data rearrangement on $kv\_state$ and $score\_state$.
    3. Group $score\_state$ by compression ratio and add $Ape$, then perform the softmax operation. Multiply the softmax result with $kv\_state$ using Mul, and then perform the ReduceSum operation.

- **Formula**:

    1. Compute matrix multiplication and dequantization (quant_mode=1, A8W8_A_HIFP8_PER_TENSOR_W_HIFP8_PER_CHANNEL):

    Perform Matmul with HIFLOAT8 input, then multiply the FLOAT32 output by the merged scaling factor (the product of x_descale and wkv/wgate_descale) to complete dequantization:

    $$
    C4A: \left[kv\_state^a, score\_state^a\right] = (X_{hif8} @ \left[W^{aKV}_{hif8}, W^{aGate}_{hif8}\right]) \cdot (x\_descale \cdot [wkv\_descale^a, wgate\_descale^a]), \left[kv\_state^b, score\_state^b\right] = (X_{hif8} @ \left[W^{bKV}_{hif8}, W^{bGate}_{hif8}\right]) \cdot (x\_descale \cdot [wkv\_descale^b, wgate\_descale^b]);
    $$

    $$
    C128A: \left[kv\_state, score\_state\right] = (X_{hif8} @ \left[W^{KV}_{hif8}, W^{Gate}_{hif8}\right]) \cdot (x\_descale \cdot [wkv\_descale, wgate\_descale])
    $$

    where x_descale is a per-tensor scalar (shape [1,]), and wkv_descale and wgate_descale are per-channel vectors (shape [coff\*D,], scaled along the output channel dimension of the weight). The two are multiplied and then merged into a per-channel scaling factor.

    2. Compute the grouped addition:

    $$
    C4A: score\_state_i^\prime = \left[score\_state_{\left[4(i-1)+1:4i,:\right]}^a; score\_state_{\left[4i+1:4(i+1),:\right]}^b\right] + Ape,~i=1,2,\cdots, \frac{s}{4};
    $$

    $$
    C128A: score\_state_i^\prime = score\_state_{\left[128(i-1)+1:128i,:\right]} + Ape,~i=1,2,\cdots, \frac{s}{128};
    $$

    3. Compute the grouped Softmax:

    $$
    C4A: S_i^\prime = softmax(score\_state_i^\prime),~i=1,2,\cdots, \frac{s}{4};
    $$

    $$
    C128A: S_i^\prime = softmax(score\_state_i^\prime),~i=1,2,\cdots, \frac{s}{128};
    $$

    4. Compute the Hadamard product:

    $$
    C4A: (S_H)_i = S_i^\prime \odot \left[kv\_state^a_{\left[4(i-1)+1:4i,:\right]} ;kv\_state^b_{\left[4i+1:4(i+1),:\right]}\right],~i=1,2,\cdots, \frac{s}{4};
    $$

    $$
    C128A: S_H = S_i^\prime \odot kv\_state;
    $$

    5. Compute the grouped sum along the compression axis:

    $$
    C4A: C_{i}^{\text{Comp}} = \left[1\right]_{1\times8} @ (S_H)_i, ~i=1,2,\cdots, \frac{s}{4};
    $$

    $$
     C128A: C_{i}^{\text{Comp}} = \left[1\right]_{1\times128} @ (S_H)_i, ~i=1,2,\cdots, \frac{s}{128};
    $$

## Function Prototype

```python
cann_ops_transformer.quant_compressor(
    x,
    wkv,
    wgate,
    state_cache,
    ape,
    quant_mode,
    cmp_ratio,
    *,
    x_descale=None,
    wkv_descale=None,
    wgate_descale=None,
    state_block_table=None,
    cu_seqlens=None,
    seqused=None,
    start_pos=None,
    coff=1,
    cache_mode=1) -> Tensor
```

## Parameters

| Name | Type | Required/Optional | Description | Data Type | Dimension (shape) |
| ---- | ---- | ---- | ---- | ---- | ---- |
| x | Tensor | Required | Raw uncompressed data, corresponding to $X$ in the formula. HIFLOAT8 quantized input. Does not support non-contiguous tensors. Data format: ND. | uint8 (HIFLOAT8) | [B,S,H] or [T,H] |
| wkv | Tensor | Required | kv compression weight, corresponding to $W^{KV}$ in the formula. HIFLOAT8 quantized input. Does not support non-contiguous tensors. Data format: ND. | uint8 (HIFLOAT8) | [coff\*D,H] |
| wgate | Tensor | Required | gate compression weight, corresponding to $W^{Gate}$ in the formula. HIFLOAT8 quantized input. Does not support non-contiguous tensors. Data format: ND. | uint8 (HIFLOAT8) | [coff\*D,H] |
| state_cache | Tensor | Required | Historical data of kv_state and score_state, corresponding to $\left[kv\_state, score\_state\right]$ in the formula. Supports non-contiguous tensors along axis 0. Data format: ND. After computation, kv_state and score_state are updated in place in this tensor. | float32 | [block_num, block_size, 2\*coff\*D], where block_num > 0 |
| ape | Tensor | Required | positional biases, corresponding to $Ape$ in the formula. Does not support non-contiguous tensors. Data format: ND. | float32 | [cmp_ratio,coff\*D] |
| quant_mode | int | Required | Quantization mode. Value range: [1]. 1 indicates A8W8_A_HIFP8_PER_TENSOR_W_HIFP8_PER_CHANNEL (HIFLOAT8 input; x is dequantized with per-tensor scaling, and wkv/wgate are dequantized with per-channel scaling). | - | - |
| cmp_ratio | int | Required | Data compression ratio. Value range: integer in [2, 128]. | - | - |
| x_descale | Tensor | Optional | Dequantization scaling factor of x, per-tensor scaling. Required when quant_mode=1. Does not support non-contiguous tensors. Data format: ND. | float32 | [1,] |
| wkv_descale | Tensor | Optional | Dequantization scaling factor of wkv, per-channel scaling. The number of channels is coff\*D. Required when quant_mode=1. Does not support non-contiguous tensors. Data format: ND. | float32 | [coff\*D,] |
| wgate_descale | Tensor | Optional | Dequantization scaling factor of wgate, per-channel scaling. The number of channels is coff\*D. Required when quant_mode=1. Does not support non-contiguous tensors. Data format: ND. | float32 | [coff\*D,] |
| state_block_table | Tensor | Optional | Block mapping table used by state_cache for storage. Does not support non-contiguous tensors. Data format: ND. | int32 | When cache_mode=1, the shape is [B,ceil(Smax/block_size)]. Smax indicates the maximum Sequence Length in each Batch. When the shape of x is [B,S,H], Smax=max(start_pos)+S. When the shape of x is [T,H], Smax=max(start_pos)+max(cu_seqlens[n+1] - cu_seqlens[n]). When cache_mode=2, the shape is [B]. In the cacheMode=1/2 scenarios supported by the current operator, this parameter is mandatory. |
| cu_seqlens | Tensor | Optional | Number of valid tokens in each Batch. Does not support non-contiguous tensors. Data format: ND.<br>When the shape of x is [B,S,H], this parameter must be empty.<br>When the shape of x is [T,H], the input shape must be [B+1,]. This parameter is a prefix-sum array. Each element must be greater than or equal to the previous element. The first element must be 0, and the last element must be T. | int32 | [B+1,] |
| seqused | Tensor | Optional | Number of tokens actually involved in compression in each Batch. Does not support non-contiguous tensors. Data format: ND.<br>If None is specified, the value equals the Sequence Length of each Batch.<br>[B,S,H] use case: 0 ≤ seqused[n] ≤ S<br>[T,H] use case: 0 ≤ seqused[n] ≤ cu_seqlens[n+1] - cu_seqlens[n]. | int32 | [B,] |
| start_pos | Tensor | Optional | Computation start position. Does not support non-contiguous tensors. Data format: ND. If the input is None, computation starts from 0. | int32 | [B,] |
| coff | int | Optional | Default value: 1. Only 1/2 is supported.<br>coff=1: overlap data rearrangement is not required<br>coff=2: overlap data rearrangement is required. | int | - |
| cache_mode | int | Optional | Storage mode of state_cache.<br>1: contiguous buffer<br>2: circular buffer<br>Default value: 1. | int | - |

## Return Values

| Name | Type | Required/Optional | Description | Data Type | Dimensions (shape) |
| ---- | ---- | ---- | ---- | ---- | ---- |
| cmp_kv | Tensor | Required | Compressed data. Does not support non-contiguous tensors. The data format supports ND; the content of the pad rows at the tail that do not form a complete compression group is undefined (the operator does not write them, and the memory content before the call is retained). The caller should consume only the valid compressed rows.<br>When the shape of x is [B,S,H], the output is concatenated as: (\<batch0\>compressed_tokens+pad0) +  (\<batch1\>compressed_tokens+pad1) + ... +  (\<batchN\>compressed_tokens+padN);<br>When the shape of x is [T,H], the output is concatenated as: \<batch0\>compressed_tokens + \<batch1\>compressed_tokens + ... + \<batchN\>compressed_tokens + pad. | bfloat16 | x=[B,S,H]: [B,ceil(S/cmp_ratio),D]<br>x=[T,H]: [min(T,T//cmp_ratio+B),D] |

## Constraints

- This API supports use in inference scenarios.

- This API can be called in single-operator mode and TorchAir graph mode (aclgraph).

- Meaning of the dimensions of parameter x: B (Batch Size) indicates the batch size of input samples, S (Sequence Length) indicates the sequence length of input samples, H (Head Size) indicates the size of the hidden layer, D (Head Dim) indicates the minimum unit size of the hidden layer, and T indicates the sum of sequence lengths of input samples across all batches.

- This API supports generalization of B and S, with the following use case restrictions:
  - In some long-sequence use cases, if the computation amount is too large, an error indicating that NPU memory is exceeded may occur. Note: the computation amount is affected by the shape of the input x; the larger the value, the greater the computation amount. Typical long-sequence use cases (that is, the product of B and S or T is large) include but are not limited to:
    <div style="overflow-x: auto;">
    <table style="undefined;table-layout: fixed; width: 400px"><colgroup>
    <col style="width: 100px">
    <col style="width: 100px">
    <col style="width: 100px">
    </colgroup><thead>
    <tr>
    <th>B</th>
    <th>S</th>
    <th>H</th>
    </tr></thead>
    <tbody>
    <tr>
    <td>100</td>
    <td>65525</td>
    <td>4096</td>
    </tr>
    <tr>
    <td>25</td>
    <td>261120</td>
    <td>4096</td>
    </tr>
    <tr>
    <td>100</td>
    <td>131072</td>
    <td>4096</td>
    </tr>
    <tr>
    <td>100</td>
    <td>261120</td>
    <td>4096</td>
    </tr>
    </tbody>
    </table>
    </div>
- This interface supports setting B, S, and T to 0. That is, input parameters whose shapes are related to the B, S, and T values allow empty tensors, while other input parameters do not support empty tensors. In this use case, state_cache is not updated, and the output cmp_kv is an empty tensor.
- The value range of state_block_table elements is [0, block_num), where block_num is the size of the 0th dimension of state_cache. Element values are directly used as the block indices of state_cache, and out-of-range values cause illegal memory access. When an element value is 0: in cache_mode=1 (continuous buffer), the state_cache write operation skips this position; in cache_mode=2 (cyclic buffer), neither read nor write operations skip it. The operator does not check for duplicates. The caller must ensure the uniqueness of element values: in cache_mode=1, the element value 0 is the "unallocated" sentinel value and can appear repeatedly, while non-zero values must be globally unique; in cache_mode=2, 0 is a valid physical block ID, and all element values must be globally unique. Duplicates cause multiple logical blocks/batches to write to the same physical block area, corrupting state_cache data.
- D supports 128/512.
- H supports 1K-10K and must be 512-aligned.
- block_size supports 1-1024.
- quant_mode is set to 1 (A8W8_A_HIFP8_PER_TENSOR_W_HIFP8_PER_CHANNEL). In this case, x_descale, wkv_descale, and wgate_descale are required inputs, where x_descale has a shape of [1,] (per-tensor scaling) and wkv_descale and wgate_descale have a shape of [coff\*D,] (per-channel scaling).
- The following three typical combination use cases are supported:
  - C4A: D=512, coff=2, cmp_ratio=4;
  - C4Li: D=128, coff=2, cmp_ratio=4;
  - C128A: D=512, coff=1, cmp_ratio=128.

## Deterministic Computation

- Defaults to a deterministic implementation.
- batch consistency: Set the deterministic level to 3 by calling torch_npu.npu.set_deterministic_level() to enable batch consistency. After batch consistency is enabled, computation results are independent of the batch size and batch position.

## Example

> **NOTE:**<br>
>
> - The following example uses the C128A use case (B=1, S=128, H=4096, D=512, coff=1, cmp_ratio=128). For more parameter combinations, see [Constraints](#constraints).

- Call in single-operator mode:

    ```python
    import torch
    import torch_npu
    import numpy as np
    from cann_ops_transformer import quant_compressor

    # Parameter settings
    B = 1
    S = 128
    H = 4096
    D = 512
    coff = 1  # 1: no overlap  2: overlap
    cmp_ratio = 128
    cache_mode = quant_compressor.CacheMode.LINEAR_BUFFER
    block_size = 128
    quant_mode = quant_compressor.QuantMode.A8W8_A_HIFP8_PER_TENSOR_W_HIFP8_PER_CHANNEL

    # block_table construction: when cache_mode=1, the shape is [B, ceil(Smax/block_size)]
    block_num = (S + block_size - 1) // block_size
    block_table = torch.zeros(size=(B, block_num), dtype=torch.int32)
    next_block_id = 1
    for i in range(B):
        for j in range(block_num):
            block_table[i][j] = next_block_id
            next_block_id = next_block_id + 1

    # Construct the input (HIFLOAT8 is carried as uint8 in torch)
    x = torch.randint(0, 256, (B, S, H), dtype=torch.uint8).npu()
    wkv = torch.randint(0, 256, (coff * D, H), dtype=torch.uint8).npu()
    wgate = torch.randint(0, 256, (coff * D, H), dtype=torch.uint8).npu()
    ape = torch.randn((cmp_ratio, coff * D), dtype=torch.float32).npu()
    state_cache = torch.zeros((torch.max(block_table).item() + 1, block_size, 2 * coff * D),
                              dtype=torch.float32).npu()
    x_descale = torch.rand((1,), dtype=torch.float32).npu()
    wkv_descale = torch.rand((coff * D,), dtype=torch.float32).npu()
    wgate_descale = torch.rand((coff * D,), dtype=torch.float32).npu()
    start_pos = torch.zeros((B,), dtype=torch.int32).npu()
    block_table = block_table.npu()

    # Call quant_compressor to perform the compression computation
    cmp_kv = quant_compressor(
        x,
        wkv,
        wgate,
        state_cache,
        ape,
        quant_mode=quant_mode,
        cmp_ratio=cmp_ratio,
        x_descale=x_descale,
        wkv_descale=wkv_descale,
        wgate_descale=wgate_descale,
        state_block_table=block_table,
        cu_seqlens=None,
        seqused=None,
        start_pos=start_pos,
        coff=coff,
        cache_mode=cache_mode
    )
    print(f"cmp_kv shape: {cmp_kv.shape}")
    ```

- Call in TorchAir graph mode:

    ```python
    import torch
    import torch_npu
    import numpy as np
    import torchair
    from cann_ops_transformer import quant_compressor
    from torchair.configs.compiler_config import CompilerConfig

    # Parameter settings
    B = 1
    S = 128
    H = 4096
    D = 512
    coff = 1
    cmp_ratio = 128
    cache_mode = quant_compressor.CacheMode.LINEAR_BUFFER
    block_size = 128
    quant_mode = quant_compressor.QuantMode.A8W8_A_HIFP8_PER_TENSOR_W_HIFP8_PER_CHANNEL

    block_num = (S + block_size - 1) // block_size
    block_table = torch.zeros(size=(B, block_num), dtype=torch.int32)
    next_block_id = 1
    for i in range(B):
        for j in range(block_num):
            block_table[i][j] = next_block_id
            next_block_id = next_block_id + 1

    x = torch.randint(0, 256, (B, S, H), dtype=torch.uint8).npu()
    wkv = torch.randint(0, 256, (coff * D, H), dtype=torch.uint8).npu()
    wgate = torch.randint(0, 256, (coff * D, H), dtype=torch.uint8).npu()
    ape = torch.randn((cmp_ratio, coff * D), dtype=torch.float32).npu()
    state_cache = torch.zeros((torch.max(block_table).item() + 1, block_size, 2 * coff * D),
                              dtype=torch.float32).npu()
    x_descale = torch.rand((1,), dtype=torch.float32).npu()
    wkv_descale = torch.rand((coff * D,), dtype=torch.float32).npu()
    wgate_descale = torch.rand((coff * D,), dtype=torch.float32).npu()
    start_pos = torch.zeros((B,), dtype=torch.int32).npu()
    block_table = block_table.npu()


    class QuantCompressorNetwork(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, wkv, wgate, state_cache, ape, block_table,
                    x_descale, wkv_descale, wgate_descale, start_pos):
            return torch.ops.cann_ops_transformer.quant_compressor(
                x, wkv, wgate, state_cache, ape,
                quant_mode=quant_mode,
                cmp_ratio=cmp_ratio,
                x_descale=x_descale,
                wkv_descale=wkv_descale,
                wgate_descale=wgate_descale,
                state_block_table=block_table,
                cu_seqlens=None,
                seqused=None,
                start_pos=start_pos,
                coff=coff,
                cache_mode=cache_mode
            )


    config = CompilerConfig()
    config.mode = "reduce-overhead"
    npu_backend = torchair.get_npu_backend(compiler_config=config)
    torch._dynamo.reset()
    npu_mode = torch.compile(QuantCompressorNetwork(), fullgraph=True, backend=npu_backend, dynamic=False)
    cmp_kv = npu_mode(x, wkv, wgate, state_cache, ape, block_table,
                     x_descale, wkv_descale, wgate_descale, start_pos)
    print(f"cmp_kv shape: {cmp_kv.shape}")
    ```
