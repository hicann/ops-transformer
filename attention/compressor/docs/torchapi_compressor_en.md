# compressor

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-21T08:27:25.835Z -->

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
- <term>Atlas 200I/500 A2 inference products</term>: Not supported
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas inference products</term>: Not supported
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas training products</term>: Not supported
<!-- end id6 -->

## Function

- **Description**: Compressor is a preprocessing operator for SMLA and QLI in inference use cases. It compresses every 4 or 128 tokens of KV cache into one, and each token then performs DSA computation with these compressed KV caches. For long sequences, Compressor effectively reduces computation overhead. The main computation process is as follows:
    1. Perform Matmul on the input $X$ and $W^{KV}$ to obtain $kv\_state$. Perform Matmul on the input $X$ and $W^{Gate}$, and then perform Add with $Ape$ to obtain $score\_state$. Update $kv\_state$ and $score\_state$ based on the input start_pos and cu_seqlens.
    2. When coff is 2, perform data rearrangement on $kv\_state$ and $score\_state$.
    3. Apply the softmax operation to $score\_state$, compute the Mul of the softmax result and $kv\_state$, and then perform the ReduceSum operation.

- **Formula**:

    1. Compute the matrix multiplication:

        $$
        C4A: \left[kv\_state^a, score\_state^a\right] = X @ \left[W^{aKV}, W^{aGate}\right]^{T}, \left[kv\_state^b, score\_state^b\right] = X @ \left[W^{bKV}, W^{bGate}\right]^{T};
        $$

        $$
        C128A: \left[kv\_state, score\_state\right] = X @ \left[W^{KV}, W^{Gate}\right]^{T}
        $$

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

    5. Perform grouped summation along the compression axis:

        $$
        C4A: C_{i}^{\text{Comp}} = \left[1\right]_{1\times8} @ (S_H)_i, ~i=1,2,\cdots, \frac{s}{4};
        $$

        $$
        C128A: C_{i}^{\text{Comp}} = \left[1\right]_{1\times128} @ (S_H)_i, ~i=1,2,\cdots, \frac{s}{128};
        $$

## Function Prototype

```python
cann_ops_transformer.compressor(
    x,
    wkv,
    wgate,
    state_cache,
    ape,
    cmp_ratio,
    *,
    state_block_table=None,
    cu_seqlens=None,
    seqused=None,
    start_pos=None,
    coff=1,
    cache_mode=1) -> Tensor
```

## Parameters

| Name | Type | Type | Description | Data Type | Dimension (shape) |
| ---- | ---- | ---- | ---- | ---- | ---- |
| x | Tensor | Required | Original uncompressed data, corresponding to $X$ in the formula. Non-contiguous tensors are not supported; the data format supports ND. | bfloat16, float16 | [B,S,H], [T,H] |
| wkv | Tensor | Required | KV compression weight, corresponding to $W^{KV}$ in the formula. Non-contiguous tensors are not supported; the data format supports ND. | bfloat16, float16 | [coff\*D,H] |
| wgate | Tensor | Required | Gate compression weight, corresponding to $W^{Gate}$ in the formula. Non-contiguous tensors are not supported; the data format supports ND. | bfloat16, float16 | [coff\*D,H] |
| state_cache | Tensor | Required | Historical data of kv_state and score_state, corresponding to $\left[kv\_state, score\_state\right]$ in the formula. Non-contiguous data on axis 0 is supported; the data format supports ND. After computation, kv_state and score_state are updated in place in this tensor. | float32 | [block_num, block_size, 2\*coff\*D], block_num > 0 required |
| ape | Tensor | Required | Positional biases, corresponding to $Ape$ in the formula. Non-contiguous tensors are not supported; the data format supports ND. | float32 | [cmp_ratio,coff\*D] |
| cmp_ratio | int | Required | Data compression ratio. An integer in the range [2, 128]. | - | - |
| state_block_table | Tensor | Optional | Block mapping table used for state_cache storage. Non-contiguous tensors are not supported; the data format supports ND. | int32 | When cache_mode=1, the shape is [B,ceil(Smax/block_size)], where Smax is the maximum Sequence Length across batches. When the shape of x is [B,S,H], Smax=max(start_pos)+S. When the shape of x is [T,H], Smax=max(start_pos)+max(cu_seqlens[n+1] - cu_seqlens[n]). When cache_mode=2, the shape is [B] |
| cu_seqlens | Tensor | Optional | Number of valid tokens in each batch. Non-contiguous tensors are not supported; the data format supports ND.<br>When the shape of x is [B,S,H], this parameter must be empty.<br>When the shape of x is [T,H], the input shape must be [B+1,]. This parameter is a prefix-sum array, where each element must be greater than or equal to the previous element, the first element must be 0, and the last element must be T. | int32 | [B+1,] |
| seqused | Tensor | Optional | Number of tokens that actually participate in compression in each batch. Non-contiguous tensors are not supported; the data format supports ND.<br>If set to None, the value equals the Sequence Length in each batch.<br>[B,S,H] use case: 0 ≤ seqused[n] ≤ S<br>[T,H] use case: 0 ≤ seqused[n] ≤ cu_seqlens[n+1] - cu_seqlens[n]. | int32 | [B,] |
| start_pos | Tensor | Optional | Start position for computation. Non-contiguous tensors are not supported; the data format supports ND. If the input is None, computation starts from 0. | int32 | [B,] |
| coff | int | Optional | Indicates whether to perform overlap data rearrangement. The default value is 1.<br>Only 1/2 is supported:<br>coff=1: no overlap data rearrangement is required.<br>coff=2: overlap data rearrangement is required. | int | - |
| cache_mode | int | Optional | Storage mode of state_cache. The default value is 1.<br>1: contiguous buffer.<br>2: circular buffer. | int | - |

<!-- npu="A3,910b" id7 -->
- <term>Atlas A3 products</term>: cmp_ratio supports only 2/4/8/16/32/64/128; gradEnabled cannot be set to true.
<!-- end id7 -->

## Return Values

| Name | Type | Type | Description | Data Type | Dimension (shape) |
| ---- | ---- | ---- | ---- | ---- | ---- |
| cmp_kv | Tensor | Required | Compressed data. Non-contiguous tensors are not supported. The supported data format is ND;<br>When the shape of x is [B,S,H], the output is the concatenation of (\<batch0\>compressed_tokens+pad0) +  (\<batch1\>compressed_tokens+pad1) + ... +  (\<batchN\>compressed_tokens+padN);<br>When the shape of x is [T,H], the output is the concatenation of \<batch0\>compressed_tokens + \<batch1\>compressed_tokens + ... + \<batchN\>compressed_tokens + pad.<br>Valid output range: compressed_tokens[n]=(start_pos[n]+Seqlen[n])/cmp_ratio-start_pos[n]/cmp_ratio, where Seqlen indicates the number of valid tokens in each batch. For the non-BS-merged layout, the valid results of each batch occupy the first compressed_tokens[n] rows; for the BS-merged layout, valid results are concatenated in batch order and the total valid range is the first sum(compressed_tokens[n]) rows. The remaining tail rows in the allocated capacity are not written by the operator. | bfloat16, float16 | x=[B,S,H]: [B,ceil(S/cmp_ratio),D]<br>x=[T,H]: [min(T,T//cmp_ratio+B),D] |

## Constraints

- This API supports use in inference scenarios.
- This API supports both the single-operator mode and the TorchAir graph mode (aclgraph).
- Dimensions of parameter x: B (Batch Size) denotes the batch size of input samples, S (Sequence Length) denotes the sequence length of input samples, H (Head Size) denotes the size of the hidden layer, D (Head Dim) denotes the size of the minimum unit of the hidden layer, and T denotes the accumulated sum of sequence lengths of input samples across all batches.
- This API supports generalized values of B and S, with the following use case restrictions:
  - In some long-sequence use cases, an error indicating that NPU memory is exceeded may be reported if the computation load is too high. Note: the computation load is affected by the shape of input x; the larger the values, the higher the computation load. Typical long-sequence use cases (that is, a large product of B and S, or a large T) include but are not limited to:
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
- This API supports B, S, and T being set to 0. That is, input parameters whose shapes depend on the values of B, S, and T allow empty tensors, while other input parameters do not support empty tensors. In this case, state_cache is not updated, and the output cmp_kv is an empty tensor.
- The value range of the state_block_table elements is [0, block_num), where block_num is the size of dimension 0 of state_cache. Element values are directly used as block indexes of state_cache. Out-of-range values cause invalid memory access. When the element value is 0: for cache_mode=1 (contiguous buffer), the write operation to state_cache skips this position; for cache_mode=2 (circular buffer), neither read nor write operations skip this position. The operator does not check for duplicates. The caller must ensure the uniqueness of element values: for cache_mode=1, the element value 0 is an "unallocated" sentinel value and can appear multiple times, while non-zero values must be globally unique; for cache_mode=2, 0 is a valid physical block number, and all element values must be globally unique. Duplicates cause multiple logical blocks/batches to write to the same physical block area, resulting in data overwrite in state_cache.
- D of 128/512 is supported.
- H in the range 1K-10K is supported and must be aligned to 512.
- block_size in the range 1-1024 is supported.
- The following three typical combination use cases are supported:
  - C4A: D=512, coff=2, cmp_ratio=4;
  - C4Li: D=128, coff=2, cmp_ratio=4;
  - C128A: D=512, coff=1, cmp_ratio=128.

## Deterministic Computation

- Defaults to a deterministic implementation.
<!-- npu="950" id8 -->
- <term>Ascend 950PR&950DT products</term>: batch consistency: Set the deterministic level to 3 using torch_npu.npu.set_deterministic_level() to enable batch consistency. After it is enabled, computation results are independent of the batch size and batch position.
<!-- end id8 -->

## Example

> **NOTE:**<br>
>
> - The following example uses the C128A use case (B=1, S=128, H=4096, D=512, coff=1, cmp_ratio=128). For more parameter combinations, see [Constraints](#constraints).

- Call in single-operator mode:

    ```python
    import torch
    import torch_npu
    import numpy as np
    import cann_ops_transformer

    # Set parameters
    B = 1
    S = 128
    H = 4096
    D = 512
    coff = 1  # 1: no overlap  2: overlap
    cmp_ratio = 128
    cache_mode = 1
    block_size = 128

    # block_table construction: when cache_mode=1, the shape is [B, ceil(Smax/block_size)]
    block_num = (S + block_size - 1) // block_size
    block_table = torch.zeros(size=(B, block_num), dtype=torch.int32)
    next_block_id = 1
    for i in range(B):
        for j in range(block_num):
            block_table[i][j] = next_block_id
            next_block_id = next_block_id + 1

    # Construct the input
    x = torch.randn((B, S, H), dtype=torch.bfloat16).npu()
    wkv = torch.randn((coff * D, H), dtype=torch.bfloat16).npu()
    wgate = torch.randn((coff * D, H), dtype=torch.bfloat16).npu()
    ape = torch.randn((cmp_ratio, coff * D), dtype=torch.float32).npu()
    state_cache = torch.zeros((torch.max(block_table).item() + 1, block_size, 2 * coff * D),
                              dtype=torch.float32).npu()
    start_pos = torch.zeros((B,), dtype=torch.int32).npu()
    block_table = block_table.npu()

    # Call compressor to perform the compression computation
    cmp_kv = cann_ops_transformer.compressor(
        x,
        wkv,
        wgate,
        state_cache,
        ape,
        cmp_ratio=cmp_ratio,
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
    import cann_ops_transformer
    from torchair.configs.compiler_config import CompilerConfig

    # Set parameters
    B = 1
    S = 128
    H = 4096
    D = 512
    coff = 1
    cmp_ratio = 128
    cache_mode = 1
    block_size = 128

    block_num = (S + block_size - 1) // block_size
    block_table = torch.zeros(size=(B, block_num), dtype=torch.int32)
    next_block_id = 1
    for i in range(B):
        for j in range(block_num):
            block_table[i][j] = next_block_id
            next_block_id = next_block_id + 1

    x = torch.randn((B, S, H), dtype=torch.bfloat16).npu()
    wkv = torch.randn((coff * D, H), dtype=torch.bfloat16).npu()
    wgate = torch.randn((coff * D, H), dtype=torch.bfloat16).npu()
    ape = torch.randn((cmp_ratio, coff * D), dtype=torch.float32).npu()
    state_cache = torch.zeros((torch.max(block_table).item() + 1, block_size, 2 * coff * D),
                              dtype=torch.float32).npu()
    start_pos = torch.zeros((B,), dtype=torch.int32).npu()
    block_table = block_table.npu()


    class CompressorNetwork(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, wkv, wgate, state_cache, ape, block_table, start_pos):
            return torch.ops.cann_ops_transformer.compressor(
                x, wkv, wgate, state_cache, ape,
                cmp_ratio=cmp_ratio,
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
    npu_mode = torch.compile(CompressorNetwork(), fullgraph=True, backend=npu_backend, dynamic=False)
    cmp_kv = npu_mode(x, wkv, wgate, state_cache, ape, block_table, start_pos)
    print(f"cmp_kv shape: {cmp_kv.shape}")
    ```
