# indexer_quant_cache

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-21T10:37:07.240Z -->

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

    Performs in-place dynamic quantization and compression updates on the KV Cache in the Epilog phase of the Indexer attention mechanism, encapsulating aclnnIndexerQuantCache. Dynamically quantizes float16/bfloat16 activation values x into FP8 (E4M3/E5M2) or MX-FP4 based on the quantization granularity (32 elements per group in mxfp8/mxfp4 mode, and an entire row per group in fp8/hifloat8 mode), and scatters the quantization results and corresponding scales to cache/cache_scale based on slot_mapping. Tokens with a value of -1 in slot_mapping are skipped. Supports four quantization modes: mxfp8, fp8, hifloat8, and mxfp4 (see quant_mode in Parameters for the meaning of each mode).

- Semantics:

    The underlying aclnn API is implemented in place for cacheRef and cacheScaleRef. This Python API also has in-place semantics: it directly updates the input cache and cache_scale, and rows not hit by slot_mapping retain their original values. In the schema, cache and cache_scale are marked as `Tensor(a!)` and `Tensor(b!)` respectively. **This API has no return value**. After the call, directly use the input cache and cache_scale.

## Function Prototype

```python
cann_ops_transformer.indexer_quant_cache(cache, cache_scale, x, slot_mapping, *, quant_mode="fp8", round_scale=True, x_scale=1.0) -> None
```

## Parameters

<table style="undefined;table-layout: fixed; width:1625px"><colgroup>
<col style="width: 147px">
<col style="width: 132px">
<col style="width: 132px">
<col style="width: 480px">
<col style="width: 330px">
<col style="width: 280px">
</colgroup>
<thead>
<tr>
    <th>Parameter</th>
    <th>Type</th>
    <th>Optional/Required</th>
    <th>Description</th>
    <th>Data Type</th>
    <th>Dimension (shape)</th>
</tr>
</thead>
<tbody>
    <tr>
        <td>cache</td>
        <td>Tensor</td>
        <td>Required</td>
        <td>Paged KV Cache of the current layer. <b>Only four-dimensional shapes are supported</b> (num_slots = blockNum × blockSize). <b>headDim must be ≥ d</b> (counted in fp4 elements for MX-FP4). <b>Non-contiguity is supported along the blockNum dimension</b> (paging). In MX-FP4 mode, cache stores FP4 data (2 fp4 values packed per byte). The data format is $ND$.</td>
        <td>float8_e4m3fn, float8_e5m2, uint8, float4_E2M1, float4_e1m2</td>
        <td>(blockNum, blockSize, 1, headDim)</td>
    </tr>
    <tr>
        <td>cache_scale</td>
        <td>Tensor</td>
        <td>Required</td>
        <td>Quantization scale factor for each block, <b>only four-dimensional shapes are supported</b>, and <b>scaleHeadDim must be ≥ scaleCol</b>. Verification applies to all quantization modes, including HiFloat8/quant_mode=2 (for details about scaleCol values and HiFloat8 rules, see Constraints). The data format is $ND$.</td>
        <td>float, float8_e8m0</td>
        <td>(blockNum, blockSize, 1, scaleHeadDim)</td>
    </tr>
    <tr>
        <td>x</td>
        <td>Tensor</td>
        <td>Required</td>
        <td>Activation input to be quantized. d must satisfy d % 32 == 0 and d ≤ 8192. The data format is $ND$.</td>
        <td>float16, bfloat16</td>
        <td>(bs, d)</td>
    </tr>
    <tr>
        <td>slot_mapping</td>
        <td>Tensor</td>
        <td>Required</td>
        <td>Index mapping from tokens to cache slots. The element value range is [-1, num_slots - 1], where -1 indicates that the token is skipped; the number of dimensions must be 1 less than that of x. The data format is $ND$.</td>
        <td>int32</td>
        <td>(bs)</td>
    </tr>
    <tr>
        <td>quant_mode</td>
        <td>str</td>
        <td>Optional</td>
        <td>Quantization mode (string, case-insensitive; internally mapped by TorchNPU through an enumeration to an operator-side int). Default value: <code>"fp8"</code>. Optional values and their meanings:
            <ul>
                <li><code>"mxfp8"</code> (internal 0): MX-FP8 microscaling quantization. x is grouped every 32 elements, the output is FP8 (e4m3/e5m2), each group has one e8m0 (power-of-2 exponent) scale, and cache_scale is float8_e8m0.</li>
                <li><code>"fp8"</code> (internal 1): per-token dynamic FP8 quantization. An entire row (headDim) is one group, the output is FP8 (e4m3/e5m2), each row has one float32 scale (scaleCol=1), and cache_scale is float.</li>
                <li><code>"hifloat8"</code> (internal 2): HiFloat8 static quantization (scaled by x_scale and then converted to HiFloat8). An entire row is one group, and cache_scale is float.</li>
                <li><code>"mxfp4"</code> (internal 3): MX-FP4 microscaling quantization. x is grouped every 32 elements, the output is FP4 (2 values packed per byte), each group has one e8m0 scale, and cache_scale is float8_e8m0.</li>
            </ul>
            In the following Constraints section, quant_mode=0/1/2/3 refers to the corresponding internal int values listed above.</td>
        <td>STR</td>
        <td>-</td>
    </tr>
    <tr>
        <td>round_scale</td>
        <td>bool</td>
        <td>Optional</td>
        <td>Whether to round the per-row scale up to a power of 2, only in fp8 mode (internal quant_mode=1, Normal). In mxfp8 mode, rounding is fixed and is not controlled by this flag. Default value: True.</td>
        <td>BOOL</td>
        <td>-</td>
    </tr>
    <tr>
        <td>x_scale</td>
        <td>float</td>
        <td>Optional</td>
        <td>Global scale multiplier in HiFloat8 mode (quant_mode=2). Default value: 1.0.</td>
        <td>float</td>
        <td>-</td>
    </tr>
</tbody>
</table>

## Return Values

This API has no return value. The computation results are written in-place directly to the input tensors `cache` and `cache_scale`. After the computation, the shapes and dtypes of both tensors remain unchanged. Rows not hit by `slot_mapping` retain their original values.

## Constraints

- The current Torch API does not support HiFloat4 (mode4) available through ACLNN/GE. `torch.uint8` stores floating-point codes: cache carries E4M3 in mxfp8/fp8 mode, HiFloat8 in hifloat8 mode, and two E2M1 values per byte in mxfp4 mode; uint8 cache_scale carries E8M0 in mxfp8/mxfp4 mode.

- This API supports use in inference scenarios.
- This API can be called in single-operator mode.
- The combination of data types of cache and cache_scale must match quant_mode: for fp8/hifloat8 mode, cache_scale is float; for mxfp8/mxfp4 mode, cache_scale is float8_e8m0; for mxfp4 mode, cache is FP4 (packed as uint8).
- Both cache and cache_scale (in all quantization modes, including HiFloat8) support only the four-dimensional shape `[blockNum, blockSize, 1, headDim]`, where the second-to-last dimension is fixed to 1 (one quantization vector per token); num_slots = blockNum × blockSize.
- cache/cache_scale supports non-contiguous layout only along the blockNum dimension: blocks need not be tightly packed, but data within each block must be contiguous.
- headDim length constraints:
  - cache.headDim ≥ d (MX-FP4 is counted in fp4 elements; d fp4 values occupy ⌈d/2⌉ bytes);
  - cache_scale.headDim ≥ scaleCol, where scaleCol = ⌈d/32⌉ for mxfp8/mxfp4, and 1 for fp8/hifloat8;
  - Example: d=128, quant_mode=0 → scaleCol=4 → cache.headDim ≥ 128 and cache_scale.headDim ≥ 4.
- The dimension of slot_mapping must equal the dimension of x minus 1, meaning slot_mapping is obtained by flattening all dimensions of x except the last dimension.
- Tokens with a value of -1 in slot_mapping are skipped without processing; other valid elements must be in the range [0, num_slots - 1] and must not be duplicated. Otherwise, result correctness is not guaranteed.

## Deterministic Computation

Defaults to a deterministic implementation.

## API Call

- Calling in single-operator mode

    ```python
    import torch
    import torch_npu
    from cann_ops_transformer.ops import indexer_quant_cache

    # cache/cache_scale is 4D [blockNum, blockSize, 1, headDim], and num_slots = blockNum*blockSize
    block_num, block_size, d = 128, 16, 128   # num_slots = 2048; mode1 headDim=d=128 >= d
    bs = 1024

    cache = torch.zeros(block_num, block_size, 1, d, dtype=torch.float8_e4m3fn).npu()
    cache_scale = torch.zeros(block_num, block_size, 1, 1, dtype=torch.float32).npu()  # mode1 scaleCol=1
    x = torch.randn(bs, d, dtype=torch.float16).npu()
    slot_mapping = torch.randint(0, block_num * block_size, (bs,), dtype=torch.int32).npu()

    indexer_quant_cache(
        cache, cache_scale, x, slot_mapping, quant_mode="fp8", round_scale=True, x_scale=1.0)
    print(cache.shape, cache.dtype, cache_scale.shape, cache_scale.dtype)
    ```

- Calling in graph mode (torchair)

    Reuse the cache/cache_scale/x/slot_mapping constructed in the preceding single-operator example, compile with the torchair backend, and then call the API:

    ```python
    import torchair

    class IndexerQuantCacheModel(torch.nn.Module):
        def forward(self, cache, cache_scale, x, slot_mapping):
            indexer_quant_cache(
                cache, cache_scale, x, slot_mapping,
                quant_mode="fp8", round_scale=True, x_scale=1.0)
            return cache, cache_scale

    model = IndexerQuantCacheModel().npu()
    npu_backend = torchair.get_npu_backend()
    model = torch.compile(model, backend=npu_backend, dynamic=False)
    model(cache, cache_scale, x, slot_mapping)
    print(cache.shape, cache.dtype, cache_scale.shape, cache_scale.dtype)
    ```
