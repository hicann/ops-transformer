# attn_epilogue

English | [简体中文](./torchapi_attn_epilogue.md)

## Product Support

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

## Description

- **Interface function**: `cann_ops_transformer.attn_epilogue` fuses inverse RoPE, two MXFP8 activation quantizations, and two-stage Matmul in the attention output projection into a single Torch custom operator call. This interface directly calls the DSL implementation and does not include communication or AllReduce. For complete grouping, layout, and scheduling details, see [AttnEpilogue](../README_en.md).

- **Computation semantics**:

    Let $A$ be the BF16 activation after inverse RoPE, $Q_{32}(\cdot)$ be MXFP8 quantization with one E8M0 scale per 32 elements along the K axis, and $W_{oA,g}$ and $W_{oB}$ be the two-stage weights. Then:

    $$
    A = \operatorname{BF16}\left(\operatorname{inverse\_rope}(x)\right)
    $$

    $$
    Y_g = \operatorname{BF16}\left(\operatorname{MXMatmul}\left(Q_{32}(A_g), W_{oA,g}^{T}\right)\right)
    $$

    $$
    out = \operatorname{BF16}\left(\operatorname{MXMatmul}\left(Q_{32}\left(\operatorname{concat}_{g}(Y_g)\right), W_{oB}^{T}\right)\right)
    $$

    inverse RoPE processes only the last 64 dimensions of each head; the first 448 dimensions are kept unchanged. The tail is paired by adjacent elements and computed in FP32 as `even' = even * cos + odd * sin` and `odd' = odd * cos - even * sin`, then rounded to BF16. The sin/cos tables use the repeated form `[s0,s0,s1,s1,...]`.

- **Grouping**: NG is the number of local groups handled by this operator. Each group contains 8 heads, with fixed `F=4096`, `O_LORA=1024`, and `DIM=5120`.

    | Scenario | Input x | Second-stage K |
    | :--- | :--- | ---: |
    | NG=8 | `[T,64,512]` | 8192 |
    | NG=2 | `[T,16,512]` | 2048 |
    | NG=1 | `[T,8,512]` | 1024 |

## Function Prototype

```python
import cann_ops_transformer

cann_ops_transformer.attn_epilogue(
    x: Tensor,
    woa: Tensor,
    wob: Tensor,
    descale_woa: Tensor,
    descale_wob: Tensor,
    rope_sin: Tensor,
    rope_cos: Tensor,
) -> Tensor
```

You can also import the interface with `from cann_ops_transformer.ops import attn_epilogue`. The registered name is `torch.ops.cann_ops_transformer.attn_epilogue`, and both single-operator mode and `torch.compile` graph mode are supported.

## Parameter Description

| Parameter | Type | Optional/Mandatory | Description | Data Type | Shape |
| :--- | :--- | :--- | :--- | :--- | :--- |
| x | Tensor | Mandatory | Attention output activation to project. inverse RoPE applies only to the last 64 dimensions of each head. The data format is ND. | bfloat16 | (T, NG*8, 512) |
| woa | Tensor | Mandatory | First-stage WoA weight. The shape is (NG, 1024, 4096) for an ND weight and the logical shape is (NG, 4096, 1024) for a private NZ weight. | float8_e4m3fn | See description |
| wob | Tensor | Mandatory | Second-stage WoB weight. The shape is (5120, NG*1024) for an ND weight and the logical shape is (NG*1024, 5120) for a private NZ weight. | float8_e4m3fn | See description |
| descale_woa | Tensor | Mandatory | First-stage weight MX scale, always ND. It is (NG, 1024, 64, 2) for ND weights and (NG, 64, 1024, 2) for NZ weights. | float8_e8m0fnu | See description |
| descale_wob | Tensor | Mandatory | Second-stage weight MX scale, always ND. It is (5120, NG*16, 2) for ND weights and (NG*16, 5120, 2) for NZ weights. | float8_e8m0fnu | See description |
| rope_sin | Tensor | Mandatory | sin table for inverse RoPE, repeated as `[s0,s0,s1,s1,...]`. The data format is ND. | float32 | (T, 64) |
| rope_cos | Tensor | Mandatory | cos table for inverse RoPE, repeated as `[c0,c0,c1,c1,...]`. The data format is ND. | float32 | (T, 64) |

## Return Value Description

| Parameter | Type | Description | Data Type | Shape |
| :--- | :--- | :--- | :--- | :--- |
| out | Tensor | Output projection result. Normal execution returns a Tensor with independent storage. The data format is ND. | bfloat16 | (T, 5120) |

## Constraints

- This interface supports training and inference scenarios.
- This interface supports single-operator mode and torch.compile (npugraph_ex) graph mode.
- `T >= 1`; `NG` supports only 1, 2, and 8.
- All inputs must be contiguous, reside on the same NPU, and match the shapes and dtypes in the parameter table. The two weight stages can independently use ND or NZ, and scales are always ND.
- NZ refers to a private tensor produced by `torch_npu.npu_format_cast(weight_kn,29)` that keeps the logical `(K,N)` axes; it is not a normal ND tensor with an extra packing dimension.
- The interface only checks the effective activation dimensions and identifies the layout from the logical weight axes. It does not scan each tensor's dtype, device, stride, or private format. Callers must guarantee these contracts when loading weights and cannot rely on the interface to detect every invalid input.
- When loading scales from a uint8 file, use `.view(torch.float8_e8m0fnu)` to restore the bit pattern; do not perform numeric conversion.
- Graph replay output buffers follow the backend's static reuse semantics; to retain results across replays, save them first with `out.clone()`.
- The current implementation provides forward only, not backward.

## Determinism/Batch Consistency

- Determinism: Deterministic computation is supported by default.
- Batch consistency: Batch consistency is supported by default.

## Example

- Single-operator mode

    ```python
    import torch
    import torch_npu
    from cann_ops_transformer.ops import attn_epilogue

    torch.npu.set_device(0)
    device = torch.device("npu:0")
    t, ng = 72, 8

    x = (torch.randn(t, ng * 8, 512, device=device) * 0.3).bfloat16()
    wa = (torch.randn(ng, 1024, 4096, device=device) * 0.3).bfloat16()
    wb = (torch.randn(5120, ng * 1024, device=device) * 0.2).bfloat16()

    woa, descale_woa = torch_npu.npu_dynamic_mx_quant(
        wa.reshape(-1, 4096), dst_type=292
    )
    wob, descale_wob = torch_npu.npu_dynamic_mx_quant(wb, dst_type=292)
    e8 = torch.float8_e8m0fnu
    woa = woa.reshape(ng, 1024, 4096)
    descale_woa = descale_woa.view(e8).reshape(ng, 1024, 64, 2)
    descale_wob = descale_wob.view(e8)

    angles = torch.outer(
        torch.arange(t).float() + 3,
        10000.0 ** (-torch.arange(0, 64, 2).float() / 64),
    )
    rope_sin = angles.sin().repeat_interleave(2, -1).to(device)
    rope_cos = angles.cos().repeat_interleave(2, -1).to(device)

    out = attn_epilogue(x, woa, wob, descale_woa, descale_wob, rope_sin, rope_cos)
    print(out.shape, out.dtype)
    ```

- Graph mode

    ```python
    import torch
    import torch_npu
    from cann_ops_transformer.ops import attn_epilogue

    class AttnEpilogueModel(torch.nn.Module):
        def forward(self, x, woa, wob, descale_woa, descale_wob, rope_sin, rope_cos):
            return attn_epilogue(
                x, woa, wob, descale_woa, descale_wob, rope_sin, rope_cos
            )

    model = torch.compile(
        AttnEpilogueModel().eval(),
        backend="npugraph_ex", fullgraph=True, dynamic=False,
        options={"clone_input": False},
    )
    out = model(x, woa, wob, descale_woa, descale_wob, rope_sin, rope_cos)
    print(out.shape, out.dtype)
    ```

    When private NZ weights are passed as call arguments, set `clone_input=False` to keep the storage addresses of NZ weights unchanged; rebuild the graph after replacing weight storage.
