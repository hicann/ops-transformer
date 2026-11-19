# AttnEpilogue

English | [简体中文](./README.md)

## Product Support

| Product | Supported |
| ---- | :------: |
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term> | × |
| <term>Atlas A2 products</term> | × |
| <term>Atlas 200I/500 A2 inference products</term> | × |
| <term>Atlas inference products</term> | × |
| <term>Atlas training products</term> | × |

## Description

- Operator function: Local attention output projection on Ascend 950PR/950DT (ARCH 3510). It fuses inverse RoPE, two MXFP8 activation quantizations, and two-stage Matmul into one device kernel. The implementation is in [attn_epilogue.py](op_dsl/attn_epilogue.py) and does not include communication or AllReduce.

- Calculation formula:

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

    inverse RoPE processes only the last 64 dimensions of each head; the first 448 dimensions are kept unchanged. The tail is paired by adjacent elements and computed in FP32 as `even' = even * cos + odd * sin` and `odd' = odd * cos - even * sin`, then rounded to BF16. The sin/cos tables use the repeated form `[s0,s0,s1,s1,...]`. NG is the number of local groups handled by this operator; the operator only computes the output projection for these groups and does not perform cross-device communication or result aggregation.

    Any $T \ge 1$ is supported, subject to device memory. Normal execution returns a BF16 `[T,5120]` output with independent storage.

- Implementation notes:

    MM1 accumulates in FP32 and rounds to BF16 before QuantY. FP32 tiles are delivered directly to the paired AIV through Fixpipe, eliminating the BF16 intermediate Y GM buffer. Internally only four byte workspaces are kept: Aq, A-scale, Yq, and Y-scale.

    The implementation uses explicit layout, RingBuffer, and copy_options from DSL 0.8, and explicitly enables MX Matmul. Constructor layout/shape differences are identified once at import time instead of being re-checked in every forward. Fixpipe delivers complete tiles as two half-frames to AIVs; each AIV independently trims the valid rows. Empty shards still consume receive slots to complete ownership handoff but do not write to GM. Tail blocks do not rely on configurations enumerated by T or platform.

    M/N/K tiling is derived from T, matrix dimensions, the current stream core quota, and the L0/L1/UB capacities of ARCH 3510; there are no configurations enumerated by T or device name. MM1 N tiles are aligned to complete MX scale byte pairs; when the producer coverage condition is met, QuantA and MM1 are pipelined by K slab. Cube buffer reuse and the QuantY → MM2 data/scale readiness dependencies keep explicit synchronization.

    Once the compilation cache hits, the handle is reused without recomputing tiling. Normal execution caches workspaces by device, stream, T, and NG; graph capture uses the graph memory pool. Cross-stream dependencies are established by the caller. The first call includes compilation overhead and should be warmed up. The current implementation provides forward only, not backward.

## Parameter Description

| Parameter | Input/Output/Attribute | Description | Data Type | Data Format |
| :--- | :--- | :--- | :--- | :--- |
| x | Input | Attention output activation to project, with shape `[T,NG*8,512]`. inverse RoPE applies only to the last 64 dimensions of each head. | BFLOAT16 | ND |
| woa | Input | First-stage WoA weight. The ND logical shape is `[NG,1024,4096]`; the NZ logical shape is `[NG,4096,1024]`. | FLOAT8_E4M3FN | ND, private NZ |
| wob | Input | Second-stage WoB weight. The ND logical shape is `[5120,NG*1024]`; the NZ logical shape is `[NG*1024,5120]`. | FLOAT8_E4M3FN | ND, private NZ |
| descale_woa | Input | First-stage weight MX scale, always ND. It is `[NG,1024,64,2]` for ND weights and `[NG,64,1024,2]` for NZ weights. | FLOAT8_E8M0FNU | ND |
| descale_wob | Input | Second-stage weight MX scale, always ND. It is `[5120,NG*16,2]` for ND weights and `[NG*16,5120,2]` for NZ weights. | FLOAT8_E8M0FNU | ND |
| rope_sin | Input | sin table for inverse RoPE, with shape `[T,64]`, repeated as `[s0,s0,s1,s1,...]`. | FLOAT32 | ND |
| rope_cos | Input | cos table for inverse RoPE, with shape `[T,64]`, repeated as `[c0,c0,c1,c1,...]`. | FLOAT32 | ND |
| out | Output | Output projection result, with shape `[T,5120]`. | BFLOAT16 | ND |

Each group contains 8 heads, with fixed `F=4096`, `O_LORA=1024`, and `DIM=5120`. The three group scenarios are as follows:

| Scenario | Input x | Second-stage K |
| :--- | :--- | ---: |
| NG=8 | `[T,64,512]` | 8192 |
| NG=2 | `[T,16,512]` | 2048 |
| NG=1 | `[T,8,512]` | 1024 |

## Constraints

- `T >= 1`; `NG` supports only 1, 2, and 8; the shape of `x` must be `[T,NG*8,512]`.
- All inputs must be contiguous, reside on the same NPU, and match the shapes and dtypes in the parameter table. The two weight stages can independently use ND or NZ, and scales are always ND.
- NZ refers to a private tensor produced by `torch_npu.npu_format_cast(weight_kn,29)` that keeps the logical `(K,N)` axes; it is not a normal ND tensor with an extra packing dimension.
- The entry only checks the effective activation dimensions and identifies the layout from the logical weight axes. It does not scan each tensor's dtype, device, stride, or private format. Callers must guarantee these contracts when loading weights and cannot rely on the entry to detect every invalid input.
- When loading scales from a uint8 file, use `.view(torch.float8_e8m0fnu)` to restore the bit pattern; do not perform numeric conversion.
- The current implementation provides forward only, not backward.

## Calling Instructions

| Calling Method | Sample Code | Description |
| :--- | :--- | :--- |
| PyTorch API | [torchapi_attn_epilogue_en.md](docs/torchapi_attn_epilogue_en.md) | Calls the AttnEpilogue operator through `cann_ops_transformer.attn_epilogue`, including the full prototype, parameters, return value, and examples. |
| Direct DSL call | [op_dsl/attn_epilogue.py](op_dsl/attn_epilogue.py) | Adds `attention/attn_epilogue/op_dsl` to `PYTHONPATH` and calls the DSL implementation through `from attn_epilogue import attn_epilogue`. |
| Graph mode | [Graph Mode](#graph-mode) | Calls through `torch.npu.NPUGraph` or the `npugraph_ex` backend of `torch.compile`. |
| Unit test | [tests/pytest](tests/pytest) | Six NPU precision cases and two Meta cases. |

### Pre-converting ND Weights to Private NZ

Conversion is performed only during initialization and must not be placed inside the measured forward. Byte-wise conversion avoids device-transpose restrictions on FP8/E8M0 in some CANN versions and does not change quantized values.

```python
import torch
import torch_npu


def weight_to_nz(weight):
    """Preserve quantized bytes while converting logical (N,K) to private NZ (K,N)."""
    kn = weight.view(torch.uint8).cpu().transpose(-2, -1).contiguous()
    return torch_npu.npu_format_cast(kn.npu().view(torch.float8_e4m3fn), 29)


wa_nz = weight_to_nz(wa_nd)
wb_nz = weight_to_nz(wb_nd)
sa_kn = sa_nd.view(torch.uint8).cpu().permute(0, 2, 1, 3).contiguous().npu()
sb_kn = sb_nd.view(torch.uint8).cpu().permute(1, 0, 2).contiguous().npu()
out = attn_epilogue(x, wa_nz, wb_nz,
    sa_kn.view(torch.float8_e8m0fnu), sb_kn.view(torch.float8_e8m0fnu), sin, cos)
```

### Graph Mode

`torch.npu.NPUGraph` and the `npugraph_ex` backend of `torch.compile` are supported. In a model, register the quantized weights and scales as fixed buffers to preserve their private NZ storage:

```python
class Projection(torch.nn.Module):
    """Keep quantized weights as fixed graph parameters."""

    def __init__(self, woa, wob, sa, sb):
        """Retain the initialized NPU tensors without converting their formats."""
        super().__init__()
        for name, value in zip(("woa", "wob", "sa", "sb"), (woa, wob, sa, sb)):
            self.register_buffer(name, value)

    def forward(self, x, sin, cos):
        """Project the current activations with the fixed quantized weights."""
        return attn_epilogue(x, self.woa, self.wob, self.sa, self.sb, sin, cos)


model = torch.compile(
    Projection(woa, wob, descale_woa, descale_wob).eval(),
    backend="npugraph_ex", fullgraph=True, dynamic=False,
)
out = model(x, rope_sin, rope_cos)
```

When the same model is used across T, set `dynamic=True`; switching among T=3, 7, 16, and 17 has been verified. Graph replay output buffers follow the backend's static reuse semantics; to retain results across replays, save them first with `out.clone()`.

To directly compile the seven-argument function and pass private NZ weights as call arguments, use the backend's `clone_input=False`:

```python
compiled_op = torch.compile(
    attn_epilogue, backend="npugraph_ex", fullgraph=True, dynamic=False,
    options={"clone_input": False},
)
out = compiled_op(x, woa, wob, descale_woa, descale_wob, rope_sin, rope_cos)
```

In this case, update data through fixed input buffers and keep the storage addresses of NZ weights unchanged; rebuild the graph after replacing weight storage. By default, the current torch_npu backend may convert private NZ to ND when cloning user inputs and then attempt an unsupported FP8 private-format copy. This happens outside the operator; both the buffer-parameter approach and disabling input cloning have been verified. ND arguments can use the backend's default configuration. The operator does not change global format switches and does not repack weights inside forward.

### Testing

The in-repo test is [test_attn_epilogue_dsl.py](tests/pytest/test_attn_epilogue_dsl.py): six NPU precision cases cover the three NG values, ND/NZ, tail blocks, NPUGraph, and one npugraph_ex regression, plus two Meta cases. It does not include performance collection, large-T sweeps, external `cann_ops_*` dependencies, or full compilation tasks, and does not modify public CI configuration.

```bash
# Use the repository's common markers to select the CPU / NPU stages.
python3 -m pytest -q attention/attn_epilogue/tests/pytest -m 'not npu'
ASCEND_RT_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 \
  python3 -m pytest -q attention/attn_epilogue/tests/pytest -m npu
```

Matching CANN, torch_npu, and DSL 0.8.0 are required; the NZ path uses `TensorSpec(storage_format="nz")`. This 0.8.0 adaptation was verified on 950DT with CANN 9.3.0, torch 2.12.0+cpu, and torch_npu 2.12.0.post2. Historical 950PR validation used DSL 0.7.1. Full precision, stress, performance, and raw profiling are maintained by external development validation scripts; online CI does not collect these scripts.

## References

- [PyTorch API documentation (Chinese)](docs/torchapi_attn_epilogue.md)
- [PyTorch API documentation (English)](docs/torchapi_attn_epilogue_en.md)
- [Operator unit test](tests/pytest/test_attn_epilogue_dsl.py)

## Performance (DSL 0.8.0)

On 2026-11-02, 16 representative points were re-measured on Ascend950DT using the implementation with on-chip capacity constraint fixes, CANN 9.3.0, DSL 0.8.0, torch 2.12.0+cpu, and torch_npu 2.12.0.post2, with 32 AIC / 64 AIV and 128 MiB L2. The 16-point re-measurement was performed; the 576-point full sweep was not re-run.

The table reports median device kernel time in microseconds; the native chain is the sum of five kernels.

| Scenario | Weight layout | T | Native chain (µs) | Fused operator (µs) | Speedup |
| :--- | :--- | ---: | ---: | ---: | ---: |
| NG=8 | ND | 1 | 48.178 | 29.924 | 1.610× |
| NG=8 | ND | 72 | 56.465 | 41.449 | 1.362× |
| NG=8 | ND | 128 | 66.463 | 49.918 | 1.331× |
| NG=8 | ND | 256 | 87.355 | 73.464 | 1.189× |
| NG=8 | NZ | 1 | 45.078 | 28.239 | 1.596× |
| NG=8 | NZ | 72 | 55.338 | 40.445 | 1.368× |
| NG=8 | NZ | 128 | 63.409 | 49.397 | 1.284× |
| NG=8 | NZ | 256 | 96.873 | 73.363 | 1.320× |
| NG=2 | ND | 1 | 35.511 | 19.578 | 1.814× |
| NG=2 | ND | 16 | 37.096 | 21.400 | 1.733× |
| NG=2 | NZ | 1 | 29.942 | 18.431 | 1.625× |
| NG=2 | NZ | 16 | 31.898 | 21.635 | 1.474× |
| NG=1 | ND | 1 | 30.793 | 14.036 | 2.194× |
| NG=1 | ND | 16 | 34.368 | 17.791 | 1.932× |
| NG=1 | NZ | 1 | 26.480 | 13.266 | 1.996× |
| NG=1 | NZ | 16 | 30.030 | 16.925 | 1.774× |

The native five-kernel chain and the fused single kernel use the same inputs and the same ND/NZ weights, with interleaved graph replays; each point uses 48 samples per side, and before each call a 256 MiB buffer is read on the same stream for software L2 eviction. Timing excludes eviction, compilation, input restoration, and host scheduling; speedup is native median / fused median. This metric does not represent end-to-end request latency. All 16 points gain, with speedups of 1.189–2.194×. Normal execution and 16 graph-capture outputs per point are bitwise identical to the native reference.
