# FusedQuantQkvProjNormRopeCache

English | [简体中文](./README.md)

## Product Support

|Product             |  Supported  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     √    |
|  <term>Atlas A3 products</term>   |     ×    |
|  <term>Atlas A2 products</term>     |     ×    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     ×    |
|  <term>Atlas inference products</term>    |     ×    |
|  <term>Atlas training products</term>    |     ×    |

## Function Description

- Operator function: takes the activation and weights that have been MXFP8-quantized by the caller, performs the fused dequantized QKV projection (MatMul), K RmsNorm, partial Q/K RoPE, and K/V cache update operations, and outputs q_out, k_out, v_out, mirror_k_out (optional), and mirror_v_out (optional), while updating k_cache and v_cache in place. This operator is designed for inference scenarios and supports both plain layers and KV mirror source layers.
- The scenarios currently supported by this operator are as follows:

  |         Scenario Type           |            Overview            |
  |:----------------------|:-----------------------------|
  |<ul><li>Plain layer (mirror outputs not provided)</li><li>Activation and weights have been MXFP8-quantized</li></ul>|Let the widths of Q and KV be QW = H<sub>q</sub> * 256 and KVW = H<sub>kv</sub> * 256 respectively; the projection output order is [Q, K, V]. The main computation process and output mapping: <br><ul><li>X, W go through Dequant, MatMul->C</li><li>Q goes through RoPE->q_out</li><li>K goes through RmsNorm, RoPE->k_out, k_cache</li><li>V goes through passthrough->v_out, v_cache</li></ul>|
  |<ul><li>Mirror source layer (mirror outputs provided)</li><li>Activation and weights have been MXFP8-quantized</li></ul>|The projection output order is [Q, K, V, mirror K, mirror V], and N = (H<sub>q</sub>+4 * H<sub>kv</sub>) * 256. The computation of Q/K/V is the same as that of a plain layer. In addition: <br><ul><li>mirror K and mirror V are the raw projection values, without RmsNorm or RoPE, and go through passthrough->mirror_k_out, mirror_v_out</li></ul>|

- Computation formula:

  (1) Dequantization and projection (MatMul):

  In the following formula, X is the quantized activation [M, K], W is the quantized weight [N, K], and S<sub>A</sub> and S<sub>B</sub> are the corresponding E8M0 scaling factors. Dequant restores floating-point values per group of 32 elements using the corresponding E8M0 scaling factor, as specified by the MX standard; BF16_RN denotes rounding to BF16 in RN mode. For plain layers, N = (H<sub>q</sub>+2 * H<sub>kv</sub>) * 256; for mirror source layers, N = (H<sub>q</sub>+4 * H<sub>kv</sub>) * 256.

  $$
  C = \text{BF16\_RN}(\mathrm{Dequant}(X, S_A) @ \mathrm{Dequant}(W, S_B)^{\mathrm{T}})
  $$

  (2) RmsNorm:

  Here, $x$ and $y$ denote the input tensor and the output tensor of RmsNorm respectively. The normalization is performed along the last dimension (feature dimension), and this computation rule applies only to the K component (each head's 256 dimensions are computed independently). Q is not normalized.

  $$
  \mathrm{squareX} = x \cdot x
  $$

  $$
  \mathrm{meanSquareX} = \mathrm{mean}(\mathrm{squareX}, \text{dim} = -1, \text{keepdim} = \text{True})
  $$

  $$
  \mathrm{rms} = \sqrt{\mathrm{meanSquareX} + \epsilon}
  $$

  $$
  y = \frac{x}{\mathrm{rms}} \cdot \gamma
  $$

  (3) RoPE (NeoX rotate-half):

  Here, $y$ denotes the input involved in the RoPE computation: the Q component is the projection result $C$, and the K component is the RmsNorm output rounded by BF16_RN. RoPE acts only on the last 64 dimensions of each head (the first 32 dimensions are $y_1$ and the last 32 dimensions are $y_2$), while the other dimensions pass through; cos and sin are given by the cos_sin table row indexed by positions (the first 32 values of each row are cos and the last 32 values are sin).

  $$
  y_1 = y[\ldots, -64:-32]
  $$

  $$
  y_2 = y[\ldots, -32:]
  $$

  $$
  y_{\mathrm{RoPE}} = \mathrm{concat}(-y_2, y_1)
  $$

  $$
  y_{\mathrm{embed}} = y \cdot \cos + y_{\mathrm{RoPE}} \cdot \sin
  $$

  (4) Cache write (Scatter) and row mapping:

  The K/V computation results are written to both the dense outputs and the cache. The cache rows to write are specified by slot_mapping; a slot of -1 skips the cache write. In chunk mode (L = tokens_per_rank, C = chunk_tokens_per_rank, a = chunk_token_offset), local input row i is mapped to the output and metadata row:

  $$
  g(i) = \lfloor i/C \rfloor \times L + a + (i \bmod C)
  $$

## Parameter Description

The interface uses pre-allocated outputs and caches, modifies them in place, and returns `None`.

| Parameter | Input/Output/Attribute | Description | Data Type | Data Format |
| --- | --- | --- | --- | --- |
| `hidden_quant` | Input | `[M, K]`. Caller-quantized activation payload. K is a positive multiple of 256. | FLOAT8_E4M3FN / FLOAT8_E5M2 | ND |
| `hidden_scale` | Input | `[M, K//64, 2]`. One positive E8M0 scale per 32 activation elements, packed as pairs along the last dimension. | FLOAT8_E8M0 / INT8 / UINT8 | ND |
| `weight_quant` | Input | `[N, K]`. Caller-quantized weight payload; its FP8 format may differ from hidden_quant. | FLOAT8_E4M3FN / FLOAT8_E5M2 | ND |
| `weight_scale` | Input | `[N, K//64, 2]`. One E8M0 scale per 32 weight elements. | FLOAT8_E8M0 / INT8 / UINT8 | ND |
| `k_gamma` | Input | `[256]`. One-dimensional K RmsNorm gamma shared by all K heads. | BFLOAT16 | ND |
| `positions` | Input | `[T]`. Position indices. Initialize every used row g(i), including each tile's first row across tile and chunk boundaries. | INT64 | ND |
| `cos_sin` | Input | `[P, 64]`. RoPE table: 32 cosine values followed by 32 sine values per row. | FLOAT32 | ND |
| `slot_mapping` | Input | `[T]`. Cache slots. Used values are -1 (skip cache write) or unique nonnegative integers in [0,S). | INT64 | ND |
| `q_out` | Input/Output | `[T, QW]`. Preallocated Q buffer. Unselected chunk rows retain input values; initialize them before reading them. | BFLOAT16 | ND |
| `k_out` | Input/Output | `[T, KVW]`. Preallocated K buffer. Unselected chunk rows retain input values; initialize them before reading them. | BFLOAT16 | ND |
| `v_out` | Input/Output | `[T, KVW]`. Preallocated V buffer. Unselected chunk rows retain input values; initialize them before reading them. | BFLOAT16 | ND |
| `k_cache` | Input/Output | `[S, KVW]`. K cache updated in place; unwritten rows retain input values. | BFLOAT16 | ND |
| `v_cache` | Input/Output | `[S, KVW]`. V cache updated in place; unwritten rows retain input values. | BFLOAT16 | ND |
| `mirror_k_out` | Optional input/output | `[T, KVW]`. Raw mirror-K projection without RmsNorm/RoPE. Both mirrors must be present together or both absent. Unselected chunk rows retain input values and must be initialized before reading. | BFLOAT16 | ND |
| `mirror_v_out` | Optional input/output | `[T, KVW]`. Raw mirror-V projection without RmsNorm/RoPE. Both mirrors must be present together or both absent. Unselected chunk rows retain input values and must be initialized before reading. | BFLOAT16 | ND |
| `epsilon` | Optional attribute | Finite, strictly positive RmsNorm epsilon. Default: 1e-6. | FLOAT32 | - |
| `positions_contiguous` | Optional attribute | Default: False. True reads the first position of each Q/K processing tile and derives the remaining tile positions by incrementing it. Initialize all used positions and guarantee continuity. | BOOL | - |
| `tokens_per_rank` | Optional attribute | Chunk output stride L. All three chunk arguments must be supplied together or omitted together. | INT64 | - |
| `chunk_tokens_per_rank` | Optional attribute | Local chunk length C. All three chunk arguments must be supplied together or omitted together. | INT64 | - |
| `chunk_token_offset` | Optional attribute | Chunk offset a. All three chunk arguments must be supplied together or omitted together. | INT64 | - |

## Constraints

- M >= 0, K is a positive multiple of 256, and head dimension is 256. Let QW = Hq*256 and KVW = Hkv*256, with Hq, Hkv > 0. N = QW+2*KVW for plain layers and N = QW+4*KVW for mirror source layers. S, P > 0.
- INT8/UINT8 `hidden_scale` and `weight_scale` tensors carry raw E8M0 encodings. Use a bitwise `view`, not an integer numeric conversion.
- `mirror_k_out` and `mirror_v_out` must both be present or both be absent.
- Without chunking, T = M. With chunking, L > 0, C > 0, M%C = 0, 0 <= a, a+C <= L, T = (M//C)*L, and g(i) = (i//C)*L+a+i%C. Only dense/mirror rows g(i) are written. Initialize all other rows that must retain meaningful values before reading them.
- Every used position is in [0,P); every used slot is -1 or an integer in [0,S). Used nonnegative slots must be unique within a call. -1 skips only cache writes; dense and mirror outputs are still produced. Unused metadata rows do not participate in computation.
- `positions_contiguous=True` promises that positions[g(i)] advance by one in local processing order. Each Q/K tile reloads its first position at g(local_row_start) and derives the remaining tile positions by their local offsets. Initialize all used positions, including across tile and chunk boundaries; initializing only the first element of the entire tensor is insufficient.
- `epsilon` is finite and strictly positive. All tensors are contiguous, on the same NPU, and have `requires_grad=False`. Writable tensors must not overlap each other or any input.
- M = 0 validates metadata and returns without modifying buffers. Index values are not checked at runtime. The caller guarantees bounds, uniqueness, continuity, cross-stream dependencies and cache-write ordering.

## Calling Instructions

| Calling Method | Calling Sample | Description |
| --- | --- | --- |
| Torch API | [test_fused_quant_qkv_proj_norm_rope_cache](examples/test_fused_quant_qkv_proj_norm_rope_cache.py) | Uses `torch.ops.cann_ops_transformer.fused_quant_qkv_proj_norm_rope_cache`. |

The Host selects tiling by M: M <= 3072 uses baseM = 128 and kL1 = 512; larger M uses baseM = 256 and kL1 = 256. Both profiles are precompiled into the RUN. Work traverses M within 4-row token-tile windows and reverses head order between adjacent windows.

### Build and Deployment

Run from the repository root with matching CANN and Torch/TorchNPU installations. Replace `/path/to` with actual paths:

```bash
source /path/to/cann/set_env.sh
bash build.sh --pkg --experimental --soc=ascend950 \
  --ops=fused_quant_qkv_proj_norm_rope_cache --vendor_name=custom
bash build.sh --torch_extension --experimental \
  --ops=fused_quant_qkv_proj_norm_rope_cache --vendor_name=custom
```

Install the RUN matching the current host architecture and its companion wheel, then load the environment:

```bash
bash build_out/cann-ops-transformer-custom_linux-*.run --quiet --install-path=/path/to/opp
python3 -m pip install --force-reinstall --no-deps build_out/cann_ops_transformer_custom-*.whl
source /path/to/opp/vendors/custom_transformer/bin/set_env.bash
```

The vendor suffix in the wheel filename follows repository packaging rules; calls use the standard `torch.ops.cann_ops_transformer` namespace.

### Run Example

Run from the repository root (`PYTHONPATH` provides the base package when the wheel is not installed and can be omitted otherwise):

```bash
export PYTHONPATH="$PWD/torch_extension:${PYTHONPATH:-}"
OP=experimental/posembedding/fused_quant_qkv_proj_norm_rope_cache
python3 "$OP/examples/test_fused_quant_qkv_proj_norm_rope_cache.py"
python3 "$OP/examples/test_fused_quant_qkv_proj_norm_rope_cache.py" --mirror --chunk --contiguous
```

The [standalone example](examples/test_fused_quant_qkv_proj_norm_rope_cache.py) constructs fully preallocated inputs and self-checks the in-place Q/K/V, mirror and cache updates; it prints PASS on success.

### Run Test Cases

Run from the repository root (the suite assembles its own import paths; `PYTHONPATH` is not required):

```bash
OP=experimental/posembedding/fused_quant_qkv_proj_norm_rope_cache
CANNBOTDSL_NATIVE_BINARY_MODE=off QKV_TEST_BACKEND=source python3 -m pytest "$OP/tests/pytest" -q
CANNBOTDSL_NATIVE_BINARY_MODE=require QKV_TEST_BACKEND=opp python3 -m pytest "$OP/tests/pytest" -q
```

Coverage includes interface validation, Meta/Fake, plain and mirror layers, dynamic dimensions, chunking, tails, cache updates and repeated invocation.
