# MlaPrologV4

## Product Support

| Product | Supported |
| --- | --- |
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term> | × |
| <term>Atlas A2 products</term> | × |
| <term>Atlas 200I/500 A2 inference products</term> | × |
| <term>Atlas inference products</term> | × |
| <term>Atlas training products</term> | × |

## Description

Performs Multi-Head Latent Attention (MLA) inference preprocessing, including projection, Root Mean Square Normalization (RMSNorm), Rotary Positional Embedding (RoPE), optional Q/Cache quantization, and in-place Cache updates.

Q denotes the query, KV the shared low-rank key/value vector, and KR the rotary key segment. QA, QB, and KVA denote the three projections. Wkb maps the non-positional (NoPE) query segment to 512 dimensions. ND is a contiguous linear layout; NZ is the FRACTAL_NZ blocked format. BF16 and FP32 denote bfloat16 and float32. HiF8 uses raw uint8 encodings.

The formulas describe mathematical dependencies; the conversion points below remain part of the numerical path. `T` is the token count, `N` the head count, `Dn` the Wkb reduction width, and `Dr` the rotary dimension. The KV low-rank dimension is fixed at 512. `B` and `S` are the Cache page count and page size.

In the formulas, `wqa`, `wqb`, and `wkva` use the ND weight orientation `(output dimension, input dimension)`. NZ storage represents the same mathematical weights.

$$
\operatorname{RMSNorm}(v,\gamma)_j=\frac{v_j\gamma_j}{\sqrt{\frac{1}{d}\sum_{i=0}^{d-1}v_i^2+\epsilon}}.
$$

$$
R=\operatorname{RMSNorm}(x wqa^{T},\gamma_{qa})\cdot\texttt{qr\_scale}.
$$

$$
U=\operatorname{reshape}_{T,N,Dn+Dr}(R wqb^{T}),\quad
Q^N_{t,n}=U_{t,n,:Dn} wkb_n,\quad Q^R=\operatorname{RoPE}(U_{:,:,Dn:}).
$$

$$
V=x wkva^{T},\quad KV=\operatorname{RMSNorm}(V_{:,:512},\gamma_{kva})\cdot\texttt{kvr\_scale},\quad KR=\operatorname{RoPE}(V_{:,512:}).
$$

Before output encoding, Q concatenates Q-NoPE and Q-RoPE. Both normalization paths share epsilon. KR is neither normalized nor multiplied by `kvr_scale`; Wkb applies only to Q-NoPE.

Matrix projections accumulate in FP32. The BF16 input implementation retains intermediate BF16 conversions of projection and normalization results. In the quantized implementation, Q-NoPE is converted to BF16 before and after Wkb, and RoPE results are converted to BF16. A real-valued matrix chain without these conversions does not reproduce the numerical path.

## Parameters

| Parameter | Input/Output/Attribute | Description | Data Type | Data Format |
| --- | --- | --- | --- | --- |
| `x` | Input | Hidden states. Shape: `(T,dim)`. | BFLOAT16 | ND |
| `wqa` | Input | Query low-rank projection from dim to q_lora. Logical shape: `ND:(q_lora,dim); NZ:(dim,q_lora)`. | BFLOAT16 | ND/FRACTAL_NZ |
| `wqb` | Input | Query up-projection. Logical shape: `ND:(N*(Dn+Dr),q_lora); NZ:(q_lora,N*(Dn+Dr))`. | BFLOAT16 | ND/FRACTAL_NZ |
| `wkva` | Input | KVA projection. Logical shape: `ND:(512+Dr,dim); NZ:(dim,512+Dr)`. | BFLOAT16 | ND/FRACTAL_NZ |
| `wkb` | Input | Per-head Q-NoPE absorption weight; not used to generate Cache. Shape: `(N,Dn,512)`. | BFLOAT16 | ND |
| `norm_weight_qa` | Input | Gamma for QA RMSNorm. Shape: `(q_lora,)`. | FLOAT | ND |
| `norm_weight_kva` | Input | Gamma for KV RMSNorm. Shape: `(512,)`. | FLOAT | ND |
| `kv_cache` | Input/Output | Combined KV/KR Cache updated in place, only at valid indexed slots. Shape: `(B,S,1,Dcache)`. | mode 0:BFLOAT16; mode 1/2:FLOAT8_E4M3FN; mode 3:UINT8 | ND carrier |
| `cache_index` | Input | Linear Cache slot for each token; negative values skip Cache writes. Shape: `(T,)`. | INT64 | ND |
| `qkv_quant_mode` | Required attribute | Q/Cache output mode, specified with this operator's QkvQuantMode enum. Required; no default. See the mode definitions in the PyTorch API. | Enum | - |
| `quant_scale_kv` | Optional input | Default: None. Required in modes 2/3; must be finite, with zero and negative values allowed. Must be None in modes 0/1. Cache quantization multiplies the value by quant_scale_kv. Shape: `(1,)`. | FLOAT | ND |
| `rope_sin` | Optional input | Default: None. Required with rope_cos when RoPE is enabled. Shape: `(T,Dr)`. | BFLOAT16 | ND |
| `rope_cos` | Optional input | Default: None. RoPE cosine values, paired with rope_sin. Shape: `(T,Dr)`. | BFLOAT16 | ND |
| `norm_eps` | Optional attribute | Default: 1e-5. Shared by both RMSNorm paths; must be finite and positive after FP32 conversion. Bool is rejected. | FLOAT | - |
| `cache_mode` | Optional attribute | CacheMode.PA_BSND or CacheMode.PA_NZ; mode 1 supports PA_BSND only. Default: PA_BSND. | Enum | - |
| `qr_scale` | Optional attribute | Default: 1.0. Multiplier after QA normalization. Must be finite in FP32; zero and negative values are allowed. Bool is rejected. | FLOAT | - |
| `kvr_scale` | Optional attribute | Default: 1.0. Multiplier after KV normalization; does not scale KR. Same value constraints as qr_scale. | FLOAT | - |
| `rope_mode` | Optional attribute | Default: RopeMode.ROPE_INTERLEAVE_HALF. Supports NO_ROPE, ROPE_INTERLEAVE_HALF, and ROPE_HALF. | Enum | - |
| `return_qr` | Optional attribute | Default: False. Fills the qr output slot when True. | BOOL | - |
| `q` | Output | Concatenated Q-NoPE and Q-RoPE, with 512+Dr values per head. Shape: `(T,N,512+Dr)`. | mode 0/1:BFLOAT16; mode 2:FLOAT8_E4M3FN; mode 3:UINT8 | ND |
| `descale_q` | Output | Q dequantization multiplier. Empty in modes 0/1 and mode 3 with empty input. Shape: `mode 2:(T,N,1); mode 3,T>0:(1,); otherwise:(0,)`. | FLOAT | ND |
| `qr` | Output | Filled only when return_qr=True. Shape: `return_qr=True:(T,q_lora); otherwise:(0,)`. | BFLOAT16 | ND |

Weight shapes in this table are Python-visible logical shapes (`Tensor.shape`). All three weights must use ND, or all must use FRACTAL_NZ. Pass ND weights as `(output dimension,input dimension)`. For NZ, pass the transposed weights as `(input dimension,output dimension)` with actual FRACTAL_NZ storage. Transpose or reshape alone does not perform NZ format conversion.

Cache uses one combined buffer. Q-NoPE and Q-RoPE form one q output.

## Constraints

- `T >= 0`; `dim` is a multiple of 256 in [1024,8192]; `q_lora` is 640, 1024, 1536, or 2048; `N` is in [1,128]; `(Dn,Dr)` is `(128,64)`, `(192,64)`, or `(256,0)`. The KV low-rank dimension is fixed at 512.
- `B > 0`; `S` is a multiple of 16 in [16,1024]. Mode 1 supports PA_BSND only; other modes support PA_BSND and PA_NZ. Input must use the merged shape `(T,dim)`; `(B,S,dim)` and PA_BLK are not supported.
- All inputs must be on the same NPU. All inputs except Cache must be contiguous. Cache inner strides must be `(Dcache,Dcache,1)`, with `stride(0) >= S*Dcache`. Gaps along axis 0 and nonzero storage offsets are supported. Unwritten slots and page gaps remain byte-for-byte unchanged.
- The caller must ensure that nonnegative Cache indices are unique and less than `B*S`. The operator does not read back or validate index values. A negative index skips only the Cache write; the token still contributes to Q/QR and global quantization statistics.
- `Dr=0` requires `NO_ROPE`. RoPE requires paired sin/cos inputs. When disabled, both may be None; supplied tensors still undergo shape and type checks.
- `norm_eps` must be finite and positive after FP32 conversion. `qr_scale/kvr_scale` must be finite FP32 values; zero and negative values are allowed. All three reject bool. In modes 2/3, `quant_scale_kv` must be finite, allowing zero and negative values; in modes 0/1, it must be None.
- `T=0` still validates metadata, returns empty outputs according to the interface, launches no computation kernel, and leaves Cache unchanged. Cache width is `512+Dr` in modes 0, 2, and 3, and `512+2*Dr+16` in mode 1.
- Supports Decode/Prefill inference. Automatic differentiation and functional Cache updates are not provided. Outputs and temporary space reused across stages are allocated internally; there are no external out/workspace parameters. Nonempty calls require valid MIX core resources on the current stream.

## Calling the Operator

| Method | Example | Description |
| --- | --- | --- |
| PyTorch API | [Eager/ACLGraph/NPUGraph-EX](docs/torchapi_mla_prolog_v2.md#调用示例) | [mla_prolog_v2](docs/torchapi_mla_prolog_v2.md) |

Call `cann_ops_transformer.ops.mla_prolog_v2` with an explicit `qkv_quant_mode`. Inputs and Cache must be on the same NPU. `kv_cache` is updated in place.

## References

- [PyTorch API](docs/torchapi_mla_prolog_v2.md)
