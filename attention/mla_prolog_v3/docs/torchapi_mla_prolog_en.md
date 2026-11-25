# mla\_prolog

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

## Description

- API function:

  `mla_prolog` is an extended `cann_ops_transformer` API. It is used to call the `MlaPrologV4WeightNz` operator to complete the query, KV cache, and Kr cache calculation and cache update in the process before the MLA (Multi-head Latent Attention) Decoder operator is called. You can determine whether to execute RoPE (Rotary Positional Embedding) by checking whether `rope_sin` and `rope_cos` are transferred in pairs.

  In the PagedAttention (PA) scenario, this API uses `kv_cache`/ `kr_cache` as the cache input and updates the cache in the operator. The interface supports the following scenarios:

  - If both `rope_sin` and `rope_cos` are not empty, RoPE is performed on the query, and the RoPE result is displayed in `query_rope`.
  - If both `rope_sin` and `rope_cos` are empty (or `None`), RoPE is not executed.
  - If `rope_sin` or `rope_cos` is empty or not empty, the input is invalid. The interface directly reports an error and intercepts the input.

- Formulas:

    `RmsNorm` formula:

    $$
    \text{RmsNorm}(x) = \gamma \cdot \frac{x_i}{\text{RMS}(x)}
    $$

    $$
    \text{RMS}(x) = \sqrt{\frac{1}{N} \sum_{i=1}^{N} x_i^2 + \epsilon}
    $$

    The computation formula of the query, including downsampling, RmsNorm, and two upsampling operations.

    $$
    c^Q = \alpha_q\cdot\mathrm{RmsNorm}(x \cdot W^{DQ})
    $$

    $$
    q^C = c^Q \cdot W^{UQ}
    $$

    $$
    q^N = q^C \cdot W^{UK}
    $$
    Where $\alpha_q$ is the scale correction parameter of the query.

    Performs rotary position encoding (ROPE) to the query.

    $$
    q^R = \mathrm{ROPE}(c^Q \cdot W^{QR})
    $$

    ROPE calculation is not performed.

    $$
    q^R = c^Q \cdot W^{QR}
    $$

    The computation formula of the key, including downsampling and RmsNorm. The computation result is stored in the cache.

    $$
    c^{KV} = \alpha_{kv}\cdot\mathrm{RmsNorm}(x \cdot W^{DKV})
    $$

    $$
    k^C = \mathrm{Cache}(c^{KV})
    $$
    $\alpha_{kv}$ is the scale correction parameter of the key.

    Performs rotary position encoding (ROPE) to the key and stores the result in the cache.

    $$
    k^R = \mathrm{Cache}(\mathrm{ROPE}(x \cdot W^{KR}))
    $$

    ROPE calculation is not performed.

    $$
    k^R = \mathrm{Cache}(x \cdot W^{KR})
    $$

    Dequant Scale Query Nope Formula

    $$
    \mathrm{dequantScaleQNope} = {\mathrm{RowMax}(\mathrm{abs}(q^{N})) / Q_{max}}
    $$

    $$
    q^{N} = {\mathrm{round}(q^{N} / \mathrm{dequantScaleQNope})}
    $$

    Query Norm and Dequant Scale Query Norm formulas (output when queryNormFlag is set to true)

    Non-quantization scenario (weightQuantMode=0):

    $$
    queryNorm = c^Q = \alpha_q\cdot\mathrm{RmsNorm}(x \cdot W^{DQ})
    $$

    In this case, queryNorm is of the BFLOAT16 type, and dequantScaleQNorm is not output (nullptr).

    In quantization scenarios (weightQuantMode=1/2/4/5, per-token dynamic quantization), smoothScaleCq is optional. If smoothScaleCq is not passed, it is considered as 1.

    $$
    \tilde{c}^Q = c^Q \cdot smoothScale
    $$

    $$
    \mathrm{dequantScaleQNorm} = {\mathrm{RowMax}(\mathrm{abs}(\tilde{c}^Q)) / Q_{max}}
    $$

    $$
    queryNorm = {\mathrm{round}(\tilde{c}^Q / \mathrm{dequantScaleQNorm})}
    $$

    mxfp8 quantization scenario (weightQuantMode=3, per-group dynamic quantization for every 32 elements, quantization parameter of the FLOAT8_E8M0 type):

    $$
    \mathrm{dequantScaleQNorm} = 2^{\lfloor \mathrm{log}_2(\mathrm{GroupMax}(\mathrm{abs}(c^Q))) \rfloor - 8}
    $$

    $$
    queryNorm = {\mathrm{cast\_to\_fp8}(c^Q / \mathrm{dequantScaleQNorm})}
    $$

    $Q_{max}$ is the maximum value of the quantized output type. The value of INT8 is 127, the value of FLOAT8_E4M3FN is 448, and the value of HIFLOAT8 is 32768. The value 8 is the maximum exponent emax of FLOAT8_E4M3FN.

## Prototype

`mla_prolog` is the in-place cache update interface, which directly updates `kv_cache`/ `kr_cache` transferred by the invoker.

```python
cann_ops_transformer.mla_prolog(
    token_x,
    weight_dq,
    weight_uq_qr,
    weight_uk,
    weight_dkv_kr,
    rmsnorm_gamma_cq,
    rmsnorm_gamma_ckv,
    kv_cache,
    kr_cache,
    *,
    rope_sin=None,
    rope_cos=None,
    cache_index=None,
    dequant_scale_x=None,
    dequant_scale_w_dq=None,
    dequant_scale_w_uq_qr=None,
    dequant_scale_w_dkv_kr=None,
    quant_scale_ckv=None,
    quant_scale_ckr=None,
    smooth_scales_cq=None,
    actual_seq_len=None,
    k_nope_clip_alpha=None,
    rmsnorm_epsilon_cq=1e-05,
    rmsnorm_epsilon_ckv=1e-05,
    cache_mode='PA_BSND',
    query_norm_flag=False,
    weight_quant_mode=0,
    kv_cache_quant_mode=0,
    query_quant_mode=0,
    ckvkr_repo_mode=0,
    quant_scale_repo_mode=0,
    tile_size=128,
    qc_qr_scale=1.0,
    kc_scale=1.0,
    token_x_dtype=None,
    weight_dq_dtype=None,
    weight_uq_qr_dtype=None,
    weight_dkv_kr_dtype=None,
    kv_cache_dtype=None,
) -> (Tensor, Tensor, Tensor, Tensor, Tensor)
```

## Parameters

| Parameter| Type| Required/Optional| Description| Data Type| Dimension (shape)|
|---|---|---|---|---|---|
| token_x | Tensor | Required| Input tensor used to calculate the query and key in the formula. Discontinuous tensors are not supported. The data format can be ND. In the non-full quantization scenario, `weight_quant_mode` supports 2-dimensional `[T, He]` (BS combined) or 3-dimensional `[B, S, He]` (BS non-combined). In the `weight_quant_mode=5` (hifloat8 full quantization) scenario, `[T, He]` is stored in `torch.uint8` mode. The underlying ACL data type needs to be specified together with `token_x_dtype`.| torch.float16/torch.bfloat16; torch.uint8 in the hifloat8 scenario| 2-dimensional or 3-dimensional, ND format|
| weight_dq | Tensor | Required| In the formula, the downsampling weight matrix $W^{DQ}$ used to calculate the query does not support non-consecutive values. The data format can be FRACTAL_NZ, the logical shape is `[He, Hcq]`, and the dtype constraint is the same as that of `token_x` (torch.uint8 in the hifloat8 scenario).| Same as `token_x`.| The format must be FRACTAL_NZ (4-dimensional).|
| weight_uq_qr | Tensor | Required| In the formula, the upsampling weight matrix $W^{UQ}$ and position encoding weight matrix $W^{QR}$ used to calculate the query are not consecutive. The data format can be FRACTAL_NZ, and the logical shape is `[Hcq, N*(D+Dr)]`. In the `weight_quant_mode=0` scenario, dtype is the same as that of `token_x`. In the `weight_quant_mode=1` (semi-quantization) scenario, the value is of the int8 type.| The value is the same as that of `token_x`. If mode is set to 1, torch.int8 is used.| The format must be FRACTAL_NZ (4-dimensional).|
| weight_uk | Tensor | Required| In the formula, the upsampling weight $W^{UK}$ is used to calculate the key. Discontinuous values are not supported. The data format can be ND, and the shape is `[N, D, Hckv]`.| torch.bfloat16 | 3-dimensional `[N, D, Hckv]`, ND format|
| weight_dkv_kr | Tensor | Required| In the formula, the downsampling weight matrix $W^{DKV}$ and position encoding weight matrix $W^{KR}$ used to calculate the key are not consecutive. The data format can be FRACTAL_NZ, the logical shape is `[He, Hckv+Dr]`, and the dtype is the same as that of `token_x`.| Same as `token_x`.| The format must be FRACTAL_NZ (4-dimensional).|
| rmsnorm_gamma_cq | Tensor | Required| $\gamma$ parameter in the RMSNorm formula for calculating $c^{Q}$. Discontinuous values are not supported. The data format can be ND and the shape is `[Hcq]`.| The value is the same as that of `token_x`.| `[Hcq]` |
| rmsnorm_gamma_ckv | Tensor | Required| $\gamma$ parameter in the RMSNorm formula for calculating $c^{KV}$. Discontinuous values are not supported. The data format can be ND and the shape is `[Hckv]`.| The value is the same as that of `token_x`.| `[Hckv]` |
| kv_cache | Tensor | Required| Cache index. The calculation result is updated in place (corresponding to $k^{C}$ in the formula). Only the first axis is discontinuous. Other axes except the first axis must be continuous. The data format can be ND. In PA mode (`cache_mode` is "PA_BSND"/"PA_NZ"/"PA_BLK_BSND"/"PA_BLK_NZ"), empty tensors are supported. In BSND or TND mode, empty tensors are not supported. Nkv is associated with N. N is a hyperparameter. Therefore, Nkv=0 is not supported.| torch.bfloat16; hifloat8: torch.uint8 for full quantization in the KV scenario| PA mode: 4-dimensional `[BlockNum, BlockSize, Nkv, Dtile]`; "BSND": 4-dimensional `[B, S, Nkv, Dtile]`; "TND": 3-dimensional `[T, Nkv, Dtile]`, ND format|
| kr_cache | Tensor | Required| Cache used for key position encoding. The calculation result is updated in place (corresponding to $k^{R}$ in the formula). Only the first axis can be discontinuous. Other axes except the first axis must be continuous. The data format can be ND. In PA mode, empty tensors are supported. In BSND or TND mode, empty tensors are not supported. If both `rope_sin` and `rope_cos` are empty (or `None`), RoPE is not executed.| torch.bfloat16 | The layout is the same as that of `kv_cache`: PA mode `[BlockNum, BlockSize, Nkv, Dr]`; "BSND" `[B, S, Nkv, Dr]`; "TND" `[T, Nkv, Dr]`, ND format.|
| rope_sin | Tensor | Optional| Sine parameter matrix used to calculate the rotation position encoding. Discontinuous is not supported. The data format can be ND. If both `rope_cos` and `rope_cos` are not empty, RoPE is executed. The dimension is the same as that of `token_x`. If both `rope_cos` and `rope_cos` are empty (or `None`), RoPE is not executed. If either `rope_cos` or `rope_cos` is empty, the input is invalid. The default value is `None`. An empty tensor with B = 0, S = 0, and T = 0 is supported.| torch.bfloat16 | 2-dimensional `[T, Dr]` or 3-dimensional `[B, S, Dr]`|
| rope_cos | Tensor | Optional| Cosine parameter matrix used to calculate the rotation position encoding. Discontinuous values are not supported. The data format can be ND, and the constraints are the same as those of `rope_sin`. The default value is `None`.| torch.bfloat16 | The value is the same as that of `rope_sin`.|
| cache_index | Tensor | The PA mode is mandatory, and the BSND or TND mode is optional.| Index used to store `kv_cache` and `kr_cache`. Discontinuous indexes are not supported. The data format can be ND. If `cache_mode` is "PA_BSND" or "PA_NZ": When the BS is combined, shape is `[T]`. When the BS is not combined, shape is `[B, S]`. The value range is `[0, BlockNum*BlockSize)`. When `cache_mode` is "PA_BLK_BSND"/"PA_BLK_NZ", the shape is `[Sum (Ceil (S_i/BlockSize))]` (S_i indicates the sequence length of the ith batch). When the BS is not combined, the shape is `[B, Ceil(S/BlockSize)]`. The value range is `[0, BlockNum)`. When `cache_mode` is set to BSND or TND, the value does not need to be transferred. In PagedAttention mode, if this parameter is not transferred, the API reports an error and intercepts the request.| torch.int32/torch.int64 | One-dimensional `[T]` or two-dimensional `[B, S]`|
| dequant_scale_x | Tensor | Optional| Dequantization parameter of `token_x`. Discontinuous values are not supported. The data format can only be ND. When `weight_quant_mode=2/4/5` is used, the shape is `[T,1]` or `[B*S,1]`. When `weight_quant_mode=3` is used, the shape is `[T, He/32]` or `[B*S, He/32]`. When mode=3, the dtype is `torch.float8_e8m0fnu`. When mode=2/4/5, the dtype is float32. The default value is `None`. An empty tensor with B = 0, S = 0, and T = 0 is supported.| torch.float8_e8m0fnu (mode=3) /torch.float32 (mode=2/4/5)| mode=2/4/5: `[T, 1]` or `[B*S, 1]`; mode=3: `[T, He/32]` or `[B*S, He/32]`|
| dequant_scale_w_dq | Tensor | Optional| Dequantization parameter of `weight_dq`. Discontinuous values are not supported. The data format can only be ND. For `weight_quant_mode=2/4/5`, the shape is `[1, Hcq]`. For `weight_quant_mode=3`, the shape is `[Hcq, He/32]`. The constraints of dtype are the same as those of `dequant_scale_x`. The default value is `None`.| Same as `dequant_scale_x`.| mode=2/4/5: `[1, Hcq]`; mode=3: `[Hcq, He/32]`|
| dequant_scale_w_uq_qr | Tensor | Optional| Dequantization parameter of `weight_uq_qr`, which is the perchannel parameter used for dequantization after MatmulQcQr matrix multiplication. Discontinuous values are not supported. The data format can be ND. For `weight_quant_mode=1/2/4/5`, the shape is `[1, N*(D+Dr)]`. For `weight_quant_mode=3`, the shape is `[N*(D+Dr), Hcq/32]`. The constraints of dtype are the same as those of `dequant_scale_x`. The default value is `None`.| Same as `dequant_scale_x`.| mode=1/2/4/5: `[1, N*(D+Dr)]`; mode=3: `[N*(D+Dr), Hcq/32]`|
| dequant_scale_w_dkv_kr | Tensor | Optional| Dequantization parameter of `weight_dkv_kr`. Discontinuous values are not supported. The data format can only be ND. For `weight_quant_mode=2/4/5`, the shape is `[1, Hckv+Dr]`. For `weight_quant_mode=3`, the shape is `[Hckv+Dr, He/32]`. The constraints of dtype are the same as those of `dequant_scale_x`. The default value is `None`.| Same as `dequant_scale_x`.| mode=2/4/5: `[1, Hckv+Dr]`; mode=3: `[Hckv+Dr, He/32]`|
| quant_scale_ckv | Tensor | Optional| Parameter used to quantize the kv_cache output data. Non-consecutive values are not supported. The data format can be ND. When `kv_cache_quant_mode=1` is used, the shape is `[1]`. When `kv_cache_quant_mode=2` is used, the shape is `[1, Hckv]`. When `kv_cache_quant_mode=3` is used, no value needs to be assigned. This parameter is mandatory in the full quantization KV per-tensor scenario (`kv_cache_quant_mode=1` and `weight_quant_mode` are set to 2, 3, 4, or 5). The default value is `None`.| torch.float32 | `kv_cache_quant_mode=1`: `[1]`; mode=2: `[1, Hckv]`|
| quant_scale_ckr | Tensor | Optional| Parameter used to quantize the kr_cache output data. Non-consecutive values are not supported. The data format can be ND. When `kv_cache_quant_mode=2` is used, the shape is `[1, Dr]`. When `kv_cache_quant_mode=1/3` is used, no value needs to be assigned. The default value is `None`.| torch.float32 | `[1, Dr]` |
| smooth_scales_cq | Tensor | Optional| Parameter used for dynamic quantization on the RMSNorm_cq output. Non-consecutive values are not supported. The data format can be ND. If `weight_quant_mode=1/2/4/5` is used, the shape is `[1, Hcq]`. The default value is `None`.| torch.float32 | `[1, Hcq]` |
| actual_seq_len | Tensor | Optional| Sequence length in each batch. The value is stored in the format of prefix and and cannot be inconsecutive. The data format can be ND. This parameter is mandatory only when the BS is combined and `cache_mode` is set to "PA_BLK_BSND"/"PA_BLK_NZ". The default value is `None`.| torch.int32 | `[B]` |
| k_nope_clip_alpha | Tensor | Optional| Scaling factor when the kv_cache performs the clip operation. This parameter is used in the pertoken-pergroup scenario for partial quantization and pertoken-pergroup scenario for int8 full quantization. In other scenarios, no value needs to be assigned. Discontinuous values are not supported. The data format can be ND. The default value is `None`.| torch.float32 | `[1]` |
| rmsnorm_epsilon_cq | float | Optional| Calculate the $\epsilon$ parameter in the RMSNorm formula of $c^{Q}$. The default value is `1e-5`.| float | - |
| rmsnorm_epsilon_ckv | float | Optional| Calculate the $\epsilon$ parameter in the RMSNorm formula of $c^{KV}$. The default value is `1e-5`.| float | - |
| cache_mode | str | Optional| kv_cache mode. The options are "PA_BSND", "PA_NZ", "PA_BLK_BSND", PA_BLK_NZ (corresponding to PagedAttention), TND (corresponding to BS combined axis), and BSND (corresponding to BS non-combined axis). The default value is PA_BSND.| str | - |
| query_norm_flag | bool | Optional| Whether to output query_norm. False indicates no output, and True indicates output (dequant_scale_q_norm is output in quantization scenarios). The default value is False.| bool | - |
| weight_quant_mode | int | Optional| Quantization mode of weight_dq, weight_uq_qr, weight_uk and weight_dkv_kr. 0: non-quantization; 1: weight_uq_qr quantization (semi-quantization); 2: weight_dq, weight_uq_qr and weight_dkv_kr int8 full quantization; 3: mxfp8 full quantization; 4: fp8 full quantization; 5: HiF8 full quantization. The default value is 0.| int | - |
| kv_cache_quant_mode | int | Optional| Quantization mode of kv_cache. 0: non-quantization; 1: per-tensor quantization; 2: per-channel quantization; 3: per-token-per-group quantization. The default value is 0.| int | - |
| query_quant_mode | int | Optional| Quantization mode of a query. 0: non-quantization; 1: per-token-head quantization. The default value is 0.| int | - |
| ckvkr_repo_mode | int | Optional| Storage mode of kv_cache and kr_cache. 0: separate storage; 1: combined storage. The default value is 0.| int | - |
| quant_scale_repo_mode | int | Optional| Storage mode of the quantization scale. 0: scale and data are stored separately. 1: scale and data are stored together. The default value is 0.| int | - |
| tile_size | int | Optional| Size of each tile during per-token-per-group quantization. This parameter is valid only when `kv_cache_quant_mode=3` is used. Currently only 128 is supported. No value needs to be assigned when `kv_cache_quant_mode=1/2` is used.| int | - |
| qc_qr_scale | float | Optional| Scale correction coefficient of the query. The default value is 1.0.| float | - |
| kc_scale | float | Optional| Scale correction coefficient of the key. The default value is 1.0.| float | - |
| token_x_dtype | int | Optional| The input dtype of token_x is `torch_npu.hifloat8` (enumerated value: 290) in the HiF8 full quantization scenario and is None in other scenarios.| int | - |
| weight_dq_dtype | int | Optional| The input dtype of weight_dq is `torch_npu.hifloat8` in the HiF8 full quantization scenario and None in other scenarios.| int | - |
| weight_uq_qr_dtype | int | Optional| The input dtype of weight_uq_qr is `torch_npu.hifloat8` in the HiF8 full quantization scenario and None in other scenarios.| int | - |
| weight_dkv_kr_dtype | int | Optional| The input dtype of weight_dkv_kr is `torch_npu.hifloat8` in the HiF8 full quantization scenario and None in other scenarios.| int | - |
| kv_cache_dtype | int | Optional| The input dtype of the kv_cache parameter is `torch_npu.hifloat8` in the hif8 kv_cache per-tensor quantization and hif8 kv_cache per-token-per-group quantization scenarios, and is None in other scenarios.| int | - |

## Return Values

Five tensors are returned for `mla_prolog`, and the `kv_cache` and `kr_cache` passed by the caller are updated in place.

| Parameter| Type| Optional/Required| Description| Data Type| Dimension (shape)|
|---|---|---|---|---|---|
| query | Tensor | Required| Q Side output. In the full quantization KV scenario (`weight_quant_mode∈{2,3,4,5}` and `kv_cache_quant_mode=1`), dtype is the same as `token_x`. In other scenarios, dtype is bfloat16.| torch.float16/torch.bfloat16; torch.uint8 in the hifloat8 scenario| `[T, N, Hckv]` or `[B, S, N, Hckv]`|
| query_rope | Tensor | Required| RoPE output. If rope_sin, `/`, and rope_cos are `empty or` None', the data is directly output without RoPE calculation.| torch.bfloat16 | `[T, N, Dr]` or `[B, S, N, Dr]`|
| dequant_scale_q_nope | Tensor | Required| Q Dequantization scale of the nope part. This parameter is valid only when `weight_quant_mode∈{2,3,4,5}` and `kv_cache_quant_mode=1` are used (full quantization KV per-tensor scenario). Otherwise, the tensor is empty.| torch.float32 | `[T, N, 1]` or `[B*S, N, 1]`. Otherwise, the value is an empty tensor.|
| query_norm | Tensor | Required| Normalized query output. This parameter is valid when `query_norm_flag=True` is specified. Otherwise, the value is an empty tensor.| The value is the same as that of `weight_uq_qr`.| `[T, Hcq]` or `[B, S, Hcq]`. Otherwise, the value is an empty tensor.|
| dequant_scale_q_norm | Tensor | Required| Dequantization scale of query normalization. This parameter is valid only when `query_norm_flag=True` and `weight_quant_mode!=0` are specified. Otherwise, this parameter is an empty tensor.| torch.float32; The value of `weight_quant_mode=3` is the same as that of `dequant_scale_x`.| `[T, 1]` or `[T, Hcq/32]`. Otherwise, the value is an empty tensor.|

## Constraints

- This API can be used in inference scenarios.

- This API supports both single-operator mode and graph mode.

- The following table describes the shape layout fields.

    | Field| Full Name/Description| Value Rule and Description|
    |--------------|--------------------------------|------------------------------------------------------------------------------|
    | B | Batch (batch size of input samples)| Value range: 0 to 65536.|
    | S | Seq-Length (sequence length of input samples)| Value range: not limited.|
    | He | Head-Size (hidden layer size)| The value is fixed at 1024, 2048, 3072, 4096, 5120, 6144, 7168, 7680, or 8192.|
    | Hcq | Dimension of the low-rank q matrix| The value is fixed at 1536 or 2048.|
    | N | Head-Num (number of heads)| Value range: 1 to 128|
    | Hckv | Dimension of the low-rank KV matrix| The value is fixed at `512`.|
    | Dtile | D-axis dimension of kv_cache| The value is fixed to 656 in the pertoken-pergroup scenario (`kv_cache_quant_mode=3`) and 512 in the non-pertoken-pergroup scenario.|
    | D | QK dimension without position encoding| The value is fixed to 128 or 192.|
    | Dr | QK positional encoding dimension| The value is fixed at `64`.|
    | Nkv | Number of KV heads| The value is fixed at `1`.|
    | BlockNum | Number of blocks in the PagedAttention scenario.| The value is rounded up to the nearest integer after the result of `B*Skv/BlockSize` is calculated. (`Skv` indicates the sequence length of KV, which can be `0`.)|
    | BlockSize | Block size in the PagedAttention scenario| Value range: 16-1024 (multiple of 16)|
    | T | Size after the fusion of the `B` and `S` axes| Value range: not limited; Note: If the BS combined axis is used, token_x and query_norm are 2-dimensional, query_out and query_rope_out are 3-dimensional, cache_index is 1-dimensional, and rope_sin and rope_cos are None (RoPE is disabled). Or two-dimensional (RoPE enabled)|

- Built-in API verification: The preceding shape format field constraints (He/Hcq/N/D/Hckv/Dr/Nkv/Dtile/BlockSize) are verified in a unified manner in the API. If the constraints are not met, an error is reported and intercepted. N (Head-Num) can be any integer between `[1, 128]`s.

- The `token_x` supports only two or three dimensions. The `weight_uk` must have three dimensions.
- The `weight_dq`, `weight_uq_qr`, and `weight_dkv_kr` must be in FRACTAL_NZ format (converted by `torch_npu.npu_format_cast(t, 29)`). The ND format will be intercepted by the interface.
- The `rope_sin` and `rope_cos` parameters must be transferred in pairs. If they are not empty at the same time, RoPE is executed. If they are empty (or `None`) at the same time, RoPE is not executed. If one of them is empty and the other is not empty, the input is invalid, and the interface reports an error and intercepts the input.
- When `cache_mode` is in PagedAttention mode (`"PA_BSND"`/ `"PA_NZ"`/ `"PA_BLK_BSND"`/ `"PA_BLK_NZ"`), `cache_index` must be transferred (not empty). Otherwise, the interface reports an error. `"BSND"`/ `"TND"` is optional.
- When `weight_quant_mode=3` (mxfp8) or `weight_quant_mode=5` (hifloat8) is used, `dequant_scale_x`, `dequant_scale_w_dq`, `dequant_scale_w_uq_qr`, and `dequant_scale_w_dkv_kr` must not be empty at the same time. When mode is set to 3, dtype is set to `torch.float8_e8m0fnu`.
- When `weight_quant_mode=5` is input as `torch.uint8`, each `*_dtype` must be explicitly specified as `torch_npu.hifloat8` (enumerated value: 290).
- When `kv_cache_quant_mode=1` (per-tensor) and `weight_quant_mode` are fully quantized (2, 3, 4, or 5), `quant_scale_ckv` must be passed.
- The value of `kv_cache_quant_mode` must match that of `weight_quant_mode`. For example, `weight_quant_mode=0` supports only `kv_cache_quant_mode=0`.
- The value of `query_quant_mode` can only be 0 or 1.
- Currently, only NPU devices (PrivateUse1 dispatch key) are supported. CPUs are not supported.

## Deterministic Computing

Deterministic computing is supported by default.

## Description

- Basic invocation (executing RoPE):

  ```python
  import torch
  import torch_npu
  import cann_ops_transformer

  # The following is an example of input parameters that meet the shape format field constraints:
  # He=1024, Hcq=1536, N=8, D=128, Dr=64, Hckv=512, Nkv=1, BlockNum=4, BlockSize=128, Dtile=512
  T = 128
  He, Hcq, N, D, Dr = 1024, 1536, 8, 128, 64
  Hckv, Nkv = 512, 1
  BlockNum, BlockSize, Dtile = 4, 128, 512

  dtype = torch.bfloat16
  token_x = torch.randn(T, He, dtype=dtype).npu()
  weight_dq = torch.randn(He, Hcq, dtype=dtype).npu()
  weight_dq_cast = torch_npu.npu_format_cast(weight_dq.contiguous(), 29)
  weight_uq_qr = torch.randn(Hcq, N * (D + Dr), dtype=dtype).npu()
  weight_uq_qr_cast = torch_npu.npu_format_cast(weight_uq_qr.contiguous(), 29)
  weight_uk = torch.randn(N, D, Hckv, dtype=dtype).npu()
  weight_dkv_kr = torch.randn(He, Hckv + Dr, dtype=dtype).npu()
  weight_dkv_kr_cast = torch_npu.npu_format_cast(weight_dkv_kr.contiguous(), 29)
  rmsnorm_gamma_cq = torch.randn(Hcq, dtype=dtype).npu()
  rmsnorm_gamma_ckv = torch.randn(Hckv, dtype=dtype).npu()
  rope_sin = torch.randn(T, Dr, dtype=dtype).npu()
  rope_cos = torch.randn(T, Dr, dtype=dtype).npu()
  kv_cache = torch.zeros(BlockNum, BlockSize, Nkv, Dtile, dtype=dtype).npu()
  kr_cache = torch.zeros(BlockNum, BlockSize, Nkv, Dr, dtype=dtype).npu()
  cache_index = torch.arange(T, dtype = torch.int64).npu() # is mandatory in PA_BSND mode by default.

  # In-place update of kv_cache/kr_cache
  query, query_rope, dequant_scale_q_nope, query_norm, dequant_scale_q_norm = (
      cann_ops_transformer.mla_prolog(
          token_x, weight_dq_cast, weight_uq_qr_cast, weight_uk, weight_dkv_kr_cast,
          rmsnorm_gamma_cq, rmsnorm_gamma_ckv, kv_cache, kr_cache,
          rope_sin=rope_sin,
          rope_cos=rope_cos,
          cache_index=cache_index,
          query_norm_flag=True,
      )
  )
  ```

- Do not invoke RoPE.

  If `rope_sin` or `rope_cos` is not transferred (or `None` is explicitly transferred at the same time), RoPE is not executed. The bottom layer calls the aclnn API using `doRope=false`.

  ```python
  # If rope_sin and rope_cos are omitted or set to None at the same time, RoPE is not performed.
  query, query_rope, dequant_scale_q_nope, query_norm, dequant_scale_q_norm = (
      cann_ops_transformer.mla_prolog(
          token_x, weight_dq_cast, weight_uq_qr_cast, weight_uk, weight_dkv_kr_cast,
          rmsnorm_gamma_cq, rmsnorm_gamma_ckv, kv_cache, kr_cache,
          cache_index=cache_index, # Mandatory in PA_BSND mode by default
      )
  )
  ```
