# MlaPrologV3

English | [简体中文](./README.md)

## Product Support

|Product      | Supported |
|:----------------------------|:-----------:|
|<term>Ascend 950PR&950DT products</term>|      √     |
|<term>Atlas A3 products</term>|      √     |
|<term>Atlas A2 products</term>|      √     |
|<term>Atlas 200I/500 A2 inference products</term>|      ×     |
|<term>Atlas inference products</term>|      ×     |
|<term>Atlas training products</term>|      ×     |

## Description

- **Function updates**: (differences from aclnnMlaPrologV2weightNz)
    - Adds scale correction factors for query and key, corresponding to qcQrScale ($\alpha_q$) and kcScale ($\alpha_{kv}$) respectively.
    - Adds optional inputs and parameters, and changes cache_mode from required to optional. Specifically:
        - actualSeqLenOptional: When the BS axes are combined and CacheMode="PA_BLK_BSND"/"PA_BLK_NZ", specifies the actual sequence lengths in the current batch.
        - kNopeClipAlphaOptional: Indicates the scaling factor used when the clip operation is performed on kv_cache.
        - queryNormFlag: Indicates whether to output query_norm, and dequant_scale_q_norm in quantization scenarios.
        - weightQuantMode: Indicates the quantization modes of weight_dq, weight_uq_qr, weight_uk, and weight_dkv_kr.
        - kvCacheQuantMode: Indicates the quantization mode of kv_cache.
        - queryQuantMode: Indicates the quantization mode of query.
        - ckvkrRepoMode: Indicates the storage modes of kv_cache and kr_cache.
        - quantScaleRepoMode: Indicates the storage mode of quantization scales.
        - tileSize: Indicates the size of each tile in pertoken-pergroup quantization.
        - queryNormOptional: Output tensor after rmsNorm is performed on tokenX in the formula (corresponding to $c^Q$).
        - dequantScaleQNormOptional: Quantization parameters of the query_norm output tensor.
    - Adjusts the name and position of the cacheIndex parameter, corresponding to the current cacheIndexOptional.
- **Operator function**: This operator performs the computation of the Multi-Head Latent Attention (MLA) pre-processing in inference scenarios. The main computation consists of seven paths. First, the input $x$ is multiplied by $W^{DQ}$ for downsampling and RmsNorm, and the result is split into two paths. The first path is multiplied by $W^{UQ}$ and $W^{UK}$ and goes through two upsampling operations to obtain $q^N$. The second path is multiplied by $W^{QR}$ and then goes through the rotary position encoding (ROPE) to obtain $q^R$. In the third path, the input $x$ is multiplied by $W^{DKV}$ for downsampling and RmsNorm, and the result is stored in the Cache to obtain $k^C$. In the fourth path, the input $x$ is multiplied by $W^{KR}$ and then goes through the rotary position encoding, and the result is stored in another Cache to obtain $k^R$. The fifth path obtains the quantization parameters after $q^N$ passes through DynamicQuant. In the sixth path, when queryNormFlag=true, the input $x$ is multiplied by $W^{DQ}$ for downsampling and RmsNorm to obtain the compressed result $c^Q$, which is output as queryNorm. In the seventh path, when queryNormFlag=true and quantization is enabled (weightQuantMode≠0), DynamicQuant is performed on $c^Q$, and finally the quantized queryNorm and the corresponding quantization parameter dequantScaleQNorm are obtained.
- **Computation formulas**:

    RmsNorm formula

    $$
    \mathrm{RmsNorm}(x) = \gamma \cdot \frac{x_i}{\mathrm{RMS}(x)}
    $$

    $$
    \mathrm{RMS}(x) = \sqrt{\frac{1}{N} \sum_{i=1}^{N} x_i^2 + \epsilon}
    $$

    Query computation formulas, including downsampling, RmsNorm, and two upsampling operations

    $$
    c^Q = \alpha_q\cdot\mathrm{RmsNorm}(x \cdot W^{DQ})
    $$

    $$
    q^C = c^Q \cdot W^{UQ}
    $$

    $$
    q^N = q^C \cdot W^{UK}
    $$
    where $\alpha_q$ is the scale correction parameter of the Query.

    ROPE rotary position encoding for the Query

    $$
    q^R = \mathrm{ROPE}(c^Q \cdot W^{QR})
    $$

    Key computation formulas, including downsampling and RmsNorm, with the computation results stored in the cache

    $$
    c^{KV} = \alpha_{kv}\cdot\mathrm{RmsNorm}(x \cdot W^{DKV})
    $$

    $$
    k^C = \mathrm{Cache}(c^{KV})
    $$
    where $\alpha_{kv}$ is the scale correction parameter of the Key.

    ROPE rotary position encoding for the Key, with the result stored in the cache

    $$
    k^R = \mathrm{Cache}(\mathrm{ROPE}(x \cdot W^{KR}))
    $$

    Dequant Scale Query Nope computation formulas

    $$
    \mathrm{dequantScaleQNope} = {\mathrm{RowMax}(\mathrm{abs}(q^{N})) / Q_{max}}
    $$

    $$
    q^{N} = {\mathrm{round}(q^{N} / \mathrm{dequantScaleQNope})}
    $$

    Query Norm and Dequant Scale Query Norm computation formulas (output when queryNormFlag=true)

    Non-quantization scenario (weightQuantMode=0):

    $$
    queryNorm = c^Q = \alpha_q\cdot\mathrm{RmsNorm}(x \cdot W^{DQ})
    $$

    In this case, queryNorm is of the BFLOAT16 type, and dequantScaleQNorm is not output (set to nullptr).

    Quantization scenario (weightQuantMode=1/2/4/5, per-token dynamic quantization), where smoothScaleCq is optional and is considered as 1 if not passed:

    $$
    \tilde{c}^Q = c^Q \cdot smoothScale
    $$

    $$
    \mathrm{dequantScaleQNorm} = {\mathrm{RowMax}(\mathrm{abs}(\tilde{c}^Q)) / Q_{max}}
    $$

    $$
    queryNorm = {\mathrm{round}(\tilde{c}^Q / \mathrm{dequantScaleQNorm})}
    $$

    mxfp8 quantization scenario (weightQuantMode=3, pergroup dynamic quantization with 32 elements in each group, and quantization parameters of the FLOAT8_E8M0 type):

    $$
    \mathrm{dequantScaleQNorm} = 2^{\lfloor \mathrm{log}_2(\mathrm{GroupMax}(\mathrm{abs}(c^Q))) \rfloor - 8}
    $$

    $$
    queryNorm = {\mathrm{cast\_to\_fp8}(c^Q / \mathrm{dequantScaleQNorm})}
    $$

    where $Q_{max}$ is the maximum value of the quantized output type: 127 for INT8, 448 for FLOAT8_E4M3FN, 32768 for HIFLOAT8, and 8 is the maximum value emax of the exponent bits of FLOAT8_E4M3FN.

## Parameters

| Parameter                  | Input/Output/Attribute | Description  | Data Type | Data Format   |
|----------------------------|-----------|-----|----------------------------------------------------------------------|----------------------------|
| token_x                     | Input      | Input tensor for computing Query and Key in the formulas. | BFLOAT16, FLOAT8_E4M3FN, INT8, HIFLOAT8 | ND         |
| weight_dq                   | Input      | Downsampling weight matrix $W^{DQ}$ for computing Query in the formulas. <br> Dimension representation in the non-transposed case: (k, n). | BFLOAT16, FLOAT8_E4M3FN, INT8, HIFLOAT8 | FRACTAL_NZ |
| weight_uq_qr                 | Input      | Upsampling weight matrix $W^{UQ}$ and position encoding weight matrix $W^{QR}$ for computing Query in the formulas. <br> Dimension representation in the non-transposed case: (k, n).| BFLOAT16, FLOAT8_E4M3FN, INT8, HIFLOAT8 | FRACTAL_NZ |
| weight_uk                   | Input      | Upsampling weight $W^{UK}$ for computing Key in the formulas. | BFLOAT16 | ND         |
| weight_dkv_kr                | Input      | Downsampling weight matrix $W^{DKV}$ and position encoding weight matrix $W^{KR}$ for computing Key in the formulas. <br> Dimension representation in the non-transposed case: (k, n).| BFLOAT16, FLOAT8_E4M3FN, INT8, HIFLOAT8 | FRACTAL_NZ |
| rmsnorm_gamma_cq             | Input      | Parameter $\gamma$ in the RmsNorm formula for computing $c^Q$. | BFLOAT16 | ND         |
| rmsnorm_gamma_ckv            | Input      | Parameter $\gamma$ in the RmsNorm formula for computing $c^{KV}$. | BFLOAT16 | ND         |
| rope_sin                    | Input      | Sine parameter matrix of the rotary position encoding. | BFLOAT16 | ND         |
| rope_cos                    | Input      | Cosine parameter matrix of the rotary position encoding. | BFLOAT16 | ND         |
| kv_cache                 | Input/Output| aclTensor of the cache index, with the computation result updated in place (corresponding to $k^C$).|  BFLOAT16, FLOAT8_E4M3FN, INT8, HIFLOAT8 | ND         |
| kr_cache                 | Input/Output| Cache of the Key position encoding, with the computation result updated in place (corresponding to $k^R$). |  BFLOAT16, INT8 | ND         |
| cache_index                 | Input      | Index for storing kvCache and krCache. | INT64| ND         |
| dequant_scale_x      | Input      | Dequantization parameters of token_x.  | FLOAT8_E8M0, FLOAT | ND         |
| dequant_scale_w_dq    | Input      | Dequantization parameters of weight_dq. | FLOAT8_E8M0, FLOAT | ND         |
| dequant_scale_w_uq_qr  | Input      | Perchannel parameters for dequantization after the MatmulQcQr matrix multiplication. | FLOAT8_E8M0, FLOAT | ND         |
| dequant_scale_w_dkv_kr | Input      | Dequantization parameters of weight_dkv_kr. | FLOAT8_E8M0, FLOAT | ND         |
| quant_scale_ckv      | Input      | Quantization parameters of the KVCache output. | FLOAT | ND         |
| quant_scale_ckr      | Input      | Quantization parameters of the KRCache output. | FLOAT | ND         |
| smooth_scales_cq     | Input      | Dynamic quantization parameters of the RmsNormCq output. | FLOAT | ND         |
| actual_seq_len                 | Input      | Reserved parameter. It is not used in the current version and must be passed as a null pointer. | INT32 | ND         |
| k_nope_clip_alpha    | Input      | Scaling factor used when the clip operation is performed on kv_cache.  | FLOAT | ND         |
| rmsnorm_epsilon_cq           | Input      | Parameter $\epsilon$ in the RmsNorm formula for computing $c^Q$. | DOUBLE | -          |
| rmsnorm_epsilon_ckv          | Input      | Parameter $\epsilon$ in the RmsNorm formula for computing $c^{KV}$. | DOUBLE | -          |
| cache_mode          | Input      | kvCache mode. | CHAR* | -          |
| query_norm_flag           | Input      | Indicates whether to output query_norm; Host-side parameter. | BOOL | -          |
| weight_quant_mode          | Input      | Indicates the quantization modes of weight_dq, weight_uq_qr, weight_uk, and weight_dkv_kr. | INT64 | -          |
| kv_cache_quant_mode           | Input      | Indicates the quantization mode of kv_cache. | INT64 | -          |
| query_quant_mode          | Input      | Indicates the quantization mode of query. | INT64 | -          |
| ckvkr_repo_mode           | Input      | Indicates the storage modes of kv_cache and kr_cache. | INT64 | -          |
| quant_scale_repo_mode           | Input      | Indicates the storage mode of quantization scales. | INT64 | -          |
| tile_size          | Input      | Indicates the size of each tile in pertoken-pergroup quantization. The value must be 128. | INT64 | -          |
| qc_qr_scale          | Input      | Scale correction parameter of the Query, corresponding to $\alpha_q$. The default value is 1.0. | DOUBLE | -          |
| kc_scale          | Input      | Scale correction parameter of the Key, corresponding to $\alpha_{kv}$. The default value is 1.0. | DOUBLE | -          |
| query                   | Output      | Output tensor of Query in the formulas (corresponding to $q^N$). | BFLOAT16, FLOAT8_E4M3FN, INT8, HIFLOAT8 | ND         |
| query_rope               | Output      | Output tensor of the Query position encoding in the formulas (corresponding to $q^R$). | BFLOAT16 | ND |
| dequant_scale_q_nope | Output     | Quantization parameters of the Query output tensor.   | FLOAT | ND         |
| query_norm               | Output      | Output tensor after rmsNorm is performed on tokenX in the formula (corresponding to $c^Q$). | BFLOAT16, FLOAT8_E4M3FN, INT8, HIFLOAT8 | ND |
| dequant_scale_q_norm | Output     | Quantization parameters of the query_norm output tensor.   | FLOAT, FLOAT8_E8M0 | ND         |

- <term>Atlas A3 products</term> and <term>Atlas A2 products</term>:
  - token_x, weight_dq, weight_uq_qr, weight_dkv_kr, kv_cache, query, and query_norm do not support the FLOAT8_E4M3FN or HIFLOAT8 data types.
  - dequant_scale_x, dequant_scale_w_dq, dequant_scale_w_uq_qr, dequant_scale_w_dkv_kr, and dequant_scale_q_norm do not support the FLOAT8_E8M0 data type.

## Constraints

- Shape constraints
    - If the dimensions of tokenX adopt combined BS axes, that is (T, He):
        - The shapes of ropeSin and ropeCos are (T, Dr).
        - When CacheMode is PA_BSND or PA_NZ, the shape of cacheIndex is (T).
        - When CacheMode is PA_BLK_BSND or PA_BLK_NZ, the shape of cacheIndex is (Sum(Ceil(S_i/BlockSize))), where S_i is the length of S in each batch.
        - When CacheMode is PA_BLK_BSND or PA_BLK_NZ, actualSeqLenOptional must be passed, with the dimension (B).
        - In int8/fp8/hif8 full quantization scenarios, the shape of dequantScaleXOptional is (T, 1); in mxfp8 full quantization scenarios, the shape of dequantScaleXOptional is (T, He/32).
        - The shape of queryOut is (T, N, Hckv).
        - The shape of queryRopeOut is (T, N, Dr).
        - In int8/mxfp8/fp8/hif8 full quantization scenarios, the shape of dequantScaleQNopeOutOptional is (T, N, 1); in other scenarios, it is nullptr.
    - If the dimensions of tokenX do not adopt combined BS axes, that is (B, S, He):
        - The shapes of ropeSin and ropeCos are (B, S, Dr).
        - When CacheMode is PA_BSND or PA_NZ, the shape of cacheIndex is (B, S).
        - When CacheMode is PA_BLK_BSND or PA_BLK_NZ, the shape of cacheIndex is (B,Ceil(S/BlockSize)).
        - In int8/fp8/hif8 full quantization scenarios, the shape of dequantScaleXOptional is (B\*S, 1); in mxfp8 full quantization scenarios, the shape of dequantScaleXOptional is (B*S, He/32).
        - The shape of queryOut is (B, S, N, Hckv).
        - The shape of queryRopeOut is (B, S, N, Dr).
        - In int8/mxfp8/fp8/hif8 full quantization scenarios, the shape of dequantScaleQNopeOutOptional is (B*S, N, 1); in other scenarios, it is nullptr.
    - One or more of the values B, S, T, and Skv can be 0. That is, input parameters whose shapes are related to the values of B, S, T, and Skv can be empty tensors, while other input parameters do not support empty tensors.
        - If B, S, or T is 0, queryOut and queryRopeOut output empty tensors, and kvCacheRef and krCacheRef are not updated.
        - If Skv is 0, queryOut, queryRopeOut, and dequantScaleQNopeOutOptional are computed normally, while kvCacheRef and krCacheRef are not updated; that is, empty tensors are output.
    - When CacheMode is BSND:
        - tokenX should not adopt combined BS axes, that is, the dimensions are (B, S, He).
        - The dimensions of kvCache are (B,S,Nkv,Dr).
    - When CacheMode is TND:
        - tokenX should adopt combined BS axes, that is, the dimensions are (T, He).
        - The dimensions of kvCache are (T,Nkv,Dr).
    - When ckvkrRepoMode=1:
        - The dimensions of krCache should contain 0, and the shape (0) is supported.

- Special constraints
  - On Atlas A2 products, Atlas A3 products, and Ascend 950PR&950DT products, kvCache and krCache support a non-contiguous first axis; the other axes except the first axis must be contiguous.
  - When actualSeqLenOptional is passed, the last value of actualSeqLenOptional must be the same as T.
  - In pertoken-pergroup quantization mode, ckvkrRepoMode and quantScaleRepoMode must both be 1; in other quantization modes and non-quantization scenarios, ckvkrRepoMode and quantScaleRepoMode must both be 0.
  - In pertoken-pergroup quantization mode, CacheMode supports only PA_BSND, BSND, and TND.
  - When ckvkrRepoMode is 1, krCache must be an empty tensor (that is, the product of the shape is 0).
  - In kvcache pertensor quantization mode, kvCacheQuantMode and queryQuantMode must both be 1.
- Scenarios supported by the aclnnMlaPrologV3WeightNz API:
    - <term>Atlas A2 products</term> and <term>Atlas A3 products</term>: The fp8/hif8/mxfp8 full quantization scenarios are not supported currently
    - <term>Ascend 950PR&950DT products</term>: All quantization scenarios are supported currently
  <table style="table-layout: auto;" border="1">
    <tr>
      <th colspan="2">Scenario</th>
      <th>Description</th>
    </tr>
    <tr>
      <td colspan="2">Non-quantization</td>
      <td>
          weight_quant_mode=0,kv_cache_quant_mode=0,query_quant_mode=0<br>
          Input: All inputs are non-quantized data <br>
          Output: All outputs are non-quantized data
      </td>
    </tr>
    <tr>
      <td rowspan="3">Partial quantization</td>
      <td>kvCache non-quantization </td>
      <td>
          weight_quant_mode=1,kv_cache_quant_mode=0,query_quant_mode=0<br>
          Input: weightUqQr is passed as pertoken quantized data, and all other inputs are non-quantized data. The dequantScaleWUqQr field must be passed, and the smoothScalesCq field is optional <br>
          Output: All outputs return non-quantized data
      </td>
    </tr>
    <tr>
      <td>kvCache perchannel quantization </td>
      <td>
          weight_quant_mode=1,kv_cache_quant_mode=2,query_quant_mode=0<br>
          Input: weightUqQr is passed as pertoken quantized data, kvCacheRef and krCacheRef are passed as perchannel quantized data, and all other inputs are non-quantized data. The dequantScaleWUqQr, quantScaleCkv, and quant_scale_ckr fields must be passed, and the smoothScalesCq field is optional <br>
          Output: kvCacheRef and krCacheRef return perchannel quantized data, and all other outputs return non-quantized data
      </td>
    </tr>
    <tr>
      <td>kvCache pertoken-pergroup quantization </td>
      <td>
          weight_quant_mode=1, kv_cache_quant_mode=3, query_quant_mode=0<br>
          Input: weightUqQr is passed as pertoken quantized data, and all other inputs are non-quantized data. The dequantScaleWUqQr field must be passed, and the smoothScalesCq field is optional <br>
          Output: kvCacheRef returns pertoken-pergroup quantized data, and all other outputs return non-quantized data
      </td>
    </tr>
    <tr>
      <td rowspan="3">int8/fp8/hif8 full quantization</td>
      <td> kvCache non-quantization</td>
      <td>
          weight_quant_mode=2/4/5,kv_cache_quant_mode=0,query_quant_mode=0<br>
          Input: tokenX is passed as pertoken quantized data, weightDq, weightUqQr, and weightDkvKr are passed as perchannel quantized data, and all other inputs are non-quantized data. The dequantScaleX, dequantScaleWDq, dequantScaleWUqQr, and dequantScaleWDkvKr fields must be passed, and the smoothScalesCq field is optional <br>
          Output: All outputs return non-quantized data
      </td>
    </tr>
    <tr>
      <td> kvCache pertensor quantization </td>
      <td>
          weight_quant_mode=2/4/5,kv_cache_quant_mode=1,query_quant_mode=1<br>
          Input: tokenX is passed as pertoken quantized data, weightDq, weightUqQr, and weightDkvKr are passed as perchannel quantized data, kvCacheRef is passed as pertensor quantized data, and all other inputs are non-quantized data. The dequantScaleX, dequantScaleWDq, dequantScaleWUqQr, dequantScaleWDkvKr, and quantScaleCkv fields must be passed, and the smoothScalesCq field is optional <br>
          Output: queryOut returns per-token-head quantized data, the kvCacheRef output returns pertensor quantized data, and all other outputs return non-quantized data
      </td>
    </tr>
    <tr>
      <td> kvCache pertoken-pergroup quantization </td>
      <td>
          weight_quant_mode=2/4/5,kv_cache_quant_mode=3,query_quant_mode=0<br>
          Input: tokenX is passed as pertoken quantized data, weightDq, weightUqQr, and weightDkvKr are passed as perchannel quantized data, and all other inputs are non-quantized data. The dequantScaleX, dequantScaleWDq, dequantScaleWUqQr, and dequantScaleWDkvKr fields must be passed, and the smoothScalesCq field is optional <br>
          Output: The kvCacheRef output returns pertoken-pergroup quantized data, and all other outputs return non-quantized data
      </td>
    </tr>
    <tr>
      <td rowspan="3">mxfp8 full quantization</td>
      <td> kvCache non-quantization</td>
      <td>
          weight_quant_mode=3,kv_cache_quant_mode=0,query_quant_mode=0<br>
          Input: tokenX is passed as pertoken quantized data, weightDq, weightUqQr, and weightDkvKr are passed as perchannel quantized data, and all other inputs are non-quantized data. The dequantScaleX, dequantScaleWDq, dequantScaleWUqQr, and dequantScaleWDkvKr fields must be passed <br>
          Output: All outputs return non-quantized data
      </td>
    </tr>
    <tr>
      <td> kvCache pertensor quantization </td>
      <td>
          weight_quant_mode=3,kv_cache_quant_mode=1,query_quant_mode=1<br>
          Input: tokenX is passed as pertoken quantized data, weightDq, weightUqQr, and weightDkvKr are passed as perchannel quantized data, kvCacheRef is passed as pertensor quantized data, and all other inputs are non-quantized data. The dequantScaleX, dequantScaleWDq, dequantScaleWUqQr, dequantScaleWDkvKr, and quantScaleCkv fields must be passed <br>
          Output: queryOut returns per-token-head quantized data, the kvCacheRef output returns pertensor quantized data, and all other outputs return non-quantized data
      </td>
    </tr>
    <tr>
      <td> kvCache pertoken-pergroup quantization </td>
      <td>
          weight_quant_mode=3,kv_cache_quant_mode=3,query_quant_mode=0<br>
          Input: tokenX is passed as pertoken quantized data, weightDq, weightUqQr, and weightDkvKr are passed as perchannel quantized data, and all other inputs are non-quantized data. The dequantScaleX, dequantScaleWDq, dequantScaleWUqQr, and dequantScaleWDkvKr fields must be passed <br>
          Output: The kvCacheRef output returns pertoken-pergroup quantized data, and all other outputs return non-quantized data
      </td>
    </tr>
  </table>

## Invocation

<table class="tg"><thead>
  <tr>
    <th class="tg-0pky">Invocation Method</th>
    <th class="tg-0pky">Sample Code</th>
    <th class="tg-0pky">Description</th>
  </tr></thead>
<tbody>
  <tr>
    <td class="tg-9wq8" rowspan="2">aclnn API</td>
    <td class="tg-0pky">
    <a href="./examples/test_aclnn_mla_prolog_v3_weight_nz.cpp">Test case code of the aclnnMlaPrologV3WeightNz API
    </a>
    </td>
    <td class="tg-lboi">
    Invokes the operator through the
    <a href="./docs/aclnnMlaPrologV3WeightNz_en.md">aclnnMlaPrologV3WeightNz
    </a>
    API
    </td>
  </tr>
  <tr>
    <td class="tg-0pky">
    <a href="./examples/test_aclnn_mla_prolog_v4_weight_nz.cpp">Test case code of the aclnnMlaPrologV4WeightNz API
    </a>
    </td>
    <td class="tg-lboi">
    Invokes the operator through the
    <a href="./docs/aclnnMlaPrologV4WeightNz_en.md">aclnnMlaPrologV4WeightNz
    </a>
    API (the newly added doRope switch controls whether to perform the rotary position encoding)
    </td>
  </tr>
  <tr>
    <td class="tg-9wq8">PyTorch API</td>
    <td class="tg-0pky">-</td>
    <td class="tg-lboi">Invokes the mla_prolog operator through the <a href="../../attention/mla_prolog_v3/docs/torchapi_mla_prolog_en.md">mla_prolog</a> API.</td>
  </tr>
</tbody></table>
