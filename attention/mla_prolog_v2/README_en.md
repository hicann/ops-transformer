# MlaPrologV2

English | [简体中文](./README.md)

## Product Support

|Product  | Supported |
|:----------------------------|:-----------:|
|<term>Ascend 950PR&950DT products</term>|      ×     |
|<term>Atlas A3 products</term>|      √     |
|<term>Atlas A2 products</term>|      √     |

## Function Description

- **Operator function**: Computes the Multi-Head Latent Attention (MLA) preprocessing in inference scenarios. The main computation consists of four paths. First, the input $x$ is multiplied by $W^{DQ}$ for downsampling and RmsNorm, and then split into two paths: the first path is multiplied by $W^{UQ}$ and $W^{UK}$ and passes through two rounds of upsampling to obtain $q^N$; the second path is multiplied by $W^{QR}$ and then passes through the rotary position embedding (ROPE) to obtain $q^R$; the third path multiplies the input $x$ by $W^{DKV}$ for downsampling and RmsNorm, and then stores the result into the Cache to obtain $k^C$; the fourth path multiplies the input $x$ by $W^{KR}$, passes it through the rotary position embedding, and then stores the result into another Cache to obtain $k^R$.
- **Computation formulas**:

    RmsNorm formula

    $$
    \text{RmsNorm}(x) = \gamma \cdot \frac{x_i}{\text{RMS}(x)}
    $$

    $$
    \text{RMS}(x) = \sqrt{\frac{1}{N} \sum_{i=1}^{N} x_i^2 + \epsilon}
    $$

    Computation formulas for the Query, including downsampling, RmsNorm, and two rounds of upsampling

    $$
    c^Q = RmsNorm(x \cdot W^{DQ})
    $$

    $$
    q^C = c^Q \cdot W^{UQ}
    $$

    $$
    q^N = q^C \cdot W^{UK}
    $$

    ROPE rotary position encoding applied to the Query

    $$
    q^R = ROPE(c^Q \cdot W^{QR})
    $$

    Computation formulas for the Key, including downsampling and RmsNorm, and storing the computation result into the cache

    $$
    c^{KV} = RmsNorm(x \cdot W^{DKV})
    $$

    $$
    k^C = Cache(c^{KV})
    $$

    ROPE rotary position encoding applied to the Key, and the result is stored into the cache

    $$
    k^R = Cache(ROPE(x \cdot W^{KR}))
    $$

## Parameter Description

| Parameter Name             | Input/Output/Attribute | Description  | Data Type       | Data Format   |
|----------------------------|-----------|----------------------------------------------------------------------|----------------|------------|
| token_x                     | Input      | Input tensor used to compute the Query and Key in the formulas | INT8, BF16 | ND         |
| weight_dq                   | Input      | Downsampling weight matrix $W^{DQ}$ used to compute the Query in the formulas | INT8, BF16 | FRACTAL_NZ |
| weight_uq_qr                 | Input      | Upsampling weight matrix $W^{UQ}$ and position-encoding weight matrix $W^{QR}$ used to compute the Query in the formulas.| INT8, BF16 | FRACTAL_NZ |
| weight_uk                   | Input      | Upsampling weight $W^{UK}$ used to compute the Key in the formulas | BF16       | ND         |
| weight_dkv_kr                | Input      | Downsampling weight matrix $W^{DKV}$ and position-encoding weight matrix $W^{KR}$ used to compute the Key in the formulas | INT8, BF16| FRACTAL_NZ |
| rmsnorm_gamma_cq             | Input      | The $\gamma$ parameter in the RmsNorm formula for computing $c^Q$ | BF16       | ND         |
| rmsnorm_gamma_ckv            | Input      | The $\gamma$ parameter in the RmsNorm formula for computing $c^{KV}$ | BF16       | ND         |
| rope_sin                    | Input      | Sine parameter matrix of the rotary position encoding | BF16       | ND         |
| rope_cos                    | Input      | Cosine parameter matrix of the rotary position encoding | BF16       | ND         |
| cache_index                 | Input      | Indexes for storing kvCache and krCache | INT64          | ND         |
| kv_cache                 | Input/Output| aclTensor for cache indexing; the computation result updates it in place (corresponding to $k^C$)| BF16, INT8 | ND         |
| kr_cache                 | Input/Output| Cache for Key position encoding; the computation result updates it in place (corresponding to $k^R$) | BF16, INT8 | ND         |
| dequant_scale_x      | Input      | Reserved parameter, not used in the current version; you must pass a null pointer  | FLOAT          | ND         |
| dequant_scale_w_dq    | Input      | Reserved parameter, not used in the current version; you must pass a null pointer | FLOAT          | ND         |
| dequant_scale_w_uq_qr  | Input      | per-channel parameter for dequantization after the MatmulQcQr matrix multiplication| FLOAT          | ND         |
| dequant_scale_w_dkv_kr | Input      | Reserved parameter, not used in the current version; you must pass a null pointer | FLOAT          | ND         |
| quant_scale_ckv      | Input      | Quantization parameter of the KVCache output | FLOAT          | ND         |
| quant_scale_ckr      | Input      | Quantization parameter of the KRCache output | FLOAT          | ND         |
| smooth_scales_cq     | Input      | Dynamic quantization parameter of the RmsNormCq output | FLOAT          | ND         |
| rmsnorm_epsilon_cq           | Input      | The $\epsilon$ parameter in the RmsNorm formula for computing $c^Q$ | DOUBLE         | -          |
| rmsnorm_epsilon_ckv          | Input      | The $\epsilon$ parameter in the RmsNorm formula for computing $c^{KV}$ | DOUBLE         | -          |
| cache_mode          | Input      | Mode of kvCache | CHAR*          | -          |
| query                   | Output      | Output tensor of the Query in the formulas (corresponding to $q^N$) | BF16, INT8 | ND         |
| query_rope               | Output      | Output tensor of the Query position encoding in the formulas (corresponding to $q^R$) | BF16, INT8       | ND |
| dequant_scale_q_nope | Output     | Quantization parameter of the Query output tensor   | FLOAT             | ND         |

## Constraints

- shape constraints
    - If the dimensions of token_x use BS merged axes, that is (T, He)
        - The shape of rope_sin and rope_cos is (T, Dr)
        - The shape of cache_index is (T,)
        - The shape of dequant_scale_x is (T, 1)
        - The shape of query is (T, N, Hckv)
        - The shape of query_rope is (T, N, Dr)
        - In the full-quantization scenario, the shape of dequantScaleQNopeOutOptional is (T, N, 1); in other scenarios, it is (1)
    - If the dimensions of token_x do not use BS merged axes, that is (B, S, He)
        - The shape of rope_sin and rope_cos is (B, S, Dr)
        - The shape of cache_index is (B, S)
        - The shape of dequant_scale_x is (B*S, 1)
        - The shape of query is (B, S, N, Hckv)
        - The shape of query_rope is (B, S, N, Dr)
        - In the full-quantization scenario, the shape of dequantScaleQNopeOutOptional is (B*S, N, 1); in other scenarios, it is (1)
    - One or more of B, S, T, and Skv can be 0, which means that input parameters whose shapes are related to the values of B, S, T, and Skv can be passed in as empty tensors, and the other input parameters do not support empty tensors.
        - If B, S, or T is 0, query and query_rope output empty tensors, and kv_cache and kr_cache are not updated.
        - If Skv is 0, query, query_rope, and dequantScaleQNopeOutOptional are computed normally, and kv_cache and kr_cache are not updated, that is, they output empty tensors.
- For weight_dq, weight_uq_qr, and weight_dkv_kr without transposition, the dimensions are represented as (k, n).
- Scenarios supported by the aclnnMlaPrologV2WeightNz API:
    <table style="table-layout: auto;" border="1">
      <tr>
        <th colspan="2">Scenario</th>
        <th>Description</th>
      </tr>
      <tr>
        <td colspan="2">Non-quantization</td>
        <td>
            Input parameters: All input parameters are non-quantized data <br>
            Output parameters: All output parameters are non-quantized data
        </td>
      </tr>
      <tr>
        <td rowspan="2">Partial quantization</td>
        <td>kv_cache is not quantized</td>
        <td>
            Input parameters: weight_uq_qr is passed in as pertoken-quantized data, and the other input parameters are all non-quantized data <br>
            Output parameters: All output parameters return non-quantized data
        </td>
      </tr>
      <tr>
        <td>kv_cache is quantized</td>
        <td>
            Input parameters: weight_uq_qr is passed in as pertoken-quantized data, kv_cache and kr_cache are passed in as perchannel-quantized data, and the other input parameters are all non-quantized data <br>
            Output parameters: kv_cache and kr_cache return perchannel-quantized data, and the other output parameters return non-quantized data
        </td>
      </tr>
      <tr>
        <td rowspan="2">Full quantization</td>
        <td> kv_cache is not quantized</td>
        <td>
            Input parameters: token_x is passed in as pertoken-quantized data, weight_dq, weight_uq_qr, and weight_dkv_kr are passed in as perchannel-quantized data, and the other input parameters are all non-quantized data <br>
            Output parameters: All output parameters are non-quantized data
        </td>
      </tr>
      <tr>
        <td> kv_cache is quantized </td>
        <td>
            Input parameters: token_x is passed in as pertoken-quantized data, weight_dq, weight_uq_qr, and weight_dkv_kr are passed in as perchannel-quantized data, kv_cache is passed in as pertensor-quantized data, and the other input parameters are all non-quantized data <br>
            Output parameters: query returns pertoken_head-quantized data, the kv_cache output returns pertensor-quantized data, and the other output parameters return non-quantized data
        </td>
      </tr>
    </table>

- In different quantization scenarios, the dtype and shape combinations of parameters must meet the following conditions:
    <div style="overflow-x: auto; width: 100%;">
    <table style="table-layout: auto;" border="1">
      <tr>
        <th rowspan="3">Parameter Name</th>
        <th rowspan="2" colspan="2">Non-Quantization Scenario</th>
        <th colspan="4">Partial Quantization Scenario</th>
        <th colspan="4">Full Quantization Scenario</th>
      </tr>
      <tr>
        <th colspan="2">kv_cache is not quantized</th>
        <th colspan="2">kv_cache is quantized</th>
        <th colspan="2">kv_cache is not quantized</th>
        <th colspan="2">kv_cache is quantized</th>
      </tr>
      <tr>
        <th>dtype</th>
        <th>shape</th>
        <th>dtype</th>
        <th>shape</th>
        <th>dtype</th>
        <th>shape</th>
        <th>dtype</th>
        <th>shape</th>
        <th>dtype</th>
        <th>shape</th>
      </tr>
      <tr>
        <td>token_x</td>
        <td>BFLOAT16</td>
        <td>· (B,S,He) <br> · (T, He)</td>
        <td>BFLOAT16</td>
        <td>· (B,S,He) <br> · (T, He)</td>
        <td>BFLOAT16</td>
        <td>· (B,S,He) <br> · (T, He)</td>
        <td>INT8</td>
        <td>· (B,S,He) <br> · (T, He)</td>
        <td>INT8</td>
        <td>· (B,S,He) <br> · (T, He)</td>
      </tr>
      <tr>
        <td>weight_dq</td>
        <td>BFLOAT16</td>
        <td> (He, Hcq)</td>
        <td>BFLOAT16</td>
        <td> (He, Hcq)</td>
        <td>BFLOAT16</td>
        <td> (He, Hcq)</td>
        <td>INT8</td>
        <td> (He, Hcq)</td>
        <td>INT8</td>
        <td> (He, Hcq)</td>
      </tr>
      <tr>
        <td>weight_uq_qr</td>
        <td>BFLOAT16</td>
        <td> (Hcq, N*(D+Dr))</td>
        <td>INT8</td>
        <td> (Hcq, N*(D+Dr))</td>
        <td>INT8</td>
        <td> (Hcq, N*(D+Dr))</td>
        <td>INT8</td>
        <td> (Hcq, N*(D+Dr))</td>
        <td>INT8</td>
        <td> (Hcq, N*(D+Dr))</td>
      </tr>
      <tr>
        <td>weight_uk</td>
        <td>BFLOAT16</td>
        <td> (N, D, Hckv)</td>
        <td>BFLOAT16</td>
        <td> (N, D, Hckv)</td>
        <td>BFLOAT16</td>
        <td> (N, D, Hckv)</td>
        <td>BFLOAT16</td>
        <td> (N, D, Hckv)</td>
        <td>BFLOAT16</td>
        <td> (N, D, Hckv)</td>
      </tr>
      <tr>
        <td>weight_dkv_kr</td>
        <td>BFLOAT16</td>
        <td> (He, Hckv+Dr)</td>
        <td>BFLOAT16</td>
        <td> (He, Hckv+Dr)</td>
        <td>BFLOAT16</td>
        <td> (He, Hckv+Dr)</td>
        <td>INT8</td>
        <td> (He, Hckv+Dr)</td>
        <td>INT8</td>
        <td> (He, Hckv+Dr)</td>
      </tr>
      <tr>
        <td> rmsnorm_gamma_cq </td>
        <td>BFLOAT16</td>
        <td> (Hcq)</td>
        <td>BFLOAT16</td>
        <td> (Hcq)</td>
        <td>BFLOAT16</td>
        <td> (Hcq)</td>
        <td>BFLOAT16</td>
        <td> (Hcq)</td>
        <td>BFLOAT16</td>
        <td> (Hcq)</td>
      </tr>
      <tr>
        <td> rmsnorm_gamma_ckv </td>
        <td>BFLOAT16</td>
        <td> (Hckv)</td>
        <td>BFLOAT16</td>
        <td> (Hckv)</td>
        <td>BFLOAT16</td>
        <td> (Hckv)</td>
        <td>BFLOAT16</td>
        <td> (Hckv)</td>
        <td>BFLOAT16</td>
        <td> (Hckv)</td>
      </tr>
      <tr>
        <td> rope_sin </td>
        <td>BFLOAT16</td>
        <td> · (B,S,Dr) <br> · (T, Dr )</td>
        <td>BFLOAT16</td>
        <td> · (B,S,Dr) <br> · (T, Dr )</td>
        <td>BFLOAT16</td>
        <td> · (B,S,Dr) <br> · (T, Dr )</td>
        <td>BFLOAT16</td>
        <td> · (B,S,Dr) <br> · (T, Dr )</td>
        <td>BFLOAT16</td>
        <td> · (B,S,Dr) <br> · (T, Dr )</td>
      </tr>
      <tr>
        <td> rope_cos </td>
        <td>BFLOAT16</td>
        <td> · (B,S,Dr) <br> · (T, Dr )</td>
        <td>BFLOAT16</td>
        <td> · (B,S,Dr) <br> · (T, Dr )</td>
        <td>BFLOAT16</td>
        <td> · (B,S,Dr) <br> · (T, Dr )</td>
        <td>BFLOAT16</td>
        <td> · (B,S,Dr) <br> · (T, Dr )</td>
        <td>BFLOAT16</td>
        <td> · (B,S,Dr) <br> · (T, Dr )</td>
      </tr>
      <tr>
        <td> cache_index </td>
        <td>INT64</td>
        <td> · (B,S) <br> · (T)</td>
        <td>INT64</td>
        <td> · (B,S) <br> · (T)</td>
        <td>INT64</td>
        <td> · (B,S) <br> · (T)</td>
        <td>INT64</td>
        <td> · (B,S) <br> · (T)</td>
        <td>INT64</td>
        <td> · (B,S) <br> · (T)</td>
      </tr>
      <tr>
        <td> kv_cache </td>
        <td>BFLOAT16</td>
        <td> (BlockNum, BlockSize, Nkv, Hckv)</td>
        <td>BFLOAT16</td>
        <td> (BlockNum, BlockSize, Nkv, Hckv)</td>
        <td>INT8</td>
        <td> (BlockNum, BlockSize, Nkv, Hckv)</td>
        <td>BFLOAT16</td>
        <td> (BlockNum, BlockSize, Nkv, Hckv)</td>
        <td>INT8</td>
        <td> (BlockNum, BlockSize, Nkv, Hckv)</td>
      </tr>
      <tr>
        <td> kr_cache </td>
        <td>BFLOAT16</td>
        <td> (BlockNum, BlockSize, Nkv, Dr)</td>
        <td>BFLOAT16</td>
        <td> (BlockNum, BlockSize, Nkv, Dr)</td>
        <td>INT8</td>
        <td> (BlockNum, BlockSize, Nkv, Dr)</td>
        <td>BFLOAT16</td>
        <td> (BlockNum, BlockSize, Nkv, Dr)</td>
        <td>BFLOAT16</td>
        <td> (BlockNum, BlockSize, Nkv, Dr)</td>
      </tr>
      <tr>
        <td> dequant_scale_x </td>
        <td>No assignment required</td>
        <td> / </td>
        <td>No assignment required</td>
        <td> / </td>
        <td>No assignment required</td>
        <td> / </td>
        <td>FLOAT</td>
        <td> · (B*S, 1) <br> · (T, 1)</td>
        <td>FLOAT</td>
        <td> · (B*S, 1) <br> · (T, 1)</td>
      </tr>
      <tr>
        <td> dequant_scale_w_dq </td>
        <td>No assignment required</td>
        <td> / </td>
        <td>No assignment required</td>
        <td> / </td>
        <td>No assignment required</td>
        <td> / </td>
        <td>FLOAT</td>
        <td> (1, Hcq)</td>
        <td>FLOAT</td>
        <td> (1, Hcq)</td>
      </tr>
      <tr>
        <td> dequant_scale_w_uq_qr </td>
        <td>No assignment required</td>
        <td> / </td>
        <td>FLOAT</td>
        <td> (1, N*(D+Dr)) </td>
        <td>FLOAT</td>
        <td> (1, N*(D+Dr)) </td>
        <td>FLOAT</td>
        <td> (1, N*(D+Dr)) </td>
        <td>FLOAT</td>
        <td> (1, N*(D+Dr)) </td>
      </tr>
      <tr>
        <td> dequant_scale_w_dkv_kr </td>
        <td> No assignment required </td>
        <td> / </td>
        <td> No assignment required </td>
        <td> / </td>
        <td> No assignment required </td>
        <td> / </td>
        <td>FLOAT</td>
        <td> (1, Hckv+Dr) </td>
        <td>FLOAT</td>
        <td> (1, Hckv+Dr) </td>
      </tr>
      <tr>
        <td> quant_scale_ckv </td>
        <td>No assignment required</td>
        <td> / </td>
        <td>No assignment required</td>
        <td> / </td>
        <td>FLOAT</td>
        <td> (1, Hckv) </td>
        <td>No assignment required</td>
        <td> / </td>
        <td>FLOAT</td>
        <td> (1, Hckv) </td>
      </tr>
      <tr>
        <td> quant_scale_ckr </td>
        <td>No assignment required</td>
        <td> / </td>
        <td>No assignment required</td>
        <td> / </td>
        <td>FLOAT</td>
        <td> (1, Dr) </td>
        <td>No assignment required</td>
        <td> / </td>
        <td>No assignment required</td>
        <td> / </td>
      </tr>
      <tr>
        <td> smooth_scales_cq </td>
        <td>No assignment required</td>
        <td> / </td>
        <td>FLOAT</td>
        <td> (1, Hcq) </td>
        <td>FLOAT</td>
        <td> (1, Hcq) </td>
        <td>FLOAT</td>
        <td> (1, Hcq) </td>
        <td>FLOAT</td>
        <td> (1, Hcq) </td>
      </tr>
      <tr>
        <td> query </td>
        <td>BFLOAT16</td>
        <td> · (B, S, N, Hckv) <br> · (T, N, Hckv)</td>
        <td>BFLOAT16</td>
        <td> · (B, S, N, Hckv) <br> · (T, N, Hckv)</td>
        <td>BFLOAT16</td>
        <td> · (B, S, N, Hckv) <br> · (T, N, Hckv)</td>
        <td>BFLOAT16</td>
        <td> · (B, S, N, Hckv) <br> · (T, N, Hckv)</td>
        <td>INT8</td>
        <td> · (B, S, N, Hckv) <br> · (T, N, Hckv)</td>
      </tr>
      <tr>
        <td> query_rope </td>
        <td>BFLOAT16</td>
        <td> · (B, S, N, Dr) <br> · (T, N, Dr)</td>
        <td>BFLOAT16</td>
        <td> · (B, S, N, Dr) <br> · (T, N, Dr)</td>
        <td>BFLOAT16</td>
        <td> · (B, S, N, Dr) <br> · (T, N, Dr)</td>
        <td>BFLOAT16</td>
        <td> · (B, S, N, Dr) <br> · (T, N, Dr)</td>
        <td>BFLOAT16</td>
        <td> · (B, S, N, Dr) <br> · (T, N, Dr)</td>
      </tr>
      <tr>
        <td> dequant_scale_q_nope </td>
        <td>No assignment required</td>
        <td>/</td>
        <td>No assignment required</td>
        <td>/</td>
        <td>No assignment required</td>
        <td>/</td>
        <td>No assignment required</td>
        <td>/</td>
        <td>FLOAT</td>
        <td> · (B*S, N, 1) <br> · (T, N, 1)</td>
      </tr>
    </table>
    </div>

## Invocation Instructions

<table class="tg"><thead>
  <tr>
    <th class="tg-0lax">Invocation Method</th>
    <th class="tg-0lax">Sample Code</th>
    <th class="tg-0lax">Description</th>
  </tr></thead>
<tbody>
  <tr>
    <td class="tg-9wq8" rowspan="10">aclnn API</td>
    <td class="tg-0pky">
    <a href="./examples/test_aclnn_mla_prolog_v2_weight_nz_nq_bsh.cpp">MlaPrologV2 non-quantization (BSH) API test case code
    </a>
    </td>
    <td class="tg-lboi" rowspan="10">Calls the operator through the
    <a href="./docs/aclnnMlaPrologV2WeightNz_en.md">aclnnMlaPrologV2WeightNz
    </a>
    API
  </td>
  </tr>
  <tr>
    <td class="tg-0lax">
    <a href="./examples/test_aclnn_mla_prolog_v2_weight_nz_fq.cpp">MlaPrologV2 full-quantization KV quantization (BSH) API test case code
    </a>
    </td>
  </tr>
</tbody></table>

<!-- ## Reference Resources
[MlaProlog Operator Design Principle](../mla_prolog/docs/MlaProlog_design_en.md) -->
