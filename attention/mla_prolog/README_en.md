# MlaProlog

English | [简体中文](./README.md)

## Product Support

|Product      | Supported |
|:----------------------------|:-----------:|
|<term>Ascend 950PR&950DT products</term>|      ×     |
|<term>Atlas A3 products</term>|      √     |
|<term>Atlas A2 products</term>|      √     |

## Function Description

- **Operator function**: Computes the preprocessing of Multi-Head Latent Attention in inference scenarios. The main computation process is divided into four paths. First, the input $x$ is multiplied by $W^{DQ}$ for downsampling and RmsNorm, and the result is split into two paths. Path 1 is multiplied by $W^{UQ}$ and $W^{UK}$ and obtains $q^N$ after two upsampling operations. Path 2 is multiplied by $W^{QR}$ and obtains $q^R$ after the rotary position embedding (ROPE). Path 3 is the input $x$ multiplied by $W^{DKV}$ for downsampling and RmsNorm, and the result is written into the cache to obtain $k^C$. Path 4 is the input $x$ multiplied by $W^{KR}$, and the result is written into another cache after the rotary position embedding to obtain $k^R$.
- **Computation formulas**:

    RmsNorm formula

    $$
    \text{RmsNorm}(x) = \gamma \cdot \frac{x_i}{\text{RMS}(x)}
    $$

    $$
    \text{RMS}(x) = \sqrt{\frac{1}{N} \sum_{i=1}^{N} x_i^2 + \epsilon}
    $$

    Query computation formulas, including downsampling, RmsNorm, and two upsampling operations

    $$
    c^Q = RmsNorm(x \cdot W^{DQ})
    $$

    $$
    q^C = c^Q \cdot W^{UQ}
    $$

    $$
    q^N = q^C \cdot W^{UK}
    $$

    ROPE rotary position embedding for the Query

    $$
    q^R = ROPE(c^Q \cdot W^{QR})
    $$

    Key computation formulas, including downsampling and RmsNorm. The computation results are stored in the cache.

    $$
    c^{KV} = RmsNorm(x \cdot W^{DKV})
    $$

    $$
    k^C = Cache(c^{KV})
    $$

    ROPE rotary position embedding for the Key, and the result is stored in the cache

    $$
    k^R = Cache(ROPE(x \cdot W^{KR}))
    $$

## Parameter Description

| Parameter name              | Input/Output/Attribute | Description                                                          | Data type       | Data format |
|----------------------------|-----------|----------------------------------------------------------------------|----------------|------------|
| token_x                     | Input      | The input tensor used to compute the Query and Key in the formulas | INT8, BF16 | ND         |
| weight_dq                   | Input      | The downsampling weight matrix $W^{DQ}$ used to compute the Query in the formulas | INT8, BF16 | FRACTAL_NZ |
| weight_uq_qr                 | Input      | The upsampling weight matrix $W^{UQ}$ and the position embedding weight matrix $W^{QR}$ used to compute the Query in the formulas.| INT8, BF16 | FRACTAL_NZ |
| weight_uk                   | Input      | The upsampling weight $W^{UK}$ used to compute the Key in the formulas | BF16       | ND         |
| weight_dkv_kr                | Input      | The downsampling weight matrix $W^{DKV}$ and the position embedding weight matrix $W^{KR}$ used to compute the Key in the formulas | INT8, BF16| FRACTAL_NZ |
| rmsnorm_gamma_cq             | Input      | The $\gamma$ parameter in the RmsNorm formula for computing $c^Q$ | FLOAT16, BF16       | ND         |
| rmsnorm_gamma_ckv            | Input      | The $\gamma$ parameter in the RmsNorm formula for computing $c^{KV}$ | FLOAT16, BF16       | ND         |
| rope_sin                    | Input      | The sine parameter matrix of the rotary position embedding | FLOAT16, BF16       | ND         |
| rope_cos                    | Input      | The cosine parameter matrix of the rotary position embedding | FLOAT16, BF16       | ND         |
| cache_index                 | Input      | The indices for storing kvCache and krCache | INT64          | ND         |
| kv_cache                 | Input/ Output| The aclTensor of the cache index; the computation result is updated in place (corresponding to $k^C$)| FLOAT16, BF16, INT8 | ND         |
| kr_cache                 | Input/ Output| The cache of the Key position embedding; the computation result is updated in place (corresponding to $k^R$) | FLOAT16, BF16, INT8 | ND         |
| dequant_scale_x      | Input      | Reserved parameter; not used in the current version; a null pointer must be passed in  | FLOAT          | ND         |
| dequant_scale_w_dq    | Input      | Reserved parameter; not used in the current version; a null pointer must be passed in | FLOAT          | ND         |
| dequant_scale_w_uq_qr  | Input      | The per-channel dequantization parameter after the MatmulQcQr matrix multiplication| FLOAT          | ND         |
| dequant_scale_w_dkv_kr | Input      | Reserved parameter; not used in the current version; a null pointer must be passed in | FLOAT          | ND         |
| quant_scale_ckv      | Input      | The quantization parameter of the KVCache output | FLOAT          | ND         |
| quant_scale_ckr      | Input      | The quantization parameter of the KRCache output | FLOAT          | ND         |
| smooth_scales_cq     | Input      | The dynamic quantization parameter of the RmsNormCq output | FLOAT          | ND         |
| rmsnorm_epsilon_cq           | Input      | The $\epsilon$ parameter in the RmsNorm formula for computing $c^Q$ | DOUBLE         | -          |
| rmsnorm_epsilon_ckv          | Input      | The $\epsilon$ parameter in the RmsNorm formula for computing $c^{KV}$ | DOUBLE         | -          |
| cache_mode          | Input      | The kvCache mode | CHAR*          | -          |
| query                   | Output      | The output tensor of the Query in the formulas (corresponding to $q^N$) | FLOAT16, BF16, INT8 | ND         |
| query_rope               | Output      | The output tensor of the Query position embedding in the formulas (corresponding to $q^R$) | FLOAT16, BF16, INT8       | ND |

## Constraint Description

- One or more of B, S, T, and Skv can be 0, that is, the input parameters whose shapes are related to B, S, T, and Skv can be passed as empty tensors, and the other input parameters do not support being passed as empty tensors.
    - If B, S, or T is 0, query and query_rope output empty tensors, and kv_cache and kr_cache are not updated.
    - If Skv is 0, query and query_rope are computed normally, and kv_cache and kr_cache are not updated, that is, empty tensors are output.
- For weight_dq, weight_uq_qr, and weight_dkv_kr, the dimensions are represented as (k, n) when they are not transposed.
- Scenarios supported by the aclnnMlaProlog interface:
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
        <td>kv_cache non-quantization </td>
        <td>
            Input parameters: weight_uq_qr is passed as pertoken-quantized data, and the other input parameters are all non-quantized data <br>
            Output parameters: All output parameters return non-quantized data
        </td>
      </tr>
      <tr>
        <td>kv_cache quantization </td>
        <td>
            Input parameters: weight_uq_qr is passed as pertoken-quantized data, kv_cache and kr_cache are passed as perchannel-quantized data, and the other input parameters are all non-quantized data <br>
            Output parameters: kv_cache and kr_cache return perchannel-quantized data, and the other output parameters return non-quantized data
        </td>
      </tr>
    </table>

- In different quantization scenarios, the dtype and shape combinations of the parameters must meet the following conditions:
    <div style="overflow-x: auto; width: 100%;">
    <table style="table-layout: auto;" border="1">
      <tr>
        <th rowspan="3">Parameter name</th>
        <th rowspan="2" colspan="2">Non-quantization scenario</th>
        <th colspan="4">Partial quantization scenario</th>
      </tr>
      <tr>
        <th colspan="2">kv_cache non-quantization</th>
        <th colspan="2">kv_cache quantization</th>
      </tr>
      <tr>
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
      </tr>
      <tr>
        <td>weight_dq</td>
        <td>BFLOAT16</td>
        <td> (He, Hcq)</td>
        <td>BFLOAT16</td>
        <td> (He, Hcq)</td>
        <td>BFLOAT16</td>
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
      </tr>
      <tr>
        <td>weight_uk</td>
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
      </tr>
      <tr>
        <td> rmsnorm_gamma_cq </td>
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
      </tr>
      <tr>
        <td> rope_sin </td>
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
      </tr>
      <tr>
        <td> cache_index </td>
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
      </tr>
      <tr>
        <td> kr_cache </td>
        <td>BFLOAT16</td>
        <td> (BlockNum, BlockSize, Nkv, Dr)</td>
        <td>BFLOAT16</td>
        <td> (BlockNum, BlockSize, Nkv, Dr)</td>
        <td>INT8</td>
        <td> (BlockNum, BlockSize, Nkv, Dr)</td>
      </tr>
      <tr>
        <td> dequant_scale_x </td>
        <td>No need to assign</td>
        <td> / </td>
        <td>No need to assign</td>
        <td> / </td>
        <td>No need to assign</td>
        <td> / </td>
      </tr>
      <tr>
        <td> dequant_scale_w_dq </td>
        <td>No need to assign</td>
        <td> / </td>
        <td>No need to assign</td>
        <td> / </td>
        <td>No need to assign</td>
        <td> / </td>
      </tr>
      <tr>
        <td> dequant_scale_w_uq_qr </td>
        <td>No need to assign</td>
        <td> / </td>
        <td> FLOAT </td>
        <td> (1, N*(D+Dr)) </td>
        <td> FLOAT </td>
        <td> (1, N*(D+Dr)) </td>
      </tr>
      <tr>
        <td> dequant_scale_w_dkv_kr </td>
        <td>No need to assign</td>
        <td> / </td>
        <td>No need to assign</td>
        <td> / </td>
        <td>No need to assign</td>
        <td> / </td>
      </tr>
      <tr>
        <td> quant_scale_ckv </td>
        <td>No need to assign</td>
        <td> / </td>
        <td>No need to assign</td>
        <td> / </td>
        <td>FLOAT</td>
        <td> (1, Hckv) </td>
      </tr>
      <tr>
        <td> quant_scale_ckr </td>
        <td>No need to assign</td>
        <td> / </td>
        <td>No need to assign</td>
        <td> / </td>
        <td>FLOAT</td>
        <td> (1, Dr) </td>
      </tr>
      <tr>
        <td> smooth_scales_cq </td>
        <td>No need to assign</td>
        <td> / </td>
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
      </tr>
      <tr>
        <td> query_rope </td>
        <td>BFLOAT16</td>
        <td> · (B, S, N, Dr) <br> · (T, N, Dr)</td>
        <td>BFLOAT16</td>
        <td> · (B, S, N, Dr) <br> · (T, N, Dr)</td>
        <td>BFLOAT16</td>
        <td> · (B, S, N, Dr) <br> · (T, N, Dr)</td>
      </tr>
    </table>
    </div>

## Invocation Description

<table class="tg"><thead>
  <tr>
    <th class="tg-0pky">Invocation Method</th>
    <th class="tg-0pky">Sample Code</th>
    <th class="tg-0pky">Description</th>
  </tr></thead>
<tbody>
  <tr>
    <td class="tg-9wq8" rowspan="6">aclnn API</td>
    <td class="tg-0pky">
    <a href="./examples//test_aclnn_mla_prolog_nq_bsh.cpp">Test case code of the MlaProlog non-quantization (BSH) interface
    </a>
    </td>
    <td class="tg-lboi" rowspan="6">
    Invokes the operator through the
    <a href="./docs/aclnnMlaProlog_en.md">aclnnMlaProlog
    </a>
    interface
    </td>
  </tr>
  <tr>
    <td class="tg-0pky">
    <a href="./examples/test_aclnn_mla_prolog_pq.cpp">Test case code of the MlaProlog partial quantization (BSH) interface
    </a>
  </td>
  </tr>
</tbody></table>

<!-- ## References
[MlaProlog operator design principles](docs/MlaProlog_design_en.md) -->
