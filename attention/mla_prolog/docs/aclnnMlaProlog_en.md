# aclnnMlaProlog

**Note: This API will be deprecated in later versions. Use the latest API aclnnMlaPrologV3WeightNz instead.**

## Supported Products

|Product     | Supported|
|:----------------------------|:-----------:|
|Ascend 950PR/Ascend 950DT|      ×     |
|<term>Atlas A3 training products/Atlas A3 inference products</term>|      √     |
|<term>Atlas A2 training products/Atlas A2 inference products</term>|      √     |

## Function

- **Interface function**: In inference scenarios, this operator performs the preprocessing computation for Multi-Head Latent Attention. The computation process consists of four paths. First, the input $x$ is multiplied by $W^{DQ}$ for downsampling and RmsNorm, and then split into two paths. In the first path, $W^{UQ}$ and $W^{UK}$ are applied, followed by two upsampling operations to obtain $q^N$. In the second path, $W^{QR}$ is applied and rotary position encoding (ROPE) is used to obtain $q^R$. The third path multiplies input $x$ by $W^{DKV}$ for downsampling and RmsNorm, and the result is stored into the cache to obtain $k^C$. The fourth path multiplies input $x$ by $W^{KR}$, applies rotary position encoding, and then stores the result into another cache to obtain $k^R$.
- **Formula**:

    RmsNorm formula

    $$
    \text{RmsNorm}(x) = \gamma \cdot \frac{x_i}{\text{RMS}(x)}
    $$

    $$
    \text{RMS}(x) = \sqrt{\frac{1}{N} \sum_{i=1}^{N} x_i^2 + \epsilon}
    $$

    The computation formula of the query, including downsampling, RmsNorm, and two upsampling operations.

    $$
    c^Q = RmsNorm(x \cdot W^{DQ})
    $$

    $$
    q^C = c^Q \cdot W^{UQ}
    $$

    $$
    q^N = q^C \cdot W^{UK}
    $$

    Performs rotary position encoding (ROPE) to the query.

    $$
    q^R = ROPE(c^Q \cdot W^{QR})
    $$

    The computation formula of the key, including downsampling and RmsNorm. The computation result is stored in the cache.

    $$
    c^{KV} = RmsNorm(x \cdot W^{DKV})
    $$

    $$
    k^C = Cache(c^{KV})
    $$

    Performs rotary position encoding (ROPE) to the key and stores the result in the cache.

    $$
    k^R = Cache(ROPE(x \cdot W^{KR}))
    $$

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnMlaPrologGetWorkspaceSize` is called to obtain the input parameters and compute the required workspace size based on the process. Then, `aclnnMlaProlog` is called to perform computation.

```cpp
aclnnStatus aclnnMlaPrologGetWorkspaceSize(
  const aclTensor *tokenX, 
  const aclTensor *weightDq, 
  const aclTensor *weightUqQr, 
  const aclTensor *weightUk, 
  const aclTensor *weightDkvKr, 
  const aclTensor *rmsnormGammaCq, 
  const aclTensor *rmsnormGammaCkv, 
  const aclTensor *ropeSin, 
  const aclTensor *ropeCos, 
  const aclTensor *cacheIndex, 
  aclTensor       *kvCacheRef, 
  aclTensor       *krCacheRef, 
  const aclTensor *dequantScaleXOptional, 
  const aclTensor *dequantScaleWDqOptional, 
  const aclTensor *dequantScaleWUqQrOptional, 
  const aclTensor *dequantScaleWDkvKrOptional, 
  const aclTensor *quantScaleCkvOptional, 
  const aclTensor *quantScaleCkrOptional, 
  const aclTensor *smoothScalesCqOptional, 
  double           rmsnormEpsilonCq, 
  double           rmsnormEpsilonCkv, 
  char            *cacheModeOptional, 
  const aclTensor *queryOut, 
  const aclTensor *queryRopeOut, 
  uint64_t        *workspaceSize, 
  aclOpExecutor  **executor)
```

``` cpp
aclnnStatus aclnnMlaProlog(
  void          *workspace, 
  uint64_t       workspaceSize, 
  aclOpExecutor *executor, 
  aclrtStream    stream)
```

## aclnnMlaPrologGetWorkspaceSize

- **Parameters**

    | Parameter| Input/Output| Description| Usage Description  | Data Type| Data Format  | Dimension (shape)| Non-contiguous tensor|
    |----------------------------|-----------|----------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------|------------|---------------------------|-------------------------|
    | tokenX                     | Input     | Input tensor used to calculate the query and key in the formula.         | Empty tensors with B=0, S=0, and T=0 are supported.                                                                                                             | BFLOAT16       | ND         | Input dimensions of A2 and A3:<br>- BS fused: (T, He)<br>- BS unfused: (B, S, He)| ×                       |
    | weightDq                   | Input     | Downsampling weight matrix $W^{DQ}$ used to calculate the query in the formula.     |  Empty tensors are not supported.                                                                                                                        | BFLOAT16       | FRACTAL_NZ | (He,Hcq)             | ×                       |
    | weightUqQr                 | Input     | Upsampling weight matrix $W^{UQ}$ and position encoding weight matrix $W^{QR}$ used to calculate the query in the formula.|  Empty tensors are not supported.<br> When dtype is INT8 (quantization scenario):<br> 1. The input must be per-tensor quantization.<br>2. dequantScaleWUqQrOptional is mandatory for non-quantized output.<br>3. dequantScaleWUqQrOptional, quantScaleCkvOptional, and quantScaleCkrOptional must be passed for quantized output.<br>4. smoothScalesCqOptional (optional)<br> dtype is BFLOAT16 (non-quantization scenario):<br>1. dequantScaleWUqQrOptional, quantScaleCkvOptional, quantScaleCkrOptional and smoothScalesCqOptional must pass a null pointer.| BFLOAT16, INT8| FRACTAL_NZ | (Hcq,N*(D+Dr))       | ×                       |
    | weightUk                   | Input     | Upsampling weight matrix $W^{UK}$ used to calculate the key in the formula.          |  Empty tensors are not supported.      | BFLOAT16       | ND         | (N,D,Hckv)           | ×                       |
    | weightDkvKr                | Input     | Downsampling weight matrix $W^{DKV}$ and position encoding weight matrix $W^{KR}$ used to calculate the key in the formula.|  Empty tensors are not supported.                                       | BFLOAT16       | FRACTAL_NZ | (He,Hckv+Dr)         | ×                      |
    | rmsnormGammaCq             | Input     | Parameter $\gamma$ in the RmsNorm formula for calculating $c^Q$.         |  Empty tensors are not supported.                                                 | BFLOAT16       | ND         | (Hcq)                | ×                       |
    | rmsnormGammaCkv            | Input     | Parameter $\gamma$ in the RmsNorm formula for calculating $c^{KV}$.       |  Empty tensors are not supported.                                                        | BFLOAT16       | ND         | (Hckv)               | ×                       |
    | ropeSin                    | Input     | Sine parameter matrix used to calculate the rotation position encoding.             |  Empty tensors with B=0, S=0, and T=0 are supported.                                                 | BFLOAT16       | ND         | Input dimensions of A2 and A3:<br>- BS fused: (T, Dr)<br>- BS unfused: (B, S, Dr)| ×                       |
    | ropeCos                    | Input     | Cosine parameter matrix used to calculate the rotation position encoding.             |  Empty tensors with B=0, S=0, and T=0 are supported.                                               | BFLOAT16       | ND         | Input dimensions of A2 and A3:<br>- BS fused: (T, Dr)<br>- BS unfused: (B, S, Dr) | ×                       |
    | cacheIndex                 | Input     | Index used to store kvCache and krCache.                 |  Empty tensors with B=0, S=0, and T=0 are supported.<br> The value range must be [0, BlockNum x BlockSize).                                                                  | INT64          | ND         | Input dimensions of A2 and A3:<br>- BS fused: (T)<br>- BS unfused: (B, S)     | ×                       |
    | kvCacheRef                 | Input     | The aclTensor used for cache indexing, with in-place update of the computation result (corresponding to $k^C$ in the formula).      |  - Empty tensors with B=0 and Skv=0 are supported. Nkv is associated with N, where N is a hyperparameter; therefore, Nkv cannot be equal to 0.                                                                        | BFLOAT16, INT8| ND         | (BlockNum,BlockSize,Nkv,Hckv) | ×                       |
    | krCacheRef                 | Input     | Cache used to calculate the key position encoding. The calculation result is updated in place (corresponding to $k^R$ in the formula).| - Empty tensors with B=0 and Skv=0 are supported. Nkv is associated with N, where N is a hyperparameter; therefore, Nkv cannot be equal to 0.                                                                         | BFLOAT16, INT8| ND         | (BlockNum,BlockSize,Nkv,Dr) | ×                       |
    | dequantScaleXOptional      | Input     | Dequantization parameter of tokenX. |  - The data format is ND.| FLOAT          | ND         | - BS fused: (T)<br>- BS unfused: (B*S, 1)                         | ×                       |
    | dequantScaleWDqOptional    | Input     | Dequantization parameter of weightDq.|  - The data format is ND.       | FLOAT          | ND         | (1,Hcq)                         | ×                       |
    | dequantScaleWUqQrOptional  | Input     | Per-channel parameter used for dequantization after MatmulQcQr matrix multiplication.|  - Non-empty tensor is required (only applicable in INT8 dtype scenarios).                                                                                                | FLOAT          | ND         | (1,N*(D+Dr))         | ×                       |
    | dequantScaleWDkvKrOptional | Input     | Dequantization parameter of weightDkvKr.|  - The data format is ND.| FLOAT          | ND         |  (1, Hckv+Dr)                        | ×                       |
    | quantScaleCkvOptional      | Input     | Parameter used to quantize the output data of kvCache.           |  - Non-empty tensor is required (only applicable in INT8 quantized output scenarios).                                                                                        | FLOAT          | ND         | (1,Hckv)             | ×                       |
    | quantScaleCkrOptional      | Input     | Parameter used to quantize the output data of krCache.           |  - Non-empty tensor is required (only applicable in INT8 quantized output scenarios).                                                                                        | FLOAT          | ND         | (1,Dr)               | ×                       |
    | smoothScalesCqOptional     | Input     | Parameter used to dynamically quantize the output of RmsNormCq.        |  - Non-empty tensor is required (optional only in INT8 dtype scenarios).                                                                                              | FLOAT          | ND         | (1,Hcq)              | ×                       |
    | rmsnormEpsilonCq           | Input     | Parameter $\epsilon$ in the RmsNorm formula for calculating $c^Q$.                 |  - If the value is not specified, you are advised to set it to 1e-05.<br> - Only the double type is supported.                                                                                | DOUBLE         | -          | -                         | -                       |
    | rmsnormEpsilonCkv          | Input     | Parameter $\epsilon$ in the RmsNorm formula for calculating $c^{KV}$.               |  - If the value is not specified, you are advised to set it to 1e-05.<br> - Only the double type is supported.                                                                                | DOUBLE         | -          | -                         | -                       |
    | cacheModeOptional          | Input     | kvCache mode.                                       |  - If not specified, you are advised to pass "PA_BSND".<br> - Only `char*` type is supported.<br> - For A2 and A3, the value can be "PA_BSND" or "PA_NZ".                                         | CHAR*          | -          | -                         | -                       |
    | queryOut                   | Output     | Output tensor of the query in the formula (corresponding to $q^N$).            | -                                                                                                                        | BFLOAT16, INT8| ND         |  Input dimensions for A2 and A3:<br>- BS fused: (T, N, Hckv)<br>- BS unfused: (B, S, N, Hckv) | ×                       |
    | queryRopeOut               | Output     | Output tensor of the position encoding in the query (corresponding to $q^R$) in the formula.     | -                                                                                                                          | BFLOAT16       | ND         | A2 and A3 input dimensions:<br>- BS fused: (T, N, Dr)<br>- BS unfused: (B, S, N, Dr) | ×                       |
    | workspaceSize              | Output     | Size of the workspace required to be allocated on the device.                                 | - Used only for output results. No input configuration is required.<br> - The data type is uint64_t*.                                                                                | -              | -          | -                         | -                       |
    | executor                   | Output     | Operator executor, containing the operator computation process.                                     |  - Used only for output results. No input configuration is required.<br> - The data type is aclOpExecutor**.                                                                          | -              | -          | -                         | -                       |

- **Returns:**

    `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).
    
    The first-phase API implements input parameter verification. The following errors may be thrown.

    | Return Value                | Error Code              | Description     |
    |------------------------|----------------------|----------------------|
    | ACLNN_ERR_PARAM_NULLPTR | 161001   | A null pointer exists in a mandatory parameter (for example, an input/output parameter on which the core of the API depends).        |
    | ACLNN_ERR_PARAM_INVALID | 161002   | The shape (dimension/sizes) or dtype (data type) of the input parameter is not supported by the API.|
    | ACLNN_ERR_RUNTIME_ERROR | 361001   | An exception occurs when the API memory calls the NPU Runtime API (for example, the Runtime service is not started or memory allocation fails).|
    | ACLNN_ERR_INNER_TILING_ERROR | 561002  | An exception occurs during tiling. The dtype or shape of the input parameter is incorrect.|

## aclnnMlaProlog

- **Parameters**

  | Parameter       | Type        | Description                                                               |
  |---------------|------------------|---------------------------------------------------------------------|
  | workspace     | void\*           | Address of the workspace to be allocated on the device.                                |
  | workspaceSize | uint64_t         | Size of the workspace to be allocated on the device, which is obtained by calling <code>aclnnMlaPrologGetWorkspaceSize</code>.|
  | executor      | aclOpExecutor\*  | Operator executor, containing the operator computation process.                                     |
  | stream        | aclrtStream      | Stream for executing the task.|

- **Returns:**

    `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computing:
  - `aclnnMlaProlog` defaults to deterministic implementation.
The details are as follows:
- Shape field description

    | Field      | Full Spelling/Description                 | Value Rule and Description                                                                |
    |--------------|--------------------------------|------------------------------------------------------------------------------|
    | B            | Batch (batch size of input samples)     | Value range: 0 to 65536.                                                          |
    | S            | Seq-Length (sequence length of input samples)| A2 and A3: no limit                   |
    | He           | Hidden-Size (hidden layer size)       | A2 and A3: 1024, 2048, 3072, 4096, 5120, 6144, 7168, 7680, and 8192  |
    | Hcq          | Dimension of the low-rank q matrix                | The value is fixed at 1536.                                                          |
    | N            | Head-Num (number of heads)            | Value range: 1, 2, 4, 8, 16, 32, 64, or 128.                                      |
    | Hckv         | Dimension of the low-rank KV matrix               | The value is fixed at 512.                                                            |
    | D            | QK without position encoding dimensions           | The value is fixed at 128.                                                            |
    | Dr           | QK positional encoding dimension               | The value is fixed at 64.                                                             |
    | Nkv          | Number of KV heads                 | The value is fixed at 1.                                                              |
    | BlockNum     | Number of blocks in the PagedAttention scenario.   | The value is rounded up to the nearest integer after the result of `B x Skv/BlockSize` is calculated. (Skv indicates the sequence length of kv and can be 0.)|
    | BlockSize    | Block size in the PagedAttention scenario | Value range: a multiple of 16 ranging from 16 to 1024<br>                                                         |
    | T            | The fused size of BS.               | A2 and A3: 0–1048576; Note: When the `B` and `S` axes are fused, tokenX, ropeSin, and ropeCos are 2D, cacheIndex is 1D, and queryOut and queryRopeOut are 3D. |

- When transposing is not performed, the dimensions of `weight_dq`, `weight_uq_qr`, and `weight_dkv_kr` are represented as (k, n).
- The aclnnMlaProlog API supports the following scenarios for A2 and A3:
  <table style="table-layout: auto;" border="1">
    <tr>
      <th colspan="2">Scenario </th>
      <th>Description</th>
    </tr>
    <tr>
      <td colspan="2">Non-quantized</td>
      <td>
          Input parameters: All input parameters are non-quantized data.<br>
          Output parameters: All output parameters are non-quantized data.
      </td>
    </tr>
    <tr>
      <td rowspan="2">Partial quantization</td>
      <td>kv_cache non-quantization</td>
      <td>
          Input parameters: The weightUqQr input parameter is pertoken quantized data, and other input parameters are non-quantized data.<br>
          Output parameters: All output parameters are non-quantized data.
      </td>
    </tr>
    <tr>
      <td>kv_cache quantization</td>
      <td> 
          Input parameters: The weightUqQr input parameter is pertoken quantized data, the kvCacheRef and krCacheRef input parameters are perchannel quantized data, and other input parameters are non-quantized data.<br>
          Output parameters: kvCacheRef and krCacheRef return perchannel quantized data, and other output parameters return non-quantized data.
      </td>
    </tr>
  </table>

- In different quantization scenarios, the combination of dtype and shape of parameters must meet the following conditions:
  <div style="overflow-x: auto; width: 100%;">
  <table style="table-layout: auto;" border="1">
    <tr>
      <th rowspan="3">Name</th>
      <th rowspan="2" colspan="2">Non-Quantization Scenario</th>
      <th colspan="4">Partial Quantization Scenario</th>
    </tr>
    <tr>
      <th colspan="2">kv_cache Non-Quantization</th>
      <th colspan="2">kv_cache Quantization</th>
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
      <td>tokenX</td>
      <td>BFLOAT16</td>
      <td>· (B,S,He) <br> · (T, He)</td>
      <td>BFLOAT16</td>
      <td>· (B,S,He) <br> · (T, He)</td>
      <td>BFLOAT16</td>
      <td>· (B,S,He) <br> · (T, He)</td>
    </tr>
    <tr>
      <td>weightDq</td>
      <td>BFLOAT16</td>
      <td> (He, Hcq)</td>
      <td>BFLOAT16</td>
      <td> (He, Hcq)</td>
      <td>BFLOAT16</td>
      <td> (He, Hcq)</td>
    </tr>
    <tr>
      <td>weightUqQr</td>
      <td>BFLOAT16</td>
      <td> (Hcq, N*(D+Dr))</td>
      <td>INT8</td>
      <td> (Hcq, N*(D+Dr))</td>
      <td>INT8</td>
      <td> (Hcq, N*(D+Dr))</td>
    </tr>
    <tr>
      <td>weightUk</td>
      <td>BFLOAT16</td>
      <td> (N, D, Hckv)</td>
      <td>BFLOAT16</td>
      <td> (N, D, Hckv)</td>
      <td>BFLOAT16</td>
      <td> (N, D, Hckv)</td>
    </tr>
    <tr>
      <td>weightDkvKr</td>
      <td>BFLOAT16</td>
      <td> (He, Hckv+Dr)</td>
      <td>BFLOAT16</td>
      <td> (He, Hckv+Dr)</td>
      <td>BFLOAT16</td>
      <td> (He, Hckv+Dr)</td>
    </tr>
    <tr>
      <td> rmsnormGammaCq </td>
      <td>BFLOAT16</td>
      <td> (Hcq)</td>
      <td>BFLOAT16</td>
      <td> (Hcq)</td>
      <td>BFLOAT16</td>
      <td> (Hcq)</td>
    </tr>
    <tr>
      <td> rmsnormGammaCkv </td>
      <td>BFLOAT16</td>
      <td> (Hckv)</td>
      <td>BFLOAT16</td>
      <td> (Hckv)</td>
      <td>BFLOAT16</td>
      <td> (Hckv)</td>
    </tr>
    <tr>
      <td> ropeSin </td>
      <td>BFLOAT16</td>
      <td> · (B,S,Dr) <br> · (T, Dr )</td>
      <td>BFLOAT16</td>
      <td> · (B,S,Dr) <br> · (T, Dr )</td>
      <td>BFLOAT16</td>
      <td> · (B,S,Dr) <br> · (T, Dr )</td>
    </tr>
    <tr>
      <td> ropeCos </td>
      <td>BFLOAT16</td>
      <td> · (B,S,Dr) <br> · (T, Dr )</td>
      <td>BFLOAT16</td>
      <td> · (B,S,Dr) <br> · (T, Dr )</td>
      <td>BFLOAT16</td>
      <td> · (B,S,Dr) <br> · (T, Dr )</td>
    </tr>
    <tr>
      <td> cacheIndex </td>
      <td>INT64</td>
      <td> · (B,S) <br> · (T)</td>
      <td>INT64</td>
      <td> · (B,S) <br> · (T)</td>
      <td>INT64</td>
      <td> · (B,S) <br> · (T)</td>
    </tr>
    <tr>
      <td> kvCacheRef </td>
      <td>BFLOAT16</td>
      <td> (BlockNum, BlockSize, Nkv, Hckv)</td>
      <td>BFLOAT16</td>
      <td> (BlockNum, BlockSize, Nkv, Hckv)</td>
      <td>INT8</td>
      <td> (BlockNum, BlockSize, Nkv, Hckv)</td>
    </tr>
    <tr>
      <td> krCacheRef </td>
      <td>BFLOAT16</td>
      <td> (BlockNum, BlockSize, Nkv, Dr)</td>
      <td>BFLOAT16</td>
      <td> (BlockNum, BlockSize, Nkv, Dr)</td>
      <td>INT8</td>
      <td> (BlockNum, BlockSize, Nkv, Dr)</td>
    </tr>
    <tr>
      <td> dequantScaleXOptional </td>
      <td>No value needs to be assigned.</td>
      <td> / </td>
      <td>No value needs to be assigned.</td>
      <td> / </td>
      <td>No value needs to be assigned.</td>
      <td> / </td>
    </tr>
    <tr>
      <td> dequantScaleWDqOptional </td>
      <td>No value needs to be assigned.</td>
      <td> / </td>
      <td>No value needs to be assigned.</td>
      <td> / </td>
      <td>No value needs to be assigned.</td>
      <td> / </td>
    </tr>
    <tr>
      <td> dequantScaleWUqQrOptional </td>
      <td>No value needs to be assigned.</td>
      <td> / </td>
      <td>FLOAT</td>
      <td> (1, N*(D+Dr)) </td>
      <td>FLOAT</td>
      <td> (1, N*(D+Dr)) </td>
    </tr>
    <tr>
      <td> dequantScaleWDkvKrOptional </td>
      <td>No value needs to be assigned.</td>
      <td> / </td>
      <td>No value needs to be assigned.</td>
      <td> / </td>
      <td>No value needs to be assigned.</td>
      <td> / </td>
    </tr>
    <tr>
      <td> quantScaleCkvOptional </td>
      <td>No value needs to be assigned.</td>
      <td> / </td>
      <td>No value needs to be assigned.</td>
      <td> / </td>
      <td>FLOAT</td>
      <td> (1, Hckv) </td>
    </tr>
    <tr>
      <td> quantScaleCkrOptional </td>
      <td>No value needs to be assigned.</td>
      <td> / </td>
      <td>No value needs to be assigned.</td>
      <td> / </td>
      <td>FLOAT</td>
      <td> (1, Dr) </td>
    </tr>
    <tr>
      <td> smoothScalesCqOptional </td>
      <td>No value needs to be assigned.</td>
      <td> / </td>
      <td>FLOAT</td>
      <td> (1, Hcq) </td>
      <td>FLOAT</td>
      <td> (1, Hcq) </td>
    </tr>
    <tr>
      <td> queryOut </td>
      <td>BFLOAT16</td>
      <td> · (B, S, N, Hckv) <br> · (T, N, Hckv)</td>
      <td>BFLOAT16</td>
      <td> · (B, S, N, Hckv) <br> · (T, N, Hckv)</td>
      <td>BFLOAT16</td>
      <td> · (B, S, N, Hckv) <br> · (T, N, Hckv)</td>
    </tr>
    <tr>
      <td> queryRopeOut </td>
      <td>BFLOAT16</td>
      <td> · (B, S, N, Dr) <br> · (T, N, Dr)</td>
      <td>BFLOAT16</td>
      <td> · (B, S, N, Dr) <br> · (T, N, Dr)</td>
      <td>BFLOAT16</td>
      <td> · (B, S, N, Dr) <br> · (T, N, Dr)</td>
    </tr>
    <tr>
      <td> dequantScaleQNopeOutOptional </td>
      <td>No value needs to be assigned.</td>
      <td>/</td>
      <td>No value needs to be assigned.</td>
      <td>/</td>
      <td>No value needs to be assigned.</td>
      <td>/</td>
    </tr>
  </table>
  </div>

  <!-- For details about the parameters, see **Operator Execution APIs**. -->

## Calling Examples

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

  ```Cpp
  #include <iostream>
  #include <vector>
  #include "acl/acl.h"
  #include "aclnnop/aclnn_mla_prolog.h"
  
  #define CHECK_RET(cond, return_expr) \
    do {                               \
      if (!(cond)) {                   \
        return_expr;                   \
      }                                \
    } while (0)
  
  #define LOG_PRINT(message, ...)     \
    do {                              \
      printf(message, ##__VA_ARGS__); \
    } while (0)
  
  int64_t GetShapeSize(const std::vector<int64_t>& shape) {
      int64_t shape_size = 1;
      for (auto i : shape) {
          shape_size *= i;
      }
      return shape_size;
  }
  
  int Init(int32_t deviceId, aclrtStream* stream) {
      // (Fixed writing) Initialize resources.
      auto ret = aclInit(nullptr);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
      ret = aclrtSetDevice(deviceId);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
      ret = aclrtCreateStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
      return 0;
  }
  
  template <typename T>
  int CreateAclTensorND(const std::vector<T>& shape, void** deviceAddr, void** hostAddr,
                      aclDataType dataType, aclTensor** tensor) {
      auto size = GetShapeSize(shape) * sizeof(T);
      // Call aclrtMalloc to allocate memory on the device.
      auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
      // Call aclrtMalloc to allocate memory on the host.
      ret = aclrtMalloc(hostAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
      // Call aclCreateTensor to create an aclTensor.
      *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, nullptr, 0, aclFormat::ACL_FORMAT_ND,
                                shape.data(), shape.size(), *deviceAddr);
      // Call aclrtMemcpy to copy the data on the host to the memory on the device.
      ret = aclrtMemcpy(*deviceAddr, size, *hostAddr, GetShapeSize(shape)*aclDataTypeSize(dataType), ACL_MEMCPY_HOST_TO_DEVICE);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
      return 0;
  }
  
  template <typename T>
  int CreateAclTensorNZ(const std::vector<T>& shape, void** deviceAddr, void** hostAddr,
                      aclDataType dataType, aclTensor** tensor) {
      auto size = GetShapeSize(shape) * sizeof(T);
      // Call aclrtMalloc to allocate memory on the device.
      auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
      // Call aclrtMalloc to allocate memory on the host.
      ret = aclrtMalloc(hostAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
      // Call aclCreateTensor to create an aclTensor.
      *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, nullptr, 0, aclFormat::ACL_FORMAT_FRACTAL_NZ,
                                shape.data(), shape.size(), *deviceAddr);
      // Call aclrtMemcpy to copy the data on the host to the memory on the device.
      ret = aclrtMemcpy(*deviceAddr, size, *hostAddr, GetShapeSize(shape)*aclDataTypeSize(dataType), ACL_MEMCPY_HOST_TO_DEVICE);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
      return 0;
  }
  
  int TransToNZShape(std::vector<int64_t> &shapeND) {
      int64_t inputParam1 = shapeND[0];
      int64_t inputParam2 = shapeND[1];
      int64_t h0 = 16;
      int64_t newParam1 = inputParam2 / h0;
      int64_t newParam2 = inputParam1 / h0;
      shapeND[0] = newParam1;
      shapeND[1] = newParam2;
      shapeND.emplace_back(h0);
      shapeND.emplace_back(h0);
      return 0;
  }
  
  int main() {
      // 1. (Fixed writing) Initialize the device and stream. For details, see the list of external AscendCL APIs.
      // Set the device ID in use.
      int32_t deviceId = 0;
      aclrtStream stream;
      auto ret = Init(deviceId, &stream);
      // Handle the check as required.
      CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
      // 2. Construct the input and output based on the API definition.
      std::vector<int64_t> tokenXShape = {8, 1, 7168};  // B,S,He
      std::vector<int64_t> weightDqShape = {7168, 1536};  // He,Hcq
      std::vector<int64_t> weightUqQrShape = {1536, 6144};  // Hcq,N*(D+Dr)
      std::vector<int64_t> weightUkShape = {32, 128, 512};  // N,D,Hckv
      std::vector<int64_t> weightDkvKrShape = {7168, 576};  // He,Hckv+Dr
      std::vector<int64_t> rmsnormGammaCqShape = {1536};  // Hcq
      std::vector<int64_t> rmsnormGammaCkvShape = {512};  // Hckv
      std::vector<int64_t> ropeSinShape = {8, 1, 64};  // B,S,Dr
      std::vector<int64_t> ropeCosShape = {8, 1, 64};  // B,S,Dr
      std::vector<int64_t> cacheIndexShape = {8, 1};  // B,S
      std::vector<int64_t> kvCacheShape = {16, 128, 1, 512};  // BlockNum,BlockSize,Nkv,Hckv
      std::vector<int64_t> krCacheShape = {16, 128, 1, 64};  // BlockNum,BlockSize,Nkv,Dr
      std::vector<int64_t> queryShape = {8, 1, 32, 512};  // B,S,N,Hckv
      std::vector<int64_t> queryRopeShape = {8, 1, 32, 64};  // B,S,N,Dr
      double rmsnormEpsilonCq = 1e-5;
      double rmsnormEpsilonCkv = 1e-5;
      char cacheMode[] = "PA_BSND";
  
      void* tokenXDeviceAddr = nullptr;
      void* weightDqDeviceAddr = nullptr;
      void* weightUqQrDeviceAddr = nullptr;
      void* weightUkDeviceAddr = nullptr;
      void* weightDkvKrDeviceAddr = nullptr;
      void* rmsnormGammaCqDeviceAddr = nullptr;
      void* rmsnormGammaCkvDeviceAddr = nullptr;
      void* ropeSinDeviceAddr = nullptr;
      void* ropeCosDeviceAddr = nullptr;
      void* cacheIndexDeviceAddr = nullptr;
      void* kvCacheDeviceAddr = nullptr;
      void* krCacheDeviceAddr = nullptr;
      void* queryDeviceAddr = nullptr;
      void* queryRopeDeviceAddr = nullptr;
  
      void* tokenXHostAddr = nullptr;
      void* weightDqHostAddr = nullptr;
      void* weightUqQrHostAddr = nullptr;
      void* weightUkHostAddr = nullptr;
      void* weightDkvKrHostAddr = nullptr;
      void* rmsnormGammaCqHostAddr = nullptr;
      void* rmsnormGammaCkvHostAddr = nullptr;
      void* ropeSinHostAddr = nullptr;
      void* ropeCosHostAddr = nullptr;
      void* cacheIndexHostAddr = nullptr;
      void* kvCacheHostAddr = nullptr;
      void* krCacheHostAddr = nullptr;
      void* queryHostAddr = nullptr;
      void* queryRopeHostAddr = nullptr;
  
      aclTensor* tokenX = nullptr;
      aclTensor* weightDq = nullptr;
      aclTensor* weightUqQr = nullptr;
      aclTensor* weightUk = nullptr;
      aclTensor* weightDkvKr = nullptr;
      aclTensor* rmsnormGammaCq = nullptr;
      aclTensor* rmsnormGammaCkv = nullptr;
      aclTensor* ropeSin = nullptr;
      aclTensor* ropeCos = nullptr;
      aclTensor* cacheIndex = nullptr;
      aclTensor* kvCache = nullptr;
      aclTensor* krCache = nullptr;
      aclTensor* query = nullptr;
      aclTensor* queryRope = nullptr;
  
      // Convert the shapes of the three variables in NZ format.
      ret = TransToNZShape(weightDqShape);
      CHECK_RET(ret == 0, LOG_PRINT("trans NZ shape failed.\n"); return ret);
      ret = TransToNZShape(weightUqQrShape);
      CHECK_RET(ret == 0, LOG_PRINT("trans NZ shape failed.\n"); return ret);
      ret = TransToNZShape(weightDkvKrShape);
      CHECK_RET(ret == 0, LOG_PRINT("trans NZ shape failed.\n"); return ret);
  
      // Create a tokenX aclTensor.
      ret = CreateAclTensorND(tokenXShape, &tokenXDeviceAddr, &tokenXHostAddr, aclDataType::ACL_BF16, &tokenX);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create a weightDq aclTensor.
      ret = CreateAclTensorNZ(weightDqShape, &weightDqDeviceAddr, &weightDqHostAddr, aclDataType::ACL_BF16, &weightDq);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create a weightUqQr aclTensor.
      ret = CreateAclTensorNZ(weightUqQrShape, &weightUqQrDeviceAddr, &weightUqQrHostAddr, aclDataType::ACL_BF16, &weightUqQr);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create a weightUk aclTensor.
      ret = CreateAclTensorND(weightUkShape, &weightUkDeviceAddr, &weightUkHostAddr, aclDataType::ACL_BF16, &weightUk);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create a weightDkvKr aclTensor.
      ret = CreateAclTensorNZ(weightDkvKrShape, &weightDkvKrDeviceAddr, &weightDkvKrHostAddr, aclDataType::ACL_BF16, &weightDkvKr);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create a ropeSin aclTensor.
      ret = CreateAclTensorND(ropeSinShape, &ropeSinDeviceAddr, &ropeSinHostAddr, aclDataType::ACL_BF16, &ropeSin);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create a ropeCos aclTensor.
      ret = CreateAclTensorND(ropeCosShape, &ropeCosDeviceAddr, &ropeCosHostAddr, aclDataType::ACL_BF16, &ropeCos);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create an rmsnormGammaCq aclTensor.
      ret = CreateAclTensorND(rmsnormGammaCqShape, &rmsnormGammaCqDeviceAddr, &rmsnormGammaCqHostAddr, aclDataType::ACL_BF16, &rmsnormGammaCq);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create an rmsnormGammaCkv aclTensor.
      ret = CreateAclTensorND(rmsnormGammaCkvShape, &rmsnormGammaCkvDeviceAddr, &rmsnormGammaCkvHostAddr, aclDataType::ACL_BF16, &rmsnormGammaCkv);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create a cacheIndex aclTensor.
      ret = CreateAclTensorND(cacheIndexShape, &cacheIndexDeviceAddr, &cacheIndexHostAddr, aclDataType::ACL_INT64, &cacheIndex);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create a kvCache aclTensor.
      ret = CreateAclTensorND(kvCacheShape, &kvCacheDeviceAddr, &kvCacheHostAddr, aclDataType::ACL_BF16, &kvCache);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create a krCache aclTensor.
      ret = CreateAclTensorND(krCacheShape, &krCacheDeviceAddr, &krCacheHostAddr, aclDataType::ACL_BF16, &krCache);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create a query aclTensor.
      ret = CreateAclTensorND(queryShape, &queryDeviceAddr, &queryHostAddr, aclDataType::ACL_BF16, &query);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create a queryRope aclTensor.
      ret = CreateAclTensorND(queryRopeShape, &queryRopeDeviceAddr, &queryRopeHostAddr, aclDataType::ACL_BF16, &queryRope);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
  
      // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
      uint64_t workspaceSize = 0;
      aclOpExecutor* executor = nullptr;
      // Call the first-phase API of aclnnMlaProlog.
      ret = aclnnMlaPrologGetWorkspaceSize(tokenX, weightDq, weightUqQr, weightUk, weightDkvKr, rmsnormGammaCq, rmsnormGammaCkv, ropeSin, ropeCos, cacheIndex, kvCache, krCache, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, rmsnormEpsilonCq, rmsnormEpsilonCkv, cacheMode, query, queryRope, &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMlaPrologGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
      // Allocate device memory based on workspaceSize computed by the first-phase API.
      void* workspaceAddr = nullptr;
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
      }
      //Call the second-phase API of aclnnMlaProlog.
      ret = aclnnMlaProlog(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMlaProlog failed. ERROR: %d\n", ret); return ret);
  
      // 4. (Fixed writing) Wait until the task execution is complete.
      ret = aclrtSynchronizeStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
  
      // 5. Obtain the output value and copy the result from the device memory to the host. Modify the code based on the API definition.
      auto size = GetShapeSize(queryShape);
      std::vector<float> resultData(size, 0);
      ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), queryDeviceAddr, size * sizeof(float),
                        ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  
      // 6. Release aclTensor and aclScalar. Modify the configuration based on the API definition.
      aclDestroyTensor(tokenX);
      aclDestroyTensor(weightDq);
      aclDestroyTensor(weightUqQr);
      aclDestroyTensor(weightUk);
      aclDestroyTensor(weightDkvKr);
      aclDestroyTensor(rmsnormGammaCq);
      aclDestroyTensor(rmsnormGammaCkv);
      aclDestroyTensor(ropeSin);
      aclDestroyTensor(ropeCos);
      aclDestroyTensor(cacheIndex);
      aclDestroyTensor(kvCache);
      aclDestroyTensor(krCache);
      aclDestroyTensor(query);
      aclDestroyTensor(queryRope);
  
      // 7. Release device resources.
      aclrtFree(tokenXDeviceAddr);
      aclrtFree(weightDqDeviceAddr);
      aclrtFree(weightUqQrDeviceAddr);
      aclrtFree(weightUkDeviceAddr);
      aclrtFree(weightDkvKrDeviceAddr);
      aclrtFree(rmsnormGammaCqDeviceAddr);
      aclrtFree(rmsnormGammaCkvDeviceAddr);
      aclrtFree(ropeSinDeviceAddr);
      aclrtFree(ropeCosDeviceAddr);
      aclrtFree(cacheIndexDeviceAddr);
      aclrtFree(kvCacheDeviceAddr);
      aclrtFree(krCacheDeviceAddr);
      aclrtFree(queryDeviceAddr);
      aclrtFree(queryRopeDeviceAddr);
  
      // 8. Release host resources.
      aclrtFree(tokenXHostAddr);
      aclrtFree(weightDqHostAddr);
      aclrtFree(weightUqQrHostAddr);
      aclrtFree(weightUkHostAddr);
      aclrtFree(weightDkvKrHostAddr);
      aclrtFree(rmsnormGammaCqHostAddr);
      aclrtFree(rmsnormGammaCkvHostAddr);
      aclrtFree(ropeSinHostAddr);
      aclrtFree(ropeCosHostAddr);
      aclrtFree(cacheIndexHostAddr);
      aclrtFree(kvCacheHostAddr);
      aclrtFree(krCacheHostAddr);
      aclrtFree(queryHostAddr);
      aclrtFree(queryRopeHostAddr);
  
      if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
      }
      aclrtDestroyStream(stream);
      aclrtResetDevice(deviceId);
      aclFinalize();
  
      return 0;
  }
  ```
