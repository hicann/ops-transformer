# aclnnMlaPrologV3WeightNz

[📄 View Source Code](https://gitcode.com/cann/ops-transformer/tree/master/attention/mla_prolog_v3)

## Supported Products

|Product     | Supported|
|:----------------------------|:-----------:|
|Ascend 950PR/Ascend 950DT|      √     |
|<term>Atlas A3 training products/Atlas A3 inference products</term>|      √     |
|<term>Atlas A2 training products/Atlas A2 inference products</term>|      √     |

## Function

- **Function updates** (compared with `aclnnMlaPrologV2weightNz`):
    - The scale correction factors of the query and key are added, corresponding to qcQrScale ($\alpha_q$) and kcScale ($\alpha_{kv}$), respectively.
    - Optional input parameters (such as actualSeqLenOptional, kNopeClipAlphaOptional, queryNormFlag, weightQuantMode, kvCacheQuantMode, queryQuantMode, ckvkrRepoMode, quantScaleRepoMode, tileSize, `queryNormOutOptional`, and `dequantScaleQNormOptional`) are added, and `cache_mode` is changed from mandatory to optional.
    - The name and position of the `cacheIndex` parameter are adjusted to match the current `cacheIndexOptional`.
- **Interface function**: In inference scenarios, this operator performs the preprocessing computation for Multi-Head Latent Attention. The main computation process consists of five paths.
    - After multiplying the input $x$ with $W^{DQ}$ for downsampling and RmsNorm, the first path multiplies the result with $W^{UQ}$ and $W^{UK}$, followed by two upsampling operations, and then multiplies the result with the query scale correction factor $\alpha_q$ to obtain $q^N$. The second path multiplies the result with $W^{QR}$ and applies rotary position encoding (ROPE) to obtain $q^R$.
    - The third path is to multiply the input $x$ by $W^{DKV}$, perform downsampling and RmsNorm, and then multiply the result by the key scale correction factor $\alpha_{kv}$ to obtain $k^C$ and pass it to the cache.
    - The fourth path multiplies the input $x$ with $W^{KR}$, applies rotary position encoding (ROPE), and then stores the result into another cache to obtain $k^R$.
    - The fifth path processes the output $q^N$ through DynamicQuant to generate quantization parameters.
    - The weight parameters `WeightDq`, `WeightUqQr`, and `WeightDkvKr` are required to be provided in NZ format.

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
    c^Q = \alpha_q\cdot\mathrm{RmsNorm}(x \cdot W^{DQ})
    $$

    $$
    q^C = c^Q \cdot W^{UQ}
    $$

    $$
    q^N = q^C \cdot W^{UK}
    $$

    Performs rotary position encoding (ROPE) to the query.

    $$
    q^R = \mathrm{ROPE}(c^Q \cdot W^{QR})
    $$

    The computation formula of the key, including downsampling and RmsNorm. The computation result is stored in the cache.

    $$
    c^{KV} = \alpha_{kv}\cdot\mathrm{RmsNorm}(x \cdot W^{DKV})
    $$

    $$
    k^C = \mathrm{Cache}(c^{KV})
    $$

    Performs rotary position encoding (ROPE) to the key and stores the result in the cache.

    $$
    k^R = \mathrm{Cache}(\mathrm{ROPE}(x \cdot W^{KR}))
    $$

    Dequant Scale Query Nope calculation formula:

    $$
    \mathrm{dequantScaleQNope} = {\mathrm{RowMax}(\mathrm{abs}(q^{N})) / 127}
    $$

    $$
    q^{N} = {\mathrm{round}(q^{N} / \mathrm{dequantScaleQNope})}
    $$

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnMlaPrologV3WeightNzGetWorkspaceSize` is called to obtain the input parameters and compute the required workspace size based on the process. Then, `aclnnMlaPrologV3WeightNz` is called to perform computation.

```cpp
aclnnStatus aclnnMlaPrologV3WeightNzGetWorkspaceSize(
    const aclTensor *tokenX,
    const aclTensor *weightDq,
    const aclTensor *weightUqQr,
    const aclTensor *weightUk,
    const aclTensor *weightDkvKr,
    const aclTensor *rmsnormGammaCq,
    const aclTensor *rmsnormGammaCkv,
    const aclTensor *ropeSin,
    const aclTensor *ropeCos,
    aclTensor       *kvCacheRef,
    aclTensor       *krCacheRef,
    const aclTensor *cacheIndexOptional,
    const aclTensor *dequantScaleXOptional,
    const aclTensor *dequantScaleWDqOptional,
    const aclTensor *dequantScaleWUqQrOptional,
    const aclTensor *dequantScaleWDkvKrOptional,
    const aclTensor *quantScaleCkvOptional,
    const aclTensor *quantScaleCkrOptional,
    const aclTensor *smoothScalesCqOptional,
    const aclTensor *actualSeqLenOptional,
    const aclTensor *kNopeClipAlphaOptional,
    double           rmsnormEpsilonCq,
    double           rmsnormEpsilonCkv,
    char            *cacheModeOptional,
    int64_t          weightQuantMode,
    int64_t          kvCacheQuantMode,
    int64_t          queryQuantMode,
    int64_t          ckvkrRepoMode,
    int64_t          quantScaleRepoMode,
    int64_t          tileSize,
    double           qcQrScale,
    double           kcScale,
    const aclTensor *queryOut,
    const aclTensor *queryRopeOut,
    const aclTensor *dequantScaleQNopeOutOptional,
    const aclTensor *queryNormOutOptional,
    const aclTensor *dequantScaleQNormOutOptional,
    uint64_t        *workspaceSize,
    aclOpExecutor  **executor)
```

```cpp
aclnnStatus aclnnMlaPrologV3WeightNz(
  void             *workspace,
  uint64_t          workspaceSize,
  aclOpExecutor    *executor,
  const aclrtStream stream)
```

## aclnnMlaPrologV3WeightNzGetWorkspaceSize

- **Parameters**

  | Parameter Name                    | Input/Output| Description            | Usage Description      | Data Type      | Data types supported by Ascend 950PR/Ascend 950DT| Data Format  | Dimension (shape)   |Non-contiguous tensor|
  |----------------------------|-----------|----------------------------------------------|----------------|----------------|-|------------|-----------------|-------|
  | tokenX          | Input     | Input tensor used to calculate the query and key in the formula.   | Empty tensors with B=0, S=0, and T=0 are supported.  | BFLOAT16, INT8| BFLOAT16, FLOAT8_E4M3FN, INT8, HIFLOAT8| ND    | - BS fused: (T, He)<br>- BS unfused: (B, S, He)        |×   |
  | weightDq        | Input     | Downsampling weight matrix $W^{DQ}$ used to calculate the query in the formula.<br>When transposing is not performed, the dimensions are represented as (k, n).| Empty tensors are not supported.     | BFLOAT16, INT8| BFLOAT16, FLOAT8_E4M3FN, INT8, HIFLOAT8| FRACTAL_NZ | (He,Hcq)                      |×   |
  | weightUqQr      | Input     | Upsampling weight matrix $W^{UQ}$ and position encoding weight matrix $W^{QR}$ used to calculate the query in the formula.<br>When transposing is not performed, the dimensions are represented as (k, n).| Empty tensors are not supported. | BFLOAT16, INT8| BFLOAT16, INT8, FLOAT8_E4M3FN, HIFLOAT8| FRACTAL_NZ | (Hcq,N*(D+Dr))                |×   |
  | weightUk        | Input     | Upsampling weight $W^{UK}$ used to calculate the key in the formula.          | Empty tensors are not supported.    | BFLOAT16       | BFLOAT16 | ND         | (N,D,Hckv)                    |×   |
  | weightDkvKr     | Input     | Downsampling weight matrix $W^{DKV}$ and position encoding weight matrix $W^{KR}$ used to calculate the key in the formula.<br>When transposing is not performed, the dimensions are represented as (k, n).| Empty tensors are not supported. | BFLOAT16, INT8| BFLOAT16, FLOAT8_E4M3FN, INT8, HIFLOAT8|  FRACTAL_NZ | (He,Hckv+Dr)                  |×   |
  | rmsnormGammaCq  | Input     | Parameter $\gamma$ in the RmsNorm formula for calculating $c^Q$.       | Empty tensors are not supported.  | BFLOAT16       | BFLOAT16 | ND         | (Hcq)                         |×   |
  | rmsnormGammaCkv | Input     | Parameter $\gamma$ in the RmsNorm formula for calculating $c^{KV}$.     | Empty tensors are not supported.| BFLOAT16       | BFLOAT16 | ND         | (Hckv)                        |×   |
  | ropeSin         | Input     | Sine parameter matrix used to calculate the rotation position encoding.             | Empty tensors with B=0, S=0, and T=0 are supported.| BFLOAT16       | BFLOAT16| ND         | - BS fused: (T, Dr)<br>- BS unfused: (B, S, Dr)        |×   |
  | ropeCos         | Input     | Cosine parameter matrix used to calculate the rotation position encoding.          | Empty tensors with B=0, S=0, and T=0 are supported. | BFLOAT16       | BFLOAT16| ND         | - BS fused: (T, Dr)<br>- BS unfused: (B, S, Dr)        |×   |
  | kvCacheRef      | Input     | The aclTensor used for cache indexing, with in-place update of the computation result (corresponding to $k^C$ in the formula). | Empty tensors with B=0 and Skv=0 are supported. Nkv is associated with N, where N is a hyperparameter; therefore, Nkv cannot be equal to 0. | BFLOAT16, INT8| BFLOAT16, INT8, FLOAT8_E4M3FN, HIFLOAT8| ND   | - CacheMode="PA_BSND"/"PA_NZ"/"PA_BLK_BSND"/"PA_BLK_NZ": (BlockNum,BlockSize,Nkv,Dtile) <br> - CacheMode="BSND": (B,S,Nkv,Dtile) <br> - CacheMode="TND": (T,Nkv,Dtile) |×   |
  | krCacheRef      | Input     | Cache used for key position encoding. The computation result is updated in place (corresponding to $k^R$ in the formula).   | Empty tensors with B=0 and Skv=0 are supported. Nkv is associated with N, where N is a hyperparameter; therefore, Nkv cannot be equal to 0.| BFLOAT16, INT8| BFLOAT16, INT8| ND         | - CacheMode="PA_BSND"/"PA_NZ"/"PA_BLK_BSND"/"PA_BLK_NZ": (BlockNum,BlockSize,Nkv,Dr) <br> - CacheMode="BSND": (B,S,Nkv,Dr) <br> - CacheMode="TND": (T,Nkv,Dr) <br> - When ckvkrRepoMode is set to 1: The dimension must contain 0, and the supported dimension is (0).|×   |
  | cacheIndexOptional | Input     | Index used to store kvCache and krCache.| Empty tensors with B=0, S=0, and T=0 are supported.<br>- CacheMode="PA_BSND"/"PA_NZ": The value range must be within [0, BlockNum x BlockSize).<br>- CacheMode="PA_BLK_BSND"/"PA_BLK_NZ": The value range must be within [0,BlockNum).<br>- CacheMode="TND"/"BSND": nullptr | INT64   | INT64 | ND  | - CacheMode="PA_BSND"/"PA_NZ": <br>1. BS fused: (T)<br>2. BS unfused: (B, S)<br>- CacheMode="PA_BLK_BSND"/"PA_BLK_NZ": <br> 1. BS fused: (Sum(Ceil(S_i/BlockSize))), where S_i is the length of S in each batch.<br> 2. BS unfused: (B, Ceil(S/BlockSize))<br>- CacheMode="TND"/"BSND": nullptr |×   |
  | dequantScaleXOptional      | Input     | Dequantization parameter of tokenX.| - Empty tensor with B=0, S=0, and T=0 (required in the scenario where weightQuantMode is set to 2, 3, 4, or 5)  | FLOAT          | FLOAT8_E8M0, FLOAT| ND         | - weightQuantMode=2/4/5:<br>1. BS fused: (T)<br>2. BS unfused: (B*S, 1)<br>- weightQuantMode=3:<br>  1. BS fused: (T, He/32)<br>  2. BS unfused: (B*S, He/32)                              |×   |
  | dequantScaleWDqOptional    | Input     | Dequantization parameter of weightDq.  | - Non-empty tensor (required in the scenario where weightQuantMode is set to 2, 3, 4, or 5)   | FLOAT          | FLOAT8_E8M0, FLOAT| ND          | - weightQuantMode=2/4/5: (1,Hcq)<br> - weightQuantMode=3: (Hcq, He/32)                                |×   |
  | dequantScaleWUqQrOptional  | Input     | Per-channel parameter used for dequantization after MatmulQcQr matrix multiplication.| - Non-empty tensor (required in the scenario where weightQuantMode is set to 1, 2, 3, 4, or 5) | FLOAT          | FLOAT, FLOAT8_E8M0| ND         | - weightQuantMode=1/2/4/5: (1,N*(D+Dr))<br> - weightQuantMode=3: (N*(D+Dr), Hcq/32)    |×   |
  | dequantScaleWDkvKrOptional | Input     | Dequantization parameter of weightDkvKr.  | - Non-empty tensor (required in the scenario where weightQuantMode is set to 2, 3, 4, or 5)  | FLOAT          | FLOAT8_E8M0, FLOAT| ND         | - weightQuantMode=2/4/5: (1,Hckv+Dr)<br> - weightQuantMode=3: (Hckv+Dr, He/32)|×   |
  | quantScaleCkvOptional      | Input     | Parameter used for quantizing the output data of kvCache.| - Non-empty tensor (required in the scenario where kvCacheQuantMode is set to 1 or 2) | FLOAT          | FLOAT| ND         | - kvCacheQuantMode=1: (1)<br> - kvCacheQuantMode=2: (1,Hckv)|×   |
  | quantScaleCkrOptional      | Input     | Parameter used for quantizing the output data of krCache.| - Non-empty tensors are supported. (This parameter is required in the scenario where kvCacheQuantMode is set to 2.)   | FLOAT    | FLOAT | ND   | (1,Dr)     |×   |
  | smoothScalesCqOptional     | Input     | Parameter used to perform dynamic quantization on the output of RmsNormCq.  | - Non-empty tensors are supported. (This parameter is optional in the scenario where weightQuantMode is set to 1, 2, 4, or 5.)| FLOAT  | FLOAT | ND | (1,Hcq)                       |×   |
  | actualSeqLenOptional     | Input     | Sequence length in each batch, stored in the form of prefix sums.| - Required when BS fusion is enabled and CacheMode is "PA_BLK_BSND" or "PA_BLK_NZ". | INT32    | INT32 | ND   | (B)     |×   |
  | kNopeClipAlphaOptional     | Input     | Scaling factor for clipping the kvCache. | - In the per-tile scenario of partial quantization and int8 full quantization, the value is 1. In other scenarios, this parameter is optional. Empty tensors are not supported.| FLOAT  | - | ND | (1)    |×   |
  | rmsnormEpsilonCq           | Input     | Parameter $\epsilon$ in the RmsNorm formula for calculating $c^Q$.       | If no specific value is required, 1e-05 is recommended. Only the double type is supported.| DOUBLE         | DOUBLE | -          | - |-   |
  | rmsnormEpsilonCkv          | Input     | Parameter $\epsilon$ in the RmsNorm formula for calculating $c^{KV}$.  | If no specific value is required, 1e-05 is recommended. Only the double type is supported.  | DOUBLE         | DOUBLE | -          | -  |-   |
  | cacheModeOptional          | Input     | kvCache mode.| - If not specified, you are advised to pass "PA_BSND".<br> - Only `char*` type is supported.<br> - The value can be "PA_BSND", "PA_NZ", "PA_BLK_BSND", "PA_BLK_NZ", "BSND", or "TND".| CHAR*          | CHAR* | -          | - |-   |
  | queryNormFlag     | Input     | Whether to output queryNormOutOptional and dequantScaleQNormOutOptional. | - false indicates that the output is not generated, and true indicates that the output is generated. The default value is false.| BOOL  | BOOL| -- | --    |-   |
  | weightQuantMode     | Input     | Quantization mode of weightDq, weightUqQr, weightUk, and weightDkvKr. | - 0: non-quantization; 1: weightUqQr quantization; 2: weightDq, weightUqQr, and weightDkvKr int8 quantization; 3: weightDq, weightUqQr, and weightDkvKr mxfp8 quantization; 4: weightDq, weightUqQr, and weightDkvKr fp8 quantization; 5: weightDq, weightUqQr, and weightDkvKr hif8 quantization. The default value is 0.| INT  | INT| -- | --    |-   |
  | kvCacheQuantMode     | Input     | Quantization mode of kvCache. | - `0`: non-quantization; `1`: per-tensor quantization; `2`: per-channel quantization; `3`: per-tile quantization. The default value is `0`.| INT64  | INT64| -- | --    |-   |
  | queryQuantMode     | Input     | Quantization mode of query. | - `0`: non-quantization; `1`: per-token-head quantization. The default value is `0`.| INT64  | INT64| -- | --    |-   |
  | ckvkrRepoMode     | Input     | Storage mode of kvCache and krCache. | - 0: kvCache and krCache are stored separately. 1: kvCache and krCache are stored together. The default value is 0.| INT64  | INT64 | -- | --    |-   |
  | quantScaleRepoMode     | Input     | Storage mode of the quantization scale. | - 0: The quantization scale and data are stored separately. 1: The quantization scale and data are stored together as the output of kvCacheRef. The default value is 0.| INT64  | INT64 | -- | --    |-   |
  | tileSize     | Input     | Size of each tile in per-tile quantization. This parameter is valid only when kvCacheQuantMode is set to 3. | - The default value is `128`.| INT64 | INT64 | -- | --    |-   |
  | qcQrScale     | Input     |   Scale correction coefficient of the query. | If no specific value is required, <code>1.0</code> is recommended.| DOUBLE | DOUBLE | -   | -  |- |
  | kcScale     | Input     |   Scale correction coefficient of the key. | If no specific value is required, <code>1.0</code> is recommended.| DOUBLE | DOUBLE | -    | -  |- |
  | queryOut                   | Output     | Output tensor of the query in the formula (corresponding to $q^N$).    | -  | BFLOAT16, INT8| BFLOAT16, FLOAT8_E4M3FN, INT8, HIFLOAT8| ND         | - BS fused: (T, N, Hckv)<br>- BS unfused: (B, S, N, Hckv)|×   |
  | queryRopeOut               | Output     | Output tensor of the position encoding in the query (corresponding to $q^R$) in the formula. | - | BFLOAT16       | BFLOAT16 | ND         | - BS fused: (T, N, Dr)<br>- BS unfused: (B, S, N, Dr)    |×   |
  | dequantScaleQNopeOutOptional  | Output          | Quantization parameter of the query output in the formula. | - Output when weightQuantMode is set to 2, 3, 4, or 5. nullptr when weightQuantMode is set to 0 or 1.    | FLOAT      | FLOAT| ND   | - BS fused: (T, N, 1)<br>- BS unfused: (B*S, N, 1)  |×   |
  | queryNormOutOptional     | Output     | Output tensor of tokenX after rmsNorm in the formula (corresponding to $c^Q$). | - Output when queryNormFlag is set to true. | BFLOAT16, INT8 | BFLOAT16, INT8, FLOAT8_E4M3FN, HIFLOAT8| ND | - BS fused: (T, Hcq)<br> - BS unfused: (B, S, Hcq) |×   |
  | dequantScaleQNormOutOptional     | Output     | Dequantization parameter of queryNormOutOptional. | - Output when queryNormFlag is set to true and weightQuantMode is set to 1, 2, 3, 4, or 5. nullptr when weightQuantMode is set to 0.| FLOAT  | FLOAT, FLOAT8_E8M0| ND | - weightQuantMode=1/2/4/5:<br> 1. BS fused: (T, 1)<br> 2. BS unfused: (B*S, 1)<br> -weightQuantMode=3:<br>  1. BS fused: (T, Hcq/32)<br>  2. BS unfused: (B\*S, Hcq/32)|×   |
  | workspaceSize              | Output     | Size of the workspace required to be allocated on the device. | Used only for output results. No input configuration is required. - Data type: uint64_t*| -              | -| -          | -                                  |-   |
  | executor                   | Output     | Operator executor, containing the operator computation process.       | - Used only for output results. No input configuration is required. - Data type: aclOpExecutor**.   | -              | -| -          | -                                  |-   |

- **Returns**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API implements input parameter verification. The following errors may be thrown.
    
    | Return Value                | Error Code              | Description                                                                |
    |------------------------|----------------------|----------------------------------------------------------------------|
    | ACLNN_ERR_PARAM_NULLPTR | 161001               | A null pointer exists in a mandatory parameter (for example, an input/output parameter on which the core of the API depends).        |
    | ACLNN_ERR_PARAM_INVALID | 161002               | The shape (dimension/sizes) or dtype (data type) of the input parameter is not supported by the API.|
    | ACLNN_ERR_RUNTIME_ERROR | 361001               | An exception occurs when the API memory calls the NPU Runtime API (for example, the Runtime service is not started or memory allocation fails).|
    | ACLNN_ERR_INNER_TILING_ERROR | 561002          | An exception occurs during tiling. The dtype or shape of the input parameter is incorrect.|

## aclnnMlaPrologV3WeightNz

- **Parameters**

  | Parameter Name       | Type        | Description                                                                |
  |---------------|------------------|----------------------------------------------------------------------|
  | workspace     | void\*           | Address of the workspace to be allocated on the device.                                 |
  | workspaceSize | uint64_t         | Size of the workspace to be allocated on the device, which is obtained by calling <code>aclnnMlaPrologV3WeightNzGetWorkspaceSize</code>.|
  | executor      | aclOpExecutor\*  | Operator executor, containing the operator computation process.                                      |
  | stream        | aclrtStream      | Stream for executing the task.                                  |

- **Returns**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - `aclnnMlaPrologV3WeightNz` defaults to a non-deterministic implementation. You can call `aclrtCtxSetSysParamOpt` to enable deterministic computation.
- Shape field description

  | Field      | Full Spelling/Description                 | Value Rule and Description                                                                |
  |--------------|--------------------------------|------------------------------------------------------------------------------|
  | B            | Batch (batch size of input samples)     | Value range: 0 to 65536.                                                          |
  | S            | Seq-Length (sequence length of input samples)| Value range: not limited                                                             |
  | He           | Head-Size (hidden layer size)       | The values of A2, A3, and A5 are fixed to 1024, 2048, 3072, 4096, 5120, 6144, 7168, 7680, and 8192. |
  | Hcq          | Dimension of the low-rank q matrix                | The value is fixed at 1536.                                                         |
  | N            | Head-Num (number of heads)            | Value range: 1, 2, 4, 8, 16, 32, 64, or 128                                      |
  | Hckv         | Dimension of the low-rank KV matrix               | The value is fixed at 512.                                                            |
  | D            | QK without position encoding dimensions           | The value is fixed at 128.                                                           |
  | Dr           | QK positional encoding dimension               | The value is fixed at 64.                                                             |
  | Nkv          | Number of KV heads                 | The value is fixed at 1.                                                              |
  | BlockNum     | Number of blocks in the PagedAttention scenario.   | 1. When CacheMode is set to "PA_BSND"/"PA_NZ", the value must be greater than or equal to the rounded-up result of `(B*S)/BlockSize`.<br> 2. When CacheMode is set to "PA_BLK_BSND"/"PA_BLK_NZ", the value must be greater than or equal to the rounded-up result of `B` * `(S / BlockSize)` (that is, `B * Ceil(S/BlockSize)`). Note: In the BS fusion scenario, the S length in each batch can be different. Therefore, the value of <code>BlockNum</code> must be greater than or equal to the sum of the rounded-up results of the S length in each batch divided by <code>BlockSize</code>. For example, if the value of actualSeqLenOptional is [47, 151, 261, 422] and blocksize is 128, the length of S in each batch is [47, 104, 110, 161]. In this case, BlockNum = Ceil(47/128) + Ceil(104/128) + Ceil(110/128) + Ceil(161/128) = 5.|
  | BlockSize    | Block size in the PagedAttention scenario | Value range: a multiple of 16 ranging from 16 to 1024<br>                                             |
  | T            | The fused size of BS.               | Range: not limited. Note: When BS fusion is used, <code>tokenX</code>, <code>ropeSin</code>, and <code>ropeCos</code> are all 2D tensors; <code>cacheIndex</code> is a 1D tensor; <code>queryOut</code> and <code>queryRopeOut</code> are 3D tensors.|
  | Dtile        | Size of the D dimension of kvCache          | - In the per-tile quantization scenario, the value is fixed to 656.<br> - In other scenarios, the value is fixed to Hckv (512).                                                      |

- Shape restrictions:
    - When BS fusion is used for `tokenX`, that is , (T, He)
        - The shape of ropeSin and ropeCos is (T, Dr).
        - When CacheMode is set to PA_BSND or PA_NZ, the shape of cacheIndex is (T).
        - When CacheMode is set to PA_BLK_BSND or PA_BLK_NZ, the shape of cacheIndex is (Sum(Ceil(S_i/BlockSize))), where S_i is the length of S in each batch.
        - When CacheMode is set to PA_BLK_BSND or PA_BLK_NZ, actualSeqLenOptional needs to be passed, and the dimension is (B).
        - In the int8/fp8/hif8 full quantization scenario, the shape of dequantScaleXOptional is (T, 1). In the mxfp8 full quantization scenario, the shape of dequantScaleXOptional is (T, He/32).
        - The shape of queryOut is (T, N, Hckv).
        - The shape of queryRopeOut is (T, N, Dr).
        - In the int8/mxfp8/fp8/hif8 full quantization scenario, the shape of dequantScaleQNopeOutOptional is (T, N, 1). In other scenarios, the shape is nullptr.
    - When BS fusion is not used for `tokenX`, that is , (B, S, He)
        - The shape of ropeSin and ropeCos is (B, S, Dr).
        - When CacheMode is set to PA_BSND or PA_NZ, the shape of cacheIndex is (B, S).
        - When CacheMode is set to PA_BLK_BSND or PA_BLK_NZ, the shape of cacheIndex is (B, Ceil(S/BlockSize)).
        - In the int8/fp8/hif8 full quantization scenario, the shape of dequantScaleXOptional is (B\*S, 1). In the mxfp8 full quantization scenario, the shape of dequantScaleXOptional is (B*S, He/32).
        - The shape of queryOut is (B, S, N, Hckv).
        - The shape of queryRopeOut is (B, S, N, Dr).
        - In the int8/mxfp8/fp8/hif8 full quantization scenario, the shape of dequantScaleQNopeOutOptional is (B*S, N, 1). In other scenarios, the shape is nullptr.
    - One or more of the B, S, T, and Skv values can be 0. That is, input parameters related to the shape and B, S, T, and Skv values can be empty tensors. Other input parameters do not support empty tensors.
        - If the values of B, S, and T are 0, `queryOut` and `queryRopeOut` output empty tensors, and `kvCacheRef` and `krCacheRef` are not updated.
        - If the value of Skv is 0, `queryOut`, `queryRopeOut`, and `dequantScaleQNopeOutOptional` are calculated normally, and `kvCacheRef` and `krCacheRef` are not updated. That is, an empty tensor is output.
    - When CacheMode is BSND
        - BS fusion should not be used for `tokenX`, that is, the dimensions are (B, S, He).
        - The dimension of kvCache is (B, S, Nkv, Dr).
    - When CacheMode is TND
        - BS fusion should be used for `tokenX`, that is, the dimension is (T, He).
        - The dimension of kvCache is (T, Nkv, Dr).
    - When ckvkrRepoMode is 1
        - The dimension of krCache must contain 0, and the supported dimension is (0).

- Special constraints
  - When actualSeqLenOptional is passed, the last number of actualSeqLenOptional must be the same as T.
  - In per-tile quantization mode, both ckvkrRepoMode and quantScaleRepoMode must be 1. In other quantization modes and non-quantization scenarios, both ckvkrRepoMode and quantScaleRepoMode must be 0.
  - In per-tile quantization mode, `CacheMode` supports only `PA_BSND`, `BSND`, and `TND`.
  - When the value of ckvkrRepoMode is 1, krCache must be an empty tensor (that is, the product of the shape is 0).
  - In kvcache per-tensor quantization mode, both kvCacheQuantMode and queryQuantMode must be 1.
- The aclnnMlaPrologV3WeightNz API supports the following scenarios: A2 and A3 do not support full quantization scenarios of fp8/hif8/mxfp8. A5 supports all quantization scenarios.
  <table style="table-layout: auto;" border="1">
    <tr>
      <th colspan="2">Scenario </th>
      <th>Description</th>
    </tr>
    <tr>
      <td colspan="2">Non-quantization</td>
      <td>
          Input parameters: All input parameters are non-quantized data.<br>
          Output parameters: All output parameters are non-quantized data.
      </td>
    </tr>
    <tr>
      <td rowspan="3">Partial quantization</td>
      <td>kv_cache non-quantized</td>
      <td>
          Input: The weightUqQr input parameter is used to pass the per-token quantized data, and other input parameters are used to pass the non-quantized data. The dequantScaleWUqQr field must be passed, and the smoothScalesCq field is optional.<br>
          Output parameters: All output parameters are non-quantized data.
      </td>
    </tr>
    <tr>
      <td>kv_cache per-channel non-quantized</td>
      <td>
          Input: The weightUqQr input parameter is used to pass the per-token quantized data, the kvCacheRef and krCacheRef input parameters are used to pass the per-channel quantized data, and other input parameters are used to pass the non-quantized data. The dequantScaleWUqQr, quantScaleCkv, and quant_scale_ckr fields must be passed, and the smoothScalesCq field is optional.<br>
          Output: The kvCacheRef and krCacheRef output parameters return the per-channel quantized data, and other output parameters return the non-quantized data.
      </td>
    </tr>
    <tr>
      <td>kv_cache per-tile quantized</td>
      <td>
          Input: The weightUqQr input parameter is used to pass the per-token quantized data, and other input parameters are used to pass the non-quantized data. The dequantScaleWUqQr field must be passed, and the smoothScalesCq field is optional.<br>
          Output: The kvCacheRef output parameter returns the per-tile quantized data, and other output parameters return the non-quantized data.
      </td>
    </tr>
    <tr>
      <td rowspan="3">int8/fp8/hif8 full quantization</td>
      <td>kv_cache non-quantized</td>
      <td>
          Inputs: <code>tokenX</code> must be provided as <code>pertoken</code> quantized data, <code>weightDq</code>, <code>weightUqQr</code>, and <code>weightDkvKr</code> must be provided as <code>perchannel</code> quantized data, and all other inputs must be non-quantized data. The dequantScaleX, dequantScaleWDq, dequantScaleWUqQr, and dequantScaleWDkvKr fields must be passed, and the smoothScalesCq field is optional.<br>
          Output parameters: All output parameters are non-quantized data.
      </td>
    </tr>
    <tr>
      <td>kv_cache per-tensor quantized</td>
      <td>
          Inputs: <code>tokenX</code> must be provided as <code>pertoken</code> quantized data, <code>weightDq</code>, <code>weightUqQr</code>, and <code>weightDkvKr</code> must be provided as <code>perchannel</code> quantized data, <code>kvCacheRef</code> must be provided as <code>pertensor</code> quantized data, and all other inputs must be non-quantized data. The dequantScaleX, dequantScaleWDq, dequantScaleWUqQr, dequantScaleWDkvKr, and quantScaleCkv fields must be passed. The smoothScalesCq field is optional.<br>
          Outputs: <code>queryOut</code> is <code>pertoken_head</code> quantized data, <code>kvCacheRef</code> is <code>pertensor</code> quantized data, and all other outputs are non-quantized data.
      </td>
    </tr>
    <tr>
      <td>kv_cache per-tile quantized</td>
      <td>
          Inputs: <code>tokenX</code> must be provided as <code>pertoken</code> quantized data, <code>weightDq</code>, <code>weightUqQr</code>, and <code>weightDkvKr</code> must be provided as <code>perchannel</code> quantized data, and all other inputs must be non-quantized data. The dequantScaleX, dequantScaleWDq, dequantScaleWUqQr, and dequantScaleWDkvKr fields must be passed. The smoothScalesCq field is optional.<br>
          Outputs: <code>kvCacheRef</code> is <code>pertile</code> quantized data, and all other outputs are non-quantized data.
      </td>
    </tr>
    <tr>
      <td rowspan="3">mxfp8 full quantization</td>
      <td>kv_cache non-quantized</td>
      <td>
          Inputs: <code>tokenX</code> must be provided as <code>pertoken</code> quantized data, <code>weightDq</code>, <code>weightUqQr</code>, and <code>weightDkvKr</code> must be provided as <code>perchannel</code> quantized data, and all other inputs must be non-quantized data. The <code>dequantScaleX</code>, <code>dequantScaleWDq</code>, <code>dequantScaleWUqQr</code>, and <code>dequantScaleWDkvKr</code> fields are required.<br>
          Output parameters: All output parameters are non-quantized data.
      </td>
    </tr>
    <tr>
      <td>kv_cache per-tensor quantized</td>
      <td>
          Inputs: <code>tokenX</code> must be provided as <code>pertoken</code> quantized data, <code>weightDq</code>, <code>weightUqQr</code>, and <code>weightDkvKr</code> must be provided as <code>perchannel</code> quantized data, <code>kvCacheRef</code> must be provided as <code>pertensor</code> quantized data, and all other inputs must be non-quantized data. The <code>dequantScaleX</code>, <code>dequantScaleWDq</code>, <code>dequantScaleWUqQr</code>, <code>dequantScaleWDkvKr</code>, and <code>quantScaleCkv</code> fields are required<br>.<br>
          Outputs: <code>queryOut</code> is <code>pertoken_head</code> quantized data, <code>kvCacheRef</code> is <code>pertensor</code> quantized data, and all other outputs are non-quantized data.
      </td>
    </tr>
    <tr>
      <td>kv_cache per-tile quantized</td>
      <td>
          Inputs: <code>tokenX</code> must be provided as <code>pertoken</code> quantized data, <code>weightDq</code>, <code>weightUqQr</code>, and <code>weightDkvKr</code> must be provided as <code>perchannel</code> quantized data, and all other inputs must be non-quantized data. The <code>dequantScaleX</code>, <code>dequantScaleWDq</code>, <code>dequantScaleWUqQr</code>, and <code>dequantScaleWDkvKr</code> fields are required.<br>
          Outputs: <code>kvCacheRef</code> is <code>pertile</code> quantized data, and all other outputs are non-quantized data.
      </td>
    </tr>
  </table>

- In different quantization scenarios, the dtype combinations of parameters must meet the following conditions:

  <div style="overflow-x: auto; width: 100%;">
  <table style="table-layout: auto;" border="1">
    <tr>
      <th rowspan="3">Name</th>
      <th rowspan="2" colspan="1">Non-Quantization</th>
      <th colspan="3">Partial Quantization</th>
      <th colspan="3">INT8 full quantization</th>
      <th colspan="3">mxFP8 full quantization</th>
      <th colspan="3">fp8 full quantization</th>
      <th colspan="3">HIF8 full quantization</th>
    </tr>
    <tr>
      <th colspan="1">kvCache Non-Quantized</th>
      <th colspan="1">kvCache Per-Channel Quantized</th>
      <th colspan="1">kvCache Per-Tile Quantized</th>
      <th colspan="1">kvCache Non-Quantized</th>
      <th colspan="1">kvCache Per-Tensor Quantized</th>
      <th colspan="1">kvCache Per-Tile Quantized</th>
      <th colspan="1">kvCache Non-Quantized</th>
      <th colspan="1">kvCache Per-Tensor Quantized</th>
      <th colspan="1">kvCache Per-Tile Quantized</th>
      <th colspan="1">kvCache Non-Quantized</th>
      <th colspan="1">kvCache Per-Tensor Quantized</th>
      <th colspan="1">kvCache Per-Tile Quantized</th>
      <th colspan="1">kvCache Non-Quantized</th>
      <th colspan="1">kvCache Per-Tensor Quantized</th>
      <th colspan="1">kvCache Per-Tile Quantized</th>
    </tr>
    <tr>
      <th>dtype</th>
      <th>dtype</th>
      <th>dtype</th>
      <th>dtype</th>
      <th>dtype</th>
      <th>dtype</th>
      <th>dtype</th>
      <th>dtype</th>
      <th>dtype</th>
      <th>dtype</th>
      <th>dtype</th>
      <th>dtype</th>
      <th>dtype</th>
      <th>dtype</th>
      <th>dtype</th>
      <th>dtype</th>
    </tr>
    <tr>
      <td>tokenX</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>INT8</td>
      <td>INT8</td>
      <td>INT8</td>
      <td>FLOAT8_E4M3FN</td>
      <td>FLOAT8_E4M3FN</td>
      <td>FLOAT8_E4M3FN</td>
      <td>FLOAT8_E4M3FN</td>
      <td>FLOAT8_E4M3FN</td>
      <td>FLOAT8_E4M3FN</td>
      <td>HIFLOAT8</td>
      <td>HIFLOAT8</td>
      <td>HIFLOAT8</td>
    </tr>
    <tr>
      <td>weightDq</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>INT8</td>
      <td>INT8</td>
      <td>INT8</td>
      <td>FLOAT8_E4M3FN</td>
      <td>FLOAT8_E4M3FN</td>
      <td>FLOAT8_E4M3FN</td>
      <td>FLOAT8_E4M3FN</td>
      <td>FLOAT8_E4M3FN</td>
      <td>FLOAT8_E4M3FN</td>
      <td>HIFLOAT8</td>
      <td>HIFLOAT8</td>
      <td>HIFLOAT8</td>
    </tr>
    <tr>
      <td>weightUqQr</td>
      <td>BFLOAT16</td>
      <td>INT8</td>
      <td>INT8</td>
      <td>INT8</td>
      <td>INT8</td>
      <td>INT8</td>
      <td>INT8</td>
      <td>FLOAT8_E4M3FN</td>
      <td>FLOAT8_E4M3FN</td>
      <td>FLOAT8_E4M3FN</td>
      <td>FLOAT8_E4M3FN</td>
      <td>FLOAT8_E4M3FN</td>
      <td>FLOAT8_E4M3FN</td>
      <td>HIFLOAT8</td>
      <td>HIFLOAT8</td>
      <td>HIFLOAT8</td>
    </tr>
    <tr>
      <td>weightUk</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
    </tr>
    <tr>
      <td>weightDkvKr</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>INT8</td>
      <td>INT8</td>
      <td>INT8</td>
      <td>FLOAT8_E4M3FN</td>
      <td>FLOAT8_E4M3FN</td>
      <td>FLOAT8_E4M3FN</td>
      <td>FLOAT8_E4M3FN</td>
      <td>FLOAT8_E4M3FN</td>
      <td>FLOAT8_E4M3FN</td>
      <td>HIFLOAT8</td>
      <td>HIFLOAT8</td>
      <td>HIFLOAT8</td>
    </tr>
    <tr>
      <td> rmsnormGammaCq </td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
    </tr>
    <tr>
      <td> rmsnormGammaCkv </td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
    </tr>
    <tr>
      <td> ropeSin </td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
    </tr>
    <tr>
      <td> ropeCos </td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
    </tr>
    <tr>
      <td> kvCacheRef </td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>INT8</td>
      <td>INT8</td>
      <td>BFLOAT16</td>
      <td>INT8</td>
      <td>INT8</td>
      <td>BFLOAT16</td>
      <td>FLOAT8_E4M3FN</td>
      <td>FLOAT8_E4M3FN</td>
      <td>BFLOAT16</td>
      <td>FLOAT8_E4M3FN</td>
      <td>FLOAT8_E4M3FN</td>
      <td>BFLOAT16</td>
      <td>HIFLOAT8</td>
      <td>HIFLOAT8</td>
    </tr>
    <tr>
      <td> krCacheRef </td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>INT8</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
    </tr>
    <tr>
      <td> cacheIndexOptional </td>
      <td>INT64</td>
      <td>INT64</td>
      <td>INT64</td>
      <td>INT64</td>
      <td>INT64</td>
      <td>INT64</td>
      <td>INT64</td>
      <td>INT64</td>
      <td>INT64</td>
      <td>INT64</td>
      <td>INT64</td>
      <td>INT64</td>
      <td>INT64</td>
      <td>INT64</td>
      <td>INT64</td>
      <td>INT64</td>
    </tr>
    <tr>
      <td> dequantScaleXOptional </td>
      <td>NULLPTR</td>
      <td>NULLPTR</td>
      <td>NULLPTR</td>
      <td>NULLPTR</td>
      <td>FLOAT</td>
      <td>FLOAT</td>
      <td>FLOAT</td>
      <td>FLOAT8_E8M0</td>
      <td>FLOAT8_E8M0</td>
      <td>FLOAT8_E8M0</td>
      <td>FLOAT</td>
      <td>FLOAT</td>
      <td>FLOAT</td>
      <td>FLOAT</td>
      <td>FLOAT</td>
      <td>FLOAT</td>
    </tr>
    <tr>
      <td> dequantScaleWDqOptional </td>
      <td>NULLPTR</td>
      <td>NULLPTR</td>
      <td>NULLPTR</td>
      <td>NULLPTR</td>
      <td>FLOAT</td>
      <td>FLOAT</td>
      <td>FLOAT</td>
      <td>FLOAT8_E8M0</td>
      <td>FLOAT8_E8M0</td>
      <td>FLOAT8_E8M0</td>
      <td>FLOAT</td>
      <td>FLOAT</td>
      <td>FLOAT</td>
      <td>FLOAT</td>
      <td>FLOAT</td>
      <td>FLOAT</td>
    </tr>
    <tr>
      <td> dequantScaleWUqQrOptional </td>
      <td>NULLPTR</td>
      <td>FLOAT</td>
      <td>FLOAT</td>
      <td>FLOAT</td>
      <td>FLOAT</td>
      <td>FLOAT</td>
      <td>FLOAT</td>
      <td>FLOAT8_E8M0</td>
      <td>FLOAT8_E8M0</td>
      <td>FLOAT8_E8M0</td>
      <td>FLOAT</td>
      <td>FLOAT</td>
      <td>FLOAT</td>
      <td>FLOAT</td>
      <td>FLOAT</td>
      <td>FLOAT</td>
    </tr>
    <tr>
      <td> dequantScaleWDkvKrOptional </td>
      <td> NULLPTR </td>
      <td> NULLPTR </td>
      <td> NULLPTR </td>
      <td> NULLPTR </td>
      <td>FLOAT</td>
      <td>FLOAT</td>
      <td>FLOAT</td>
      <td>FLOAT8_E8M0</td>
      <td>FLOAT8_E8M0</td>
      <td>FLOAT8_E8M0</td>
      <td>FLOAT</td>
      <td>FLOAT</td>
      <td>FLOAT</td>
      <td>FLOAT</td>
      <td>FLOAT</td>
      <td>FLOAT</td>
    </tr>
    <tr>
      <td> quantScaleCkvOptional </td>
      <td>NULLPTR</td>
      <td>NULLPTR</td>
      <td>FLOAT</td>
      <td>NULLPTR</td>
      <td>NULLPTR</td>
      <td>FLOAT</td>
      <td>NULLPTR</td>
      <td>NULLPTR</td>
      <td>FLOAT</td>
      <td>NULLPTR</td>
      <td>NULLPTR</td>
      <td>FLOAT</td>
      <td>NULLPTR</td>
      <td>NULLPTR</td>
      <td>FLOAT</td>
      <td>NULLPTR</td>
    </tr>
    <tr>
      <td> quantScaleCkrOptional </td>
      <td>NULLPTR</td>
      <td>NULLPTR</td>
      <td>FLOAT</td>
      <td>NULLPTR</td>
      <td>NULLPTR</td>
      <td>NULLPTR</td>
      <td>NULLPTR</td>
      <td>NULLPTR</td>
      <td>NULLPTR</td>
      <td>NULLPTR</td>
      <td>NULLPTR</td>
      <td>NULLPTR</td>
      <td>NULLPTR</td>
      <td>NULLPTR</td>
      <td>NULLPTR</td>
      <td>NULLPTR</td>
    </tr>
    <tr>
      <td> smoothScalesCqOptional </td>
      <td>NULLPTR</td>
      <td>FLOAT</td>
      <td>FLOAT</td>
      <td>FLOAT</td>
      <td>FLOAT</td>
      <td>FLOAT</td>
      <td>FLOAT</td>
      <td>NULLPTR</td>
      <td>NULLPTR</td>
      <td>NULLPTR</td>
      <td>FLOAT</td>
      <td>FLOAT</td>
      <td>FLOAT</td>
      <td>FLOAT</td>
      <td>FLOAT</td>
      <td>FLOAT</td>
    </tr>
    <tr>
      <td> kNopeClipAlphaOptional </td>
      <td>NULLPTR</td>
      <td>NULLPTR</td>
      <td>NULLPTR</td>
      <td>FLOAT</td>
      <td>NULLPTR</td>
      <td>NULLPTR</td>
      <td>FLOAT</td>
      <td>NULLPTR</td>
      <td>NULLPTR</td>
      <td>NULLPTR</td>
      <td>NULLPTR</td>
      <td>NULLPTR</td>
      <td>FLOAT</td>
      <td>NULLPTR</td>
      <td>NULLPTR</td>
      <td>FLOAT</td>
    </tr>
    <tr>
      <td> queryOut </td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>INT8</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>FLOAT8_E4M3FN</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>FLOAT8_E4M3FN</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>HIFLOAT8</td>
      <td>BFLOAT16</td>
    </tr>
    <tr>
      <td> queryRopeOut </td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
    </tr>
    <tr>
      <td> dequantScaleQNopeOutOptional </td>
      <td>NULLPTR</td>
      <td>NULLPTR</td>
      <td>NULLPTR</td>
      <td>NULLPTR</td>
      <td>NULLPTR</td>
      <td>FLOAT</td>
      <td>NULLPTR</td>
      <td>NULLPTR</td>
      <td>FLOAT</td>
      <td>NULLPTR</td>
      <td>NULLPTR</td>
      <td>FLOAT</td>
      <td>NULLPTR</td>
      <td>NULLPTR</td>
      <td>FLOAT</td>
      <td>NULLPTR</td>
    </tr>
    <tr>
      <td> queryNormOutOptional </td>
      <td>BFLOAT16</td>
      <td>INT8</td>
      <td>INT8</td>
      <td>INT8</td>
      <td>INT8</td>
      <td>INT8</td>
      <td>INT8</td>
      <td>FLOAT8_E4M3FN</td>
      <td>FLOAT8_E4M3FN</td>
      <td>FLOAT8_E4M3FN</td>
      <td>FLOAT8_E4M3FN</td>
      <td>FLOAT8_E4M3FN</td>
      <td>FLOAT8_E4M3FN</td>
      <td>HIFLOAT8</td>
      <td>HIFLOAT8</td>
      <td>HIFLOAT8</td>
    </tr>
    <tr>
      <td> dequantScaleQNormOutOptional </td>
      <td>NULLPTR</td>
      <td>FLOAT</td>
      <td>FLOAT</td>
      <td>FLOAT</td>
      <td>FLOAT</td>
      <td>FLOAT</td>
      <td>FLOAT</td>
      <td>FLOAT8_E8M0</td>
      <td>FLOAT8_E8M0</td>
      <td>FLOAT8_E8M0</td>
      <td>FLOAT</td>
      <td>FLOAT</td>
      <td>FLOAT</td>
      <td>FLOAT</td>
      <td>FLOAT</td>
      <td>FLOAT</td>
    </tr>
  </table>
  </div>

## Example

The following sample code for A2 and A3 is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

  ```Cpp
  #include <iostream>
  #include <cstring>
  #include <vector>
  #include <cstdint>
  #include "acl/acl.h"
  #include "aclnnop/aclnn_mla_prolog_v3_weight_nz.h"
  #include<unistd.h>

  #define CHECK_RET(cond, return_expr) \
    do {                               \
      if (!(cond)) {                   \
        return_expr;                   \
      }                                \
    } while (0)

#define LOG_PRINT(message, ...)      \
  do {                               \
    printf(message, ##__VA_ARGS__);  \
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
      ret = aclrtMallocHost(hostAddr, size);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMallocHost failed. ERROR: %d\n", ret); return ret);
      memset(*hostAddr, 0, size);
      // Call aclCreateTensor to create an aclTensor.
      *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, nullptr, 0, aclFormat::ACL_FORMAT_ND,
                                shape.data(), shape.size(), *deviceAddr);
      // Call aclrtMemcpy to copy host data to the device memory.
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
      ret = aclrtMallocHost(hostAddr, size);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMallocHost failed. ERROR: %d\n", ret); return ret);
      memset(*hostAddr, 0, size);
      // Call aclCreateTensor to create an aclTensor.
      *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, nullptr, 0, aclFormat::ACL_FORMAT_FRACTAL_NZ,
                                shape.data(), shape.size(), *deviceAddr);
      // Call aclrtMemcpy to copy host data to the device memory.
      ret = aclrtMemcpy(*deviceAddr, size, *hostAddr, GetShapeSize(shape) * aclDataTypeSize(dataType), ACL_MEMCPY_HOST_TO_DEVICE);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
      return 0;
  }

  int TransToNZShape(std::vector<int64_t> &shapeND, size_t typeSize) {
      if (typeSize == static_cast<size_t>(0)) {
        return 0;
      }
      int64_t h = shapeND[0];
      int64_t w = shapeND[1];
      int64_t h0 = static_cast<int64_t>(16);
      int64_t w0 = static_cast<int64_t>(32) / static_cast<int64_t>(typeSize);
      int64_t h1 = h / h0;
      int64_t w1 = w / w0;
      shapeND[0] = w1;
      shapeND[1] = h1;
      shapeND.emplace_back(h0);
      shapeND.emplace_back(w0);
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
      // 2. Construct the input and output based on the API.
      std::vector<int64_t> tokenXShape = {8, 1, 7168};            // B,S,He
      std::vector<int64_t> weightDqShape = {7168, 1536};          // He,Hcq
      std::vector<int64_t> weightUqQrShape = {1536, 6144};        // Hcq,N*(D+Dr)
      std::vector<int64_t> weightUkShape = {32, 128, 512};        // N,D,Hckv
      std::vector<int64_t> weightDkvKrShape = {7168, 576};        // He,Hckv+Dr
      std::vector<int64_t> rmsnormGammaCqShape = {1536};          // Hcq
      std::vector<int64_t> rmsnormGammaCkvShape = {512};          // Hckv
      std::vector<int64_t> ropeSinShape = {8, 1, 64};             // B,S,Dr
      std::vector<int64_t> ropeCosShape = {8, 1, 64};             // B,S,Dr
      std::vector<int64_t> cacheIndexShape = {8, 1};              // B,S
      std::vector<int64_t> kvCacheShape = {16, 128, 1, 512};      // BlockNum,BlockSize,Nkv,Hckv
      std::vector<int64_t> krCacheShape = {16, 128, 1, 64};       // BlockNum,BlockSize,Nkv,Dr
      std::vector<int64_t> dequantScaleXShape = {8, 1};           // B*S, 1
      std::vector<int64_t> dequantScaleWDqShape = {1, 1536};      // 1, Hcq
      std::vector<int64_t> dequantScaleWUqQrShape = {1, 6144};    // 1, N*(D+Dr)
      std::vector<int64_t> dequantScaleWDkvKrShape = {1, 576};    // 1, Hckv+Dr
      std::vector<int64_t> quantScaleCkvShape = {1};              // 1
      std::vector<int64_t> smoothScalesCqShape = {1, 1536};       // 1, Hcq
      std::vector<int64_t> queryShape = {8, 1, 32, 512};          // B,S,N,Hckv
      std::vector<int64_t> queryRopeShape = {8, 1, 32, 64};       // B,S,N,Dr
      std::vector<int64_t> dequantScaleQNopeShape = {8, 32, 1};   // B*S, N, 1
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
      void* dequantScaleXDeviceAddr = nullptr;
      void* dequantScaleWDqDeviceAddr = nullptr;
      void* dequantScaleWUqQrDeviceAddr = nullptr;
      void* dequantScaleWDkvKrDeviceAddr = nullptr;
      void* quantScaleCkvDeviceAddr = nullptr;
      void* smoothScalesCqDeviceAddr = nullptr;
      void* queryDeviceAddr = nullptr;
      void* queryRopeDeviceAddr = nullptr;
      void* dequantScaleQNopeDeviceAddr = nullptr;

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
      void* dequantScaleXHostAddr = nullptr;
      void* dequantScaleWDqHostAddr = nullptr;
      void* dequantScaleWUqQrHostAddr = nullptr;
      void* dequantScaleWDkvKrHostAddr = nullptr;
      void* quantScaleCkvHostAddr = nullptr;
      void* smoothScalesCqHostAddr = nullptr;
      void* queryHostAddr = nullptr;
      void* queryRopeHostAddr = nullptr;
      void* dequantScaleQNopeHostAddr = nullptr;

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
      aclTensor* dequantScaleX = nullptr;
      aclTensor* dequantScaleWDq = nullptr;
      aclTensor* dequantScaleWUqQr = nullptr;
      aclTensor* dequantScaleWDkvKr = nullptr;
      aclTensor* quantScaleCkv = nullptr;
      aclTensor* smoothScalesCq = nullptr;
      int64_t weightQuantMode = 2;
      int64_t kvQuantMode = 1;
      int64_t queryQuantMode = 1;
      int64_t ckvkrRepoMode = 0;
      int64_t quantScaleRepoMode = 0;
      int64_t tileSize = 128;
      double kNopeClipAlpha = 1.0f;
      double qcQrScale = 1.0f;
      double kcScale = 1.0f;
      aclTensor* query = nullptr;
      aclTensor* queryRope = nullptr;
      aclTensor* dequantScaleQNope = nullptr;

      // Convert the shapes of the three variables in NZ format.
      constexpr size_t EXAMPLE_INT8_SIZE = sizeof(int8_t);
      constexpr size_t EXAMPLE_BFLOAT16_SIZE = sizeof(int16_t);
      ret = TransToNZShape(weightDqShape, EXAMPLE_INT8_SIZE);
      CHECK_RET(ret == 0, LOG_PRINT("trans NZ shape failed.\n"); return ret);
      ret = TransToNZShape(weightUqQrShape, EXAMPLE_INT8_SIZE);
      CHECK_RET(ret == 0, LOG_PRINT("trans NZ shape failed.\n"); return ret);
      ret = TransToNZShape(weightDkvKrShape, EXAMPLE_INT8_SIZE);
      CHECK_RET(ret == 0, LOG_PRINT("trans NZ shape failed.\n"); return ret);

      // Create a tokenX aclTensor.
      ret = CreateAclTensorND(tokenXShape, &tokenXDeviceAddr, &tokenXHostAddr, aclDataType::ACL_INT8, &tokenX);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create a weightDq aclTensor.
      ret = CreateAclTensorNZ(weightDqShape, &weightDqDeviceAddr, &weightDqHostAddr, aclDataType::ACL_INT8, &weightDq);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create a weightUqQr aclTensor.
      ret = CreateAclTensorNZ(weightUqQrShape, &weightUqQrDeviceAddr, &weightUqQrHostAddr, aclDataType::ACL_INT8, &weightUqQr);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create a weightUk aclTensor.
      ret = CreateAclTensorND(weightUkShape, &weightUkDeviceAddr, &weightUkHostAddr, aclDataType::ACL_BF16, &weightUk);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create a weightDkvKr aclTensor.
      ret = CreateAclTensorNZ(weightDkvKrShape, &weightDkvKrDeviceAddr, &weightDkvKrHostAddr, aclDataType::ACL_INT8, &weightDkvKr);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create an rmsnormGammaCq aclTensor.
      ret = CreateAclTensorND(rmsnormGammaCqShape, &rmsnormGammaCqDeviceAddr, &rmsnormGammaCqHostAddr, aclDataType::ACL_BF16, &rmsnormGammaCq);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create an rmsnormGammaCkv aclTensor.
      ret = CreateAclTensorND(rmsnormGammaCkvShape, &rmsnormGammaCkvDeviceAddr, &rmsnormGammaCkvHostAddr, aclDataType::ACL_BF16, &rmsnormGammaCkv);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create a ropeSin aclTensor.
      ret = CreateAclTensorND(ropeSinShape, &ropeSinDeviceAddr, &ropeSinHostAddr, aclDataType::ACL_BF16, &ropeSin);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create a ropeCos aclTensor.
      ret = CreateAclTensorND(ropeCosShape, &ropeCosDeviceAddr, &ropeCosHostAddr, aclDataType::ACL_BF16, &ropeCos);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create a cacheIndex aclTensor.
      ret = CreateAclTensorND(cacheIndexShape, &cacheIndexDeviceAddr, &cacheIndexHostAddr, aclDataType::ACL_INT64, &cacheIndex);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create a kvCache aclTensor.
      ret = CreateAclTensorND(kvCacheShape, &kvCacheDeviceAddr, &kvCacheHostAddr, aclDataType::ACL_INT8, &kvCache);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create a krCache aclTensor.
      ret = CreateAclTensorND(krCacheShape, &krCacheDeviceAddr, &krCacheHostAddr, aclDataType::ACL_BF16, &krCache);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create a dequantScaleX aclTensor.
      ret = CreateAclTensorND(dequantScaleXShape, &dequantScaleXDeviceAddr, &dequantScaleXHostAddr, aclDataType::ACL_FLOAT, &dequantScaleX);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create a dequantScaleWDq aclTensor.
      ret = CreateAclTensorND(dequantScaleWDqShape, &dequantScaleWDqDeviceAddr, &dequantScaleWDqHostAddr, aclDataType::ACL_FLOAT, &dequantScaleWDq);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create a dequantScaleWUqQr aclTensor.
      ret = CreateAclTensorND(dequantScaleWUqQrShape, &dequantScaleWUqQrDeviceAddr, &dequantScaleWUqQrHostAddr, aclDataType::ACL_FLOAT, &dequantScaleWUqQr);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create a dequantScaleWDkvKr aclTensor.
      ret = CreateAclTensorND(dequantScaleWDkvKrShape, &dequantScaleWDkvKrDeviceAddr, &dequantScaleWDkvKrHostAddr, aclDataType::ACL_FLOAT, &dequantScaleWDkvKr);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create a quantScaleCkv aclTensor.
      ret = CreateAclTensorND(quantScaleCkvShape, &quantScaleCkvDeviceAddr, &quantScaleCkvHostAddr, aclDataType::ACL_FLOAT, &quantScaleCkv);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create a smoothScalesCq aclTensor.
      ret = CreateAclTensorND(smoothScalesCqShape, &smoothScalesCqDeviceAddr, &smoothScalesCqHostAddr, aclDataType::ACL_FLOAT, &smoothScalesCq);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create a query aclTensor.
      ret = CreateAclTensorND(queryShape, &queryDeviceAddr, &queryHostAddr, aclDataType::ACL_INT8, &query);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create a queryRope aclTensor.
      ret = CreateAclTensorND(queryRopeShape, &queryRopeDeviceAddr, &queryRopeHostAddr, aclDataType::ACL_BF16, &queryRope);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create a dequantScaleQNope aclTensor.
      ret = CreateAclTensorND(dequantScaleQNopeShape, &dequantScaleQNopeDeviceAddr, &dequantScaleQNopeHostAddr, aclDataType::ACL_FLOAT, &dequantScaleQNope);
      CHECK_RET(ret == ACL_SUCCESS, return ret);

      // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
      uint64_t workspaceSize = 0;
      aclOpExecutor* executor = nullptr;
      // Call the first-phase API of aclnnMlaPrologV3WeightNz.
      ret = aclnnMlaPrologV3WeightNzGetWorkspaceSize(tokenX, weightDq, weightUqQr, weightUk, weightDkvKr, rmsnormGammaCq, rmsnormGammaCkv, ropeSin, ropeCos, kvCache, krCache, cacheIndex,
        dequantScaleX, dequantScaleWDq, dequantScaleWUqQr, dequantScaleWDkvKr, quantScaleCkv, nullptr, smoothScalesCq, nullptr, nullptr,rmsnormEpsilonCq, rmsnormEpsilonCkv, cacheMode,
        weightQuantMode, kvQuantMode, queryQuantMode, ckvkrRepoMode, quantScaleRepoMode, tileSize, qcQrScale, kcScale,
        query, queryRope, dequantScaleQNope, nullptr, nullptr, &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMlaPrologV3WeightNzGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
      // Allocate device memory based on the computed workspaceSize.
      void* workspaceAddr = nullptr;
      if (workspaceSize > static_cast<uint64_t>(0)) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
      }
      // Call the second-phase API of aclnnMlaPrologV3WeightNz.
      ret = aclnnMlaPrologV3WeightNz(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMlaPrologV3WeightNz failed. ERROR: %d\n", ret); return ret);

      // 4. (Fixed writing) Wait until the task execution is complete.
      ret = aclrtSynchronizeStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

      // 5. Obtain the output value and copy the result from the device to the host. Modify the code based on the API definition.
      auto size = GetShapeSize(queryShape);
      std::vector<float> resultData(size, 0);
      ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), queryDeviceAddr, size * sizeof(float),
                        ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
      for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
      }
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
      aclDestroyTensor(dequantScaleX);
      aclDestroyTensor(dequantScaleWDq);
      aclDestroyTensor(dequantScaleWUqQr);
      aclDestroyTensor(dequantScaleWDkvKr);
      aclDestroyTensor(quantScaleCkv);
      aclDestroyTensor(smoothScalesCq);
      aclDestroyTensor(query);
      aclDestroyTensor(queryRope);
      aclDestroyTensor(dequantScaleQNope);

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
      aclrtFree(dequantScaleXDeviceAddr);
      aclrtFree(dequantScaleWDqDeviceAddr);
      aclrtFree(dequantScaleWUqQrDeviceAddr);
      aclrtFree(dequantScaleWDkvKrDeviceAddr);
      aclrtFree(quantScaleCkvDeviceAddr);
      aclrtFree(smoothScalesCqDeviceAddr);
      aclrtFree(queryDeviceAddr);
      aclrtFree(queryRopeDeviceAddr);
      aclrtFree(dequantScaleQNopeDeviceAddr);

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
      aclrtFree(dequantScaleXHostAddr);
      aclrtFree(dequantScaleWDqHostAddr);
      aclrtFree(dequantScaleWUqQrHostAddr);
      aclrtFree(dequantScaleWDkvKrHostAddr);
      aclrtFree(quantScaleCkvHostAddr);
      aclrtFree(smoothScalesCqHostAddr);
      aclrtFree(queryHostAddr);
      aclrtFree(queryRopeHostAddr);
      aclrtFree(dequantScaleQNopeHostAddr);

      if (workspaceSize > static_cast<uint64_t>(0)) {
          aclrtFree(workspaceAddr);
      }
      aclrtDestroyStream(stream);
      aclrtResetDevice(deviceId);
      aclFinalize();

      return 0;
  }
  ```

The sample code for A5 is as follows:

  ```Cpp
#include <iostream>
#include <cstring>
#include <vector>
#include <cstdint>
#include "acl/acl.h"
#include "aclnnop/aclnn_mla_prolog_v3_weight_nz.h"
#include <unistd.h>

#define CHECK_RET(cond, return_expr)    \
    do {                                \
        if (!(cond)) {                  \
            return_expr;                \
        }                               \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
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
    ret = aclrtMallocHost(hostAddr, size);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMallocHost failed. ERROR: %d\n", ret); return ret);
    memset(*hostAddr, 0, size);
    // Call aclCreateTensor to create an aclTensor.
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, nullptr, 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    // Call aclrtMemcpy to copy host data to the device memory.
    ret = aclrtMemcpy(*deviceAddr, size, *hostAddr, GetShapeSize(shape) * aclDataTypeSize(dataType), ACL_MEMCPY_HOST_TO_DEVICE);
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
    ret = aclrtMallocHost(hostAddr, size);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMallocHost failed. ERROR: %d\n", ret); return ret);
    memset(*hostAddr, 0, size);
    // Call aclCreateTensor to create an aclTensor.
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, nullptr, 0, aclFormat::ACL_FORMAT_FRACTAL_NZ,
                              shape.data(), shape.size(), *deviceAddr);
    // Call aclrtMemcpy to copy host data to the device memory.
    ret = aclrtMemcpy(*deviceAddr, size, *hostAddr, GetShapeSize(shape) * aclDataTypeSize(dataType), ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
    return 0;
}

int TransToNZShape(std::vector<int64_t> &shapeND, size_t typeSize) {
    if (typeSize == static_cast<size_t>(0)) {
        return 0;
    }
    int64_t h = shapeND[0];
    int64_t w = shapeND[1];
    int64_t h0 = static_cast<int64_t>(16);
    int64_t w0 = static_cast<int64_t>(32) / static_cast<int64_t>(typeSize);
    int64_t h1 = h / h0;
    int64_t w1 = w / w0;
    shapeND[0] = w1;
    shapeND[1] = h1;
    shapeND.emplace_back(h0);
    shapeND.emplace_back(w0);
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
    // 2. Construct the input and output based on the API.
    std::vector<int64_t> tokenXShape = {8, 1, 7168};            // B,S,He
    std::vector<int64_t> weightDqShape = {7168, 1536};          // He,Hcq
    std::vector<int64_t> weightUqQrShape = {1536, 24576};        // Hcq,N*(D+Dr)
    std::vector<int64_t> weightUkShape = {128, 128, 512};        // N,D,Hckv
    std::vector<int64_t> weightDkvKrShape = {7168, 576};        // He,Hckv+Dr
    std::vector<int64_t> rmsnormGammaCqShape = {1536};          // Hcq
    std::vector<int64_t> rmsnormGammaCkvShape = {512};          // Hckv
    std::vector<int64_t> ropeSinShape = {8, 1 64};             // B,S,Dr
    std::vector<int64_t> ropeCosShape = {8, 1, 64};             // B,S,Dr
    std::vector<int64_t> kvCacheShape = {1, 16, 1, 512};      // BlockNum,BlockSize,Nkv,Hckv
    std::vector<int64_t> krCacheShape = {1, 16, 1, 64};       // BlockNum,BlockSize,Nkv,Dr
    std::vector<int64_t> cacheIndexShape = {8, 1};              // B,S
    std::vector<int64_t> dequantScaleXShape = {8, 224};           // B*S, 1
    std::vector<int64_t> dequantScaleWDqShape = {1536, 224};      // 1, Hcq
    std::vector<int64_t> dequantScaleWUqQrShape = {24576, 48};    // 1, N*(D+Dr)
    std::vector<int64_t> dequantScaleWDkvKrShape = {576, 224};    // 1, Hckv+Dr
    std::vector<int64_t> quantScaleCkvShape = {1};    // 1
    std::vector<int64_t> queryShape = {8, 1, 128, 512};          // B,S,N,Hckv
    std::vector<int64_t> queryRopeShape = {8, 1, 128, 64};       // B,S,N,Dr
    std::vector<int64_t> dequantScaleQNopeShape = {8, 128, 1};   // B*S, N, 1
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
    void* dequantScaleXDeviceAddr = nullptr;
    void* dequantScaleWDqDeviceAddr = nullptr;
    void* dequantScaleWUqQrDeviceAddr = nullptr;
    void* dequantScaleWDkvKrDeviceAddr = nullptr;
    void* quantScaleCkvDeviceAddr = nullptr;
    void* queryDeviceAddr = nullptr;
    void* queryRopeDeviceAddr = nullptr;
    void* dequantScaleQNopeDeviceAddr = nullptr;

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
    void* dequantScaleXHostAddr = nullptr;
    void* dequantScaleWDqHostAddr = nullptr;
    void* dequantScaleWUqQrHostAddr = nullptr;
    void* dequantScaleWDkvKrHostAddr = nullptr;
    void* quantScaleCkvHostAddr = nullptr;
    void* queryHostAddr = nullptr;
    void* queryRopeHostAddr = nullptr;
    void* dequantScaleQNopeHostAddr = nullptr;

    aclTensor* tokenX = nullptr;
    aclTensor* weightDq = nullptr;
    aclTensor* weightUqQr = nullptr;
    aclTensor* weightUk = nullptr;
    aclTensor* weightDkvKr = nullptr;
    aclTensor* rmsnormGammaCq = nullptr;
    aclTensor* rmsnormGammaCkv = nullptr;
    aclTensor* ropeSin = nullptr;
    aclTensor* ropeCos = nullptr;
    aclTensor* kvCache = nullptr;
    aclTensor* krCache = nullptr;
    aclTensor* cacheIndex = nullptr;
    aclTensor* dequantScaleX = nullptr;
    aclTensor* dequantScaleWDq = nullptr;
    aclTensor* dequantScaleWUqQr = nullptr;
    aclTensor* dequantScaleWDkvKr = nullptr;
    aclTensor* quantScaleCkv = nullptr;
    int64_t weightQuantMode = 3;
    int64_t kvQuantMode = 1;
    int64_t queryQuantMode = 1;
    int64_t ckvkrRepoMode = 0;
    int64_t quantScaleRepoMode = 0;
    int64_t tileSize = 128;
    double qcQrScale = 1.0f;
    double kcScale = 1.0f;
    aclTensor* query = nullptr;
    aclTensor* queryRope = nullptr;
    aclTensor* dequantScaleQNope = nullptr;

    // Convert the shapes of the three variables in NZ format.
    constexpr size_t EXAMPLE_INT8_SIZE = sizeof(int8_t);
    constexpr size_t EXAMPLE_BFLOAT16_SIZE = sizeof(int16_t);
    ret = TransToNZShape(weightDqShape, EXAMPLE_INT8_SIZE);
    CHECK_RET(ret == 0, LOG_PRINT("trans NZ shape failed.\n"); return ret);
    ret = TransToNZShape(weightUqQrShape, EXAMPLE_INT8_SIZE);
    CHECK_RET(ret == 0, LOG_PRINT("trans NZ shape failed.\n"); return ret);
    ret = TransToNZShape(weightDkvKrShape, EXAMPLE_INT8_SIZE);
    CHECK_RET(ret == 0, LOG_PRINT("trans NZ shape failed.\n"); return ret);

    // Create a tokenX aclTensor.
    ret = CreateAclTensorND(tokenXShape, &tokenXDeviceAddr, &tokenXHostAddr, aclDataType::ACL_FLOAT8_E4M3FN, &tokenX);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create a weightDq aclTensor.
    ret = CreateAclTensorNZ(weightDqShape, &weightDqDeviceAddr, &weightDqHostAddr, aclDataType::ACL_FLOAT8_E4M3FN, &weightDq);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create a weightUqQr aclTensor.
    ret = CreateAclTensorNZ(weightUqQrShape, &weightUqQrDeviceAddr, &weightUqQrHostAddr, aclDataType::ACL_FLOAT8_E4M3FN, &weightUqQr);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create a weightUk aclTensor.
    ret = CreateAclTensorND(weightUkShape, &weightUkDeviceAddr, &weightUkHostAddr, aclDataType::ACL_BF16, &weightUk);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create a weightDkvKr aclTensor.
    ret = CreateAclTensorNZ(weightDkvKrShape, &weightDkvKrDeviceAddr, &weightDkvKrHostAddr, aclDataType::ACL_FLOAT8_E4M3FN, &weightDkvKr);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create an rmsnormGammaCq aclTensor.
    ret = CreateAclTensorND(rmsnormGammaCqShape, &rmsnormGammaCqDeviceAddr, &rmsnormGammaCqHostAddr, aclDataType::ACL_BF16, &rmsnormGammaCq);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create an rmsnormGammaCkv aclTensor.
    ret = CreateAclTensorND(rmsnormGammaCkvShape, &rmsnormGammaCkvDeviceAddr, &rmsnormGammaCkvHostAddr, aclDataType::ACL_BF16, &rmsnormGammaCkv);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create a ropeSin aclTensor.
    ret = CreateAclTensorND(ropeSinShape, &ropeSinDeviceAddr, &ropeSinHostAddr, aclDataType::ACL_BF16, &ropeSin);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create a ropeCos aclTensor.
    ret = CreateAclTensorND(ropeCosShape, &ropeCosDeviceAddr, &ropeCosHostAddr, aclDataType::ACL_BF16, &ropeCos);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create a cacheIndex aclTensor.
    ret = CreateAclTensorND(cacheIndexShape, &cacheIndexDeviceAddr, &cacheIndexHostAddr, aclDataType::ACL_INT64, &cacheIndex);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create a kvCache aclTensor.
    ret = CreateAclTensorND(kvCacheShape, &kvCacheDeviceAddr, &kvCacheHostAddr, aclDataType::ACL_FLOAT8_E4M3FN, &kvCache);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create a krCache aclTensor.
    ret = CreateAclTensorND(krCacheShape, &krCacheDeviceAddr, &krCacheHostAddr, aclDataType::ACL_BF16, &krCache);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create a dequantScaleX aclTensor.
    ret = CreateAclTensorND(dequantScaleXShape, &dequantScaleXDeviceAddr, &dequantScaleXHostAddr, aclDataType::ACL_FLOAT8_E8M0, &dequantScaleX);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create a dequantScaleWDq aclTensor.
    ret = CreateAclTensorND(dequantScaleWDqShape, &dequantScaleWDqDeviceAddr, &dequantScaleWDqHostAddr, aclDataType::ACL_FLOAT8_E8M0, &dequantScaleWDq);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create a dequantScaleWUqQr aclTensor.
    ret = CreateAclTensorND(dequantScaleWUqQrShape, &dequantScaleWUqQrDeviceAddr, &dequantScaleWUqQrHostAddr, aclDataType::ACL_FLOAT8_E8M0, &dequantScaleWUqQr);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create a dequantScaleWDkvKr aclTensor.
    ret = CreateAclTensorND(dequantScaleWDkvKrShape, &dequantScaleWDkvKrDeviceAddr, &dequantScaleWDkvKrHostAddr, aclDataType::ACL_FLOAT8_E8M0, &dequantScaleWDkvKr);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create a quantScaleCkv aclTensor.
    ret = CreateAclTensorND(quantScaleCkvShape, &quantScaleCkvDeviceAddr, &quantScaleCkvHostAddr, aclDataType::ACL_FLOAT, &quantScaleCkv);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create a query aclTensor.
    ret = CreateAclTensorND(queryShape, &queryDeviceAddr, &queryHostAddr, aclDataType::ACL_FLOAT8_E4M3FN, &query);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create a queryRope aclTensor.
    ret = CreateAclTensorND(queryRopeShape, &queryRopeDeviceAddr, &queryRopeHostAddr, aclDataType::ACL_BF16, &queryRope);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create a dequantScaleQNope aclTensor.
    ret = CreateAclTensorND(dequantScaleQNopeShape, &dequantScaleQNopeDeviceAddr, &dequantScaleQNopeHostAddr, aclDataType::ACL_FLOAT, &dequantScaleQNope);
    CHECK_RET(ret == ACL_SUCCESS, return ret);


    // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    // Call the first-phase API of aclnnMlaPrologV3WeightNz.
    ret = aclnnMlaPrologV3WeightNzGetWorkspaceSize(tokenX, weightDq, weightUqQr, weightUk, weightDkvKr, rmsnormGammaCq, rmsnormGammaCkv, ropeSin, ropeCos, kvCache, krCache, cacheIndex,
        dequantScaleX, dequantScaleWDq, dequantScaleWUqQr, dequantScaleWDkvKr, quantScaleCkv, nullptr, nullptr, nullptr, nullptr, rmsnormEpsilonCq, rmsnormEpsilonCkv, cacheMode,
        weightQuantMode, kvQuantMode, queryQuantMode, ckvkrRepoMode, quantScaleRepoMode, tileSize, qcQrScale, kcScale,
        query, queryRope, dequantScaleQNope, nullptr, nullptr, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMlaPrologV3WeightNzGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // Allocate device memory based on the computed workspaceSize.
    void* workspaceAddr = nullptr;
    if (workspaceSize > static_cast<uint64_t>(0)) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    // Call the second-phase API of aclnnMlaPrologV3WeightNz.
    ret = aclnnMlaPrologV3WeightNz(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMlaPrologV3WeightNz failed. ERROR: %d\n", ret); return ret);

    // 4. (Fixed writing) Wait until the task execution is complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. Obtain the output value and copy the result from the device to the host. Modify the code based on the API definition.
    auto size = GetShapeSize(queryShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), queryDeviceAddr, size * sizeof(float),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }
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
    aclDestroyTensor(dequantScaleX);
    aclDestroyTensor(dequantScaleWDq);
    aclDestroyTensor(dequantScaleWUqQr);
    aclDestroyTensor(dequantScaleWDkvKr);
    aclDestroyTensor(quantScaleCkv);
    aclDestroyTensor(query);
    aclDestroyTensor(queryRope);
    aclDestroyTensor(dequantScaleQNope);

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
    aclrtFree(dequantScaleXDeviceAddr);
    aclrtFree(dequantScaleWDqDeviceAddr);
    aclrtFree(dequantScaleWUqQrDeviceAddr);
    aclrtFree(dequantScaleWDkvKrDeviceAddr);
    aclrtFree(quantScaleCkvDeviceAddr);
    aclrtFree(queryDeviceAddr);
    aclrtFree(queryRopeDeviceAddr);
    aclrtFree(dequantScaleQNopeDeviceAddr);

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
    aclrtFree(dequantScaleXHostAddr);
    aclrtFree(dequantScaleWDqHostAddr);
    aclrtFree(dequantScaleWUqQrHostAddr);
    aclrtFree(dequantScaleWDkvKrHostAddr);
    aclrtFree(quantScaleCkvHostAddr);
    aclrtFree(queryHostAddr);
    aclrtFree(queryRopeHostAddr);
    aclrtFree(dequantScaleQNopeHostAddr);

    if (workspaceSize > static_cast<uint64_t>(0)) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    _exit(0);
}
  ```
