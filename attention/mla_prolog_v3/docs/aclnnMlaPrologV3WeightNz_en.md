# aclnnMlaPrologV3WeightNz

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-21T11:50:32.048Z -->

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

## Function

- **Function updates**: (differences from aclnnMlaPrologV2weightNz)
    - Added scale correction factors for Query and Key, corresponding to qcQrScale ($\alpha_q$) and kcScale ($\alpha_{kv}$), respectively.
    - Added parameters such as actualSeqLenOptional, kNopeClipAlphaOptional, weightQuantMode, kvCacheQuantMode, queryQuantMode, ckvkrRepoMode, quantScaleRepoMode, and tileSize, as well as optional outputs queryNormOutOptional and dequantScaleQNormOutOptional.
    - Adjusted the name and position of the cacheIndex parameter, corresponding to the current cacheIndexOptional.
- **Description**: Inference use case. Computes the preprocessing of Multi-Head Latent Attention. The main computation process consists of seven branches:
    - First, the input $x$ is multiplied by $W^{DQ}$, downsampled, and normalized with RmsNorm. The result is then split into two branches. In the first branch, it is multiplied by $W^{UQ}$ and $W^{UK}$, upsampled twice, and then multiplied by the Query scale correction factor $\alpha_q$ to obtain $q^N$. In the second branch, it is multiplied by $W^{QR}$ and then goes through rotary position encoding (ROPE) to obtain $q^R$.
    - The third branch: the input $x$ is multiplied by $W^{DKV}$, downsampled, and normalized with RmsNorm. The result is then multiplied by the Key scale correction factor $\alpha_{kv}$ and written into the Cache to obtain $k^C$.
    - The fourth path: the input $x$ is multiplied by $W^{KR}$, then undergoes rotary position encoding and is passed into another Cache to obtain $k^R$;
    - The fifth path: the quantization parameter obtained after the output $q^N$ undergoes DynamicQuant.
    - The sixth branch: when queryNormOutOptional is non-null and weightQuantMode is 0, the input $x$ is multiplied by $W^{DQ}$, downsampled, and normalized with RmsNorm to obtain the compressed result $c^Q$, which is output as queryNormOutOptional.
    - The seventh branch: when queryNormOutOptional is non-null and in the quantization scenario (weightQuantMode≠0), DynamicQuant is performed on $c^Q$ to obtain the quantized queryNormOutOptional and the corresponding quantization parameter dequantScaleQNormOutOptional.
    - The weight parameters WeightDq, WeightUqQr, and WeightDkvKr must be passed in NZ format.

- **Formula**:

    RmsNorm formula

    $$
    \text{RmsNorm}(x) = \gamma \cdot \frac{x_i}{\text{RMS}(x)}
    $$

    $$
    \text{RMS}(x) = \sqrt{\frac{1}{N} \sum_{i=1}^{N} x_i^2 + \epsilon}
    $$

    Formula for Query, including downsampling, RmsNorm, and two upsampling operations.

    $$
    c^Q = \alpha_q\cdot\mathrm{RmsNorm}(x \cdot W^{DQ})
    $$

    $$
    q^C = c^Q \cdot W^{UQ}
    $$

    $$
    q^N = q^C \cdot W^{UK}
    $$
    where $\alpha_q$ is the Query scale correction parameter.

    Perform ROPE rotary positional encoding on Query.

    $$
    q^R = \mathrm{ROPE}(c^Q \cdot W^{QR})
    $$

    Key computation formula, including downsampling and RmsNorm, with the computation result stored in cache.

    $$
    c^{KV} = \alpha_{kv}\cdot\mathrm{RmsNorm}(x \cdot W^{DKV})
    $$

    $$
    k^C = \mathrm{Cache}(c^{KV})
    $$
    where $\alpha_{kv}$ is the scale correction parameter of Key.

    Perform ROPE rotary positional encoding on Key and store the result in cache.

    $$
    k^R = \mathrm{Cache}(\mathrm{ROPE}(x \cdot W^{KR}))
    $$

    Dequant Scale Query Nope Formula

    $$
    \mathrm{dequantScaleQNope} = {\mathrm{RowMax}(\mathrm{abs}(q^{N})) / Q_{max}}
    $$

    $$
    q^{N} = {\mathrm{round}(q^{N} / \mathrm{dequantScaleQNope})}
    $$

    Query Norm and Dequant Scale Query Norm Formula (output when queryNormOutOptional is non-null)

    Non-quantized field (weightQuantMode=0):

    $$
    queryNorm = c^Q = \alpha_q\cdot\mathrm{RmsNorm}(x \cdot W^{DQ})
    $$

    In this case, queryNorm is of the BFLOAT16 type, and dequantScaleQNorm is not output (it is nullptr).

    In the quantization scenario (weightQuantMode=1/2/4/5, per-token dynamic quantization), smoothScaleCq is optional. If it is not passed, it is regarded as 1:

    $$
    \tilde{c}^Q = c^Q \cdot smoothScale
    $$

    $$
    \mathrm{dequantScaleQNorm} = {\mathrm{RowMax}(\mathrm{abs}(\tilde{c}^Q)) / Q_{max}}
    $$

    $$
    queryNorm = {\mathrm{round}(\tilde{c}^Q / \mathrm{dequantScaleQNorm})}
    $$

    mxfp8 quantization scenario (weightQuantMode=3, pergroup dynamic quantization in groups of 32 elements, with quantization parameters of the FLOAT8_E8M0 type):

    $$
    \mathrm{dequantScaleQNorm} = 2^{\lfloor \mathrm{log}_2(\mathrm{GroupMax}(\mathrm{abs}(c^Q))) \rfloor - 8}
    $$

    $$
    queryNorm = {\mathrm{cast\_to\_fp8}(c^Q / \mathrm{dequantScaleQNorm})}
    $$

    Here, $Q_{max}$ is the maximum value of the quantized output data type: 127 for INT8, 448 for FLOAT8_E4M3FN, and 32768 for HIFLOAT8. 8 is the maximum value emax of the exponent bits of FLOAT8_E4M3FN.

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). You must first call the "aclnnMlaPrologV3WeightNzGetWorkspaceSize" API to obtain the input parameters and calculate the required workspace size based on the process, and then call the "aclnnMlaPrologV3WeightNz" API to perform the computation.

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

    <table style="undefined;table-layout: fixed; width: 1625px"><colgroup>
    <col style="width: 247px">
    <col style="width: 132px">
    <col style="width: 232px">
    <col style="width: 293px">
    <col style="width: 185px">
    <col style="width: 119px">
    <col style="width: 272px">
    <col style="width: 145px">
    </colgroup>
    <thead>
      <tr>
        <th>Parameter</th>
        <th>Input/Output</th>
        <th>Description</th>
        <th>Instruction</th>
        <th>Data Type</th>
        <th>Data Format</th>
        <th>Dimension (shape)</th>
        <th>Non-contiguous Tensor</th>
      </tr>
    </thead>
      <tr>
        <td>tokenX</td>
        <td>Input</td>
        <td>Input tensor used in the formula for calculating Query and Key.</td>
        <td>Supports an empty Tensor with B=0, S=0, and T=0.</td>
        <td>BFLOAT16, FLOAT8_E4M3FN, INT8, HIFLOAT8</td>
        <td>ND</td>
        <td>          <ul>
            <li>BS coalesced axis: (T, He)</li>
            <li>BS non-coalesced axis: (B, S, He)</li>
          </ul>
        </td>
        <td>×</td>
      </tr>
      <tr>
        <td>weightDq</td>
        <td>Input</td>
        <td>Downsampling weight matrix <i>W<sup><i>DQ</i></sup></i> used in the formula for calculating Query.<br>Each dimension without transposing: (k, n)</td>
        <td>Does not support empty tensors.</td>
        <td>BFLOAT16, FLOAT8_E4M3FN, INT8, HIFLOAT8</td>
        <td>FRACTAL_NZ</td>
        <td>(He, Hcq)</td>
        <td>×</td>
      </tr>
      <tr>
        <td>weightUqQr</td>
        <td>Input</td>
        <td>Upsampling weight matrix <i>W<sup><i>UQ</i></sup></i> and position encoding weight matrix <i>W<sup><i>QR</i></sup></i> used in the formula for calculating Query.<br>Representation of each dimension without transposing: (k, n)</td>
        <td>Does not support empty tensors.</td>
        <td>BFLOAT16, INT8, FLOAT8_E4M3FN, HIFLOAT8</td>
        <td>FRACTAL_NZ</td>
        <td>(Hcq, N*(D+Dr))</td>
        <td>×</td>
      </tr>
      <tr>
        <td>weightUk</td>
        <td>Input</td>
        <td>Upsampling weight <i>W<sup><i>UK</i></sup></i> used to compute Key in the formula.</td>
        <td>Does not support empty tensors</td>
        <td>BFLOAT16</td>
        <td>ND</td>
        <td>(N, D, Hckv)</td>
        <td>×</td>
      </tr>
      <tr>
        <td>weightDkvKr</td>
        <td>Input</td>
        <td>Downsampling weight matrix <i>W<sup><i>DKV</i></sup></i> and position encoding weight matrix <i>W<sup><i>KR</i></sup></i> for computing Key, used in the formula.<br>Meaning of each dimension without transposing: (k, n)</td>
        <td>Does not support empty Tensor</td>
        <td>BFLOAT16, FLOAT8_E4M3FN, INT8, HIFLOAT8</td>
        <td>FRACTAL_NZ</td>
        <td>(He,Hckv+Dr)</td>
        <td>×</td>
      </tr>
      <tr>
        <td>rmsnormGammaCq</td>
        <td>Input</td>
        <td>The <i>γ</i> parameter in the RmsNorm formula for computing <i>c<sup><i>Q</i></sup></i>.</td>
        <td>Does not support empty Tensor.</td>
        <td>BFLOAT16</td>
        <td>ND</td>
        <td>(Hcq)</td>
        <td>×</td>
      </tr>
      <tr>
        <td>rmsnormGammaCkv</td>
        <td>Input</td>
        <td>The <i>γ</i> parameter in the RmsNorm formula for computing <i>c<sup><i>KV</i></sup></i>.</td>
        <td>Does not support empty tensors.</td>
        <td>BFLOAT16</td>
        <td>ND</td>
        <td>(Hckv)</td>
        <td>×</td>
      </tr>
      <tr>
        <td>ropeSin</td>
        <td>Input</td>
        <td>Sine parameter matrix used to compute the rotary position encoding (ROPE).</td>
        <td>Supports empty tensors with B=0, S=0, T=0.</td>
        <td>BFLOAT16</td>
        <td>ND</td>
        <td>          <ul>
            <li>BS coalesced axis: (T,Dr)</li>
            <li>BS non-coalesced axis: (B,S,Dr)</li>
          </ul>
        </td>
        <td>×</td>
      </tr>
      <tr>
        <td>ropeCos</td>
        <td>Input</td>
        <td>Cosine parameter matrix used to compute the rotary position encoding.</td>
        <td>Supports empty tensors with B=0,S=0,T=0.</td>
        <td>BFLOAT16</td>
        <td>ND</td>
        <td>          <ul>
            <li>BS coalesced axis: (T,Dr)</li>
            <li>BS non-coalesced axis: (B,S,Dr)</li>
          </ul>
        </td>
        <td>×</td>
      </tr>
      <tr>
        <td>kvCacheRef</td>
        <td>Input/Output</td>
        <td>aclTensor used for cache indexing. The computation result is updated in place (corresponding to <i>k<sup><i>C</i></sup></i> in the formula).</td>
        <td>        <ul style="margin:0; padding-left:20px;">
            <li>Supports empty tensors with B=0, Skv=0;</li>
            <li>Nkv is associated with N, and N is a hyperparameter. Therefore, Nkv cannot be 0.</li>
        </ul>
        </td>
        <td>BFLOAT16, INT8, FLOAT8_E4M3FN, HIFLOAT8</td>
        <td>ND</td>
        <td>          <ul>
            <li>CacheMode="PA_BSND"/"PA_NZ"/"PA_BLK_BSND"/"PA_BLK_NZ": (BlockNum,BlockSize,Nkv,Dtile)</li>
            <li>CacheMode="BSND": (B,S,Nkv,Dtile)</li>
            <li>CacheMode="TND": (T,Nkv,Dtile)</li>
          </ul>
        </td>
        <td>Only the first axis can be non-contiguous; all other axes must be contiguous.</td>
      </tr>
      <tr>
        <td>krCacheRef</td>
        <td>Input/Output</td>
        <td>Cache for key position encoding. The calculation result is updated in place (corresponding to <i>k<sup><i>R</i></sup></i> in the formula).</td>
        <td>        <ul style="margin:0; padding-left:20px;">
            <li>Supports an empty Tensor with B=0 and Skv=0;</li>
            <li>Nkv is associated with N, and N is a hyperparameter. Therefore, Nkv cannot be 0.</li>
        </ul>
        </td>
        <td>BFLOAT16, INT8</td>
        <td>ND</td>
        <td>          <ul>
            <li>CacheMode="PA_BSND"/"PA_NZ"/"PA_BLK_BSND"/"PA_BLK_NZ": (BlockNum,BlockSize,Nkv,Dr)</li>
            <li>CacheMode="BSND": (B,S,Nkv,Dr)</li>
            <li>When CacheMode="TND": (T,Nkv,Dr)</li>
            <li>When ckvkrRepoMode=1: the dimension should include 0, and shape (0) is supported.</li>
          </ul>
        </td>
        <td>Only the first axis can be non-contiguous; all other axes must be contiguous.</td>
      </tr>
      <tr>
        <td>cacheIndexOptional</td>
        <td>Input</td>
        <td>Stores the indices of kvCache and krCache.</td>
        <td>        <ul>
            <li>Supports empty tensors with B=0, S=0, T=0.</li>
            <li>cacheMode="PA_BSND"/"PA_NZ": value range must be within [0,BlockNum*BlockSize).</li>
            <li>cacheMode="PA_BLK_BSND"/"PA_BLK_NZ": value range must be within [0,BlockNum).</li>
            <li>cacheMode="TND"/"BSND": nullptr.</li>
        </ul>
        </td>
        <td>INT64</td>
        <td>ND</td>
        <td>          <ul>
            <li>CacheMode="PA_BSND"/"PA_NZ":
              <ol>
                <li>BS coalesced axis: (T)</li>
                <li>BS non-coalesced axis: (B,S)</li>
              </ol>
            </li>
            <li>CacheMode="PA_BLK_BSND"/"PA_BLK_NZ":
              <ol>
                <li>BS coalesced axis: (Sum(⌈S<sub>i</sub>/BlockSize⌉)), where S<sub>i</sub> is the S length in each Batch</li>
                <li>BS non-coalesced axis: (B,⌈S/BlockSize⌉)</li>
              </ol>
            </li>
            <li>CacheMode="TND"/"BSND": nullptr</li>
          </ul>
        </td>
        <td>×</td>
      </tr>
      <tr>
        <td>dequantScaleXOptional</td>
        <td>Input</td>
        <td>Dequantization parameter of tokenX.</td>
        <td>Supports empty tensors with B=0, S=0, T=0 (must be passed in the weightQuantMode=2/3/4/5 use case).</td>
        <td>FLOAT8_E8M0, FLOAT</td>
        <td>ND</td>
        <td>          <ul>
            <li>weightQuantMode=2/4/5:
              <ol>
                <li>BS coalesced axis: (T, 1)</li>
                <li>BS non-coalesced axis: (B*S,1)</li>
              </ol>
            </li>
            <li>weightQuantMode=3:
              <ol>
                <li>BS coalesced axis: (T, He/32)</li>
                <li>BS non-coalesced axis: (B*S, He/32)</li>
              </ol>
            </li>
          </ul>
        </td>
        <td>×</td>
      </tr>
      <tr>
        <td>dequantScaleWDqOptional</td>
        <td>Input</td>
        <td>Dequantization parameter of weightDq.</td>
        <td>Supports non-empty Tensor (must be passed for weightQuantMode=2/3/4/5 use cases).</td>
        <td>FLOAT8_E8M0, FLOAT</td>
        <td>ND</td>
        <td>          <ul>
            <li>weightQuantMode=2/4/5: (1,Hcq)</li>
            <li>weightQuantMode=3: (Hcq, He/32)</li>
          </ul>
        </td>
        <td>×</td>
      </tr>
      <tr>
        <td>dequantScaleWUqQrOptional</td>
        <td>Input</td>
        <td>Per-channel parameter used for dequantization after the MatmulQcQr matrix multiplication.</td>
        <td>Supports non-empty tensor (need to pass this parameter for use cases with weightQuantMode=1/2/3/4/5)</td>
        <td>FLOAT, FLOAT8_E8M0</td>
        <td>ND</td>
        <td>          <ul>
            <li>weightQuantMode=1/2/4/5: (1,N*(D+Dr))</li>
            <li>weightQuantMode=3: (N*(D+Dr), Hcq/32)</li>
          </ul>
        </td>
        <td>×</td>
      </tr>
      <tr>
        <td>dequantScaleWDkvKrOptional</td>
        <td>Input</td>
        <td>Dequantization parameter of weightDkvKr.</td>
        <td>Supports non-empty Tensor (need to pass for weightQuantMode=2/3/4/5 use cases)</td>
        <td>FLOAT8_E8M0, FLOAT</td>
        <td>ND</td>
        <td>          <ul>
            <li>weightQuantMode=2/4/5: (1,Hckv+Dr)</li>
            <li>weightQuantMode=3: (Hckv+Dr, He/32)</li>
          </ul>
        </td>
        <td>×</td>
      </tr>
      <tr>
        <td>quantScaleCkvOptional</td>
        <td>Input</td>
        <td>Parameter for quantizing kvCache output data.</td>
        <td>Supports non-empty tensor (need to pass for the kvCacheQuantMode=1/2 use case)</td>
        <td>FLOAT</td>
        <td>ND</td>
        <td>          <ul>
            <li>kvCacheQuantMode=1: (1)</li>
            <li>kvCacheQuantMode=2: (1,Hckv)</li>
          </ul>
        </td>
        <td>×</td>
      </tr>
      <tr>
        <td>quantScaleCkrOptional</td>
        <td>Input</td>
        <td>Parameter for the quantization operation on the krCache output data.</td>
        <td>Supports non-empty tensor (need to pass for the kvCacheQuantMode=2 use case)</td>
        <td>FLOAT</td>
        <td>ND</td>
        <td>(1,Dr)</td>
        <td>×</td>
      </tr>
      <tr>
        <td>smoothScalesCqOptional</td>
        <td>Input</td>
        <td>Parameter for performing dynamic quantization on the RmsNormCq output.</td>
        <td>Supports non-empty tensors (optional to pass in use cases where weightQuantMode=1/2/4/5).</td>
        <td>FLOAT</td>
        <td>ND</td>
        <td>(1,Hcq)</td>
        <td>×</td>
      </tr>
      <tr>
        <td>actualSeqLenOptional</td>
        <td>Input</td>
        <td>Indicates the sequence length in each batch, stored as prefix sums.</td>
        <td>Required to pass when BS axes are coalesced and CacheMode="PA_BLK_BSND"/"PA_BLK_NZ"</td>
        <td>INT32</td>
        <td>ND</td>
        <td>(B)</td>
        <td>×</td>
      </tr>
      <tr>
        <td>kNopeClipAlphaOptional</td>
        <td>Input</td>
        <td>Indicates the scaling factor for the clip operation on kvCache.</td>
        <td>The shape is 1 in partial quantization pertoken-pergroup scenarios and int8 full quantization pertoken-pergroup scenarios. Not supported in other scenarios. Does not support empty tensors.</td>
        <td>FLOAT</td>
        <td>ND</td>
        <td>(1)</td>
        <td>×</td>
      </tr>
      <tr>
        <td>rmsnormEpsilonCq</td>
        <td>Input</td>
        <td>The <i>ε</i> parameter in the RmsNorm formula for computing <i>c<sup><i>Q</i></sup></i>.</td>
        <td>        <ul>
            <li>If the user does not specifically specify this, it is recommended to pass 1e-05</li>
            <li>Only the double data type is supported</li>
        </ul>
        </td>
        <td>DOUBLE</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>rmsnormEpsilonCkv</td>
        <td>Input</td>
        <td>The <i>ε</i> parameter in the RmsNorm formula for computing <i>c<sup><i>KV</i></sup></i>.</td>
        <td>        <ul>
            <li>If the user does not specifically specify this, it is recommended to pass 1e-05</li>
            <li>Only the double data type is supported</li>
        </ul>
        </td>
        <td>DOUBLE</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>cacheModeOptional</td>
        <td>Input</td>
        <td>Indicates the mode of kvCache.</td>
        <td>        <ul>
            <li>If not specifically specified by the user, it is recommended to pass "PA_BSND"</li>
            <li>Only the char* type is supported</li>
            <li>Optional values: "PA_BSND", "PA_NZ", "PA_BLK_BSND", "PA_BLK_NZ", "BSND", "TND"</li>
        </ul>
        </td>
        <td>CHAR*</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>weightQuantMode</td>
        <td>Input</td>
        <td>Indicates the quantization mode of weightDq, weightUqQr, weightUk, and weightDkvKr.</td>
        <td>        <ul>
            <li>0 indicates non-quantized</li>
            <li>1 indicates quantized weightUqQr</li>
            <li>2 indicates int8 quantization of weightDq, weightUqQr, and weightDkvKr</li>
            <li>3 indicates mxfp8 quantization of weightDq, weightUqQr, and weightDkvKr</li>
            <li>4 indicates fp8 quantization of weightDq, weightUqQr, and weightDkvKr</li>
            <li>5 indicates hif8 quantization of weightDq, weightUqQr, and weightDkvKr</li>
            <li>Default value is 0</li>
        </ul>
        </td>
        <td>INT</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>kvCacheQuantMode</td>
        <td>Input</td>
        <td>Indicates the quantization mode of kvCache.</td>
        <td>0 indicates non-quantized, 1 indicates pertensor quantization, 2 indicates perchannel quantization, 3 indicates pertoken-pergroup quantization, and the default value is 0.</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>queryQuantMode</td>
        <td>Input</td>
        <td>Indicates the quantization mode of query.</td>
        <td>0 indicates non-quantized, 1 indicates per-token-head quantization, and the default value is 0.</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>ckvkrRepoMode</td>
        <td>Input</td>
        <td>Indicates the storage mode of kvCache and krCache.</td>
        <td>0 indicates that kvCache and krCache are stored separately, and 1 indicates that kvCache and krCache are stored in a merged manner. Default value is 0.</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>quantScaleRepoMode</td>
        <td>Input</td>
        <td>Indicates the storage mode of the quantization scale.</td>
        <td>0 indicates that the quantization scale and data are stored separately. 1 indicates that the quantization scale and data are stored together and output as kvCacheRef. The default value is 0.</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>tileSize</td>
        <td>Input</td>
        <td>Indicates the size of each tile in pertoken-pergroup quantization. This parameter is valid only when kvCacheQuantMode is 3.</td>
        <td>Default value is 128.</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>qcQrScale</td>
        <td>Input</td>
        <td>Indicates the scale correction factor of Query.</td>
        <td>If the user does not specifically specify it, 1.0 needs to be passed.</td>
        <td>DOUBLE</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>kcScale</td>
        <td>Input</td>
        <td>Represents the scale correction factor of Key.</td>
        <td>If the user does not specifically specify it, 1.0 needs to be passed.</td>
        <td>DOUBLE</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>queryOut</td>
        <td>Output</td>
        <td>Output tensor of Query in the formula (corresponding to <i>q<sup><i>N</i></sup></i>).</td>
        <td>-</td>
        <td>BFLOAT16, FLOAT8_E4M3FN, INT8, HIFLOAT8</td>
        <td>ND</td>
        <td>          <ul>
            <li>BS coalesced axis: (T,N,Hckv)</li>
            <li>BS non-coalesced axis: (B,S,N,Hckv)</li>
          </ul>
        </td>
        <td>×</td>
      </tr>
      <tr>
        <td>queryRopeOut</td>
        <td>Output</td>
        <td>Output tensor of the Query position encoding in the formula (corresponding to <i>q<sup><i>R</i></sup></i>).</td>
        <td>-</td>
        <td>BFLOAT16</td>
        <td>ND</td>
        <td>          <ul>
            <li>BS coalesced axis: (T,N,Dr)</li>
            <li>BS non-coalesced axis: (B,S,N,Dr)</li>
          </ul>
        </td>
        <td>×</td>
      </tr>
      <tr>
        <td>dequantScaleQNopeOutOptional</td>
        <td>Output</td>
        <td>Dequantization parameters of the Query output in the formula.</td>
        <td>Output when weightQuantMode=2/3/4/5; nullptr when weightQuantMode=0/1.</td>
        <td>FLOAT</td>
        <td>ND</td>
        <td>          <ul>
            <li>BS coalesced axis: (T,N,1)</li>
            <li>BS non-coalesced axis: (B*S,N,1)</li>
          </ul>
        </td>
        <td>×</td>
      </tr>
      <tr>
        <td>queryNormOutOptional</td>
        <td>Output</td>
        <td>Output tensor of tokenX after rmsNorm in the formula (corresponding to <i>c<sup><i>Q</i></sup></i>).</td>
        <td>Optional output. Pass nullptr when this output is not needed.</td>
        <td>BFLOAT16, INT8, FLOAT8_E4M3FN, HIFLOAT8</td>
        <td>ND</td>
        <td>          <ul>
            <li>BS coalesced axis: (T,Hcq)</li>
            <li>BS non-coalesced axis: (B,S,Hcq)</li>
          </ul>
        </td>
        <td>×</td>
      </tr>
      <tr>
        <td>dequantScaleQNormOutOptional</td>
        <td>Output</td>
        <td>Dequantization parameter of queryNormOutOptional.</td>
        <td>Required when queryNormOutOptional is non-null and weightQuantMode is nonzero; must be nullptr otherwise.</td>
        <td>FLOAT, FLOAT8_E8M0</td>
        <td>ND</td>
        <td>          <ul>
            <li>weightQuantMode=1/2/4/5:
              <ol>
                <li>BS coalesced axis: (T,1)</li>
                <li>BS non-coalesced axis: (B*S,1)</li>
              </ol>
            </li>
            <li>weightQuantMode=3:
              <ol>
                <li>BS coalesced axis: (T, Hcq/32)</li>
                <li>BS non-coalesced axis: (B*S, Hcq/32)</li>
              </ol>
            </li>
          </ul>
        </td>
        <td>×</td>
      </tr>
      <tr>
        <td>workspaceSize</td>
        <td>Output</td>
        <td>Returns the workspace size that needs to be allocated on the Device side.</td>
        <td>Used only for output. No input configuration is required. The data type is uint64_t*.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>executor</td>
        <td>Output</td>
        <td>Returns the op executor, containing the operator computation flow.</td>
        <td>Used only for the output result. No input configuration required. The data type is aclOpExecutor**.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
  </table>

<!-- npu="A3,910b" id4 -->
- x1 is INT8, x2 is INT8, x1Scale is FLOAT32, and x2Scale is FLOAT32.
  - tokenX, weightDq, weightUqQr, weightDkvKr, kvCacheRef, queryOut, and queryNormOutOptional do not support the FLOAT8_E4M3FN or HIFLOAT8 data types.
  - dequantScaleXOptional, dequantScaleWDqOptional, dequantScaleWUqQrOptional, dequantScaleWDkvKrOptional, and dequantScaleQNormOutOptional do not support the FLOAT8_E8M0 data type.

<!-- end id4 -->
- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

    <table style="undefined;table-layout: fixed;width: 1155px"><colgroup>
    <col style="width: 319px">
    <col style="width: 144px">
    <col style="width: 671px">
    </colgroup>
      <thead>
        <tr>
          <th>Return Values</th>
          <th>Error Code</th>
          <th>Description</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>ACLNN_ERR_PARAM_NULLPTR</td>
          <td>161001</td>
          <td>A null pointer exists in a parameter that must be passed (such as the input/output parameters on which the API core depends).</td>
        </tr>
        <tr>
          <td>ACLNN_ERR_PARAM_INVALID</td>
          <td>161002</td>
          <td>The shape (dimension/size) or dtype (data type) of the input parameter is not within the range supported by the API.</td>
        </tr>
        <tr>
          <td>ACLNN_ERR_RUNTIME_ERROR</td>
          <td>361001</td>
          <td>An exception occurs when the API calls an NPU Runtime interface for memory operations (for example, the Runtime service is not started or memory allocation fails).</td>
        </tr>
        <tr>
          <td>ACLNN_ERR_INNER_TILING_ERROR</td>
          <td>561002</td>
          <td>An exception occurs during tiling. The dtype or shape of the input parameter is incorrect.</td>
        </tr>
      </tbody>
    </table>

## aclnnMlaPrologV3WeightNz

- **Parameters**

    <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
    <col style="width: 168px">
    <col style="width: 128px">
    <col style="width: 854px">
    </colgroup>
      <thead>
        <tr>
          <th>Parameter</th>
          <th>Parameter Type</th>
          <th>Description</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>workspace</td>
          <td>void*</td>
          <td>The workspace memory address allocated on the Device side.</td>
        </tr>
        <tr>
          <td>workspaceSize</td>
          <td>uint64_t</td>
          <td>The workspace size allocated on the Device side, obtained by the first-phase API aclnnMlaPrologV3WeightNzGetWorkspaceSize.</td>
        </tr>
        <tr>
          <td>executor</td>
          <td>aclOpExecutor*</td>
          <td>Operator executor that contains the operator computation process.</td>
        </tr>
        <tr>
          <td>stream</td>
          <td>aclrtStream</td>
          <td>Stream for executing tasks.</td>
        </tr>
      </tbody>
    </table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnMlaPrologV3WeightNz defaults to a deterministic implementation.

<details>
  <summary><a id="shapeDesc"></a>Description of shape format fields</summary>
    &nbsp;&nbsp;<table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
        <col style="width: 165px">
        <col style="width: 625px">
        <col style="width: 360px">
        </colgroup>
    <thead>
      <tr>
        <th>Field Name</th>
        <th>Full Name/Meaning</th>
        <th>Value Rules and Description</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>B</td>
        <td>Batch (input sample batch size)</td>
        <td>Value range: 0-65536</td>
      </tr>
      <tr>
        <td>S</td>
        <td>Seq-Length (input sample sequence length)</td>
        <td>Value range: unlimited</td>
      </tr>
      <tr>
        <td>He</td>
        <td>Head-Size (hidden layer size)</td>
        <td>Fixed values: 1024, 2048, 3072, 4096, 5120, 6144, 7168, 7680, 8192</td>
      </tr>
      <tr>
        <td>Hcq</td>
        <td>Dimension of the q low-rank matrix</td>
        <td>Fixed values: 1536, 2048</td>
      </tr>
      <tr>
        <td>N</td>
        <td>Head-Num (number of heads)</td>
        <td>Value range: 1-128</td>
      </tr>
      <tr>
        <td>Hckv</td>
        <td>Dimension of the kv low-rank matrix</td>
        <td>Fixed value: 512</td>
      </tr>
      <tr>
        <td>D</td>
        <td>qk dimension excluding position encoding</td>
        <td>Fixed values: 128, 192</td>
      </tr>
      <tr>
        <td>Dr</td>
        <td>qk position encoding dimension</td>
        <td>Fixed value: 64</td>
      </tr>
      <tr>
        <td>Nkv</td>
        <td>Number of kv heads</td>
        <td>Fixed value: 1</td>
      </tr>
      <tr>
        <td>BlockNum</td>
        <td>Number of blocks in the PagedAttention use case</td>
        <td>
          <ul>
            <li>If CacheMode="PA_BSND"/"PA_NZ", the value is greater than or equal to ⌈(B * S) / BlockSize⌉.</li>
            <li>If CacheMode="PA_BLK_BSND"/"PA_BLK_NZ", the value is greater than or equal to B * ⌈S / BlockSize⌉.</li>
          </ul>
          <p>Note: In the BS coalescing use case, the S length in each batch can be different. Therefore, BlockNum must be greater than or equal to the sum of the S length divided by BlockSize and rounded up for each batch.</p>
          <p>Example: If actualSeqLenOptional is [47, 151, 261, 422] and blocksize=128, the lengths in each batch are [47, 104, 110, 161]. Then BlockNum = ⌈47/128⌉ + ⌈104/128⌉ + ⌈110/128⌉ + ⌈161/128⌉ = 5</p>
        </td>
      </tr>
      <tr>
        <td>BlockSize</td>
        <td>Block size in the PagedAttention use case</td>
        <td>Value range: 16-1024, and a multiple of 16</td>
      </tr>
      <tr>
        <td>T</td>
        <td>Size after BS coalescing</td>
        <td>
          <ul>
            <li>Value range: unlimited</li>
            <li>Note: If BS coalescing is used, tokenX, ropeSin, and ropeCos are 2D, cacheIndex is 1D, and queryOut and queryRopeOut are 3D.</li>
          </ul>
        </td>
      </tr>
      <tr>
        <td>Dtile</td>
        <td>Size of the D dimension of kvCache</td>
        <td>
          <ul>
            <li>In the pertoken-pergroup quantization scenario, the fixed value is 656, which equals 512(Hckv) + 64(Dr)*2 + 4(Hckv/tileSize)*4.</li>
            <li>In other scenarios, the fixed value is Hckv (512).</li>
          </ul>
        </td>
      </tr>
    </tbody>
  </table>
</details>

  <details>
  <summary><a id="shapeConstraint"></a>shape constraints</summary>

  - If the dimension of tokenX uses BS concatenated axis, i.e., (T, He)
    - The shape of ropeSin and ropeCos is (T, Dr)
    - If CacheMode is PA_BSND or PA_NZ, the shape of cacheIndex is (T).
    - If CacheMode is PA_BLK_BSND or PA_BLK_NZ, the shape of cacheIndex is (Sum(⌈<i>S<sub>i</sub></i>/BlockSize⌉)), where S_i is the length of S in each batch.
    - If CacheMode is PA_BLK_BSND or PA_BLK_NZ, actualSeqLenOptional needs to be passed in, with the dimension of (B).
    - In int8/fp8/hif8 full quantization scenarios, the shape of dequantScaleXOptional is (T, 1); in mxfp8 full quantization scenarios, the shape of dequantScaleXOptional is (T, He/32).
    - The shape of queryOut is (T, N, Hckv).
    - The shape of queryRopeOut is (T, N, Dr).
    - In int8/mxfp8/fp8/hif8 full quantization and kvCache pertensor quantization scenarios, the shape of dequantScaleQNopeOutOptional is (T, N, 1); in other scenarios, it is nullptr.
  - If the dimension of tokenX does not use BS concatenated axis, i.e., (B, S, He)
      - The shape of ropeSin and ropeCos is (B, S, Dr)
      - If CacheMode is PA_BSND or PA_NZ, the shape of cacheIndex is (B, S).
      - If CacheMode is PA_BLK_BSND or PA_BLK_NZ, the shape of cacheIndex is (B, ⌈S/BlockSize⌉).
      - In int8/fp8/hif8 full quantization scenarios, the shape of dequantScaleXOptional is (B\*S, 1); in mxfp8 full quantization scenarios, the shape of dequantScaleXOptional is (B*S, He/32).
      - The shape of queryOut is (B, S, N, Hckv)
      - The shape of queryRopeOut is (B, S, N, Dr).
      - In int8/mxfp8/fp8/hif8 full quantization and kvCache pertensor quantization scenarios, the shape of dequantScaleQNopeOutOptional is (B*S, N, 1); in other scenarios, it is nullptr.
  - One or more of B, S, T, and Skv may be 0, meaning that input parameters whose shapes are related to B, S, T, or Skv may be null tensors. The remaining input parameters do not support null tensors.
      - If B, S, and T are 0, queryOut and queryRopeOut output null tensors, and kvCacheRef and krCacheRef are not updated.
      - If Skv is 0, queryOut, queryRopeOut, and dequantScaleQNopeOutOptional are computed normally, while kvCacheRef and krCacheRef are not updated, i.e., they output null tensors.
  - When CacheMode is BSND
      - tokenX must not use BS coalescing, that is, the dimension is (B, S, He)
      - The dimension of kvCache is (B,S,Nkv,Dr)
  - When CacheMode is TND
      - tokenX must use BS coalescing, that is, the dimension is (T, He)
      - The dimension of kvCache is (T,Nkv,Dr)
  - When ckvkrRepoMode=1
      - The dimension of krCache must contain 0, and the supported shape is (0)

  </details>

  <details>
  <summary><a id="specialConstraint"></a>Special constraints</summary>

  - If actualSeqLenOptional is passed, the last value of actualSeqLenOptional must be consistent with T.
  - In pertoken-pergroup quantization mode, ckvkrRepoMode and quantScaleRepoMode must simultaneously be 1; in other quantization modes and non-quantized fields, ckvkrRepoMode and quantScaleRepoMode must simultaneously be 0.
  - In pertoken-pergroup quantization mode, CacheMode supports only PA_BSND, BSND, and TND.
  - When the value of ckvkrRepoMode is 1, krCache must be an empty tensor (that is, the product of its shape is 0).
  - In kvcache pertensor quantization mode, kvCacheQuantMode and queryQuantMode must simultaneously be 1.

  </details>

  <details>
  <summary><a id="SupportScenes"></a>Supported use cases of aclnnMlaPrologV3WeightNz</summary>
  <ul>
    <li><term>Atlas A2 products</term> and <term>Atlas A3 products</term>: the fp8/hif8/mxfp8 fully quantized field is currently not supported</li>
    <li><term>Ascend 950PR&950DT products</term>: all quantization scenarios are currently supported</li>
  </ul>
  <table style="table-layout: auto;" border="1">
    <tr>
      <th colspan="2">Use Case</th>
      <th>Description</th>
    </tr>
    <tr>
      <td colspan="2">Non-quantized</td>
      <td>          weight_quant_mode=0,kv_cache_quant_mode=0,query_quant_mode=0<br>
          Input parameters: all input parameters are non-quantized data. <br>
          Output parameters: all output parameters are non-quantized data.
      </td>
    </tr>
    <tr>
      <td rowspan="3">Partial quantization</td>
      <td>Non-quantized kvCache </td>
      <td>          weight_quant_mode=1,kv_cache_quant_mode=0,query_quant_mode=0<br>
          Input parameters: weightUqQr takes per-token quantized data, and other input parameters are all non-quantized data. The dequantScaleWUqQr field must be passed, and the smoothScalesCq field is optional. <br>
          Output parameters: all output parameters return non-quantized data.
      </td>
    </tr>
    <tr>
      <td>kvCache per-channel quantization </td>
      <td>          weight_quant_mode=1,kv_cache_quant_mode=2,query_quant_mode=0<br>
          Input parameters: weightUqQr takes per-token quantized data, kvCacheRef and krCacheRef take per-channel quantized data, and other input parameters are all non-quantized data. The dequantScaleWUqQr, quantScaleCkv, and quant_scale_ckr fields must be passed, and the smoothScalesCq field is optional. <br>
          Output parameters: kvCacheRef and krCacheRef return per-channel quantized data, and other output parameters return non-quantized data.
      </td>
    </tr>
    <tr>
      <td>kvCache per-token-per-group quantization </td>
      <td>          weight_quant_mode=1,kv_cache_quant_mode=3,query_quant_mode=0<br>
          Input parameters: weightUqQr takes per-token quantized data, and other input parameters are all non-quantized data. The dequantScaleWUqQr field must be passed, and the smoothScalesCq field is optional. <br>
          Output parameters: kvCacheRef returns per-token-per-group quantized data, and other output parameters return non-quantized data.
      </td>
    </tr>
    <tr>
      <td rowspan="3">int8/fp8/hif8 full quantization</td>
      <td> Non-quantized kvCache</td>
      <td>          weight_quant_mode=2/4/5,kv_cache_quant_mode=0,query_quant_mode=0<br>
          Input parameters: pertoken quantized data is passed for tokenX, perchannel quantized data is passed for weightDq, weightUqQr, and weightDkvKr, and other input parameters are all non-quantized data. The dequantScaleX, dequantScaleWDq, dequantScaleWUqQr, and dequantScaleWDkvKr fields must be passed, and the smoothScalesCq field is optional. <br>
          Output parameters: all output parameters return non-quantized data.
      </td>
    </tr>
    <tr>
      <td> kvCache pertensor quantization </td>
      <td>          weight_quant_mode=2/4/5,kv_cache_quant_mode=1,query_quant_mode=1<br>
          Input parameters: pertoken quantized data is passed for tokenX, perchannel quantized data is passed for weightDq, weightUqQr, and weightDkvKr, pertensor quantized data is passed for kvCacheRef, and other input parameters are all non-quantized data. The dequantScaleX, dequantScaleWDq, dequantScaleWUqQr, dequantScaleWDkvKr, and quantScaleCkv fields must be passed, and the smoothScalesCq field is optional. <br>
          Output parameters: queryOut returns per-token-head quantized data, the kvCacheRef output parameter returns pertensor quantized data, and other output parameters return non-quantized data.
      </td>
    </tr>
    <tr>
      <td> kvCache pertoken-pergroup quantization </td>
      <td>          weight_quant_mode=2/4/5,kv_cache_quant_mode=3,query_quant_mode=0<br>
          Input parameters: pertoken quantized data is passed for tokenX, perchannel quantized data is passed for weightDq, weightUqQr, and weightDkvKr, and other input parameters are all non-quantized data. The dequantScaleX, dequantScaleWDq, dequantScaleWUqQr, and dequantScaleWDkvKr fields must be passed, and the smoothScalesCq field is optional. <br>
          Output parameters: the kvCacheRef output parameter returns pertoken-pergroup quantized data, and other output parameters return non-quantized data.
      </td>
    </tr>
    <tr>
      <td rowspan="3">mxfp8 full quantization</td>
      <td> kvCache non-quantization</td>
      <td>          weight_quant_mode=3,kv_cache_quant_mode=0,query_quant_mode=0<br>
          Input parameters: pertoken quantized data is passed for tokenX, perchannel quantized data is passed for weightDq, weightUqQr, and weightDkvKr, and other input parameters are all non-quantized data. The dequantScaleX, dequantScaleWDq, dequantScaleWUqQr, and dequantScaleWDkvKr fields must be passed. <br>
          Output parameters: all output parameters return non-quantized data.
      </td>
    </tr>
    <tr>
      <td> kvCache pertensor quantization </td>
      <td>          weight_quant_mode=3,kv_cache_quant_mode=1,query_quant_mode=1<br>
          Input parameters: pertoken quantized data is passed for tokenX, perchannel quantized data is passed for weightDq, weightUqQr, and weightDkvKr, pertensor quantized data is passed for kvCacheRef, and other input parameters are all non-quantized data. The dequantScaleX, dequantScaleWDq, dequantScaleWUqQr, dequantScaleWDkvKr, and quantScaleCkv fields must be passed. <br>
          Output parameters: queryOut returns per-token-head quantized data, the kvCacheRef output parameter returns pertensor quantized data, and other output parameters return non-quantized data.
      </td>
    </tr>
    <tr>
      <td> kvCache pertoken-pergroup quantization </td>
      <td>          weight_quant_mode=3,kv_cache_quant_mode=3,query_quant_mode=0<br>
          Input parameters: tokenX is passed as pertoken quantized data, weightDq, weightUqQr, and weightDkvKr are passed as perchannel quantized data, and other input parameters are all non-quantized data. The dequantScaleX, dequantScaleWDq, dequantScaleWUqQr, and dequantScaleWDkvKr fields must be passed <br>
          Output parameters: the kvCacheRef output parameter returns pertoken-pergroup quantized data, and other output parameters return non-quantized data.
      </td>
    </tr>
  </table>
  </details>

  <details>
  <summary><a id="SupportScenes"></a>dtype constraints for parameters in different quantization scenarios</summary>

  <div style="overflow-x: auto; width: 100%;">
  <table style="table-layout: auto;" border="1">
    <tr>
      <th rowspan="3">Parameter</th>
      <th rowspan="2" colspan="1">Non-quantized field</th>
      <th colspan="3">Partial quantization scenario</th>
      <th colspan="3">int8 fully quantized field</th>
      <th colspan="3">mxfp8 fully quantized field</th>
      <th colspan="3">fp8 fully quantized scenario</th>
      <th colspan="3">hif8 fully quantized scenario</th>
    </tr>
    <tr>
      <th colspan="1">non-quantized kvCache</th>
      <th colspan="1">kvCache perchannel quantization</th>
      <th colspan="1">kvCache pertoken-pergroup quantization</th>
      <th colspan="1">non-quantized kvCache</th>
      <th colspan="1">kvCache pertensor quantization</th>
      <th colspan="1">kvCache pertoken-pergroup quantization</th>
      <th colspan="1">non-quantized kvCache</th>
      <th colspan="1">kvCache pertensor quantization</th>
      <th colspan="1">kvCache pertoken-pergroup quantization</th>
      <th colspan="1">non-quantized kvCache</th>
      <th colspan="1">kvCache pertensor quantization</th>
      <th colspan="1">kvCache pertoken-pergroup quantization</th>
      <th colspan="1">non-quantized kvCache</th>
      <th colspan="1">kvCache pertensor quantization</th>
      <th colspan="1">kvCache pertoken-pergroup quantization</th>
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
      <td>NULLPTR</td>
      <td>NULLPTR</td>
      <td>NULLPTR</td>
      <td>NULLPTR</td>
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
  </details>

## Example

The following sample code applies to <term>Atlas A2 products</term> and <term>Atlas A3 products</term>. It is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

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
      // Boilerplate for resource initialization.
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
      // Call aclrtMalloc to allocate device memory.
      auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
      // Call aclrtMalloc to allocate host memory.
      ret = aclrtMallocHost(hostAddr, size);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMallocHost failed. ERROR: %d\n", ret); return ret);
      memset(*hostAddr, 0, size);
      // Call aclCreateTensor to create an aclTensor.
      *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, nullptr, 0, aclFormat::ACL_FORMAT_ND,
                                shape.data(), shape.size(), *deviceAddr);
      // Call aclrtMemcpy to copy host data to device memory.
      ret = aclrtMemcpy(*deviceAddr, size, *hostAddr, GetShapeSize(shape)*aclDataTypeSize(dataType), ACL_MEMCPY_HOST_TO_DEVICE);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
      return 0;
  }

  template <typename T>
  int CreateAclTensorNZ(const std::vector<T>& shape, void** deviceAddr, void** hostAddr,
                      aclDataType dataType, aclTensor** tensor) {
      auto size = GetShapeSize(shape) * sizeof(T);
      // Call aclrtMalloc to allocate device memory.
      auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
      // Call aclrtMalloc to allocate host memory.
      ret = aclrtMallocHost(hostAddr, size);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMallocHost failed. ERROR: %d\n", ret); return ret);
      memset(*hostAddr, 0, size);
      // Call aclCreateTensor to create an aclTensor.
      *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, nullptr, 0, aclFormat::ACL_FORMAT_FRACTAL_NZ,
                                shape.data(), shape.size(), *deviceAddr);
      // Call aclrtMemcpy to copy host data to device memory.
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
      // 1. Boilerplate for device/stream initialization. For details, see the AscendCL external API list.
      // Fill in deviceId based on your actual device.
      int32_t deviceId = 0;
      aclrtStream stream;
      auto ret = Init(deviceId, &stream);
      // Handle check as needed.
      CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
      // 2. Construct inputs and outputs according to the API definition.
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

      // Convert the shapes of the three NZ-format variables.
      constexpr size_t EXAMPLE_INT8_SIZE = sizeof(int8_t);
      constexpr size_t EXAMPLE_BFLOAT16_SIZE = sizeof(int16_t);
      ret = TransToNZShape(weightDqShape, EXAMPLE_INT8_SIZE);
      CHECK_RET(ret == 0, LOG_PRINT("trans NZ shape failed.\n"); return ret);
      ret = TransToNZShape(weightUqQrShape, EXAMPLE_INT8_SIZE);
      CHECK_RET(ret == 0, LOG_PRINT("trans NZ shape failed.\n"); return ret);
      ret = TransToNZShape(weightDkvKrShape, EXAMPLE_INT8_SIZE);
      CHECK_RET(ret == 0, LOG_PRINT("trans NZ shape failed.\n"); return ret);

      // Create tokenX aclTensor.
      ret = CreateAclTensorND(tokenXShape, &tokenXDeviceAddr, &tokenXHostAddr, aclDataType::ACL_INT8, &tokenX);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create weightDq aclTensor.
      ret = CreateAclTensorNZ(weightDqShape, &weightDqDeviceAddr, &weightDqHostAddr, aclDataType::ACL_INT8, &weightDq);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create weightUqQr aclTensor.
      ret = CreateAclTensorNZ(weightUqQrShape, &weightUqQrDeviceAddr, &weightUqQrHostAddr, aclDataType::ACL_INT8, &weightUqQr);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create weightUk aclTensor.
      ret = CreateAclTensorND(weightUkShape, &weightUkDeviceAddr, &weightUkHostAddr, aclDataType::ACL_BF16, &weightUk);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create weightDkvKr aclTensor.
      ret = CreateAclTensorNZ(weightDkvKrShape, &weightDkvKrDeviceAddr, &weightDkvKrHostAddr, aclDataType::ACL_INT8, &weightDkvKr);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create rmsnormGammaCq aclTensor.
      ret = CreateAclTensorND(rmsnormGammaCqShape, &rmsnormGammaCqDeviceAddr, &rmsnormGammaCqHostAddr, aclDataType::ACL_BF16, &rmsnormGammaCq);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create rmsnormGammaCkv aclTensor.
      ret = CreateAclTensorND(rmsnormGammaCkvShape, &rmsnormGammaCkvDeviceAddr, &rmsnormGammaCkvHostAddr, aclDataType::ACL_BF16, &rmsnormGammaCkv);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create ropeSin aclTensor.
      ret = CreateAclTensorND(ropeSinShape, &ropeSinDeviceAddr, &ropeSinHostAddr, aclDataType::ACL_BF16, &ropeSin);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create ropeCos aclTensor.
      ret = CreateAclTensorND(ropeCosShape, &ropeCosDeviceAddr, &ropeCosHostAddr, aclDataType::ACL_BF16, &ropeCos);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create cacheIndex aclTensor.
      ret = CreateAclTensorND(cacheIndexShape, &cacheIndexDeviceAddr, &cacheIndexHostAddr, aclDataType::ACL_INT64, &cacheIndex);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create kvCache aclTensor.
      ret = CreateAclTensorND(kvCacheShape, &kvCacheDeviceAddr, &kvCacheHostAddr, aclDataType::ACL_INT8, &kvCache);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create krCache aclTensor.
      ret = CreateAclTensorND(krCacheShape, &krCacheDeviceAddr, &krCacheHostAddr, aclDataType::ACL_BF16, &krCache);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create dequantScaleX aclTensor.
      ret = CreateAclTensorND(dequantScaleXShape, &dequantScaleXDeviceAddr, &dequantScaleXHostAddr, aclDataType::ACL_FLOAT, &dequantScaleX);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create dequantScaleWDq aclTensor.
      ret = CreateAclTensorND(dequantScaleWDqShape, &dequantScaleWDqDeviceAddr, &dequantScaleWDqHostAddr, aclDataType::ACL_FLOAT, &dequantScaleWDq);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create dequantScaleWUqQr aclTensor.
      ret = CreateAclTensorND(dequantScaleWUqQrShape, &dequantScaleWUqQrDeviceAddr, &dequantScaleWUqQrHostAddr, aclDataType::ACL_FLOAT, &dequantScaleWUqQr);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create dequantScaleWDkvKr aclTensor.
      ret = CreateAclTensorND(dequantScaleWDkvKrShape, &dequantScaleWDkvKrDeviceAddr, &dequantScaleWDkvKrHostAddr, aclDataType::ACL_FLOAT, &dequantScaleWDkvKr);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create quantScaleCkv aclTensor.
      ret = CreateAclTensorND(quantScaleCkvShape, &quantScaleCkvDeviceAddr, &quantScaleCkvHostAddr, aclDataType::ACL_FLOAT, &quantScaleCkv);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create smoothScalesCq aclTensor.
      ret = CreateAclTensorND(smoothScalesCqShape, &smoothScalesCqDeviceAddr, &smoothScalesCqHostAddr, aclDataType::ACL_FLOAT, &smoothScalesCq);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create query aclTensor.
      ret = CreateAclTensorND(queryShape, &queryDeviceAddr, &queryHostAddr, aclDataType::ACL_INT8, &query);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create queryRope aclTensor.
      ret = CreateAclTensorND(queryRopeShape, &queryRopeDeviceAddr, &queryRopeHostAddr, aclDataType::ACL_BF16, &queryRope);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create dequantScaleQNope aclTensor.
      ret = CreateAclTensorND(dequantScaleQNopeShape, &dequantScaleQNopeDeviceAddr, &dequantScaleQNopeHostAddr, aclDataType::ACL_FLOAT, &dequantScaleQNope);
      CHECK_RET(ret == ACL_SUCCESS, return ret);

      // 3. Call the CANN operator library API, which needs to be changed to the specific API.
      uint64_t workspaceSize = 0;
      aclOpExecutor* executor = nullptr;
      // Call the first-phase API of aclnnMlaPrologV3WeightNz.
      ret = aclnnMlaPrologV3WeightNzGetWorkspaceSize(tokenX, weightDq, weightUqQr, weightUk, weightDkvKr, rmsnormGammaCq, rmsnormGammaCkv, ropeSin, ropeCos, kvCache, krCache, cacheIndex,
        dequantScaleX, dequantScaleWDq, dequantScaleWUqQr, dequantScaleWDkvKr, quantScaleCkv, nullptr, smoothScalesCq, nullptr, nullptr,rmsnormEpsilonCq, rmsnormEpsilonCkv, cacheMode,
        weightQuantMode, kvQuantMode, queryQuantMode, ckvkrRepoMode, quantScaleRepoMode, tileSize, qcQrScale, kcScale,
        query, queryRope, dequantScaleQNope, nullptr, nullptr, &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMlaPrologV3WeightNzGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
      // Allocate device memory based on the workspaceSize calculated by the first-phase API.
      void* workspaceAddr = nullptr;
      if (workspaceSize > static_cast<uint64_t>(0)) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
      }
      // Call the second-phase API of aclnnMlaPrologV3WeightNz.
      ret = aclnnMlaPrologV3WeightNz(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMlaPrologV3WeightNz failed. ERROR: %d\n", ret); return ret);

      // 4. Boilerplate, synchronously wait for the task execution to complete.
      ret = aclrtSynchronizeStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

      // 5. Obtain the output values and copy the results on the device to the host. Modify based on the specific API definition.
      auto size = GetShapeSize(queryShape);
      std::vector<float> resultData(size, 0);
      ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), queryDeviceAddr, size * sizeof(float),
                        ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
      for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
      }
      // 6. Release aclTensor and aclScalar. Modify based on the specific API definition.
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

The sample code for <term>Ascend 950PR&950DT products</term> is as follows, for reference only.

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
    // Boilerplate, resource initialization.
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
    // Call aclrtMalloc to allocate device memory.
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // Call aclrtMalloc to allocate host memory.
    ret = aclrtMallocHost(hostAddr, size);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMallocHost failed. ERROR: %d\n", ret); return ret);
    memset(*hostAddr, 0, size);
    // Call aclCreateTensor to create an aclTensor.
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, nullptr, 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    // Call aclrtMemcpy to copy host data to device memory.
    ret = aclrtMemcpy(*deviceAddr, size, *hostAddr, GetShapeSize(shape) * aclDataTypeSize(dataType), ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
    return 0;
}

template <typename T>
int CreateAclTensorNZ(const std::vector<T>& shape, void** deviceAddr, void** hostAddr,
                    aclDataType dataType, aclTensor** tensor) {
    auto size = GetShapeSize(shape) * sizeof(T);
    // Call aclrtMalloc to allocate device-side memory.
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // Call aclrtMalloc to allocate host-side memory.
    ret = aclrtMallocHost(hostAddr, size);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMallocHost failed. ERROR: %d\n", ret); return ret);
    memset(*hostAddr, 0, size);
    // Call the aclCreateTensor API to create an aclTensor.
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, nullptr, 0, aclFormat::ACL_FORMAT_FRACTAL_NZ,
                              shape.data(), shape.size(), *deviceAddr);
    // Call aclrtMemcpy to copy host-side data to device-side memory.
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
    // 1. Boilerplate. Initialize the device/stream. For details, see the AscendCL external API list.
    // Set deviceId based on the actual device.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // Handle the check as needed.
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    // 2. Construct the inputs and outputs according to the API definition.
    std::vector<int64_t> tokenXShape = {8, 1, 7168};            // B,S,He
    std::vector<int64_t> weightDqShape = {7168, 1536};          // He,Hcq
    std::vector<int64_t> weightUqQrShape = {1536, 24576};        // Hcq,N*(D+Dr)
    std::vector<int64_t> weightUkShape = {128, 128, 512};        // N,D,Hckv
    std::vector<int64_t> weightDkvKrShape = {7168, 576};        // He,Hckv+Dr
    std::vector<int64_t> rmsnormGammaCqShape = {1536};          // Hcq
    std::vector<int64_t> rmsnormGammaCkvShape = {512};          // Hckv
    std::vector<int64_t> ropeSinShape = {8, 1, 64};             // B,S,Dr
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

    // Convert the shapes of the three NZ-format variables.
    constexpr size_t EXAMPLE_INT8_SIZE = sizeof(int8_t);
    constexpr size_t EXAMPLE_BFLOAT16_SIZE = sizeof(int16_t);
    ret = TransToNZShape(weightDqShape, EXAMPLE_INT8_SIZE);
    CHECK_RET(ret == 0, LOG_PRINT("trans NZ shape failed.\n"); return ret);
    ret = TransToNZShape(weightUqQrShape, EXAMPLE_INT8_SIZE);
    CHECK_RET(ret == 0, LOG_PRINT("trans NZ shape failed.\n"); return ret);
    ret = TransToNZShape(weightDkvKrShape, EXAMPLE_INT8_SIZE);
    CHECK_RET(ret == 0, LOG_PRINT("trans NZ shape failed.\n"); return ret);

    // Create the tokenX aclTensor.
    ret = CreateAclTensorND(tokenXShape, &tokenXDeviceAddr, &tokenXHostAddr, aclDataType::ACL_FLOAT8_E4M3FN, &tokenX);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create weightDq aclTensor.
    ret = CreateAclTensorNZ(weightDqShape, &weightDqDeviceAddr, &weightDqHostAddr, aclDataType::ACL_FLOAT8_E4M3FN, &weightDq);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create weightUqQr aclTensor.
    ret = CreateAclTensorNZ(weightUqQrShape, &weightUqQrDeviceAddr, &weightUqQrHostAddr, aclDataType::ACL_FLOAT8_E4M3FN, &weightUqQr);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create weightUk aclTensor.
    ret = CreateAclTensorND(weightUkShape, &weightUkDeviceAddr, &weightUkHostAddr, aclDataType::ACL_BF16, &weightUk);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create weightDkvKr aclTensor.
    ret = CreateAclTensorNZ(weightDkvKrShape, &weightDkvKrDeviceAddr, &weightDkvKrHostAddr, aclDataType::ACL_FLOAT8_E4M3FN, &weightDkvKr);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create rmsnormGammaCq aclTensor.
    ret = CreateAclTensorND(rmsnormGammaCqShape, &rmsnormGammaCqDeviceAddr, &rmsnormGammaCqHostAddr, aclDataType::ACL_BF16, &rmsnormGammaCq);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create rmsnormGammaCkv aclTensor.
    ret = CreateAclTensorND(rmsnormGammaCkvShape, &rmsnormGammaCkvDeviceAddr, &rmsnormGammaCkvHostAddr, aclDataType::ACL_BF16, &rmsnormGammaCkv);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create ropeSin aclTensor.
    ret = CreateAclTensorND(ropeSinShape, &ropeSinDeviceAddr, &ropeSinHostAddr, aclDataType::ACL_BF16, &ropeSin);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create ropeCos aclTensor.
    ret = CreateAclTensorND(ropeCosShape, &ropeCosDeviceAddr, &ropeCosHostAddr, aclDataType::ACL_BF16, &ropeCos);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create cacheIndex aclTensor.
    ret = CreateAclTensorND(cacheIndexShape, &cacheIndexDeviceAddr, &cacheIndexHostAddr, aclDataType::ACL_INT64, &cacheIndex);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create kvCache aclTensor.
    ret = CreateAclTensorND(kvCacheShape, &kvCacheDeviceAddr, &kvCacheHostAddr, aclDataType::ACL_FLOAT8_E4M3FN, &kvCache);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create krCache aclTensor.
    ret = CreateAclTensorND(krCacheShape, &krCacheDeviceAddr, &krCacheHostAddr, aclDataType::ACL_BF16, &krCache);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create dequantScaleX aclTensor.
    ret = CreateAclTensorND(dequantScaleXShape, &dequantScaleXDeviceAddr, &dequantScaleXHostAddr, aclDataType::ACL_FLOAT8_E8M0, &dequantScaleX);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create dequantScaleWDq aclTensor.
    ret = CreateAclTensorND(dequantScaleWDqShape, &dequantScaleWDqDeviceAddr, &dequantScaleWDqHostAddr, aclDataType::ACL_FLOAT8_E8M0, &dequantScaleWDq);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create dequantScaleWUqQr aclTensor.
    ret = CreateAclTensorND(dequantScaleWUqQrShape, &dequantScaleWUqQrDeviceAddr, &dequantScaleWUqQrHostAddr, aclDataType::ACL_FLOAT8_E8M0, &dequantScaleWUqQr);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create dequantScaleWDkvKr aclTensor.
    ret = CreateAclTensorND(dequantScaleWDkvKrShape, &dequantScaleWDkvKrDeviceAddr, &dequantScaleWDkvKrHostAddr, aclDataType::ACL_FLOAT8_E8M0, &dequantScaleWDkvKr);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create quantScaleCkv aclTensor.
    ret = CreateAclTensorND(quantScaleCkvShape, &quantScaleCkvDeviceAddr, &quantScaleCkvHostAddr, aclDataType::ACL_FLOAT, &quantScaleCkv);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create query aclTensor.
    ret = CreateAclTensorND(queryShape, &queryDeviceAddr, &queryHostAddr, aclDataType::ACL_FLOAT8_E4M3FN, &query);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create queryRope aclTensor.
    ret = CreateAclTensorND(queryRopeShape, &queryRopeDeviceAddr, &queryRopeHostAddr, aclDataType::ACL_BF16, &queryRope);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create dequantScaleQNope aclTensor.
    ret = CreateAclTensorND(dequantScaleQNopeShape, &dequantScaleQNopeDeviceAddr, &dequantScaleQNopeHostAddr, aclDataType::ACL_FLOAT, &dequantScaleQNope);
    CHECK_RET(ret == ACL_SUCCESS, return ret);


    // 3. Call the CANN operator library API. Modify it to the specific API.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    // Call the first-phase API of aclnnMlaPrologV3WeightNz.
    ret = aclnnMlaPrologV3WeightNzGetWorkspaceSize(tokenX, weightDq, weightUqQr, weightUk, weightDkvKr, rmsnormGammaCq, rmsnormGammaCkv, ropeSin, ropeCos, kvCache, krCache, cacheIndex,
        dequantScaleX, dequantScaleWDq, dequantScaleWUqQr, dequantScaleWDkvKr, quantScaleCkv, nullptr, nullptr, nullptr, nullptr, rmsnormEpsilonCq, rmsnormEpsilonCkv, cacheMode,
        weightQuantMode, kvQuantMode, queryQuantMode, ckvkrRepoMode, quantScaleRepoMode, tileSize, qcQrScale, kcScale,
        query, queryRope, dequantScaleQNope, nullptr, nullptr, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMlaPrologV3WeightNzGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // Allocate device memory based on the workspaceSize calculated by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > static_cast<uint64_t>(0)) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    // Call the second-phase API of aclnnMlaPrologV3WeightNz.
    ret = aclnnMlaPrologV3WeightNz(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMlaPrologV3WeightNz failed. ERROR: %d\n", ret); return ret);

    // 4. Boilerplate: synchronize and wait for the task execution to complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. Obtain the output value and copy the result on the device to the host. Modify according to the specific API definition.
    auto size = GetShapeSize(queryShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), queryDeviceAddr, size * sizeof(float),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }
    // 6. Release aclTensor and aclScalar. Modify according to the specific API definition.
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
