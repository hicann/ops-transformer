# aclnnMlaPrologV4WeightNz

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-21T12:13:33.168Z -->

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

## Function

- **Function updates**: (Differences from aclnnMlaPrologV3weightNz)
    - Added the optional parameter doRope: controls whether to perform rotary position encoding on queryRopeOut and krCache. The default value is true. When doRope=true, non-empty valid ropeSin/ropeCos Tensors must be passed in. When doRope=false, RoPE is skipped and pass-through results are written; in this case, ropeSin/ropeCos must both be empty (nullptr or empty Tensors).
    - The range of N:Head-Num (number of heads) is adjusted to 1-128
- **Description**: For inference use cases, performs the pre-processing computation of Multi-Head Latent Attention. The main computation process is divided into seven paths:
    - First, the input $x$ is multiplied by $W^{DQ}$ for downsampling and RmsNorm and then split into two paths. In the first path, the result is multiplied by $W^{UQ}$ and $W^{UK}$ through two upsampling operations and then multiplied by the Query scale correction factor $\alpha_q$ to obtain $q^N$. In the second path, the result is multiplied by $W^{QR}$; when doRope=true, rotary position encoding is applied to obtain $q^R$; when doRope=false, $c^Q \cdot W^{QR}$ is directly written out.
    - In the third path, the input $x$ is multiplied by $W^{DKV}$ for downsampling and RmsNorm, then multiplied by the Key scale correction factor $\alpha_{kv}$ and written into the Cache to obtain $k^C$;
    - In the fourth path, after the input $x$ is multiplied by $W^{KR}$, rotary position encoding is applied when doRope=true and the result is written into the Cache to obtain $k^R$; when doRope=false, $x \cdot W^{KR}$ is directly written into the Cache;
    - The fifth path: the quantization parameter obtained after the output $q^N$ undergoes DynamicQuant.
    - In the sixth path, when queryNormFlag=true, the input $x$ is multiplied by $W^{DQ}$ for downsampling and RmsNorm to obtain the compressed result $c^Q$, which is output as queryNormOutOptional.
    - In the seventh path, when queryNormFlag=true and in the quantization scenario (weightQuantMode≠0), DynamicQuant is performed on $c^Q$ to finally obtain the quantized queryNormOutOptional and the corresponding quantization parameter dequantScaleQNormOutOptional.
    - The weight parameters WeightDq, WeightUqQr, and WeightDkvKr need to be passed in NZ format.

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
    where $\alpha_q$ is the scale correction parameter of the Query.

    Perform ROPE rotary position encoding on the Query (doRope=true)

    $$
    q^R = \mathrm{ROPE}(c^Q \cdot W^{QR})
    $$

    When doRope=false

    $$
    q^R = c^Q \cdot W^{QR}
    $$

    Key computation formula, including downsampling and RmsNorm, with the computation result stored in cache.

    $$
    c^{KV} = \alpha_{kv}\cdot\mathrm{RmsNorm}(x \cdot W^{DKV})
    $$

    $$
    k^C = \mathrm{Cache}(c^{KV})
    $$
    where $\alpha_{kv}$ is the scale correction parameter of the Key.

    Perform ROPE rotary position encoding on the Key, and store the result in the cache (doRope=true)

    $$
    k^R = \mathrm{Cache}(\mathrm{ROPE}(x \cdot W^{KR}))
    $$

    When doRope=false

    $$
    k^R = \mathrm{Cache}(x \cdot W^{KR})
    $$

    Formula of Dequant Scale Query Nope

    $$
    \mathrm{dequantScaleQNope} = {\mathrm{RowMax}(\mathrm{abs}(q^{N})) / Q_{max}}
    $$

    $$
    q^{N} = {\mathrm{round}(q^{N} / \mathrm{dequantScaleQNope})}
    $$

    Formula of Query Norm and Dequant Scale Query Norm (output when queryNormFlag=true)

    Non-quantization scenario (weightQuantMode=0):

    $$
    queryNorm = c^Q = \alpha_q\cdot\mathrm{RmsNorm}(x \cdot W^{DQ})
    $$

    In this case, queryNorm is of the BFLOAT16 type, and dequantScaleQNorm is not output (nullptr).

    Quantization scenario (weightQuantMode=1/2/4/5, per-token dynamic quantization). smoothScalesCq is optional. If not passed in, it is treated as 1:

    $$
    \tilde{c}^Q = c^Q \cdot smoothScale
    $$

    $$
    \mathrm{dequantScaleQNorm} = {\mathrm{RowMax}(\mathrm{abs}(\tilde{c}^Q)) / Q_{max}}
    $$

    $$
    queryNorm = {\mathrm{round}(\tilde{c}^Q / \mathrm{dequantScaleQNorm})}
    $$

    mxfp8 quantization scenario (weightQuantMode=3, pergroup dynamic quantization with every 32 elements as a group, where the quantization parameter is of the FLOAT8_E8M0 type):

    $$
    \mathrm{dequantScaleQNorm} = 2^{\lfloor \mathrm{log}_2(\mathrm{GroupMax}(\mathrm{abs}(c^Q))) \rfloor - 8}
    $$

    $$
    queryNorm = {\mathrm{cast\_to\_fp8}(c^Q / \mathrm{dequantScaleQNorm})}
    $$

    where $Q_{max}$ is the maximum value of the quantized output type: 127 for INT8, 448 for FLOAT8_E4M3FN, and 32768 for HIFLOAT8. 8 is emax, the maximum value of the exponent bits of FLOAT8_E4M3FN.

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). You must first call the "aclnnMlaPrologV4WeightNzGetWorkspaceSize" API to obtain the input parameters and calculate the required workspace size based on the process, and then call the "aclnnMlaPrologV4WeightNz" API to perform the computation.

```cpp
aclnnStatus aclnnMlaPrologV4WeightNzGetWorkspaceSize(
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
    bool             doRope,
    const aclTensor *queryOut,
    const aclTensor *queryRopeOut,
    const aclTensor *dequantScaleQNopeOutOptional,
    const aclTensor *queryNormOutOptional,
    const aclTensor *dequantScaleQNormOutOptional,
    uint64_t        *workspaceSize,
    aclOpExecutor  **executor)
```

```cpp
aclnnStatus aclnnMlaPrologV4WeightNz(
  void             *workspace,
  uint64_t          workspaceSize,
  aclOpExecutor    *executor,
  const aclrtStream stream)
```

## aclnnMlaPrologV4WeightNzGetWorkspaceSize

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
        <td>Input tensor used in the formula to calculate Query and Key.</td>
        <td>Supports empty tensors with B=0, S=0, T=0</td>
        <td>BFLOAT16, FLOAT8_E4M3FN, INT8, HIFLOAT8</td>
        <td>ND</td>
        <td>          <ul>
            <li>BS merged: (T, He)</li>
            <li>BS unmerged: (B, S, He)</li>
          </ul>
        </td>
        <td>×</td>
      </tr>
      <tr>
        <td>weightDq</td>
        <td>Input</td>
        <td>Downsampling weight matrix <i>W<sup><i>DQ</i></sup></i> used in the formula to calculate Query.<br>Without transposing, each dimension is represented as: (k, n)</td>
        <td>Does not support empty tensors.</td>
        <td>BFLOAT16, FLOAT8_E4M3FN, INT8, HIFLOAT8</td>
        <td>FRACTAL_NZ</td>
        <td>(He, Hcq)</td>
        <td>×</td>
      </tr>
      <tr>
        <td>weightUqQr</td>
        <td>Input</td>
        <td>Upsampling weight matrix <i>W<sup><i>UQ</i></sup></i> and position encoding weight matrix <i>W<sup><i>QR</i></sup></i> used in the formula to compute the Query.<br>Representation of each dimension without transposing: (k, n)</td>
        <td>Does not support empty tensors.</td>
        <td>BFLOAT16, INT8, FLOAT8_E4M3FN, HIFLOAT8</td>
        <td>FRACTAL_NZ</td>
        <td>(Hcq, N*(D+Dr))</td>
        <td>×</td>
      </tr>
      <tr>
        <td>weightUk</td>
        <td>Input</td>
        <td>Upsampling weight <i>W<sup><i>UK</i></sup></i> used in the formula to calculate Key.</td>
        <td>Does not support empty tensors</td>
        <td>BFLOAT16</td>
        <td>ND</td>
        <td>(N, D, Hckv)</td>
        <td>×</td>
      </tr>
      <tr>
        <td>weightDkvKr</td>
        <td>Input</td>
        <td>Downsampling weight matrix <i>W<sup><i>DKV</i></sup></i> of Key and position encoding weight matrix <i>W<sup><i>KR</i></sup></i> used in the formula for calculation.<br>Each dimension without transposing is represented as: (k, n)</td>
        <td>Empty tensors are not supported.</td>
        <td>BFLOAT16, FLOAT8_E4M3FN, INT8, HIFLOAT8</td>
        <td>FRACTAL_NZ</td>
        <td>(He,Hckv+Dr)</td>
        <td>×</td>
      </tr>
      <tr>
        <td>rmsnormGammaCq</td>
        <td>Input</td>
        <td>The <i>γ</i> parameter in the RmsNorm formula for computing <i>c<sup><i>Q</i></sup></i>.</td>
        <td>Does not support empty tensors.</td>
        <td>BFLOAT16</td>
        <td>ND</td>
        <td>(Hcq)</td>
        <td>×</td>
      </tr>
      <tr>
        <td>rmsnormGammaCkv</td>
        <td>Input</td>
        <td>The <i>γ</i> parameter in the RmsNorm formula for computing <i>c<sup><i>KV</i></sup></i>.</td>
        <td>Does not support empty tensor.</td>
        <td>BFLOAT16</td>
        <td>ND</td>
        <td>(Hckv)</td>
        <td>×</td>
      </tr>
      <tr>
        <td>ropeSin</td>
        <td>Input</td>
        <td>Sine parameter matrix used to calculate rotary position encoding.</td>
        <td>Mandatory when doRope=true. Empty tensors with B=0, S=0, and T=0 are supported (indicating that the dimension length is 0); when doRope=false, must simultaneously be empty (nullptr).</td>
        <td>BFLOAT16</td>
        <td>ND</td>
        <td>          <ul>
            <li>BS axis-merged: (T,Dr)</li>
            <li>BS axis-unmerged: (B,S,Dr)</li>
          </ul>
        </td>
        <td>×</td>
      </tr>
      <tr>
        <td>ropeCos</td>
        <td>Input</td>
        <td>Cosine parameter matrix used to calculate rotary position encoding.</td>
        <td>Mandatory when doRope=true. Supports empty tensors with B=0, S=0, T=0 (indicating that the dimension length is 0); when doRope=false, must simultaneously be empty (nullptr)</td>
        <td>BFLOAT16</td>
        <td>ND</td>
        <td>          <ul>
            <li>BS axis-merged: (T,Dr)</li>
            <li>BS axis-unmerged: (B,S,Dr)</li>
          </ul>
        </td>
        <td></td>
      </tr>
      <tr>
        <td>kvCacheRef</td>
        <td>Input/Output</td>
        <td>aclTensor used for cache indexing. The computation result is updated in place (corresponding to <i>k<sup><i>C</i></sup></i> in the formula).</td>
        <td>        <ul style="margin:0; padding-left:20px;">
            <li>Supports empty tensors with B=0, Skv=0.</li>
            <li>Nkv is associated with N, and N is a hyperparameter. Therefore, Nkv=0 is not supported.</li>
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
        <td>Only the first axis supports non-contiguity; all other axes must be contiguous.</td>
      </tr>
      <tr>
        <td>krCacheRef</td>
        <td>Input/Output</td>
        <td>Cache for key position encoding. The computation result is updated in place (corresponding to <i>k<sup><i>R</i></sup></i> in the formula).</td>
        <td>        <ul style="margin:0; padding-left:20px;">
            <li>Supports empty tensors with B=0, Skv=0;</li>
            <li>Nkv is associated with N, and N is a hyperparameter. Therefore, 0 is not supported for Nkv</li>
            <li>When doRope=false: RoPE is skipped and <i>x·W<sup>KR</sup></i> is directly written to the cache. The shape is not affected</li>
        </ul>
        </td>
        <td>BFLOAT16, INT8</td>
        <td>ND</td>
        <td>          <ul>
            <li>CacheMode="PA_BSND"/"PA_NZ"/"PA_BLK_BSND"/"PA_BLK_NZ": (BlockNum,BlockSize,Nkv,Dr)</li>
            <li>CacheMode="BSND": (B,S,Nkv,Dr)</li>
            <li>When CacheMode="TND": (T,Nkv,Dr)</li>
            <li>When ckvkrRepoMode=1: the dimensions should contain 0, and shape (0) is supported</li>
          </ul>
        </td>
        <td>Only the first axis can be non-contiguous; all other axes must be contiguous.</td>
      </tr>
      <tr>
        <td>cacheIndexOptional</td>
        <td>Input</td>
        <td>Indexes used to store kvCache and krCache.</td>
        <td>        <ul>
            <li>Supports empty tensors with B=0, S=0, T=0.</li>
            <li>cacheMode="PA_BSND"/"PA_NZ": value range must be within [0,BlockNum*BlockSize)</li>
            <li>cacheMode="PA_BLK_BSND"/"PA_BLK_NZ": value range must be within [0,BlockNum)</li>
            <li>cacheMode="TND"/"BSND": nullptr</li>
        </ul>
        </td>
        <td>INT64</td>
        <td>ND</td>
        <td>          <ul>
            <li>CacheMode="PA_BSND"/"PA_NZ":
              <ol>
                <li>Merged BS axes: (T)</li>
                <li>Unmerged BS axes: (B,S)</li>
              </ol>
            </li>
            <li>CacheMode="PA_BLK_BSND"/"PA_BLK_NZ":
              <ol>
                <li>Merged BS axes: (Sum(⌈S<sub>i</sub>/BlockSize⌉)), where S<sub>i</sub> is the length of S in each batch</li>
                <li>Unmerged BS axes: (B,⌈S/BlockSize⌉)</li>
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
        <td>Supports empty tensors with B=0, S=0, T=0 (needs to be passed in for weightQuantMode=2/3/4/5 use cases)</td>
        <td>FLOAT8_E8M0, FLOAT</td>
        <td>ND</td>
        <td>          <ul>
            <li>weightQuantMode=2/4/5:
              <ol>
                <li>With BS axis merging: (T, 1)</li>
                <li>Without BS axis merging: (B*S,1)</li>
              </ol>
            </li>
            <li>weightQuantMode=3:
              <ol>
                <li>With BS axis merging: (T, He/32)</li>
                <li>Without BS axis merging: (B*S, He/32)</li>
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
        <td>Supports non-empty Tensor (the weightQuantMode=2/3/4/5 scenario requires this parameter)</td>
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
        <td>Per-channel parameter for dequantization after the MatmulQcQr matrix multiplication.</td>
        <td>Supports non-empty tensors (needs to be passed in for weightQuantMode=1/2/3/4/5 scenarios).</td>
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
        <td>Supports non-empty Tensor (the weightQuantMode=2/3/4/5 scenario requires passing it in)</td>
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
        <td>Parameter for the quantization operation on kvCache output data.</td>
        <td>Supports non-empty Tensor (needs to be passed in for the scenario where kvCacheQuantMode=1/2)</td>
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
        <td>Parameters for the quantization operation on the krCache output data.</td>
        <td>Supports non-empty Tensor (needs to be passed in for the scenario where kvCacheQuantMode=2)</td>
        <td>FLOAT</td>
        <td>ND</td>
        <td>(1,Dr)</td>
        <td>×</td>
      </tr>
      <tr>
        <td>smoothScalesCqOptional</td>
        <td>Input</td>
        <td>Parameter for performing dynamic quantization on the RmsNormCq output.</td>
        <td>Supports non-empty tensor (optional for weightQuantMode=1/2/4/5 use cases)</td>
        <td>FLOAT</td>
        <td>ND</td>
        <td>(1,Hcq)</td>
        <td>×</td>
      </tr>
      <tr>
        <td>actualSeqLenOptional</td>
        <td>Input</td>
        <td>Indicates the sequence length in each batch, stored as prefix sums.</td>
        <td>Needs to be passed in when BS axes are fused and CacheMode="PA_BLK_BSND"/"PA_BLK_NZ"</td>
        <td>INT32</td>
        <td>ND</td>
        <td>(B)</td>
        <td>×</td>
      </tr>
      <tr>
        <td>kNopeClipAlphaOptional</td>
        <td>Input</td>
        <td>Represents the scaling factor for the clip operation on kvCache.</td>
        <td>The shape is 1 in the partially quantized pertoken-pergroup scenario and the int8 fully quantized pertoken-pergroup scenario, and is not supported in other scenarios. Empty Tensor is not supported.</td>
        <td>FLOAT</td>
        <td>ND</td>
        <td>(1)</td>
        <td>×</td>
      </tr>
      <tr>
        <td>rmsnormEpsilonCq</td>
        <td>Input</td>
        <td>The <i>ε</i> parameter in the RmsNorm formula for calculating <i>c<sup><i>Q</i></sup></i>.</td>
        <td>        <ul>
            <li>If the user does not specify, it is recommended to pass in 1e-05.</li>
            <li>Only the double data type is supported.</li>
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
        <td>The <i>ε</i> parameter in the RmsNorm formula for calculating <i>c<sup><i>KV</i></sup></i>.</td>
        <td>        <ul>
            <li>If the user does not specify, it is recommended to pass in 1e-05.</li>
            <li>Only the double data type is supported.</li>
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
            <li>If the user does not specify intentionally, it is recommended to pass in "PA_BSND"</li>
            <li>Only the char* data type is supported</li>
            <li>Optional values: "PA_BSND", "PA_NZ", "PA_BLK_BSND", "PA_BLK_NZ", "BSND", "TND"</li>
        </ul>
        </td>
        <td>CHAR*</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>queryNormFlag</td>
        <td>Input</td>
        <td>Indicates whether to output queryNormOutOptional and dequantScaleQNormOutOptional.</td>
        <td>false indicates no output, true indicates output. Default value is false</td>
        <td>BOOL</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>weightQuantMode</td>
        <td>Input</td>
        <td>Indicates the quantization mode of weightDq, weightUqQr, weightUk, and weightDkvKr.</td>
        <td>        <ul>
            <li>0 indicates non-quantized.</li>
            <li>1 indicates weightUqQr quantized.</li>
            <li>2 indicates weightDq, weightUqQr, and weightDkvKr int8 quantized.</li>
            <li>3 indicates weightDq, weightUqQr, and weightDkvKr mxfp8 quantized.</li>
            <li>4 indicates weightDq, weightUqQr, and weightDkvKr fp8 quantized.</li>
            <li>5 indicates weightDq, weightUqQr, and weightDkvKr hif8 quantized.</li>
            <li>The default value is 0.</li>
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
        <td>0 indicates non-quantized, 1 indicates pertensor quantization, 2 indicates perchannel quantization, 3 indicates pertoken-pergroup quantization, default value is 0</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>queryQuantMode</td>
        <td>Input</td>
        <td>Indicates the quantization mode of query.</td>
        <td>0 indicates non-quantized, 1 indicates per-token-head quantization, default value is 0</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>ckvkrRepoMode</td>
        <td>Input</td>
        <td>Indicates the storage mode of kvCache and krCache.</td>
        <td>0 indicates that kvCache and krCache are stored separately, and 1 indicates that kvCache and krCache are stored together. The default value is 0.</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>quantScaleRepoMode</td>
        <td>Input</td>
        <td>Indicates the storage mode of the quantization scale.</td>
        <td>0 indicates that the quantization scale and data are stored separately, and 1 indicates that the quantization scale and data are merged and stored, then output as kvCacheRef. Default value is 0.</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>tileSize</td>
        <td>Input</td>
        <td>Indicates the size of each tile during pertoken-pergroup quantization. Valid only when kvCacheQuantMode is 3.</td>
        <td>Default value is 128</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>qcQrScale</td>
        <td>Input</td>
        <td>Indicates the scale correction coefficient of Query.</td>
        <td>Needs to be passed in as 1.0 when the user does not specify it.</td>
        <td>DOUBLE</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>kcScale</td>
        <td>Input</td>
        <td>Indicates the scale correction coefficient of Key.</td>
        <td>Needs to be passed in as 1.0 when the user does not specify it.</td>
        <td>DOUBLE</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>doRope</td>
        <td>Input</td>
        <td>Specifies whether to perform rotary position encoding (RoPE) on queryRopeOut and krCache.</td>
        <td><ul>
          <li>Default true: RoPE is performed. In this case, ropeSin/ropeCos must be non-empty valid tensors.</li>
          <li>false: RoPE is skipped and the pass-through result is written. In this case, ropeSin/ropeCos must both be empty (nullptr).</li>
        </ul></td>
        <td>BOOL</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>queryOut</td>
        <td>Output</td>
        <td>Output tensor of Query in the Formula (corresponding to <i>q<sup><i>N</i></sup></i>).</td>
        <td>-</td>
        <td>BFLOAT16, FLOAT8_E4M3FN, INT8, HIFLOAT8</td>
        <td>ND</td>
        <td>          <ul>
            <li>BS merged axes: (T,N,Hckv)</li>
            <li>BS non-merged axes: (B,S,N,Hckv)</li>
          </ul>
        </td>
        <td>×</td>
      </tr>
      <tr>
        <td>queryRopeOut</td>
        <td>Output</td>
        <td>The output tensor of the Query position encoding in the formula (corresponding to <i>q<sup><i>R</i></sup></i>).</td>
        <td>When doRope=false: RoPE is skipped and the pass-through result of <i>c<sup>Q</sup>·W<sup>QR</sup></i> is output directly. The shape is not affected</td>
        <td>BFLOAT16</td>
        <td>ND</td>
        <td>          <ul>
            <li>BS merged axes: (T,N,Dr)</li>
            <li>BS non-merged axes: (B,S,N,Dr)</li>
          </ul>
        </td>
        <td>×</td>
      </tr>
      <tr>
        <td>dequantScaleQNopeOutOptional</td>
        <td>Output</td>
        <td>The dequantization parameter of the Query output in the formula.</td>
        <td>Output when weightQuantMode=2/3/4/5; nullptr when weightQuantMode=0/1</td>
        <td>FLOAT</td>
        <td>ND</td>
        <td>          <ul>
            <li>BS axes merged: (T,N,1)</li>
            <li>BS axes not merged: (B*S,N,1)</li>
          </ul>
        </td>
        <td>×</td>
      </tr>
      <tr>
        <td>queryNormOutOptional</td>
        <td>Output</td>
        <td>Output tensor after rmsNorm is performed on tokenX in the formula (corresponding to <i>c<sup><i>Q</i></sup></i>).</td>
        <td>Output when queryNormFlag=true</td>
        <td>BFLOAT16, INT8, FLOAT8_E4M3FN, HIFLOAT8</td>
        <td>ND</td>
        <td>          <ul>
            <li>BS axes merged: (T,Hcq)</li>
            <li>BS axes not merged: (B,S,Hcq)</li>
          </ul>
        </td>
        <td>×</td>
      </tr>
      <tr>
        <td>dequantScaleQNormOutOptional</td>
        <td>Output</td>
        <td>Dequantization parameter of queryNormOutOptional.</td>
        <td>Output when queryNormFlag=true and weightQuantMode=1/2/3/4/5; nullptr when weightQuantMode=0</td>
        <td>FLOAT, FLOAT8_E8M0</td>
        <td>ND</td>
        <td>          <ul>
            <li>weightQuantMode=1/2/4/5:
              <ol>
                <li>BS axes merged: (T,1)</li>
                <li>BS axes not merged: (B*S,1)</li>
              </ol>
            </li>
            <li>weightQuantMode=3:
              <ol>
                <li>BS axes merged: (T, Hcq/32)</li>
                <li>BS axes not merged: (B*S, Hcq/32)</li>
              </ol>
            </li>
          </ul>
        </td>
        <td>×</td>
      </tr>
      <tr>
        <td>workspaceSize</td>
        <td>Output</td>
        <td>Returns the workspace size to be applied on the Device side.</td>
        <td>Used only for output. No input configuration is required. Data type is uint64_t*.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>executor</td>
        <td>Output</td>
        <td>Returns an op executor containing the operator computation flow.</td>
        <td>Used only for output. No input configuration is required. Data type: aclOpExecutor**</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
  </table>

<!-- npu="A3,910b" id7 -->
- <term>Atlas A3 products</term>,<term>Atlas A3 products</term>:
  - The FLOAT8_E4M3FN and HIFLOAT8 data types are not supported for tokenX, weightDq, weightUqQr, weightDkvKr, kvCacheRef, queryOut, or queryNormOutOptional.
  - The FLOAT8_E8M0 data type is not supported for dequantScaleXOptional, dequantScaleWDqOptional, dequantScaleWUqQrOptional, dequantScaleWDkvKrOptional, or dequantScaleQNormOutOptional.
<!-- end id7 -->

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
          <th>Return Value</th>
          <th>Error Code</th>
          <th>Description</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>ACLNN_ERR_PARAM_NULLPTR</td>
          <td>161001</td>
          <td>A null pointer exists in a required parameter (such as an input or output parameter that the API core depends on).</td>
        </tr>
        <tr>
          <td>ACLNN_ERR_PARAM_INVALID</td>
          <td>161002</td>
          <td>The shape (dimension/size) or dtype (data type) of an input parameter is not within the range supported by the API.</td>
        </tr>
        <tr>
          <td>ACLNN_ERR_RUNTIME_ERROR</td>
          <td>361001</td>
          <td>An exception occurs when the API calls NPU Runtime APIs for memory operations (for example, the Runtime service is not started, or memory allocation fails).</td>
        </tr>
        <tr>
          <td>ACLNN_ERR_INNER_TILING_ERROR</td>
          <td>561002</td>
          <td>An exception occurs during tiling. The input dtype or shape is incorrect.</td>
        </tr>
      </tbody>
    </table>

## aclnnMlaPrologV4WeightNz

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
          <td>Memory address of the workspace applied on the Device side.</td>
        </tr>
        <tr>
          <td>workspaceSize</td>
          <td>uint64_t</td>
          <td>Size of the workspace applied on the Device side, obtained by the first-phase API aclnnMlaPrologV4WeightNzGetWorkspaceSize.</td>
        </tr>
        <tr>
          <td>executor</td>
          <td>aclOpExecutor*</td>
          <td>Op executor, containing the computation flow of the operator.</td>
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
  - aclnnMlaPrologV4WeightNz defaults to a deterministic implementation.

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
        <td>Batch (batch size of input samples)</td>
        <td>Value range: 0-65536</td>
      </tr>
      <tr>
        <td>S</td>
        <td>Seq-Length (sequence length of input samples)</td>
        <td>Value range: unlimited</td>
      </tr>
      <tr>
        <td>He</td>
        <td>Head-Size (hidden layer size)</td>
        <td>Fixed value: 1024, 2048, 3072, 4096, 5120, 6144, 7168, 7680, 8192</td>
      </tr>
      <tr>
        <td>Hcq</td>
        <td>Dimension of the low-rank q matrix</td>
        <td>Fixed value: 1536, 2048</td>
      </tr>
      <tr>
        <td>N</td>
        <td>Head-Num (number of heads)</td>
        <td>Value range: 1-128</td>
      </tr>
      <tr>
        <td>Hckv</td>
        <td>Dimension of the low-rank kv matrix</td>
        <td>Fixed value: 512</td>
      </tr>
      <tr>
        <td>D</td>
        <td>qk dimension excluding position encoding</td>
        <td>Fixed value: 128, 192</td>
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
            <li>When CacheMode="PA_BSND"/"PA_NZ", the value must be greater than or equal to ⌈(B * S) / BlockSize⌉.</li>
            <li>When CacheMode="PA_BLK_BSND"/"PA_BLK_NZ", the value must be greater than or equal to B * ⌈S / BlockSize⌉.</li>
          </ul>
          <p>Note: In the BS axis fusion use case, the S length of each Batch can be different. Therefore, the value of BlockNum must be greater than or equal to the sum of the values obtained by rounding up the result of dividing the S length of each Batch by BlockSize.</p>
          <p>Example: If the value of actualSeqLenOptional is [47, 151, 261, 422] and blocksize=128, the lengths in each Batch are [47, 104, 110, 161] respectively. In this case, BlockNum = ⌈47/128⌉ + ⌈104/128⌉ + ⌈110/128⌉ + ⌈161/128⌉ = 5.</p>
        </td>
      </tr>
      <tr>
        <td>BlockSize</td>
        <td>Block size in the PagedAttention use case</td>
        <td>Value range: 16-1024, and the value must be a multiple of 16</td>
      </tr>
      <tr>
        <td>T</td>
        <td>Size after BS axis fusion</td>
        <td>
          <ul>
            <li>Value range: unlimited</li>
            <li>Note: If BS axis fusion is used, tokenX is 2-dimensional, cacheIndex is 1-dimensional, and queryOut and queryRopeOut are 3-dimensional. When doRope=true, ropeSin and ropeCos are 2-dimensional. When doRope=false, ropeSin and ropeCos must simultaneously be empty.</li>
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
    - When doRope=true, the shapes of ropeSin and ropeCos are (T, Dr); when doRope=false, ropeSin and ropeCos must simultaneously be empty.
    - When CacheMode is PA_BSND or PA_NZ, the shape of cacheIndex is (T).
    - When CacheMode is PA_BLK_BSND or PA_BLK_NZ, the shape of cacheIndex is (Sum(⌈<i>S<sub>i</sub></i>/BlockSize⌉)), where S_i is the length of S in each batch.
    - When CacheMode is PA_BLK_BSND or PA_BLK_NZ, actualSeqLenOptional needs to be passed in, with dimension (B).
    - In the int8/fp8/hif8 fully quantized field, the shape of dequantScaleXOptional is (T, 1); in the mxfp8 fully quantized field, the shape of dequantScaleXOptional is (T, He/32).
    - The shape of queryOut is (T, N, Hckv).
    - The shape of queryRopeOut is (T, N, Dr).
    - In the int8/mxfp8/fp8/hif8 fully quantized field with kvCache pertensor quantization, the shape of dequantScaleQNopeOutOptional is (T, N, 1); in other scenarios, it is nullptr.
  - If the dimension of tokenX does not use BS concatenated axis, i.e., (B, S, He)
      - When doRope=true, the shapes of ropeSin and ropeCos are (B, S, Dr); when doRope=false, ropeSin and ropeCos must simultaneously be empty.
      - When CacheMode is PA_BSND or PA_NZ, the shape of cacheIndex is (B, S).
      - When CacheMode is PA_BLK_BSND or PA_BLK_NZ, the shape of cacheIndex is (B, ⌈S/BlockSize⌉).
      - In the int8/fp8/hif8 fully quantized scenario, the shape of dequantScaleXOptional is (B\*S, 1); in the mxfp8 fully quantized scenario, the shape of dequantScaleXOptional is (B*S, He/32)
      - The shape of queryOut is (B, S, N, Hckv)
      - The shape of queryRopeOut is (B, S, N, Dr).
      - In the int8/mxfp8/fp8/hif8 fully quantized and kvCache pertensor quantization scenario, the shape of dequantScaleQNopeOutOptional is (B*S, N, 1); in other scenarios, it is nullptr
  - One or more of B, S, T, and Skv may be 0, meaning that input parameters whose shapes are related to B, S, T, or Skv may be null tensors. The remaining input parameters do not support null tensors.
      - If B, S, and T are 0, queryOut and queryRopeOut output null tensors, and kvCacheRef and krCacheRef are not updated.
      - If Skv is 0, queryOut, queryRopeOut, and dequantScaleQNopeOutOptional are computed normally, while kvCacheRef and krCacheRef are not updated, i.e., they output null tensors.
  - When CacheMode is BSND
      - tokenX should not use BS axis merging, that is, the dimensions are (B, S, He)
      - The dimensions of kvCache are (B,S,Nkv,Dr)
  - When CacheMode is TND
      - tokenX should use BS axis merging, that is, the dimensions are (T, He)
      - The dimensions of kvCache are (T,Nkv,Dr)
  - When ckvkrRepoMode is 1
      - The dimensions of krCache should include 0, and the supported shape is (0)

  </details>

  <details>
  <summary><a id="specialConstraint"></a>Special constraints</summary>

  - When actualSeqLenOptional is passed in, its last value must be consistent with T.
  - When doRope=false, ropeSin and ropeCos must simultaneously be empty (nullptr).
  - In pertoken-pergroup quantization mode, ckvkrRepoMode and quantScaleRepoMode must simultaneously be 1; in other quantization modes and the non-quantized field, ckvkrRepoMode and quantScaleRepoMode must simultaneously be 0.
  - In pertoken-pergroup quantization mode, CacheMode supports only PA_BSND, BSND, and TND.
  - When the value of ckvkrRepoMode is 1, krCache must be an empty tensor (that is, the product of its shape is 0).
  - In kvcache pertensor quantization mode, kvCacheQuantMode and queryQuantMode must simultaneously be 1.

  </details>

  <details>
  <summary><a id="SupportScenes"></a>Scenarios supported by the aclnnMlaPrologV4WeightNz API</summary>
  <ul>
    <li><term>Atlas A2 products</term> and <term>Atlas A3 products</term>: do not currently support fp8/hif8/mxfp8 fully quantized fields</li>
    <li><term>Ascend 950PR&950DT products</term>: currently supports all quantization scenarios</li>
  </ul>
  <table style="table-layout: auto;" border="1">
    <tr>
      <th colspan="2">Use Case</th>
      <th>Meaning</th>
    </tr>
    <tr>
      <td colspan="2">Non-quantized</td>
      <td>          weight_quant_mode=0,kv_cache_quant_mode=0,query_quant_mode=0<br>
          Input: all input parameters are non-quantized data <br>
          Output: all output parameters are non-quantized data
      </td>
    </tr>
    <tr>
      <td rowspan="3">Partial quantization</td>
      <td>Non-quantized kvCache </td>
      <td>          weight_quant_mode=1,kv_cache_quant_mode=0,query_quant_mode=0<br>
          Input: pertoken quantized data is passed in for weightUqQr, and the remaining input parameters are all non-quantized data. The dequantScaleWUqQr field must be passed in, and the smoothScalesCq field is optional <br>
          Output: all output parameters return non-quantized data
      </td>
    </tr>
    <tr>
      <td>kvCache perchannel quantization </td>
      <td>          weight_quant_mode=1,kv_cache_quant_mode=2,query_quant_mode=0<br>
          Input: pertoken quantized data is passed in for weightUqQr, perchannel quantized data is passed in for kvCacheRef and krCacheRef, and the remaining input parameters are all non-quantized data. The dequantScaleWUqQr, quantScaleCkv, and quant_scale_ckr fields must be passed in, and the smoothScalesCq field is optional <br>
          Output: kvCacheRef and krCacheRef return perchannel quantized data, and the remaining output parameters return non-quantized data
      </td>
    </tr>
    <tr>
      <td>kvCache pertoken-pergroup quantization </td>
      <td>          weight_quant_mode=1,kv_cache_quant_mode=3,query_quant_mode=0<br>
          Input parameters: weightUqQr takes pertoken quantized data, and remaining input parameters are all non-quantized data. dequantScaleWUqQr field must be passed in, and smoothScalesCq field is optional. <br>
          Output parameters: kvCacheRef returns pertoken-pergroup quantized data, and remaining output parameters return non-quantized data.
      </td>
    </tr>
    <tr>
      <td rowspan="3">int8/fp8/hif8 full quantization</td>
      <td> Non-quantized kvCache</td>
      <td>          weight_quant_mode=2/4/5,kv_cache_quant_mode=0,query_quant_mode=0<br>
          Input parameters: tokenX takes pertoken quantized data, weightDq, weightUqQr, and weightDkvKr take perchannel quantized data, and remaining input parameters are all non-quantized data. dequantScaleX, dequantScaleWDq, dequantScaleWUqQr, and dequantScaleWDkvKr fields must be passed in, and smoothScalesCq field is optional. <br>
          Output parameters: all output parameters return non-quantized data.
      </td>
    </tr>
    <tr>
      <td> kvCache pertensor quantization </td>
      <td>          weight_quant_mode=2/4/5,kv_cache_quant_mode=1,query_quant_mode=1<br>
          Input parameters: tokenX takes pertoken quantized data, weightDq, weightUqQr, and weightDkvKr take perchannel quantized data, kvCacheRef takes pertensor quantized data, and remaining input parameters are all non-quantized data. dequantScaleX, dequantScaleWDq, dequantScaleWUqQr, dequantScaleWDkvKr, and quantScaleCkv fields must be passed in, and smoothScalesCq field is optional. <br>
          Output parameters: queryOut returns per-token-head quantized data, kvCacheRef output parameter returns pertensor quantized data, and remaining output parameters return non-quantized data.
      </td>
    </tr>
    <tr>
      <td> kvCache pertoken-pergroup quantization </td>
      <td>          weight_quant_mode=2/4/5,kv_cache_quant_mode=3,query_quant_mode=0<br>
          Input parameters: tokenX takes pertoken quantized data, weightDq, weightUqQr, and weightDkvKr take perchannel quantized data, and remaining input parameters are all non-quantized data. dequantScaleX, dequantScaleWDq, dequantScaleWUqQr, and dequantScaleWDkvKr fields must be passed in, and smoothScalesCq field is optional. <br>
          Output parameters: kvCacheRef output parameter returns pertoken-pergroup quantized data, and remaining output parameters return non-quantized data.
      </td>
    </tr>
    <tr>
      <td rowspan="3">mxfp8 full quantization</td>
      <td> Non-quantized kvCache </td>
      <td>          weight_quant_mode=3,kv_cache_quant_mode=0,query_quant_mode=0<br>
          Input parameters: tokenX uses pertoken quantized data, weightDq, weightUqQr, and weightDkvKr use perchannel quantized data, and the remaining input parameters all are non-quantized data. The dequantScaleX, dequantScaleWDq, dequantScaleWUqQr, and dequantScaleWDkvKr fields must be passed in <br>
          Output parameters: all output parameters return non-quantized data
      </td>
    </tr>
    <tr>
      <td> kvCache pertensor quantization </td>
      <td>          weight_quant_mode=3,kv_cache_quant_mode=1,query_quant_mode=1<br>
          Input parameters: tokenX uses pertoken quantized data, weightDq, weightUqQr, and weightDkvKr use perchannel quantized data, kvCacheRef uses pertensor quantized data, and the remaining input parameters all are non-quantized data. The dequantScaleX, dequantScaleWDq, dequantScaleWUqQr, dequantScaleWDkvKr, and quantScaleCkv fields must be passed in <br>
          Output parameters: queryOut returns per-token-head quantized data, the kvCacheRef output parameter returns pertensor quantized data, and the remaining output parameters return non-quantized data
      </td>
    </tr>
    <tr>
      <td> kvCache pertoken-pergroup quantization </td>
      <td>          weight_quant_mode=3,kv_cache_quant_mode=3,query_quant_mode=0<br>
          Input parameters: tokenX uses pertoken quantized data, weightDq, weightUqQr, and weightDkvKr use perchannel quantized data, and the remaining input parameters all are non-quantized data. The dequantScaleX, dequantScaleWDq, dequantScaleWUqQr, and dequantScaleWDkvKr fields must be passed in <br>
          Output parameters: the kvCacheRef output parameter returns pertoken-pergroup quantized data, and the remaining output parameters return non-quantized data
      </td>
    </tr>
  </table>
  </details>

  <details>
  <summary><a id="SupportScenes"></a>dtype constraints on parameters in different quantization scenarios</summary>

  <div style="overflow-x: auto; width: 100%;">
  <table style="table-layout: auto;" border="1">
    <tr>
      <th rowspan="3">Parameter</th>
      <th rowspan="2" colspan="1">Non-quantized scenario</th>
      <th colspan="3">Partial quantization scenario</th>
      <th colspan="3">int8 fully quantized scenario</th>
      <th colspan="3">mxfp8 fully quantized scenario</th>
      <th colspan="3">fp8 fully quantized scenario</th>
      <th colspan="3">hif8 fully quantized scenario</th>
    </tr>
    <tr>
      <th colspan="1">kvCache non-quantization</th>
      <th colspan="1">kvCache perchannel quantization</th>
      <th colspan="1">kvCache pertoken-pergroup quantization</th>
      <th colspan="1">kvCache non-quantization</th>
      <th colspan="1">kvCache pertensor quantization</th>
      <th colspan="1">kvCache pertoken-pergroup quantization</th>
      <th colspan="1">kvCache non-quantization</th>
      <th colspan="1">kvCache pertensor quantization</th>
      <th colspan="1">kvCache pertoken-pergroup quantization</th>
      <th colspan="1">kvCache non-quantization</th>
      <th colspan="1">kvCache pertensor quantization</th>
      <th colspan="1">kvCache pertoken-pergroup quantization</th>
      <th colspan="1">kvCache non-quantization</th>
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

The following sample code for <term>Atlas A2 products</term> and <term>Atlas A3 products</term> is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

  ```Cpp
  #include <iostream>
  #include <cstring>
  #include <vector>
  #include <cstdint>
  #include "acl/acl.h"
  #include "aclnnop/aclnn_mla_prolog_v4_weight_nz.h"
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
      // Call aclrtMalloc to allocate device-side memory.
      auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
      // Call aclrtMalloc to allocate host-side memory.
      ret = aclrtMallocHost(hostAddr, size);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMallocHost failed. ERROR: %d\n", ret); return ret);
      memset(*hostAddr, 0, size);
      // Call the aclCreateTensor API to create an aclTensor.
      *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, nullptr, 0, aclFormat::ACL_FORMAT_ND,
                                shape.data(), shape.size(), *deviceAddr);
      // Call aclrtMemcpy to copy host-side data to device-side memory.
      ret = aclrtMemcpy(*deviceAddr, size, *hostAddr, GetShapeSize(shape)*aclDataTypeSize(dataType), ACL_MEMCPY_HOST_TO_DEVICE);
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
      // Call aclrtMemcpy to copy host-side data to device memory.
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
      // 1. Boilerplate: device/stream initialization. For details, see the AscendCL external API list.
      // Set deviceId to the actual device.
      int32_t deviceId = 0;
      aclrtStream stream;
      auto ret = Init(deviceId, &stream);
      // Handle check as needed.
      CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
      // 2. Construct the inputs and outputs according to the API definition.
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
      bool doRope = true;
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
      ret = CreateAclTensorND(tokenXShape, &tokenXDeviceAddr, &tokenXHostAddr, aclDataType::ACL_INT8, &tokenX);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create the weightDq aclTensor.
      ret = CreateAclTensorNZ(weightDqShape, &weightDqDeviceAddr, &weightDqHostAddr, aclDataType::ACL_INT8, &weightDq);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create the weightUqQr aclTensor.
      ret = CreateAclTensorNZ(weightUqQrShape, &weightUqQrDeviceAddr, &weightUqQrHostAddr, aclDataType::ACL_INT8, &weightUqQr);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create the weightUk aclTensor.
      ret = CreateAclTensorND(weightUkShape, &weightUkDeviceAddr, &weightUkHostAddr, aclDataType::ACL_BF16, &weightUk);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create the weightDkvKr aclTensor.
      ret = CreateAclTensorNZ(weightDkvKrShape, &weightDkvKrDeviceAddr, &weightDkvKrHostAddr, aclDataType::ACL_INT8, &weightDkvKr);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create the rmsnormGammaCq aclTensor.
      ret = CreateAclTensorND(rmsnormGammaCqShape, &rmsnormGammaCqDeviceAddr, &rmsnormGammaCqHostAddr, aclDataType::ACL_BF16, &rmsnormGammaCq);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create the rmsnormGammaCkv aclTensor.
      ret = CreateAclTensorND(rmsnormGammaCkvShape, &rmsnormGammaCkvDeviceAddr, &rmsnormGammaCkvHostAddr, aclDataType::ACL_BF16, &rmsnormGammaCkv);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create the ropeSin aclTensor.
      ret = CreateAclTensorND(ropeSinShape, &ropeSinDeviceAddr, &ropeSinHostAddr, aclDataType::ACL_BF16, &ropeSin);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create the ropeCos aclTensor.
      ret = CreateAclTensorND(ropeCosShape, &ropeCosDeviceAddr, &ropeCosHostAddr, aclDataType::ACL_BF16, &ropeCos);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create the cacheIndex aclTensor.
      ret = CreateAclTensorND(cacheIndexShape, &cacheIndexDeviceAddr, &cacheIndexHostAddr, aclDataType::ACL_INT64, &cacheIndex);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create the kvCache aclTensor.
      ret = CreateAclTensorND(kvCacheShape, &kvCacheDeviceAddr, &kvCacheHostAddr, aclDataType::ACL_INT8, &kvCache);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create the krCache aclTensor.
      ret = CreateAclTensorND(krCacheShape, &krCacheDeviceAddr, &krCacheHostAddr, aclDataType::ACL_BF16, &krCache);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create the dequantScaleX aclTensor.
      ret = CreateAclTensorND(dequantScaleXShape, &dequantScaleXDeviceAddr, &dequantScaleXHostAddr, aclDataType::ACL_FLOAT, &dequantScaleX);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create the dequantScaleWDq aclTensor.
      ret = CreateAclTensorND(dequantScaleWDqShape, &dequantScaleWDqDeviceAddr, &dequantScaleWDqHostAddr, aclDataType::ACL_FLOAT, &dequantScaleWDq);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create the dequantScaleWUqQr aclTensor.
      ret = CreateAclTensorND(dequantScaleWUqQrShape, &dequantScaleWUqQrDeviceAddr, &dequantScaleWUqQrHostAddr, aclDataType::ACL_FLOAT, &dequantScaleWUqQr);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create the dequantScaleWDkvKr aclTensor.
      ret = CreateAclTensorND(dequantScaleWDkvKrShape, &dequantScaleWDkvKrDeviceAddr, &dequantScaleWDkvKrHostAddr, aclDataType::ACL_FLOAT, &dequantScaleWDkvKr);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create the quantScaleCkv aclTensor.
      ret = CreateAclTensorND(quantScaleCkvShape, &quantScaleCkvDeviceAddr, &quantScaleCkvHostAddr, aclDataType::ACL_FLOAT, &quantScaleCkv);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create the smoothScalesCq aclTensor.
      ret = CreateAclTensorND(smoothScalesCqShape, &smoothScalesCqDeviceAddr, &smoothScalesCqHostAddr, aclDataType::ACL_FLOAT, &smoothScalesCq);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create the query aclTensor.
      ret = CreateAclTensorND(queryShape, &queryDeviceAddr, &queryHostAddr, aclDataType::ACL_INT8, &query);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create the queryRope aclTensor.
      ret = CreateAclTensorND(queryRopeShape, &queryRopeDeviceAddr, &queryRopeHostAddr, aclDataType::ACL_BF16, &queryRope);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create the dequantScaleQNope aclTensor.
      ret = CreateAclTensorND(dequantScaleQNopeShape, &dequantScaleQNopeDeviceAddr, &dequantScaleQNopeHostAddr, aclDataType::ACL_FLOAT, &dequantScaleQNope);
      CHECK_RET(ret == ACL_SUCCESS, return ret);

      // 3. Call the CANN operator library API. Replace it with the specific API.
      uint64_t workspaceSize = 0;
      aclOpExecutor* executor = nullptr;
      // Call the first-phase API of aclnnMlaPrologV4WeightNz.
      ret = aclnnMlaPrologV4WeightNzGetWorkspaceSize(tokenX, weightDq, weightUqQr, weightUk, weightDkvKr, rmsnormGammaCq, rmsnormGammaCkv, ropeSin, ropeCos, kvCache, krCache, cacheIndex,
        dequantScaleX, dequantScaleWDq, dequantScaleWUqQr, dequantScaleWDkvKr, quantScaleCkv, nullptr, smoothScalesCq, nullptr, nullptr,rmsnormEpsilonCq, rmsnormEpsilonCkv, cacheMode,
        weightQuantMode, kvQuantMode, queryQuantMode, ckvkrRepoMode, quantScaleRepoMode, tileSize, qcQrScale, kcScale, doRope,
        query, queryRope, dequantScaleQNope, nullptr, nullptr, &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMlaPrologV4WeightNzGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
      // Allocate device memory based on the workspaceSize calculated by the first-phase API.
      void* workspaceAddr = nullptr;
      if (workspaceSize > static_cast<uint64_t>(0)) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
      }
      // Call the second-phase API of aclnnMlaPrologV4WeightNz.
      ret = aclnnMlaPrologV4WeightNz(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMlaPrologV4WeightNz failed. ERROR: %d\n", ret); return ret);

      // 4. Boilerplate: wait synchronously until the task is complete.
      ret = aclrtSynchronizeStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

      // 5. Obtain the output values and copy the results from the device side to the host side. Modify as required by the specific API definition.
      auto size = GetShapeSize(queryShape);
      std::vector<float> resultData(size, 0);
      ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), queryDeviceAddr, size * sizeof(float),
                        ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
      for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
      }
      // 6. Release aclTensor and aclScalar. Modify as required by the specific API definition.
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

<term>Ascend 950PR&950DT products</term> sample code is as follows, for reference only.

  ```Cpp
#include <iostream>
#include <cstring>
#include <vector>
#include <cstdint>
#include "acl/acl.h"
#include "aclnnop/aclnn_mla_prolog_v4_weight_nz.h"
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
    // Boilerplate: resource initialization.
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
    // Call aclrtMalloc to allocate device-side memory.
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // Call aclrtMalloc to allocate host-side memory.
    ret = aclrtMallocHost(hostAddr, size);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMallocHost failed. ERROR: %d\n", ret); return ret);
    memset(*hostAddr, 0, size);
    // Call the aclCreateTensor API to create aclTensor.
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, nullptr, 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    // Call aclrtMemcpy to copy host-side data to device-side memory.
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
    // Call the aclCreateTensor API to create aclTensor.
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
    // 1. Boilerplate for device/stream initialization. For details, see the AscendCL public API list.
    // Fill in deviceId based on the actual device.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // Handle check as required.
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    // 2. Construct the input and output according to the API definition.
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
    bool doRope = true;
    aclTensor* query = nullptr;
    aclTensor* queryRope = nullptr;
    aclTensor* dequantScaleQNope = nullptr;

    // Convert the shapes of the three NZ-format variables
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
    // Create the weightDq aclTensor.
    ret = CreateAclTensorNZ(weightDqShape, &weightDqDeviceAddr, &weightDqHostAddr, aclDataType::ACL_FLOAT8_E4M3FN, &weightDq);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the weightUqQr aclTensor.
    ret = CreateAclTensorNZ(weightUqQrShape, &weightUqQrDeviceAddr, &weightUqQrHostAddr, aclDataType::ACL_FLOAT8_E4M3FN, &weightUqQr);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the weightUk aclTensor.
    ret = CreateAclTensorND(weightUkShape, &weightUkDeviceAddr, &weightUkHostAddr, aclDataType::ACL_BF16, &weightUk);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the weightDkvKr aclTensor.
    ret = CreateAclTensorNZ(weightDkvKrShape, &weightDkvKrDeviceAddr, &weightDkvKrHostAddr, aclDataType::ACL_FLOAT8_E4M3FN, &weightDkvKr);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the rmsnormGammaCq aclTensor.
    ret = CreateAclTensorND(rmsnormGammaCqShape, &rmsnormGammaCqDeviceAddr, &rmsnormGammaCqHostAddr, aclDataType::ACL_BF16, &rmsnormGammaCq);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the rmsnormGammaCkv aclTensor.
    ret = CreateAclTensorND(rmsnormGammaCkvShape, &rmsnormGammaCkvDeviceAddr, &rmsnormGammaCkvHostAddr, aclDataType::ACL_BF16, &rmsnormGammaCkv);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the ropeSin aclTensor.
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
    // Call the first-phase API of aclnnMlaPrologV4WeightNz.
    ret = aclnnMlaPrologV4WeightNzGetWorkspaceSize(tokenX, weightDq, weightUqQr, weightUk, weightDkvKr, rmsnormGammaCq, rmsnormGammaCkv, ropeSin, ropeCos, kvCache, krCache, cacheIndex,
        dequantScaleX, dequantScaleWDq, dequantScaleWUqQr, dequantScaleWDkvKr, quantScaleCkv, nullptr, nullptr, nullptr, nullptr, rmsnormEpsilonCq, rmsnormEpsilonCkv, cacheMode,
        weightQuantMode, kvQuantMode, queryQuantMode, ckvkrRepoMode, quantScaleRepoMode, tileSize, qcQrScale, kcScale, doRope,
        query, queryRope, dequantScaleQNope, nullptr, nullptr, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMlaPrologV4WeightNzGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // Allocate device memory based on the workspaceSize computed by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > static_cast<uint64_t>(0)) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    // Call the second-phase API of aclnnMlaPrologV4WeightNz.
    ret = aclnnMlaPrologV4WeightNz(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMlaPrologV4WeightNz failed. ERROR: %d\n", ret); return ret);

    // 4. Fixed boilerplate. Synchronously wait for the task execution to complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. Obtain the output value and copy the result from device memory to the host. Modify it according to the specific API definition.
    auto size = GetShapeSize(queryShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), queryDeviceAddr, size * sizeof(float),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }
    // 6. Release aclTensor and aclScalar. Modify it according to the specific API definition.
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
