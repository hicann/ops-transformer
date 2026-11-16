# aclnnMlaProlog

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-21T11:41:41.897Z -->

**NOTE: This API will be deprecated in later versions. Use the latest API [aclnnMlaPrologV3WeightNz](../../mla_prolog_v3/docs/aclnnMlaPrologV3WeightNz_en.md).**

## Applicable Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Not supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 products</term>: Supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 products</term>: Supported
<!-- end id3 -->

## Function

- **Description**: In inference scenarios, this API performs the pre-processing computation for Multi-Head Latent Attention. The main computation process is divided into four paths. First, the input $x$ is multiplied by $W^{DQ}$ for downsampling and RmsNorm, and then split into two paths. The first path is multiplied by $W^{UQ}$ and $W^{UK}$ and undergoes two upsampling operations to obtain $q^N$. The second path is multiplied by $W^{QR}$ and then undergoes rotary position encoding (ROPE) to obtain $q^R$. The third path involves multiplying the input $x$ by $W^{DKV}$ for downsampling and RmsNorm, and then passing the result into a cache to obtain $k^C$. The fourth path involves multiplying the input $x$ by $W^{KR}$, undergoing rotary position encoding, and then passing the result into another cache to obtain $k^R$.
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
    c^Q = RmsNorm(x \cdot W^{DQ})
    $$

    $$
    q^C = c^Q \cdot W^{UQ}
    $$

    $$
    q^N = q^C \cdot W^{UK}
    $$

    Perform ROPE rotary positional encoding on Query.

    $$
    q^R = ROPE(c^Q \cdot W^{QR})
    $$

    Key computation formula, including downsampling and RmsNorm, with the computation result stored in cache.

    $$
    c^{KV} = RmsNorm(x \cdot W^{DKV})
    $$

    $$
    k^C = Cache(c^{KV})
    $$

    Perform ROPE rotary positional encoding on Key and store the result in cache.

    $$
    k^R = Cache(ROPE(x \cdot W^{KR}))
    $$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call the "aclnnMlaPrologGetWorkspaceSize" API first to obtain the input parameters and calculate the required workspace size based on the process. Then call the "aclnnMlaProlog" API to perform the computation.

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
      <tbody>
        <tr>
          <td>tokenX</td>
          <td>Input</td>
          <td>Input tensor used to calculate Query and Key in the formula.</td>
          <td>            <ul>
              <li>Supports empty tensors where B=0, S=0, T=0.</li>
            </ul>
          </td>
          <td>BFLOAT16</td>
          <td>ND</td>
          <td>            <ul>
              <li>BS axes merged: (T,He)</li>
              <li>BS axes not merged: (B,S,He)</li>
            </ul>
          </td>
          <td>×</td>
        </tr>
        <tr>
          <td>weightDq</td>
          <td>Input</td>
          <td>Downsampling weight matrix <i>W<sup>DQ</sup></i> used to calculate Query in the formula.</td>
          <td>            <ul>
              <li>Empty tensors are not supported.</li>
            </ul>
          </td>
          <td>BFLOAT16</td>
          <td>FRACTAL_NZ</td>
          <td>(He,Hcq)</td>
          <td>×</td>
        </tr>
        <tr>
          <td>weightUqQr</td>
          <td>Input</td>
          <td>Upsampling weight matrix <i>W<sup>UQ</sup></i> for computing the Query and position encoding weight matrix <i>W<sup>QR</sup></i> in the formula.</td>
          <td>            <ul>
              <li>Empty tensors are not supported.</li>
              <li>INT8 dtype (quantization use case):
                <ul>
                  <li>Per-tensor quantization input is required.</li>
                  <li>dequantScaleWUqQrOptional is mandatory for non-quantized output.</li>
                  <li>dequantScaleWUqQrOptional, quantScaleCkvOptional, and quantScaleCkrOptional are mandatory for quantized output.</li>
                  <li>smoothScalesCqOptional is optional.</li>
                </ul>
              </li>
              <li>BFLOAT16 dtype (non-quantization use case): dequantScaleWUqQrOptional, quantScaleCkvOptional, quantScaleCkrOptional, and smoothScalesCqOptional must be null pointers.</li>
            </ul>
          </td>
          <td>BFLOAT16, INT8</td>
          <td>FRACTAL_NZ</td>
          <td>(Hcq,N*(D+Dr))</td>
          <td>×</td>
        </tr>
        <tr>
          <td>weightUk</td>
          <td>Input</td>
          <td>Upsampling weight <i>W<sup>UK</sup></i> used to compute Key in the formula.</td>
          <td>            Empty tensor not supported.
          </td>
          <td>BFLOAT16</td>
          <td>ND</td>
          <td>(N,D,Hckv)</td>
          <td>×</td>
        </tr>
        <tr>
          <td>weightDkvKr</td>
          <td>Input</td>
          <td>Downsampling weight matrix <i>W<sup>DKV</sup></i> for computing the Key and position encoding weight matrix <i>W<sup>KR</sup></i> in the formula.</td>
          <td>            Empty tensors are not supported
          </td>
          <td>BFLOAT16</td>
          <td>FRACTAL_NZ</td>
          <td>(He,Hckv+Dr)</td>
          <td>×</td>
        </tr>
        <tr>
          <td>rmsnormGammaCq</td>
          <td>Input</td>
          <td>The <i>γ</i> parameter in the RmsNorm formula for computing <i>c<sup>Q</sup></i>.</td>
          <td>            Empty tensors are not supported.
          </td>
          <td>BFLOAT16</td>
          <td>ND</td>
          <td>(Hcq)</td>
          <td>×</td>
        </tr>
        <tr>
          <td>rmsnormGammaCkv</td>
          <td>Input</td>
          <td>The <i>γ</i> parameter in the RmsNorm formula for computing <i>c<sup>KV</sup></i>.</td>
          <td>Does not support empty tensors</td>
          <td>BFLOAT16</td>
          <td>ND</td>
          <td>(Hckv)</td>
          <td>×</td>
        </tr>
        <tr>
          <td>ropeSin</td>
          <td>Input</td>
          <td>Sine parameter matrix used to compute the rotary position embedding.</td>
          <td>Supports empty tensors with B=0, S=0, T=0</td>
          <td>BFLOAT16</td>
          <td>ND</td>
          <td>            <ul>
              <li>BS merged: (T,Dr)</li>
              <li>BS non-merged: (B,S,Dr)</li>
            </ul>
          </td>
          <td>×</td>
        </tr>
        <tr>
          <td>ropeCos</td>
          <td>Input</td>
          <td>Cosine parameter matrix used to calculate the rotary position embedding.</td>
          <td>Empty tensors with B=0, S=0, T=0 are supported</td>
          <td>BFLOAT16</td>
          <td>ND</td>
          <td>            <ul>
              <li>BS merged: (T,Dr)</li>
              <li>BS non-merged: (B,S,Dr)</li>
            </ul>
          </td>
          <td>×</td>
        </tr>
        <tr>
          <td>cacheIndex</td>
          <td>Input</td>
          <td>Index used to store kvCache and krCache.</td>
          <td>            <ul>
              <li>Supports empty tensors with B=0,S=0,T=0.</li>
              <li>The value must be within [0,BlockNum*BlockSize).</li>
            </ul>
          </td>
          <td>INT64</td>
          <td>ND</td>
          <td>            <ul>
              <li>BS merged: (T)</li>
              <li>BS not merged: (B,S)</li>
            </ul>
          </td>
          <td>×</td>
        </tr>
        <tr>
          <td>kvCacheRef</td>
          <td>Input</td>
          <td>aclTensor used for cache indexing. The calculation result is updated in place (corresponding to <i>k<sup>C</sup></i> in the formula).</td>
          <td>            <ul>
              <li>Supports empty tensors with B=0, Skv=0.</li>
              <li>Nkv is associated with N. N is a hyperparameter, so Nkv cannot be 0.</li>
            </ul>
          </td>
          <td>BFLOAT16, INT8</td>
          <td>ND</td>
          <td>(BlockNum,BlockSize,Nkv,Hckv)</td>
          <td>×</td>
        </tr>
        <tr>
          <td>krCacheRef</td>
          <td>Input</td>
          <td>Cache used for key positional encoding. The calculation result is updated in place (corresponding to <i>k<sup>R</sup></i> in the formula).</td>
          <td>            <ul>
              <li>Supports empty tensors with B=0 and Skv=0.</li>
              <li>Nkv is associated with N. N is a hyperparameter. Therefore, Nkv cannot be set to 0.</li>
            </ul>
          </td>
          <td>BFLOAT16, INT8</td>
          <td>ND</td>
          <td>(BlockNum,BlockSize,Nkv,Dr)</td>
          <td>×</td>
        </tr>
        <tr>
          <td>dequantScaleXOptional</td>
          <td>Input</td>
          <td>Dequantization parameters of tokenX.</td>
          <td>Data format supports ND.</td>
          <td>FLOAT</td>
          <td>ND</td>
          <td>            <ul>
              <li>BS with axes combined: (T)</li>
              <li>BS with axes not combined: (B*S,1)</li>
            </ul>
          </td>
          <td>×</td>
        </tr>
        <tr>
          <td>dequantScaleWDqOptional</td>
          <td>Input</td>
          <td>Dequantization parameter of weightDq.</td>
          <td>Supports ND data format.</td>
          <td>FLOAT</td>
          <td>ND</td>
          <td>(1,Hcq)</td>
          <td>×</td>
        </tr>
        <tr>
          <td>dequantScaleWUqQrOptional</td>
          <td>Input</td>
          <td>Per-channel parameter for dequantization after the MatmulQcQr matrix multiplication.</td>
          <td>            A non-empty tensor (required only for the INT8 dtype use case).
          </td>
          <td>FLOAT</td>
          <td>ND</td>
          <td>(1,N*(D+Dr))</td>
          <td>×</td>
        </tr>
        <tr>
          <td>dequantScaleWDkvKrOptional</td>
          <td>Input</td>
          <td>Dequantization parameters of weightDkvKr.</td>
          <td>            The supported data format is ND.
          </td>
          <td>FLOAT</td>
          <td>ND</td>
          <td>(1, Hckv+Dr)</td>
          <td>×</td>
        </tr>
        <tr>
          <td>quantScaleCkvOptional</td>
          <td>Input</td>
          <td>Parameters for quantizing the kvCache output data.</td>
          <td>Supports non-empty Tensor (required to be passed only for the INT8 dtype use case).</td>
          <td>FLOAT</td>
          <td>ND</td>
          <td>(1,Hckv)</td>
          <td>×</td>
        </tr>
        <tr>
          <td>quantScaleCkrOptional</td>
          <td>Input</td>
          <td>Parameter for quantizing the krCache output data.</td>
          <td>Supports non-empty Tensor (required to be passed only for the INT8 dtype use case).</td>
          <td>FLOAT</td>
          <td>ND</td>
          <td>(1,Dr)</td>
          <td>×</td>
        </tr>
        <tr>
          <td>smoothScalesCqOptional</td>
          <td>Input</td>
          <td>Parameter for performing dynamic quantization on the RmsNormCq output.</td>
          <td>            Non-empty Tensor (required only for the INT8 dtype use case)
          </td>
          <td>FLOAT</td>
          <td>ND</td>
          <td>(1,Hcq)</td>
          <td>×</td>
        </tr>
        <tr>
          <td>rmsnormEpsilonCq</td>
          <td>Input</td>
          <td>The <i>ε</i> parameter in the RmsNorm formula for computing <i>c<sup>Q</sup></i>.</td>
          <td>            <ul>
              <li>1e-05 is recommended if not specified by the user.</li>
              <li>Only the double type is supported.</li>
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
          <td>The <i>ε</i> parameter in the RmsNorm formula for computing <i>c<sup>KV</sup></i>.</td>
          <td>            <ul>
              <li>1e-05 is recommended if not specified by the user.</li>
              <li>Only double is supported.</li>
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
          <td>            <ul>
              <li>If the user does not specify a value, "PA_BSND" is recommended.</li>
              <li>Only the char* type is supported.</li>
              <li>Options are "PA_BSND" and "PA_NZ".</li>
            </ul>
          </td>
          <td>CHAR*</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td>queryOut</td>
          <td>Output</td>
          <td>Output tensor of Query in the formula (corresponding to <i>q<sup>N</sup></i>).</td>
          <td>-</td>
          <td>BFLOAT16, INT8</td>
          <td>ND</td>
          <td>            <ul>
              <li>BS merged: (T,N,Hckv)</li>
              <li>BS unmerged: (B,S,N,Hckv)</li>
            </ul>
          </td>
          <td>×</td>
        </tr>
        <tr>
          <td>queryRopeOut</td>
          <td>Output</td>
          <td>Output tensor of Query position encoding in the formula (corresponding to <i>q<sup>R</sup></i>).</td>
          <td>-</td>
          <td>BFLOAT16</td>
          <td>ND</td>
          <td>            <ul>
              <li>BS merged: (T,N,Dr)</li>
              <li>BS unmerged: (B,S,N,Dr)</li>
            </ul>
          </td>
          <td>×</td>
        </tr>
        <tr>
          <td>workspaceSize</td>
          <td>Output</td>
          <td>Returns the size of the workspace to be allocated on the Device.</td>
          <td>            <ul>
              <li>Used for output only; no input configuration required.</li>
              <li>Data type: uint64_t*</li>
            </ul>
          </td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td>executor</td>
          <td>Output</td>
          <td>Returns the op executor containing the operator computation flow.</td>
          <td>            <ul>
              <li>For output only. No input configuration required.</li>
              <li>Data type: aclOpExecutor**.</li>
            </ul>
          </td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
      </tbody>
    </table>

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
          <th>Returns</th>
          <th>Error Codes</th>
          <th>Description</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>ACLNN_ERR_PARAM_NULLPTR</td>
          <td>161001</td>
          <td>A required parameter (for example, an input/output parameter on which the API depends) contains a null pointer.</td>
        </tr>
        <tr>
          <td>ACLNN_ERR_PARAM_INVALID</td>
          <td>161002</td>
          <td>The shape (dimension/size) or dtype (data type) of the input parameter is out of the range supported by the API.</td>
        </tr>
        <tr>
          <td>ACLNN_ERR_RUNTIME_ERROR</td>
          <td>361001</td>
          <td>An exception occurs when the API calls NPU Runtime APIs for memory operations (for example, the Runtime service is not started or memory allocation fails).</td>
        </tr>
        <tr>
          <td>ACLNN_ERR_INNER_TILING_ERROR</td>
          <td>561002</td>
          <td>An exception occurs during tiling. The dtype or shape of the input parameter is incorrect.</td>
        </tr>
      </tbody>
    </table>

## aclnnMlaProlog

- **Parameters**

    <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
    <col style="width: 168px">
    <col style="width: 128px">
    <col style="width: 854px">
    </colgroup>
      <thead>
        <tr>
          <th>Parameter</th>
          <th>Type</th>
          <th>Description</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>workspace</td>
          <td>void*</td>
          <td>Address of the workspace memory allocated on the device.</td>
        </tr>
        <tr>
          <td>workspaceSize</td>
          <td>uint64_t</td>
          <td>Size of the workspace memory allocated on the device, obtained by calling the first-phase API aclnnMlaPrologGetWorkspaceSize.</td>
        </tr>
        <tr>
          <td>executor</td>
          <td>aclOpExecutor*</td>
          <td>op executor, containing the operator computation flow.</td>
        </tr>
        <tr>
          <td>stream</td>
          <td>aclrtStream</td>
          <td>Stream for task execution.</td>
        </tr>
      </tbody>
    </table>

- **Return Value**

    aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnMlaProlog defaults to a deterministic implementation.

<details>
  <summary><a id="shapeDesc"></a>Meaning of shape format fields</summary>
    &nbsp;&nbsp;<table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
        <col style="width: 165px">
        <col style="width: 625px">
        <col style="width: 360px">
        </colgroup>
  <thead>
    <tr>
      <th>Field</th>
      <th>Full English Name/Meaning</th>
      <th>Value Rules and Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>B</td>
      <td>Batch (batch size of input samples).</td>
      <td>Range: 0-65536</td>
    </tr>
    <tr>
      <td>S</td>
      <td>Seq-Length (sequence length of input samples).</td>
      <td>
        Range: unlimited
      </td>
    </tr>
    <tr>
      <td>He</td>
      <td>Hidden-Size (hidden layer size).</td>
      <td>
        Fixed values: 1024, 2048, 3072, 4096, 5120, 6144, 7168, 7680, 8192
      </td>
    </tr>
    <tr>
      <td>Hcq</td>
      <td>Dimension of the q low-rank matrix.</td>
      <td>Fixed value: 1536</td>
    </tr>
    <tr>
      <td>N</td>
      <td>Head-Num (number of heads).</td>
      <td>Range: 1, 2, 4, 8, 16, 32, 64, 128</td>
    </tr>
    <tr>
      <td>Hckv</td>
      <td>Dimension of the kv low-rank matrix.</td>
      <td>Fixed value: 512</td>
    </tr>
    <tr>
      <td>D</td>
      <td>qk dimension excluding positional encoding.</td>
      <td>Fixed value: 128</td>
    </tr>
    <tr>
      <td>Dr</td>
      <td>qk positional encoding dimension.</td>
      <td>Fixed value: 64</td>
    </tr>
    <tr>
      <td>Nkv</td>
      <td>Number of kv heads.</td>
      <td>Fixed value: 1</td>
    </tr>
    <tr>
      <td>BlockNum</td>
      <td>Number of blocks in a PagedAttention use case.</td>
      <td>Value: ⌈B*Skv/BlockSize⌉ (rounded-up result of B*Skv/BlockSize; Skv indicates the sequence length of kv and can be 0)</td>
    </tr>
    <tr>
      <td>BlockSize</td>
      <td>Block size in a PagedAttention use case.</td>
      <td>Range: 16-1024, in multiples of 16</td>
    </tr>
    <tr>
      <td>T</td>
      <td>Size after BS axis merging.</td>
      <td>
        <ul>
          <li>Range: 0-1048576</li>
          <li>Note: If BS axis merging is used, tokenX, ropeSin, and ropeCos are 2D, cacheIndex is 1D, and queryOut and queryRopeOut are 3D.</li>
        </ul>
      </td>
    </tr>
  </tbody>
</table>
</details>

  <details>
  <summary><a id="specialConstraint"></a>Special Constraints</summary>

  - Dimensions of weight_dq, weight_uq_qr, and weight_dkv_kr when not transposed: (k, n).

  </details>

  <details>
  <summary><a id="SupportScenes"></a>Use cases supported by the aclnnMlaProlog API</summary>

  <table style="table-layout: auto;" border="1">
    <tr>
      <th colspan="2">Use Case</th>
      <th>Description</th>
    </tr>
    <tr>
      <td colspan="2">Non-quantization</td>
      <td>          Input: all inputs are non-quantized data <br>
          Output: all outputs are non-quantized data
      </td>
    </tr>
    <tr>
      <td rowspan="2">Partial quantization</td>
      <td>kv_cache is non-quantized </td>
      <td>          Input: weightUqQr is per-token quantized data, and the other inputs are non-quantized data <br>
          Output: all outputs return non-quantized data
      </td>
    </tr>
    <tr>
      <td>kv_cache quantization </td>
      <td>          Input: weightUqQr passes in per-token quantized data, kvCacheRef and krCacheRef pass in per-channel quantized data, and all other inputs are non-quantized data. <br>
          Output: kvCacheRef and krCacheRef return per-channel quantized data, and all other outputs return non-quantized data.
      </td>
    </tr>
  </table>
  </details>

  <details>
  <summary><a id="SupportScenes1"></a>dtype and shape constraints for parameters in different quantization use cases</summary>

  In different quantization use cases, dtype and shape combinations of parameters must meet the following conditions:
  <div style="overflow-x: auto; width: 100%;">
  <table style="table-layout: auto;" border="1">
    <tr>
      <th rowspan="3">Parameter</th>
      <th rowspan="2" colspan="2">Non-quantization use case</th>
      <th colspan="4">Partial quantization use case</th>
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
      <td>No value assignment required.</td>
      <td>-</td>
      <td>No value assignment required.</td>
      <td>-</td>
      <td>No value assignment required.</td>
      <td>-</td>
    </tr>
    <tr>
      <td> dequantScaleWDqOptional </td>
      <td>No value assignment required.</td>
      <td>-</td>
      <td>No value assignment required.</td>
      <td>-</td>
      <td>No value assignment required.</td>
      <td>-</td>
    </tr>
    <tr>
      <td> dequantScaleWUqQrOptional </td>
      <td>No value assignment required.</td>
      <td>-</td>
      <td>FLOAT</td>
      <td> (1, N*(D+Dr)) </td>
      <td>FLOAT</td>
      <td> (1, N*(D+Dr)) </td>
    </tr>
    <tr>
      <td> dequantScaleWDkvKrOptional </td>
      <td> No assignment required. </td>
      <td>-</td>
      <td> No assignment required. </td>
      <td>-</td>
      <td> No assignment required. </td>
      <td>-</td>
    </tr>
    <tr>
      <td> quantScaleCkvOptional </td>
      <td>No assignment required.</td>
      <td>-</td>
      <td>No assignment required.</td>
      <td>-</td>
      <td>FLOAT</td>
      <td> (1, Hckv) </td>
    </tr>
    <tr>
      <td> quantScaleCkrOptional </td>
      <td>No assignment required.</td>
      <td>-</td>
      <td>No assignment required.</td>
      <td>-</td>
      <td>FLOAT</td>
      <td> (1, Dr) </td>
    </tr>
    <tr>
      <td> smoothScalesCqOptional </td>
      <td>No assignment required</td>
      <td>-</td>
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
      <td>No value assignment required.</td>
      <td>-</td>
      <td>No value assignment required.</td>
      <td>-</td>
      <td>No value assignment required.</td>
      <td>-</td>
    </tr>
  </table>
  </div>
  </details>

  <!-- For parameter descriptions, see **Operator Execution APIs**. -->

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

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
      // Call aclrtMalloc to allocate device memory.
      auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
      // Call aclrtMalloc to allocate host memory.
      ret = aclrtMalloc(hostAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
      // Call the aclCreateTensor API to create an aclTensor.
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
      ret = aclrtMalloc(hostAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
      // Call the aclCreateTensor API to create an aclTensor.
      *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, nullptr, 0, aclFormat::ACL_FORMAT_FRACTAL_NZ,
                                shape.data(), shape.size(), *deviceAddr);
      // Call aclrtMemcpy to copy host-side data to device-side memory.
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
      // 1. Boilerplate: initialize the device/stream. For details, see the list of AscendCL external APIs.
      // Set deviceId based on the actual device.
      int32_t deviceId = 0;
      aclrtStream stream;
      auto ret = Init(deviceId, &stream);
      // Handle the check as needed.
      CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
      // 2. Construct inputs and outputs based on the API definition.
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

      // Convert the shapes of the three NZ-format variables.
      ret = TransToNZShape(weightDqShape);
      CHECK_RET(ret == 0, LOG_PRINT("trans NZ shape failed.\n"); return ret);
      ret = TransToNZShape(weightUqQrShape);
      CHECK_RET(ret == 0, LOG_PRINT("trans NZ shape failed.\n"); return ret);
      ret = TransToNZShape(weightDkvKrShape);
      CHECK_RET(ret == 0, LOG_PRINT("trans NZ shape failed.\n"); return ret);

      // Create the tokenX aclTensor.
      ret = CreateAclTensorND(tokenXShape, &tokenXDeviceAddr, &tokenXHostAddr, aclDataType::ACL_BF16, &tokenX);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create the weightDq aclTensor.
      ret = CreateAclTensorNZ(weightDqShape, &weightDqDeviceAddr, &weightDqHostAddr, aclDataType::ACL_BF16, &weightDq);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create the weightUqQr aclTensor.
      ret = CreateAclTensorNZ(weightUqQrShape, &weightUqQrDeviceAddr, &weightUqQrHostAddr, aclDataType::ACL_BF16, &weightUqQr);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create weightUk aclTensor.
      ret = CreateAclTensorND(weightUkShape, &weightUkDeviceAddr, &weightUkHostAddr, aclDataType::ACL_BF16, &weightUk);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create weightDkvKr aclTensor.
      ret = CreateAclTensorNZ(weightDkvKrShape, &weightDkvKrDeviceAddr, &weightDkvKrHostAddr, aclDataType::ACL_BF16, &weightDkvKr);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create ropeSin aclTensor.
      ret = CreateAclTensorND(ropeSinShape, &ropeSinDeviceAddr, &ropeSinHostAddr, aclDataType::ACL_BF16, &ropeSin);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create ropeCos aclTensor.
      ret = CreateAclTensorND(ropeCosShape, &ropeCosDeviceAddr, &ropeCosHostAddr, aclDataType::ACL_BF16, &ropeCos);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create rmsnormGammaCq aclTensor.
      ret = CreateAclTensorND(rmsnormGammaCqShape, &rmsnormGammaCqDeviceAddr, &rmsnormGammaCqHostAddr, aclDataType::ACL_BF16, &rmsnormGammaCq);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create rmsnormGammaCkv aclTensor.
      ret = CreateAclTensorND(rmsnormGammaCkvShape, &rmsnormGammaCkvDeviceAddr, &rmsnormGammaCkvHostAddr, aclDataType::ACL_BF16, &rmsnormGammaCkv);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create cacheIndex aclTensor.
      ret = CreateAclTensorND(cacheIndexShape, &cacheIndexDeviceAddr, &cacheIndexHostAddr, aclDataType::ACL_INT64, &cacheIndex);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create kvCache aclTensor.
      ret = CreateAclTensorND(kvCacheShape, &kvCacheDeviceAddr, &kvCacheHostAddr, aclDataType::ACL_BF16, &kvCache);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create krCache aclTensor.
      ret = CreateAclTensorND(krCacheShape, &krCacheDeviceAddr, &krCacheHostAddr, aclDataType::ACL_BF16, &krCache);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create query aclTensor.
      ret = CreateAclTensorND(queryShape, &queryDeviceAddr, &queryHostAddr, aclDataType::ACL_BF16, &query);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create queryRope aclTensor.
      ret = CreateAclTensorND(queryRopeShape, &queryRopeDeviceAddr, &queryRopeHostAddr, aclDataType::ACL_BF16, &queryRope);
      CHECK_RET(ret == ACL_SUCCESS, return ret);

      // 3. Call the CANN operator library API. Change it to the specific API.
      uint64_t workspaceSize = 0;
      aclOpExecutor* executor = nullptr;
      // Call the first-phase API of aclnnMlaProlog.
      ret = aclnnMlaPrologGetWorkspaceSize(tokenX, weightDq, weightUqQr, weightUk, weightDkvKr, rmsnormGammaCq, rmsnormGammaCkv, ropeSin, ropeCos, cacheIndex, kvCache, krCache, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, rmsnormEpsilonCq, rmsnormEpsilonCkv, cacheMode, query, queryRope, &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMlaPrologGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
      // Request device memory based on the workspaceSize calculated by the first-phase API.
      void* workspaceAddr = nullptr;
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
      }
      // Call the second-phase API of aclnnMlaProlog.
      ret = aclnnMlaProlog(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMlaProlog failed. ERROR: %d\n", ret); return ret);

      // 4. Boilerplate code. Wait synchronously for task execution to complete.
      ret = aclrtSynchronizeStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

      // 5. Obtain the output value and copy the result from device memory to the host side. Modify it according to the specific API definition.
      auto size = GetShapeSize(queryShape);
      std::vector<float> resultData(size, 0);
      ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), queryDeviceAddr, size * sizeof(float),
                        ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);

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
