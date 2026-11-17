# aclnnQuantMatmulAllReduceV5

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T13:38:14.303Z -->

## Applicable Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 products</term>: Not supported
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

- **Description**: Compatible with the functions supported by `aclnnQuantMatmulAllReduce`, `aclnnQuantMatmulAllReduceV2`, `aclnnQuantMatmulAllReduceV3`, and `aclnnQuantMatmulAllReduceV4`. Adds the communication engine parameter commMode to allow selection of a communication engine.
- **Formula**: Same as `aclnnQuantMatmulAllReduceV4`. For details, see [aclnnQuantMatmulAllReduceV4 Formula](aclnnQuantMatmulAllReduceV4_en.md#function).

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call `aclnnQuantMatmulAllReduceV5GetWorkspaceSize` first to obtain the workspace size required for computation and the executor that contains the computation process of the operator, and then call `aclnnQuantMatmulAllReduceV5` to execute the computation.

```cpp
aclnnStatus aclnnQuantMatmulAllReduceV5GetWorkspaceSize(
    const aclTensor *x1,
    const aclTensor *x2,
    const aclTensor *biasOptional,
    const aclTensor *x3Optional,
    const aclTensor *x1ScaleOptional,
    const aclTensor *x2ScaleOptional,
    const aclTensor *commQuantScale1Optional,
    const aclTensor *commQuantScale2Optional,
    const char      *group,
    const char      *reduceOp,
    const char      *commMode,
    int64_t          commTurn,
    int64_t          streamMode,
    int64_t          groupSize,
    int64_t          commQuantMode,
    const aclTensor *output,
    uint64_t        *workspaceSize,
    aclOpExecutor  **executor)
```

```cpp
aclnnStatus aclnnQuantMatmulAllReduceV5(
    void              *workspace,
    uint64_t           workspaceSize,
    aclOpExecutor     *executor,
    const aclrtStream  stream)
```

## aclnnQuantMatmulAllReduceV5GetWorkspaceSize

- **Parameters**
    <table style="undefined;table-layout: fixed; width: 1567px"><colgroup>
      <col style="width: 170px">
      <col style="width: 120px">
      <col style="width: 300px">
      <col style="width: 330px">
      <col style="width: 212px">
      <col style="width: 100px">
      <col style="width: 190px">
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
        </tr></thead>
      <tbody>
        <tr>
          <td>x1</td>
          <td>Input</td>
          <td>Left matrix of the MatMul computation, that is, x1 in the Formula.</td>
          <td><ul><li>Current version supports only 2D or 3D input.</li><li>Non-transposed use cases are supported.</li></ul></td>
          <td>INT8, FLOAT8_E4M3FN, FLOAT8_E5M2, HIFLOAT8, FLOAT4_E2M1.</td>
          <td>ND</td>
          <td>2-3</td>
          <td>×</td>
        </tr>
        <tr>
          <td>x2</td>
          <td>Input</td>
          <td>Right matrix of the MatMul computation, that is, x2 in the Formula.</td>
          <td><ul><li>Current version supports only 2D input.</li><li>Transpose/non-transpose use cases are supported.</li><li>In ND format, non-contiguous tensors with the last two axes transposed are supported. Other non-contiguous tensors are not supported.</li><li>The n axis cannot be 1.</li></ul></td>
          <td>INT8, FLOAT8_E4M3FN, FLOAT8_E5M2, HIFLOAT8, FLOAT4_E2M1.</td>
          <td>ND</td>
          <td>2</td>
          <td>×</td>
        </tr>
        <tr>
          <td>biasOptional</td>
          <td>Input</td>
          <td>The bias offset, which is biasOptional in the Formula.</td>
          <td><ul><li>Current version supports only 1D input.</li></ul></td>
          <td>INT32, FLOAT32</td>
          <td>ND</td>
          <td>1</td>
          <td>√</td>
        </tr>
        <tr>
          <td>x3Optional</td>
          <td>Input</td>
          <td>The add computation after the MatMul computation, that is, x3Optional in the Formula.</td>
          <td><ul><li>In the low-bit communication use case, only the BFLOAT16 output is supported, and only non-empty input is supported. The data type and dimensions must be the same as the dimensions of output.</li></ul></td>
          <td>FLOAT16, BFLOAT16, FLOAT32</td>
          <td>ND</td>
          <td>2-3</td>
          <td>√</td>
        </tr>
        <tr>
          <td>x1ScaleOptional</td>
          <td>Input</td>
          <td>Pertoken dequantization coefficient after MatMul computation, that is, x1ScaleOptional in the formula.</td>
          <td><ul><li>In the pertoken use case, if x1 is (b, m, k), the shape is (b*m); if x1 is (m, k), the shape is (m).</li><li>In the perblock use case, if x1 is (b, m, k), the shape is [b, ceilDiv(m, 128), ceilDiv(k, 128)]; if x1 is (m, k), the shape is [ceilDiv(m, 128), ceilDiv(k, 128)].</li><li>If the data type is FLOAT8_E8M0, the shape is [m, ceilDiv(k, 64), 2]; if x1 is FLOAT4_E2M1, ceilDiv(k, 32) must be an even number.</li></ul></td>
          <td>FLOAT32, FLOAT8_E8M0</td>
          <td>ND</td>
          <td>1-3</td>
          <td>√</td>
        </tr>
        <tr>
          <td>x2ScaleOptional</td>
          <td>Input</td>
          <td>Dequantization coefficient after MatMul computation, that is, x2Scale in the formula.</td>
          <td><ul><li>The shape is (1) in the pertensor use case and (n)/(1, n) in the perchannel use case.</li><li>If the input is int8 and the output is BFLOAT16, directly pass x2ScaleOptional of the BFLOAT16 type to this API.</li><li>If the output is FLOAT16 and the input is INT8, directly pass x2ScaleOptional of the FLOAT32 type to this API when x1ScaleOptional is not empty; if x1ScaleOptional is empty, call the aclnn API of the TransQuantParamV2 operator in advance to convert x2ScaleOptional to the INT64/UINT64 data type.</li><li>If the data type is FLOAT8_E8M0, only the transposed layout is supported, and the shape is [n, ceilDiv(k, 64), 2].</li> <li>If x2 is FLOAT4_E2M1, ensure that ceilDiv(k, 32) is an even number.</li><li>In the perblock use case, the shape of x2 is [ceilDiv(k, 128), ceilDiv(n, 128)]. If x2 is transposed, the shape of x2ScaleOptional is [ceilDiv(n, 128), ceilDiv(k, 128)].</li><li>When the dtype is FLOAT8_E8M0, the ceilDiv(k, 64) axis cannot be 1.</li></ul></td>
          <td>INT64, UINT64, FLOAT32, BFLOAT16, FLOAT8_E8M0</td>
          <td>ND</td>
          <td>1-3</td>
          <td>√</td>
        </tr>
        <tr>
          <td>commQuantScale1Optional</td>
          <td>Input</td>
          <td>Perchannel quantization coefficient after MatMul+Add computation, that is, commQuantScale1Optional in the formula.</td>
          <td><ul><li>Supported only when the input is of the int8 type in the current version. Pass in an empty value in other use cases.</li><li>When x2 is (k, n), the shape can be (n) or (1,n).</li></ul></td>
          <td>BFLOAT16, FLOAT16</td>
          <td>ND</td>
          <td>1-2</td>
          <td>√</td>
        </tr>
        <tr>
          <td>commQuantScale2Optional</td>
          <td>Input</td>
          <td>Perchannel quantization coefficients after the AllGather computation, that is, commQuantScale2Optional in the Formula.</td>
          <td><ul><li>Supported only when the input is of the int8 type in the current version. Pass in an empty value in other use cases.</li><li>When x2 is (k, n), the shape can be (n) or (1,n).</li></ul></td>
          <td>BFLOAT16, FLOAT16</td>
          <td>ND</td>
          <td>1-2</td>
          <td>√</td>
        </tr>
        <tr>
          <td>group</td>
          <td>Input</td>
          <td>Communicator name.</td>
          <td><ul><li>Obtained through the Hccl API "extern HcclResult HcclGetCommName(HcclComm comm, char* commName);", where commName is group.</li></ul></td>
          <td>String</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td>reduceOp</td>
          <td>Input</td>
          <td>reduce operation type.</td>
          <td><ul><li>The current version supports only "sum".</li></ul></td>
          <td>String</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td>commMode</td>
          <td>Input</td>
          <td>Communication engine parameter.</td>
          <td>Specifies the communication engine. Input restrictions vary by chip model. For a brief introduction to the communication engine and input constraints, see <a href="#constraints">Constraints</a>.</td>
          <td>String</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td>commTurn</td>
          <td>Input</td>
          <td>Number of times the communication data is split, that is, total data volume/communication volume per communication.</td>
          <td><ul><li>The current version supports only input 0.</li></ul></td>
          <td>INT64</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td>streamMode</td>
          <td>Input</td>
          <td>Enumeration of stream modes.</td>
          <td><ul><li>The current version supports only enumeration value 1.</li></ul></td>
          <td>INT64</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td>groupSize</td>
          <td>Input</td>
          <td>Indicates, in dequantization, how many x1/x2 input values along the corresponding dimension can be dequantized with one x1Scale/x2Scale input value.</td>
          <td><ul><li>The groupSize input consists of 3 values: groupSizeM, groupSizeN, and groupSizeK, one for each of the 3 directions. Each value occupies 16 bits. Formula: groupSize = groupSizeK | groupSizeN << 16 | groupSizeM << 32.</li><li>In the perblock scenario, only groupSizeM, groupSizeN, groupSizeK = 128,128,128 is supported.</li><li>In the MXFP scenario, only groupSizeM, groupSizeN, groupSizeK = 1,1,32 is supported.</li><li>Automatic parameter derivation is supported. If any parameter is set to 0, the operator derives its value. If all are set to 0, the operator derives all values.</li></ul></td>
          <td>INT64</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td>commQuantMode</td>
          <td>Input</td>
          <td>Flag indicating static quantization or dynamic quantization.</td>
          <td><ul><li>Values are 0 and 1. 1 is supported only when x1 and x2 are FLOAT8_E4M3FN or FLOAT8_E5M2. When set to 1, the Pertile quantization Fp8 communication use case applies.</li></ul></td>
          <td>INT64</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td>Output</td>
          <td>Output</td>
          <td>Result of the MatMul computation and AllReduce communication, that is, the output in the Formula.</td>
          <td><ul><li>The dimensions of output are the same as those of x1.</li></ul></td>
          <td>FLOAT16, BFLOAT16, FLOAT32</td>
          <td>ND</td>
          <td>2-3</td>
          <td>√</td>
        </tr>
        <tr>
          <td>workspaceSize</td>
          <td>Output</td>
          <td>Returns the workspace size required on the Device side.</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td>executor</td>
          <td>Output</td>
          <td>Returns the op executor, which contains the operator computation flow.</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
      </tbody>
    </table>

- **Return Value**

  Returns an aclnnStatus status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table style="undefined;table-layout: fixed; width: 1149px"><colgroup>
  <col style="width: 282px">
  <col style="width: 120px">
  <col style="width: 747px">
  </colgroup>
  <thead>
  <tr>
      <th>Return Value</th>
      <th>Error Code</th>
      <th>Description</th>
  </tr></thead>
  <tbody>
  <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>x1, x2, x2Scale, reduceOp, or output is a null pointer.</td>
  </tr>
  <tr>
      <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="3">161002</td>
      <td>The data type of x1, x2, biasOptional, x1ScaleOptional, x2Scale, x3Optional, commQuantScale1Optional, commQuantScale2Optional, or output is not within the supported range.</td>
  </tr>
  <tr>
      <td>streamMode is not within the valid range.</td>
  </tr>
  <tr>
      <td>The shape of x1, x2, biasOptional, x1ScaleOptional, x2Scale, x3Optional, commQuantScale1Optional, commQuantScale2Optional, or output does not meet the constraints.</td>
  </tr>
  </tbody>
  </table>

## aclnnQuantMatmulAllReduceV5

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
  <col style="width: 168px">
  <col style="width: 128px">
  <col style="width: 854px">
  </colgroup>
  <thead>
  <tr>
      <th>Parameter</th>
      <th>Input/Output</th>
      <th>Description</th>
  </tr></thead>
  <tbody>
  <tr>
      <td>workspace</td>
      <td>Input</td>
      <td>Workspace memory address allocated on the Device.</td>
  </tr>
  <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Workspace size allocated on the Device, obtained by the first-phase API <code>aclnnQuantMatmulAllReduceV5GetWorkspaceSize</code>.</td>
  </tr>
  <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Op executor that contains the operator computation flow.</td>
  </tr>
  <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream that executes tasks.</td>
  </tr>
  </tbody></table>

- **Return Value**

  Returns an aclnnStatus status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Communication engine commMode support:

  <!-- npu="910b" id7 -->
  - <term>Atlas A2 products</term>: Specifying a communication engine is not currently supported. commMode only allows "ai_cpu", which uses the AICPU communication engine.
  <!-- end id7 -->
  <!-- npu="950" id8 -->
  - <term>Ascend 950PR&950DT products</term>: Currently, the communication engine supports AICPU and CCU, and commMode allows "ai_cpu" or "ccu". CCU supports only single-machine UB domain interconnection, while AICPU supports cross-machine UB domain interconnection.
  <!-- end id8 -->
  - Within the same communication link, only one type of communication engine can be selected.
  - Brief introduction to the AICPU and CCU communication engines:
    - AICPU: Does not occupy compute cores and delivers high communication efficiency, but incurs large static communication overhead. Not suitable for small-data communication use cases. Suitable for large-data, high-bandwidth use cases.
    - CCU: Reduces memory access bandwidth and compute core usage, but supports a limited number of communication domains due to on-chip resource limits. Suitable for high-bandwidth, low-latency communication use cases.
    - For more details about communication engines, see [communication engine](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910beta1/programug/commopdev/hcclopdev_000005.html).
- Deterministic computation:

  <!-- npu="910b" id9 -->
  - <term>Atlas A2 products</term>: `aclnnQuantMatmulAllReduceV5` defaults to a non-deterministic implementation. Deterministic computation can be enabled by setting the `HCCL_DETERMINISTIC` environment variable to true.
  <!-- end id9 -->
  <!-- npu="950" id10 -->
  - Ascend 950PR&950DT products: `aclnnQuantMatmulAllReduceV5` defaults to a deterministic implementation.
  <!-- end id10 -->

- MC2 is disabled in incremental scenarios and enabled in full scenarios.
- The input x1 can be 2-dimensional or 3-dimensional, with its shape being (b, s, k) or (m, k). x2 must be 2-dimensional, with its shape being (k, n). The k axis must satisfy the mm operator input requirements, meaning the k axes must be equal.
- In quantization scenarios, the n axis of x2 cannot be 1. When the dtype of x2Scale is FLOAT8_E8M0, the ceilDiv(k, 64) axis in shape [n, ceilDiv(k, 64), 2] cannot be 1. The pertensor shape (1) is not affected by this constraint.
- The size of m must not exceed 2147483647. The last dimension size of x1 and x2 must not exceed 65535. The last dimension of x1 refers to k, and the last dimension of x2 refers to k when transposed or n when not transposed.
- The passed x1, x2, x2Scale, or output must not be null pointers.
- The data types and data formats of x1, x2, dequantScale, output, bias (when not null), and x3 (when not null) must fall within the supported ranges.
- The passed commQuantScale1 and commQuantScale2 must be either both null pointers or both non-null pointers. If both commQuantScale1 and commQuantScale2 are non-null pointers, the shapes of the two quantization parameters must be consistent, their types must be consistent with the operator output type, and the inputs on each card must be consistent.
- Only HCCS all-mesh networking is supported.

    <!-- npu="910b" id11 -->
    - <term>Atlas A2 products</term>: Supports 1, 2, 4, or 8 cards.
    <!-- end id11 -->
    <!-- npu="950" id12 -->
    - <term>Ascend 950PR&950DT products</term>: Supports 1, 2, 4, 8, 16, 32, or 64 cards.
    <!-- end id12 -->

- The general-purpose computation fused MC2 operators in a model support only the same communication domain.
- INT8 and FP8 low-bit communication provides performance benefits only in communication-bound scenarios. In computation-bound scenarios, enabling INT8 or FP8 low-bit communication is not recommended, meaning that commQuantScale1 and commQuantScale2 should not be input, and commQuantMode should be set to 0. (Note: INT8 low-bit communication refers to the case where the input is int8 and commQuantScale1Optional and commQuantScale2Optional are enabled; FP8 low-bit communication refers to the case where the input is FLOAT8_E4M3FN/FLOAT8_E5M2 and commQuantMode is set to 1.)
- Null tensor support:
  - Empty tensors are not supported.
- Constraints on groupSize:
  - The groupSize value is valid only when both x1Scale and x2Scale inputs are 2-dimensional or higher. In other scenarios, pass 0.
  - The input groupSize is internally decomposed into groupSizeM, groupSizeN, and groupSizeK using the following formula. When one or more of them are 0, groupSizeM, groupSizeN, and groupSizeK are reset based on the input shapes of x1, x2, x1Scale, and x2Scale for computation. Principle: If groupSizeM = 0, it indicates that the quantization group value in the m direction is inferred by the API. The inference formula is groupSizeM = m / scaleM (m must be divisible by scaleM), where m is consistent with the m in the x1 shape, and scaleM is consistent with the m in the x1Scale shape.
    $$
    groupSize = groupSizeK | groupSizeN << 16 | groupSizeM << 32
    $$

The input and output support the following data type combinations:

<!-- npu="910b" id13 -->
- <term>Atlas A2 products</term>:
    <table>
    <thead>
        <tr>
        <th>x1</th>
        <th>x2</th>
        <th>biasOptional</th>
        <th>x3Optional</th>
        <th>x1ScaleOptional</th>
        <th>x2ScaleOptional</th>
        <th>commQuantScale1Optional</th>
        <th>commQuantScale2Optional</th>
        <th>output</th>
        <th>Restrictions</th>
        </tr>
    </thead>
    <tbody>
        <tr>
        <td>INT8</td>
        <td>INT8</td>
        <td>null, INT32</td>
        <td>null, FLOAT16</td>
        <td>null</td>
        <td>INT64, UINT64</td>
        <td>null, FLOAT16</td>
        <td>null, FLOAT16</td>
        <td>FLOAT16</td>
        <td>commQuantScale1Optional, commQuantScale2Optional must be both empty or both non-empty.</td>
        </tr>
        <tr>
        <td>INT8</td>
        <td>INT8</td>
        <td>null, INT32</td>
        <td>null, FLOAT16</td>
        <td>FLOAT32</td>
        <td>FLOAT32</td>
        <td>null, FLOAT16</td>
        <td>null, FLOAT16</td>
        <td>FLOAT16</td>
        <td>commQuantScale1Optional and commQuantScale2Optional must be both empty or both non-empty.</td>
        </tr>
        <tr>
        <td>INT8</td>
        <td>INT8</td>
        <td>null, INT32</td>
        <td>null, BFLOAT16</td>
        <td>null, FLOAT32</td>
        <td>BFLOAT16</td>
        <td>null, BFLOAT16</td>
        <td>null, BFLOAT16</td>
        <td>BFLOAT16</td>
        <td>commQuantScale1Optional and commQuantScale2Optional must both be null or both be non-null.</td>
        </tr>
    </tbody>
    </table>

<!-- end id13 -->
<!-- npu="950" id14 -->
- <term>Ascend 950PR&950DT products</term>:

    For int8 input, supports pertoken-perchannel quantization && pertensor-perchannel quantization.
    <table>
    <thead>
        <tr>
        <th>x1</th>
        <th>x2</th>
        <th>biasOptional</th>
        <th>x3Optional</th>
        <th>x1ScaleOptional</th>
        <th>x2ScaleOptional</th>
        <th>commQuantScale1Optional</th>
        <th>commQuantScale2Optional</th>
        <th>output</th>
        <th>Restrictions</th>
        </tr>
    </thead>
    <tbody>
        <tr>
        <td>INT8</td>
        <td>INT8</td>
        <td>null, INT32</td>
        <td>null, FLOAT16</td>
        <td>null</td>
        <td>INT64, UINT64</td>
        <td>null, FLOAT16</td>
        <td>null, FLOAT16</td>
        <td>FLOAT16</td>
        <td>commQuantScale1Optional, commQuantScale2Optional are both empty or both non-empty</td>
        </tr>
        <tr>
        <td>INT8</td>
        <td>INT8</td>
        <td>null, INT32</td>
        <td>null, FLOAT16</td>
        <td>FLOAT32</td>
        <td>FLOAT32</td>
        <td>null, FLOAT16</td>
        <td>null, FLOAT16</td>
        <td>FLOAT16</td>
        <td>commQuantScale1Optional and commQuantScale2Optional are both null or both non-null.</td>
        </tr>
        <tr>
        <td>INT8</td>
        <td>INT8</td>
        <td>null, INT32</td>
        <td>null, BFLOAT16</td>
        <td>null, FLOAT32</td>
        <td>BFLOAT16</td>
        <td>null, BFLOAT16</td>
        <td>null, BFLOAT16</td>
        <td>BFLOAT16</td>
        <td>commQuantScale1Optional and commQuantScale2Optional must both be empty or both be non-empty.</td>
        </tr>
    </tbody>
    </table>

    pertoken-perchannel quantization && pertoken-pertensor quantization && perblock-perblock quantization
    <table>
    <thead>
        <tr>
        <th>x1</th>
        <th>x2</th>
        <th>biasOptional</th>
        <th>x3Optional</th>
        <th>x1ScaleOptional</th>
        <th>x2ScaleOptional</th>
        <th>output</th>
        <th>Restrictions</th>
        </tr>
    </thead>
    <tbody>
        <tr>
        <td>FLOAT8_E4M3FN</td>
        <td>FLOAT8_E4M3FN</td>
        <td>null, FLOAT32</td>
        <td>null, FLOAT16, BFLOAT16, FLOAT32</td>
        <td>FLOAT32</td>
        <td>FLOAT32</td>
        <td>FLOAT16, BFLOAT16, FLOAT32</td>
        <td>The B-B quantization scenario supports only null for biasOptional.</td>
        </tr>
        <tr>
        <td>FLOAT8_E5M2</td>
        <td>FLOAT8_E5M2</td>
        <td>null, FLOAT32</td>
        <td>null, FLOAT16, BFLOAT16, FLOAT32</td>
        <td>FLOAT32</td>
        <td>FLOAT32</td>
        <td>FLOAT16, BFLOAT16, FLOAT32</td>
        <td>In the B-B quantization scenario, only null is supported for biasOptional.</td>
        </tr>
        <tr>
        <td>HIFLOAT8</td>
        <td>HIFLOAT8</td>
        <td>null, FLOAT32</td>
        <td>null, FLOAT16, BFLOAT16, FLOAT32</td>
        <td>FLOAT32</td>
        <td>FLOAT32</td>
        <td>FLOAT16, BFLOAT16, FLOAT32</td>
        <td>The B-B quantization scenario only supports biasOptional set to null.</td>
        </tr>
    </tbody>
    </table>

    perchannel quantization && pertensor quantization
    <table>
    <thead>
        <tr>
        <th>x1</th>
        <th>x2</th>
        <th>biasOptional</th>
        <th>x3Optional</th>
        <th>x1ScaleOptional</th>
        <th>x2ScaleOptional</th>
        <th>output</th>
        <th>Restrictions</th>
        </tr>
    </thead>
    <tbody>
        <tr>
        <td>FLOAT8_E4M3FN</td>
        <td>FLOAT8_E4M3FN</td>
        <td>null, FLOAT32</td>
        <td>null, FLOAT16, BFLOAT16, FLOAT32</td>
        <td>null</td>
        <td>UINT64</td>
        <td>FLOAT16, BFLOAT16</td>
        <td>-</td>
        </tr>
        <tr>
        <td>FLOAT8_E5M2</td>
        <td>FLOAT8_E5M2</td>
        <td>null, FLOAT32</td>
        <td>null, FLOAT16, BFLOAT16, FLOAT32</td>
        <td>null</td>
        <td>UINT64</td>
        <td>FLOAT16, BFLOAT16</td>
        <td>-</td>
        </tr>
        <tr>
        <td>HIFLOAT8</td>
        <td>HIFLOAT8</td>
        <td>FLOAT32</td>
        <td>FLOAT16, BFLOAT16, FLOAT32</td>
        <td>null</td>
        <td>UINT64</td>
        <td>FLOAT16, BFLOAT16</td>
        <td>-</td>
        </tr>
    </tbody>
    </table>

    MXFP quantization
    <table>
    <thead>
        <tr>
        <th>x1</th>
        <th>x2</th>
        <th>biasOptional</th>
        <th>x3Optional</th>
        <th>x1ScaleOptional</th>
        <th>x2ScaleOptional</th>
        <th>output</th>
        <th>Restrictions</th>
        </tr>
    </thead>
    <tbody>
        <tr>
        <td>FLOAT4_E2M1</td>
        <td>FLOAT4_E2M1</td>
        <td>null, FLOAT32</td>
        <td>null, FLOAT16, BFLOAT16, FLOAT32</td>
        <td>FLOAT8_E8M0</td>
        <td>FLOAT8_E8M0</td>
        <td>FLOAT16, BFLOAT16, FLOAT32</td>
        <td>-</td>
        </tr>
        <tr>
        <td>FLOAT8_E4M3FN</td>
        <td>FLOAT8_E4M3FN</td>
        <td>null, FLOAT32</td>
        <td>null, FLOAT16, BFLOAT16, FLOAT32</td>
        <td>FLOAT8_E8M0</td>
        <td>FLOAT8_E8M0</td>
        <td>FLOAT16, BFLOAT16, FLOAT32</td>
        <td>-</td>
        </tr>
        <tr>
        <td>FLOAT8_E5M2</td>
        <td>FLOAT8_E5M2</td>
        <td>null, FLOAT32</td>
        <td>null, FLOAT16, BFLOAT16, FLOAT32</td>
        <td>FLOAT8_E8M0</td>
        <td>FLOAT8_E8M0</td>
        <td>FLOAT16, BFLOAT16, FLOAT32</td>
        <td>-</td>
        </tr>
    </tbody>
    </table>

<!-- end id14 -->

## Example

The following sample code is for reference only. For details about the compilation and execution process, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

NOTE: This sample code calls some HCCL APIs: HcclGetCommName, HcclCommInitAll, and HcclCommDestroy. For details, see [<<HCCL API (C)>>](https://hiascend.com/en/document/redirect/CannCommunityHcclCppApi).

<!-- npu="950,910b" id15 -->
- <term>Atlas A2 products</term>, <term>Ascend 950PR&950DT products</term>:

  ```Cpp
  #include <iostream>
  #include <vector>
  #include <thread>
  #include "hccl/hccl.h"
  #include "aclnn/opdev/fp16_t.h"
  #include "aclnnop/aclnn_trans_matmul_weight.h"
  #include "aclnnop/aclnn_quant_matmul_all_reduce_v4.h"

  #define ACL_CHECK(ret)                                                                                     \
      do {                                                                                                   \
          auto retcode = ret;                                                                                \
          if (retcode != ACL_SUCCESS) {                                                                      \
              printf("[ERROR] acl interface return err %s:%d, retcode: %d \n", __FILE__, __LINE__, retcode); \
              return retcode;                                                                                \
          }                                                                                                  \
      } while (0)

  #define CHECK_RET(cond, return_expr) \
      do {                             \
          if (!(cond)) {               \
              return_expr;             \
          }                            \
      } while (0)

  #define LOG_PRINT(message, ...)         \
      do {                                \
          printf(message, ##__VA_ARGS__); \
      } while (0)

  constexpr int DEV_NUM = 2;

  int64_t GetShapeSize(const std::vector<int64_t> &shape)
  {
      int64_t shape_size = 1;
      for (auto i : shape) {
          shape_size *= i;
      }
      return shape_size;
  }

  struct Args {
      uint32_t rankId;
      HcclComm hcclComm;
      aclrtStream stream;
      aclrtContext context;
      std::string format;
  };

  template<typename T>
  int CreateWeightNzAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                              aclDataType dataType, aclTensor **tensor, Args &args) {
      auto size = GetShapeSize(shape) * sizeof(T);
      const aclIntArray *mat2Size = aclCreateIntArray(shape.data(), shape.size());
      auto ret = aclnnCalculateMatmulWeightSizeV2(mat2Size, ACL_INT8, &size);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCalculateMatmulWeightSizeV2 failed. ERROR: %d\n", ret); return ret);

      // Call aclrtMalloc to allocate device memory.
      ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

      // Call aclrtMemcpy to copy host-side data to device-side memory.
      ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

      // Compute the strides of the contiguous tensor.
      std::vector<int64_t> strides(shape.size(), 1);
      for (int64_t i = shape.size() - 2; i >= 0; i--) {
          strides[i] = shape[i + 1] * strides[i + 1];
      }
      // Call the aclCreateTensor API to create an aclTensor.
      *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                                shape.data(), shape.size(), *deviceAddr);

      uint64_t transWorkspaceSize;
      aclOpExecutor *executor;
      void *transWorkspaceAddr = nullptr;
      ret = aclnnTransMatmulWeightGetWorkspaceSize(*tensor, &transWorkspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS && transWorkspaceSize > 0,
                printf("[ERROR] aclnnTransMatmulWeightGetWorkspaceSize failed. ret = %d \n", ret); return ret);
      ACL_CHECK(aclrtMalloc(&transWorkspaceAddr, transWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
      ret = aclnnTransMatmulWeight(transWorkspaceAddr, transWorkspaceSize, executor, args.stream);
      CHECK_RET(ret == ACL_SUCCESS, printf("[ERROR] aclnnTransMatmulWeight failed. ret = %d \n", ret);return ret);
      ACL_CHECK(aclrtSynchronizeStreamWithTimeout(args.stream, 20000000));

      return 0;
  }

  template<typename T>
  int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                      aclDataType dataType, aclTensor **tensor) {
      auto size = GetShapeSize(shape) * sizeof(T);
      // Call aclrtMalloc to allocate device-side memory.
      auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
      // Call aclrtMemcpy to copy host-side data to device-side memory.
      ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
      // Compute the strides of the contiguous tensor.
      std::vector<int64_t> strides(shape.size(), 1);
      for (int64_t i = shape.size() - 2; i >= 0; i--) {
          strides[i] = shape[i + 1] * strides[i + 1];
      }
      // Call the aclCreateTensor API to create an aclTensor.
      *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                                shape.data(), shape.size(), *deviceAddr);
      return 0;
  }

  int launchOneThreadQuantMatmulAllReduce(Args &args) {
      int ret;
      ret = aclrtSetCurrentContext(args.context);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetCurrentContext failed. ERROR: %d\n", ret); return ret);
      char hcom_name[128];
      ret = HcclGetCommName(args.hcclComm, hcom_name);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclGetCommName failed. ret = %d \n", ret); return -1);
      LOG_PRINT("[INFO] rank %d hcom: %s stream: %p, context : %p\n", args.rankId, hcom_name, args.stream,
                args.context);

      std::vector<int64_t> x1Shape = {32, 64};
      std::vector<int64_t> x2Shape = {64, 128};
      std::vector<int64_t> biasShape = {128};
      std::vector<int64_t> dequantScaleShape = {128};
      std::vector<int64_t> x1ScaleShape = {32};
      std::vector<int64_t> commQuantScale1Shape = {128};
      std::vector<int64_t> commQuantScale2Shape = {128};
      std::vector<int64_t> x3Shape = {32, 128};
      std::vector<int64_t> outShape = {32, 128};
      void *x1DeviceAddr = nullptr;
      void *x2DeviceAddr = nullptr;
      void *biasDeviceAddr = nullptr;
      void *dequantScaleDeviceAddr = nullptr;
      void *x1ScaleDeviceAddr = nullptr;
      void *commQuantScale1DeviceAddr = nullptr;
      void *commQuantScale2DeviceAddr = nullptr;
      void *x3DeviceAddr = nullptr;
      void *outDeviceAddr = nullptr;
      aclTensor *x1 = nullptr;
      aclTensor *x2 = nullptr;
      aclTensor *bias = nullptr;
      aclTensor *x2ScaleOptional = nullptr;
      aclTensor *x1Scale = nullptr;
      aclTensor *commQuantScale1 = nullptr;
      aclTensor *commQuantScale2 = nullptr;
      aclTensor *x3 = nullptr;
      aclTensor *out = nullptr;

      int64_t commTurn = 0;
      int64_t streamMode = 1;
      uint64_t workspaceSize = 0;
      aclOpExecutor *executor;
      void *workspaceAddr = nullptr;

      long long x1ShapeSize = GetShapeSize(x1Shape);
      long long x2ShapeSize = GetShapeSize(x2Shape);
      long long biasShapeSize = GetShapeSize(biasShape);
      long long dequantScaleShapeSize = GetShapeSize(dequantScaleShape);
      long long x1ScaleShapeSize = GetShapeSize(x1ScaleShape);
      long long commQuantScale1ShapeSize = GetShapeSize(commQuantScale1Shape);
      long long commQuantScale2ShapeSize = GetShapeSize(commQuantScale2Shape);
      long long x3ShapeSize = GetShapeSize(x3Shape);
      long long outShapeSize = GetShapeSize(outShape);

      std::vector<int8_t> x1HostData(x1ShapeSize, 1);
      std::vector<int8_t> x2HostData(x2ShapeSize, 1);
      std::vector<int32_t> biasHostData(biasShapeSize, 1);
      std::vector<float> dequantScaleHostData(dequantScaleShapeSize, 1);
      std::vector<float> x1ScaleHostData(x1ScaleShapeSize, 1);
      std::vector<op::fp16_t> commQuantScale1HostData(commQuantScale1ShapeSize, 1);
      std::vector<op::fp16_t> commQuantScale2HostData(commQuantScale2ShapeSize, 1);
      std::vector<op::fp16_t> x3HostData(x3ShapeSize, 1);
      std::vector<op::fp16_t> outHostData(outShapeSize, 0);
      // Create the tensor.
      ret = CreateAclTensor(x1HostData, x1Shape, &x1DeviceAddr, aclDataType::ACL_INT8, &x1);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      if (args.format == "NZ") {
          ret = CreateWeightNzAclTensor(x2HostData, x2Shape, &x2DeviceAddr, aclDataType::ACL_INT8, &x2, args);
      } else {
          ret = CreateAclTensor(x2HostData, x2Shape, &x2DeviceAddr, aclDataType::ACL_INT8, &x2);
      }
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      ret = CreateAclTensor(biasHostData, biasShape, &biasDeviceAddr, aclDataType::ACL_INT32, &bias);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      ret = CreateAclTensor(dequantScaleHostData, dequantScaleShape, &dequantScaleDeviceAddr,
                            aclDataType::ACL_FLOAT, &x2ScaleOptional);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      ret = CreateAclTensor(x1ScaleHostData, x1ScaleShape, &x1ScaleDeviceAddr,
                            aclDataType::ACL_FLOAT, &x1Scale);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      ret = CreateAclTensor(commQuantScale1HostData, commQuantScale1Shape, &commQuantScale1DeviceAddr,
                            aclDataType::ACL_FLOAT16, &commQuantScale1);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      ret = CreateAclTensor(commQuantScale2HostData, commQuantScale2Shape, &commQuantScale2DeviceAddr,
                            aclDataType::ACL_FLOAT16, &commQuantScale2);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      ret = CreateAclTensor(x3HostData, x3Shape, &x3DeviceAddr, aclDataType::ACL_FLOAT16, &x3);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT16, &out);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Call the first-phase API.
      ret = aclnnQuantMatmulAllReduceV5GetWorkspaceSize(x1, x2, bias, x3, x1Scale, x2ScaleOptional,
                                                        commQuantScale1, commQuantScale2, hcom_name,
                                                        "sum", "", commTurn, streamMode, 0, 0, out,
                                                        &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS,
                LOG_PRINT("aclnnQuantMatmulAllReduceV5GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
      // Allocate device memory using the workspaceSize computed by the first-phase API.
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
      }
      // Call the second-phase API.
      ret = aclnnQuantMatmulAllReduceV5(workspaceAddr, workspaceSize, executor, args.stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMatmulAllReduceV5 failed. ERROR: %d\n", ret); return ret);
      // (boilerplate) Synchronize and wait for the task execution to complete.
      ret = aclrtSynchronizeStreamWithTimeout(args.stream, 10000);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
      LOG_PRINT("device%d aclnnQuantMatmulAllReduceV5 execute success \n", args.rankId);
      // Release device resources. Modify the code based on the definition of the specific API.
      if (x1 != nullptr) {
          aclDestroyTensor(x1);
      }
      if (x2 != nullptr) {
          aclDestroyTensor(x2);
      }
      if (bias != nullptr) {
          aclDestroyTensor(bias);
      }
      if (x2ScaleOptional != nullptr) {
          aclDestroyTensor(x2ScaleOptional);
      }
      if (x1Scale != nullptr) {
          aclDestroyTensor(x1Scale);
      }
      if (commQuantScale1 != nullptr) {
          aclDestroyTensor(commQuantScale1);
      }
      if (commQuantScale2 != nullptr) {
          aclDestroyTensor(commQuantScale2);
      }
      if (x3 != nullptr) {
          aclDestroyTensor(x3);
      }
      if (out != nullptr) {
          aclDestroyTensor(out);
      }
      if (x1DeviceAddr != nullptr) {
          aclrtFree(x1DeviceAddr);
      }
      if (x2DeviceAddr != nullptr) {
          aclrtFree(x2DeviceAddr);
      }
      if (biasDeviceAddr != nullptr) {
          aclrtFree(biasDeviceAddr);
      }
      if (dequantScaleDeviceAddr != nullptr) {
          aclrtFree(dequantScaleDeviceAddr);
      }
      if (x1ScaleDeviceAddr != nullptr) {
          aclrtFree(x1ScaleDeviceAddr);
      }
      if (commQuantScale1DeviceAddr != nullptr) {
          aclrtFree(commQuantScale1DeviceAddr);
      }
      if (commQuantScale2DeviceAddr != nullptr) {
          aclrtFree(commQuantScale2DeviceAddr);
      }
      if (x3DeviceAddr != nullptr) {
          aclrtFree(x3DeviceAddr);
      }
      if (outDeviceAddr != nullptr) {
          aclrtFree(outDeviceAddr);
      }
      if (workspaceSize > 0) {
          aclrtFree(workspaceAddr);
      }
      aclrtDestroyStream(args.stream);
      HcclCommDestroy(args.hcclComm);
      aclrtDestroyContext(args.context);
      aclrtResetDevice(args.rankId);
      return 0;
  }

  int main(int argc, char *argv[])
  {
      int ret;
      int32_t devices[DEV_NUM];
      for (int i = 0; i < DEV_NUM; i++) {
          devices[i] = i;
      }
      HcclComm comms[128];
      ret = aclInit(nullptr);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
      // Initialize the communicator.
      for (int i = 0; i < DEV_NUM; i++) {
          ret = aclrtSetDevice(devices[i]);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
      }
      ret = HcclCommInitAll(DEV_NUM, devices, comms);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("HcclCommInitAll failed. ERROR: %d\n", ret); return ret);
      Args args[DEV_NUM];
      aclrtStream stream[DEV_NUM];
      aclrtContext context[DEV_NUM];
      for (uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
          ret = aclrtSetDevice(rankId);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
          ret = aclrtCreateContext(&context[rankId], rankId);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateContext failed. ERROR: %d\n", ret); return ret);
          ret = aclrtCreateStream(&stream[rankId]);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
      }
      // Start multiple threads.
      std::vector<std::unique_ptr<std::thread>> threads(DEV_NUM);
      for (uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
          args[rankId].rankId = rankId;
          args[rankId].hcclComm = comms[rankId];
          args[rankId].stream = stream[rankId];
          args[rankId].context = context[rankId];
          threads[rankId].reset(
                  new(std::nothrow) std::thread(&launchOneThreadQuantMatmulAllReduce, std::ref(args[rankId])));
      }
      for (uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
          threads[rankId]->join();
      }
      aclFinalize();
      return 0;
  }
  ```

<!-- end id15 -->
