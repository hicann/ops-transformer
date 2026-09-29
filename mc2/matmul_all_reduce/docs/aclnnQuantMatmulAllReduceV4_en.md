# aclnnQuantMatmulAllReduceV4

[📄 View Source Code](https://gitcode.com/cann/ops-transformer/tree/master/mc2/matmul_all_reduce)

## Supported Products

| Product                                                                                    | Supported|
| :--------------------------------------------------------------------------------------- | :------: |
| Ascend 950PR/Ascend 950DT                                                                     |    √    |
| <term>Atlas A3 training products/Atlas A3 inference products</term>                       |    ×    |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √    |
| <term>Atlas 200I/500 A2 inference products</term>                                        |    ×    |
| <term>Atlas inference products</term>                                                |    ×    |
| <term>Atlas training products</term>                                                |    ×    |

## Function

- API function: This API is compatible with the functions supported by aclnnQuantMatmulAllReduce, aclnnQuantMatmulAllReduceV2, and aclnnQuantMatmulAllReduceV3.
  - <term>Atlas A2 training products/Atlas A2 inference products</term>: No new features are added.
  - Ascend 950PR/Ascend 950DT: The perblock, pertile, and mxfp quantization modes are added. The x1 and x2 inputs support the data type of FLOAT8_E4M3FN, FLOAT8_E5M2, HIFLOAT8 and FLOAT4_E2M1.
- **Formula**:

  - Formula 1 and formula 2 or 3 for enabling low-bit communication:
  
    When x1 and x2 are of type INT8 and commQuantScale1Optional and commQuantScale2Optional are not empty:

    $$
    matmulAddOutput = (x2ScaleOptional * x1ScaleOptional * (x1_{int8}@x2_{int8} + biasOptional_{int32}) + x3Optional);
    $$

    $$
    alltoallOutput_{int8} = AllToAll(matmulAddOutput / commQuantScale1Optional);
    $$

    $$
    reduceSumOutput_{int8} = (add(alltoallOutput_{int8}) * (commQuantScale1Optional / commQuantScale2Optional));
    $$

    $$
    output = (AllGather(reduceSumOutput_{int8}) * commQuantScale2Optional);
    $$

  - Formula 2 (per-channel quantization and per-tensor quantization):
  
    x1 and x2 are of type INT8. x1ScaleOptional is not supported. x2ScaleOptional is of type INT64 or UINT64. biasOptional is of type INT32 (optional). out is of type BFLOAT16 or FLOAT16.

    $$
    output = AllReduce((x1@x2 + biasOptional) * x2ScaleOptional + x3Optional)
    $$

  - Formula 3 (per-token per-channel quantization and per-token per-tensor quantization):
    
    x1 and x2 are of type INT8. x1ScaleOptional is of type FLOAT32. x2Scale is of type FLOAT32 or BFLOAT16. biasOptional is of type INT32 (optional). out is of type FLOAT16 or BFLOAT16.

    $$
    output = AllReduce((x1@x2 + biasOptional) * x2ScaleOptional * x1ScaleOptional + x3Optional)
    $$

  - Formula 4 (MXFP quantization):
    
    x1 and x2 are of type FLOAT4_E2M1/FLOAT8_E4M3FN/FLOAT8_E5M2. x1ScaleOptional is of type FLOAT8_E8M0. x2ScaleOptional is of type FLOAT8_E8M0. biasOptional is of type FLOAT32 (optional). out is of type FLOAT16, BFLOAT16, or FLOAT32.

    $$
    output = AllReduce((x1* x1ScaleOptional)@(x2* x2ScaleOptional) + biasOptional + x3Optional)
    $$

  - Formula 5 (per-channel quantization and per-tensor quantization):

    x1 and x2 are of type FLOAT8_E4M3FN/FLOAT8_E5M2/HIFLOAT8, x2ScaleOptional is of type UINT/INT64, the optional bias is of type FLOAT32, and out is of type FLOAT16/BFLOAT16/FLOAT32.

  $$
  output = AllReduce((x1@x2 + biasOptional) * x2ScaleOptional  + x3Optional)
  $$

  - Formula 6 (per-token-per-channel quantization and per-token-per-tensor quantization):
  
    x1 and x2 are of type FLOAT8_E4M3FN/FLOAT8_E5M2/HIFLOAT8, x1ScaleOptional and x2ScaleOptional are of type FLOAT32, the optional bias is of type FLOAT32, and out is of type FLOAT16/BFLOAT16/FLOAT32.

    $$
    output = AllReduce((x1@x2 + biasOptional) * x2ScaleOptional * x1ScaleOptional + x3Optional)
    $$

  - Formula 7 (per-block-per-block quantization):
  
    x1 and x2 are of type FLOAT8_E4M3FN/FLOAT8_E5M2/HIFLOAT8, x1ScaleOptional is of type FLOAT32, x2Scale is of type FLOAT32, and there is no biasOptional. When x1 is (a0, a1) and x2 is (b0, b1), x1ScaleOptional is (ceilDiv(a0, 128), ceilDiv(a1, 128)), x2Scale is (ceilDiv(b0, 128), ceilDiv(b1, 128)), and out is of type FLOAT16/BFLOAT16/FLOAT32.

    $$
    output_{pq} = AllReduce(\sum_{0}^{\left \lfloor \frac{k}{128} \right \rfloor} (x1_{pr}@x2_{rq}*(x1ScaleOptional_{pr}*x2Scale_{rq})) + x3Optional)
    $$

  - Formula 8 (enabling low-bit communication and per-tile quantization):

      x1 and x2 are of type FLOAT8_E4M3FN/FLOAT8_E5M2, x1ScaleOptional is of type FLOAT32, x2Scale is of type FLOAT32, biasOptional is of type FLOAT32 (optional), commQuantMode is 1, and out is of type FLOAT16/BFLOAT16/FLOAT32.

    $$
    matmulAddOutput_{fp32} = (x2ScaleOptional * x1ScaleOptional * (x1_{fp8}@x2_{fp8} + biasOptional_{fp32}) + x3Optional);
    $$

    $$
    scaleOut_{fp32} = (matmulAddOutput_{fp32} / (reduceMax(abs(matmulAddOutput_{fp32})) / FP32\_MAX));
    $$

    $$
    quantOutput_{fp8} = (append((matmulAddOutput_{fp32} * scaleOut{fp32})@scaleOut_{fp32}));
    $$

    $$
    alltoallOutput_{fp8} = (AllToAll(quantOut_{fp8}));
    $$

    $$
    dequantOutput_{fp32} = (alltoallOutput_{fp8} / scaleOut_{fp32});
    $$

    $$
    reduceSumOutput_{fp32} = (reduceSum(dequantOutput_{fp32}));
    $$

    $$
    preAllGatherQuantScale_{fp32} = (reduceSumOutput_{fp32} / (reduceMax(abs(reduceSumOutput_{fp32})) / FP8\_MAX));
    $$

    $$
    preAllGatherQuantOutput_{fp8} = (append((reduceSumOutput_{fp32} * preAllGatherQuantScale_{fp32})@preAllGatherQuantScale_{fp32}));
    $$

    $$
    allGatherOutput_{fp8} = (AllGather(preAllGatherQuantOutput_{fp8}));
    $$

    $$
    output = (cast(allGatherOutput_{fp32} / preAllGatherQuantScale_{fp32}));
    $$

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnQuantMatmulAllReduceV4GetWorkspaceSize` is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, `aclnnQuantMatmulAllReduceV4` is called to perform computation.

```cpp
aclnnStatus aclnnQuantMatmulAllReduceV4GetWorkspaceSize(
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
    int64_t          commTurn,
    int64_t          streamMode,
    int64_t          groupSize,
    int64_t          commQuantMode,
    const aclTensor *output,
    uint64_t        *workspaceSize,
    aclOpExecutor  **executor)
```

```cpp
aclnnStatus aclnnQuantMatmulAllReduceV4(
    void              *workspace,
    uint64_t           workspaceSize,
    aclOpExecutor     *executor,
    const aclrtStream  stream)
```

## aclnnQuantMatmulAllReduceV4GetWorkspaceSize

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
          <th>Name</th>
          <th>Input/Output</th>
          <th>Description</th>
          <th>Usage</th>
          <th>Data Type</th>
          <th>Data Format</th>
          <th>Dimension (Shape)</th>
          <th>Non-contiguous Tensor</th>
        </tr></thead>
      <tbody>
        <tr>
          <td>x1</td>
          <td>Input</td>
          <td>Left matrix of MatMul computation, that is, <code>x1</code> in the formula.</td>
          <td><ul><li>The current version supports only 2D or 3D inputs. </li><li>The non-transpose scenario is supported.</li></ul></td>
          <td>INT8, FLOAT8_E4M3FN, FLOAT8_E5M2, HIFLOAT8, FLOAT4_E2M1.</td>
          <td>ND</td>
          <td>2-3</td>
          <td>×</td>
        </tr>
        <tr>
          <td>x2</td>
          <td>Input</td>
          <td>Right matrix of MatMul computation, that is, <code>x2</code> in the formula.</td>
          <td><ul><li>The current version supports only 2D inputs. </li><li>The transpose and non-transpose scenarios are supported. </li><li>In ND format, only discontinuous tensors with the last two axes transposed are supported. Other discontinuous tensors are not supported.</li></ul></td>
          <td>INT8, FLOAT8_E4M3FN, FLOAT8_E5M2, HIFLOAT8, FLOAT4_E2M1.</td>
          <td>ND</td>
          <td>2</td>
          <td>×</td>
        </tr>
        <tr>
          <td>biasOptional</td>
          <td>Input</td>
          <td>Bias offset, that is, biasOptional in the formula.</td>
          <td><ul><li>The current version supports only one-dimensional inputs.</li></ul></td>
          <td>INT32, FLOAT32</td>
          <td>ND</td>
          <td>1</td>
          <td>√</td>
        </tr>
        <tr>
          <td>x3Optional</td>
          <td>Input</td>
          <td>Addition after MatMul computation, that is, x3Optional in the formula.</td>
          <td>In low-bit communication scenarios, only the output in BFLOAT16 format is supported, and only non-empty inputs are supported. The data type and dimension must be the same as those of the output</td>
          <td>FLOAT16, BFLOAT16, FLOAT32</td>
          <td>ND</td>
          <td>2-3</td>
          <td>√</td>
        </tr>
        <tr>
          <td>x1ScaleOptional</td>
          <td>Input</td>
          <td>Pertoken dequantized coefficient after MatMul computation, that is, x1ScaleOptional in the formula.</td>
          <td><ul><li>In the pertoken scenario, if x1 is (b, m, k), the shape is (b*m); if x1 is (m, k), the shape is (m). </li><li>In the perblock scenario, if x1 is (b, m, k), the shape is [b, ceilDiv(m, 128), ceilDiv(k, 128)]; if x1 is (m, k), the shape is [ceilDiv(m, 128), ceilDiv(k, 128)]. </li><li>If the data type is FLOAT8_E8M0, the shape is [m, ceilDiv(k, 64), 2]. If x1 is FLOAT4_E2M1, ceilDiv(k, 32) must be an even number.</li></ul></td>
          <td>FLOAT32, FLOAT8_E8M0</td>
          <td>ND</td>
          <td>1-3</td>
          <td>√</td>
        </tr>
        <tr>
          <td>x2ScaleOptional</td>
          <td>Input</td>
          <td>Dequantized coefficient after MatMul computation, that is, x2Scale in the formula.</td>
          <td><ul><li>In the pertensor scenario, the shape is (1); in the perchannel scenario, the shape is (n)/(1, n). </li><li>When the input is int8 and the output is BFLOAT16, the x2ScaleOptional of the BFLOAT16 type is directly passed to this API. </li><li> If the output is FLOAT16 and the input is INT8, x1ScaleOptional is not empty. x2ScaleOptional of the FLOAT32 type can be directly transferred to this API. If x1ScaleOptional is empty, in this case, you need to call the aclnn API of the TransQuantParamV2 operator in advance to convert x2ScaleOptional to the INT64/UINT64 data type. </li><li> When the data type is FLOAT8_E8M0, only transpose is supported, and the shape is [n, ceilDiv(k, 64), 2]. </li> <li>When x2 is of type FLOAT4_E2M1, ensure that ceilDiv(k, 32) is an even number. </li><li>In the perblock scenario, the shape of x2 is [ceilDiv(k, 128), ceilDiv(n, 128)]. When x2 is transposed, the shape of x2ScaleOptional is [ceilDiv(n, 128), ceilDiv(k, 128)].</li></ul></td>
          <td>INT64, UINT64, FLOAT32, BFLOAT16, FLOAT8_E8M0</td>
          <td>ND</td>
          <td>1-3</td>
          <td>√</td>
        </tr>
        <tr>
          <td>commQuantScale1Optional</td>
          <td>Input</td>
          Perchannel quantization coefficient after <td>MatMul+Add calculation, that is, commQuantScale1Optional in the formula.</td>
          <td><ul><li> In the current version, this parameter is supported only when the input is of the int8 type. In other scenarios, this parameter is left empty. </li><li>If x2 is (k, n), shape can be (n) or (1,n).</li></ul></td>
          <td>BFLOAT16, FLOAT16</td>
          <td>ND</td>
          <td>1-2</td>
          <td>√</td>
        </tr>
        <tr>
          <td>commQuantScale2Optional</td>
          <td>Input</td>
          <td>Per-channel quantization coefficient after AllGather computation, that is, commQuantScale2Optional in the formula.</td>
          <td><ul><li>In the current version, this parameter is supported only when the input is of the int8 type. In other scenarios, this parameter is left empty. </li><li>When x2 is (k, n), the shape can be (n) or (1,n).</li></ul></td>
          <td>BFLOAT16, FLOAT16</td>
          <td>ND</td>
          <td>1-2</td>
          <td>√</td>
        </tr>
        <tr>
          <td>group</td>
          <td>Input</td>
          <td>Communication domain name.</td>
          <td><ul><li>Obtain the value using the extern HcclResult HcclGetCommName(HcclComm comm, char* commName) API provided by HCCL. commName indicates the group.</li></ul></td>
          <td>String</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td>reduceOp</td>
          <td>Input</td>
          <td><code>reduce</code> operation type.</td>
          <td><ul><li>In the current version, only the input "sum" is supported.</li></ul></td>
          <td>String</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td>commTurn</td>
          <td>Input</td>
          <td>Number of communication data splits, that is, the total data volume divided by single communication volume.</td>
          <td><ul><li>In the current version, only 0 can be entered.</li></ul></td>
          <td>INT64</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td>streamMode</td>
          <td>Input</td>
          <td>Enumeration of the stream mode.</td>
          <td><ul><li>In the current version, only the enumerated value 1 is supported.</li></ul></td>
          <td>INT64</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td>groupSize</td>
          <td>Input</td>
          <td>Number of times that the x1Scale/x2Scale input in the dequantization process can be used for the x1/x2 input in the corresponding dimension direction.</td>
          <td><ul><li>The groupSize input consists of three values: groupSizeM, groupSizeN, and groupSizeK. Each value occupies 16 bits. The calculation formula is as follows: groupSize = groupSizeK | groupSizeN << 16 | groupSizeM << 32. </li><li>In the perblock scenario, only groupSizeM and groupSizeN are supported. In the groupSizeK = 128,128,128</li><li>MXFP scenario, only groupSizeM and groupSizeN are supported. In the groupSizeK = 1,1,32</li><li> scenario, parameters can be automatically derived. If any parameter is set to 0, the operator automatically derives the value of the parameter. If all parameters are set to 0, the operator automatically derives the values of all parameters.</li></ul></td>
          <td>INT64</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td>commQuantMode</td>
          <td>Input</td>
          <td>Static quantization and dynamic quantization flag.</td>
          <td><ul><li>The value can be 0 or 1. The value 1 is supported only when x1 and x2 are FLOAT8_E4M3FN or FLOAT8_E5M2. When the value is 1, the quantization is performed in the Pertile quantization Fp8 communication scenario.</li></ul></td>
          <td>INT64</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td>output</td>
          <td>Output</td>
          <td>Result of MatMul computation and AllReduce communication, that is, <code>output</code> in the formula.</td>
          <td><ul><li>The number of dimensions of output is the same as that of x1.</li></ul></td>
          <td>FLOAT16, BFLOAT16, FLOAT32</td>
          <td>ND</td>
          <td>2-3</td>
          <td>√</td>
        </tr>
        <tr>
          <td>workspaceSize</td>
          <td>Output</td>
          <td>Size of the workspace to be allocated on the device.</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td>executor</td>
          <td>Output</td>
          <td>Operator executor, containing the operator computation process.</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
      </tbody>
    </table>

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API implements input parameter verification. The following errors may be thrown.

  <table style="undefined;table-layout: fixed; width: 1149px"><colgroup>
  <col style="width: 282px">
  <col style="width: 120px">
  <col style="width: 747px">
  </colgroup>
  <thead>
  <tr>
      <th>Return</th>
      <th>Error Code</th>
      <th>Description</th>
  </tr></thead>
  <tbody>
  <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>The input x1, x2, x2Scale, reduceOp, or output is a null pointer.</td>
  </tr>
  <tr>
      <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="3">161002</td>
      <td>x1, x2, biasOptional, x1ScaleOptional, x2Scale, x3Optional, commQuantScale1Optional, commQuantScale2Optional, or output is not of the supported data type.</td>
  </tr>
  <tr>
      <td>The value of <code>streamMode</code> is invalid.</td>
  </tr>
  <tr>
      <td>The shape of x1, x2, biasOptional, x1ScaleOptional, x2Scale, x3Optional, commQuantScale1Optional, commQuantScale2Optional, or output does not meet the requirements.</td>
  </tr>
  </tbody>
  </table>

## aclnnQuantMatmulAllReduceV4

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
  <col style="width: 168px">
  <col style="width: 128px">
  <col style="width: 854px">
  </colgroup>
  <thead>
  <tr>
      <th>Name</th>
      <th>Input/Output</th>
      <th>Description</th>
  </tr></thead>
  <tbody>
  <tr>
      <td>workspace</td>
      <td>Input</td>
      <td>Memory address of the workspace to be allocated on the device.</td>
  </tr>
  <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace to be allocated on the device, which is obtained by calling the first-phase API `aclnnQuantMatmulAllReduceV4GetWorkspaceSize`.</td>
  </tr>
  <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Operator executor, containing the operator computation process.</td>
  </tr>
  <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream for executing the task.</td>
  </tr>
  </tbody></table>

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - <term>Atlas A2 training products/Atlas A2 inference products</term>: aclnnQuantMatmulAllReduceV4 is implemented in non-deterministic mode by default. You can enable deterministic computing by setting the environment variable HCCL_DETERMINISTIC to true.
  - Ascend 950PR/Ascend 950DT: aclnnQuantMatmulAllReduceV4 is implemented in deterministic mode by default.
- MC2 is disabled in incremental scenarios but enabled in full scenarios.
- The input x1 can be 2D or 3D, and its shape is (b, s, k) or (m, k). **x2** must be 2-dimensional. Its shape is (k, n). The k axes meet the input parameter requirements of the mm operator and are equal.
- The value of `m` cannot exceed `2147483647`. The last dimension of `x1` is `k`, and that of `x2` is `k` in the transpose scenario or `n` in the non-transpose scenario. The size of the last dimensions of `x1` and `x2` cannot exceed `65535`.
- The input x1, x2, x2Scale, or output is not a null pointer.
- The data types and formats of `x1`, `x2`, `dequantScale`, `output`, `bias` (when not empty), and `x3` (when not empty) must be supported.
- If the passed `commQuantScale1` and `commQuantScale2` are not null pointers, their shapes must be the same, their types must be the same as the operator output type, and the inputs for each rank must be the same.
- Only the all-mesh networking of HCCS links is supported.
    - <term>Atlas A2 training products/Atlas A2 inference products</term>: 1, 2, 4, and 8 ranks are supported.
    - Ascend 950PR/Ascend 950DT: 1, 2, 4, 8, 16, 32, and 64 cards are supported.
- The merged compute and communication (MC2) operators in a model support only the same communicator.
- The performance gain of INT8 and FP8 low-bit communication is available only when the communication bound is reached. In the case of the computation bound, you are advised not to enable INT8 or FP8 low-bit communication, that is, you are advised not to set commQuantScale1 and commQuantScale2 and set commQuantMode to 0. (Note: INT8 low-bit communication refers to the scenario where the input is int8 and commQuantScale1Optional and commQuantScale2Optional are enabled. FP8 low-bit communication refers to the scenario where the input is FLOAT8_E4M3FN/FLOAT8_E5M2 and commQuantMode is set to 1.)
- Support for empty tensors:
  - Empty tensors are not supported.
- Restrictions on groupSize:
  - The value of groupSize is valid only when both x1Scale and x2Scale are 2D or higher-dimensional data. In other scenarios, 0 needs to be passed.
  - The input groupSize is decomposed into groupSizeM, groupSizeN, and groupSizeK according to the following formulas. If one or more of them are 0, groupSizeM, groupSizeN, and groupSizeK are reset based on the input shape of x1, x2, x1Scale, or x2Scale for calculation. Principle: Assume that groupSizeM = 0, indicating that the quantization group size in the m direction is inferred by the API. The inference formula is groupSizeM = m / scaleM (ensure that m is exactly divisible by scaleM). m is the same as that in the x1 shape, and scaleM is the same as that in the x1Scale shape.
    $$
    groupSize = groupSizeK | groupSizeN << 16 | groupSizeM << 32
    $$

The following table describes the supported input and output data type combinations.

- <term>Atlas A2 training products/Atlas A2 inference products</term>:
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
        <th>Restriction</th>
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
        <td>commQuantScale1Optional and commQuantScale2Optional are either both empty or both not empty.</td>
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
        <td>Both commQuantScale1Optional and commQuantScale2Optional are empty or not empty.</td>
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
        <td>Both commQuantScale1Optional and commQuantScale2Optional are empty or not empty.</td>
        </tr>
    </tbody>
    </table>

- Ascend 950PR/Ascend 950DT:
    
    When int8 is used as the input, per-token-per-channel quantization and per-tensor-per-channel quantization are supported.
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
        <th>Restriction</th>
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
        <td>commQuantScale1Optional and commQuantScale2Optional are both empty or not empty.</td>
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
        <td>commQuantScale1Optional and commQuantScale2Optional are both empty or not empty.</td>
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
        <td>commQuantScale1Optional and commQuantScale2Optional are both empty or not empty.</td>
        </tr>
    </tbody>
    </table>

    per-token-per-channel quantization && per-token-per-tensor quantization && per-block-per-block quantization
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
        <th>Restriction</th>
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
        <td>In B-B quantization scenarios, biasOptional can only be set to null</td>.
        </tr>
        <tr>
        <td>FLOAT8_E5M2</td>
        <td>FLOAT8_E5M2</td>
        <td>null, FLOAT32</td>
        <td>null, FLOAT16, BFLOAT16, FLOAT32</td>
        <td>FLOAT32</td>
        <td>FLOAT32</td>
        <td>FLOAT16, BFLOAT16, FLOAT32</td>
        <td>In B-B quantization scenarios, biasOptional can only be set to null</td>.
        </tr>
        <tr>
        <td>HIFLOAT8</td>
        <td>HIFLOAT8</td>
        <td>null, FLOAT32</td>
        <td>null, FLOAT16, BFLOAT16, FLOAT32</td>
        <td>FLOAT32</td>
        <td>FLOAT32</td>
        <td>FLOAT16, BFLOAT16, FLOAT32</td>
        <td>In B-B quantization scenarios, biasOptional can only be set to null</td>.
        </tr>
    </tbody>
    </table>

    Per-channel quantization and per-tensor quantization
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
        <th>Restriction</th>
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
        <th>Limit</th>
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

## Examples

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

Note: In this sample code, some HCCL collective communication library APIs are called, including HcclGetCommName, HcclCommInitAll, and HcclCommDestroy. For details, see [<<HCCL API (C)>>](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/900/API/hcclapiref/hcclcpp_07_0001.html).

- For the <term>Atlas A2 training products/Atlas A2 inference products</term> and Ascend 950PR/Ascend 950DT:

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
      } while(0)

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

      // Call aclrtMalloc to allocate memory on the device.
      ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

      // Call aclrtMemcpy to copy the data on the host to the memory on the device. 
      ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

      // Compute the strides of the contiguous tensor.
      std::vector<int64_t> strides(shape.size(), 1);
      for (int64_t i = shape.size() - 2; i >= 0; i--) {
          strides[i] = shape[i + 1] * strides[i + 1];
      }
      // Call aclCreateTensor to create an aclTensor.
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
      // Call aclrtMalloc to allocate memory on the device.
      auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
      // Call aclrtMemcpy to copy the data on the host to the memory on the device. 
      ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
      // Compute the strides of the contiguous tensor.
      std::vector<int64_t> strides(shape.size(), 1);
      for (int64_t i = shape.size() - 2; i >= 0; i--) {
          strides[i] = shape[i + 1] * strides[i + 1];
      }
      // Call aclCreateTensor to create an aclTensor.
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
      // Create a tensor.
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
      ret = aclnnQuantMatmulAllReduceV4GetWorkspaceSize(x1, x2, bias, x3, x1Scale, x2ScaleOptional,
                                                        commQuantScale1, commQuantScale2, hcom_name,
                                                        "sum", commTurn, streamMode, 0, 0, out,
                                                        &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS,
                LOG_PRINT("aclnnQuantMatmulAllReduceV4GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
      // Allocate device memory based on workspaceSize computed by the first-phase API.
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
      }
      // Call the second-phase API.
      ret = aclnnQuantMatmulAllReduceV4(workspaceAddr, workspaceSize, executor, args.stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMatmulAllReduceV4 failed. ERROR: %d\n", ret); return ret);
      // (Boilerplate) Wait until the task execution is complete.
      ret = aclrtSynchronizeStreamWithTimeout(args.stream, 10000);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
      LOG_PRINT("device%d aclnnQuantMatmulAllReduceV4 execute success \n", args.rankId);
      // Release device resources. Modify the code based on the API definition.
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
      // Initialize the collective communication domain.
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
