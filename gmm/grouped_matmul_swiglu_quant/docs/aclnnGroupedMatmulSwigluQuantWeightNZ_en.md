# aclnnGroupedMatmulSwigluQuantWeightNZ

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T12:30:36.987Z -->

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

- Description: Fuses GroupedMatmul, dequant, swiglu, and quant. For details, see the formula. This is a weightNZ-specific version of the [aclnnGroupedMatmulSwigluQuant](./aclnnGroupedMatmulSwigluQuant_en.md) API.

- Formula:

  <details>
    <summary>Quantization scenario A8W8 (A refers to the activation matrix, W refers to the weight matrix, and 8 refers to the INT8 data type):</summary>
    <a id="quantization-scenario-a8w8"></a>

    - **Definition**:

      - **⋅** denotes matrix multiplication.
      - **⊙** indicates per-element multiplication.
      - $\left \lfloor x\right \rceil$ indicates rounding x to the nearest integer.
      - $\mathbb{Z_8} = \{ x \in \mathbb{Z} | −128≤x≤127 \}$
      - $\mathbb{Z_{32}} = \{ x \in \mathbb{Z} | -2147483648≤x≤2147483647 \}$
    - **Input**:

      - $X∈\mathbb{Z_8}^{M \times K}$: activation matrix (left matrix), where M is the total number of tokens and K is the feature dimension.
      - $W∈\mathbb{Z_8}^{E \times K \times N}$: grouped weight matrix (right matrix), where E is the number of experts, K is the feature dimension, and N is the output dimension.
      - $w\_scale∈\mathbb{R}^{E \times N}$: Per-channel scale factor of the grouped weight matrix (right matrix), where E is the number of experts and N is the output dimension.
      - $x\_scale∈\mathbb{R}^{M}$: Per-token scale factor of the activation matrix (left matrix), where M is the total number of tokens.
      - $groupList∈\mathbb{N}^{E}$: Group index list of cumsum.
    - **Output**:

      - $Q∈\mathbb{Z_8}^{M \times N / 2}$: quantized output matrix.
      - $Q\_scale∈\mathbb{R}^{M}$: quantization scale factor.

    - **Computation Process**

      - 1.Determine the tokens in the current group based on groupList[i], $i \in [0,Len(groupList))$.

        >Example: groupList=[3,4,4,6].
        >
        > The 0th right matrix `W[0,:,:]` corresponds to tokens `x[0:3]` at index positions [0,3) (3-0=3 tokens in total), corresponding to `x_scale[0:3]`, `w_scale[0]`, `Q[0:3]`, `Q_scale[0:3]`;
        >
        > The 1st right matrix `W[1,:,:]` corresponds to tokens `x[1:4]` at index positions [3,4) (4-3=1 token in total), corresponding to `x_scale[3:4]`, `w_scale[1]`, `Q[3:4]`, `Q_scale[3:4]`;
        >
        > The 2nd right matrix `W[2,:,:]` corresponds to tokens `x[2:4]` at index positions [4,4) (4-4=0 tokens in total), corresponding to `x_scale[4:4]`, `w_scale[2]`, `Q[4:4]`, `Q_scale[4:4]`;
        >
        > The 3rd right matrix `W[3,:,:]` corresponds to tokens `x[3:6]` at index positions [4,6) (6-4=2 tokens in total), corresponding to `x_scale[4:6]`, `w_scale[3]`, `Q[4:6]`, `Q_scale[4:6]`;
        >
        > Note: The parts not specified in groupList do not participate in the update.
        > For example, when groupList=[12,14,18] and the shape of x is [30,N/2]:
        >
        > The shape of the first output Q is [30,N/2], where the part Q[18:, :] is not updated or initialized, and the data in it is the original data from when the memory space was allocated.
        >
        > Similarly, the shape of the second output Q_scale is [30], where the part Q_scale[18:] is not updated or initialized, and the data in it is the original data from when the memory space was allocated.
        >
        > That is, Q[:groupList[-1],:] and Q_scale[:groupList[-1]] in the output are the valid data parts.

      - Perform the following computation based on the input parameters determined by the group:

        $C_{i} = (X_{i}\cdot W_{i} )\odot x\_scale_{i\,\text{Broadcast}} \odot w\_scale_{i\,\text{Broadcast}}$

        $C_{i,act}, gate_{i} = split(C_{i})$

        $S_{i}=Swish(C_{i,act})\odot gate_{i}$  &nbsp;&nbsp;where $Swish(x)=\frac{x}{1+e^{-x}}$

      - Quantize the output result

        $Q\_scale_{i} = \frac{max(|S_{i}|)}{127}$

        $Q_{i} = \left\lfloor \frac{S_{i}}{Q\_scale_{i}} \right\rceil$
    </details>

    <details>
    <summary>MSD scenario A8W4 (A refers to activation matrix, W refers to weight matrix, 8 refers to INT8 data type, 4 refers to INT4 data type):</summary>
    <a id="msd-scenario-a8w4"></a>

    - **Definition**:
      - **⋅** denotes matrix multiplication.
      - **⊙** indicates per-element multiplication.
      - $\left \lfloor x\right \rceil$ indicates rounding x to the nearest integer.
      - $\mathbb{Z_8} = \{ x \in \mathbb{Z} | −128≤x≤127 \}$
      - $\mathbb{Z_4} = \{ x \in \mathbb{Z} | −8≤x≤7 \}$
      - $\mathbb{Z_{32}} = \{ x \in \mathbb{Z} | -2147483648≤x≤2147483647 \}$
    - **Input**:
      - $X∈\mathbb{Z_8}^{M \times K}$: activation matrix (left matrix), where M is the total number of tokens and K is the feature dimension.
      - $W∈\mathbb{Z_4}^{E \times K \times N}$: Grouped weight matrix (right matrix), where E is the number of experts, K is the feature dimension, and N is the output dimension.
      - $weightAssistMatrix∈\mathbb{R}^{E \times N}$: Auxiliary matrix used in matrix multiplication (see the computation process for generating the auxiliary matrix below).
      - $w\_scale∈\mathbb{R}^{E \times K\_group\_num \times N}$: Per-channel scale factor of the grouped weight matrix (right matrix), where E is the number of experts, K\_group\_num is the number of groups along the K axis, and N is the output dimension.
      - $x\_scale∈\mathbb{R}^{M}$: Per-token scale factor of the activation matrix (left matrix), where M is the total number of tokens.
      - $groupList∈\mathbb{N}^{E}$: Group index list of cumsum.
    - **Output**:
      - $Q∈\mathbb{Z_8}^{M \times N / 2}$: quantized output matrix.
      - $Q\_scale∈\mathbb{R}^{M}$: quantization scale factor.
    - **Computation Process**
      - Determine the tokens of the current group based on groupList[i], where $i \in [0,Len(groupList))$.
        - The grouping logic is the same as that of A8W8.
      - Computation process for generating the auxiliary matrix (weightAssistMatrix) (note that the weightAssistMatrix is partially computed offline and provided as input, not completed inside the operator):
        - For per-channel quantization ($w\_scale$ is 2-dimensional):

          $weightAssistMatrix_{i} = 8 × w\_scale × Σ_{k=0}^{K-1} weight[:,k,:]$

        - For per-group quantization ($w\_scale$ is 3-dimensional):

          $weightAssistMatrix_{i} = 8 × Σ_{k=0}^{K-1} (weight[:,k,:] × w\_scale[:, ⌊k/num\_per\_group⌋, :])$

          NOTE: $num\_per\_group = K // K\_group\_num$

      - Perform the following computation based on the inputs determined by grouping:

        - 3.1. Split the left matrix $\mathbb{Z_8}$ into the high and low bit parts $\mathbb{Z_4}$
          $X\_high\_4bits_{i} = \lfloor \frac{X_{i}}{16} \rfloor$
          $X\_low\_4bits_{i} = X_{i} \& 0x0f - 8$
        - 3.2. Enable per-channel or per-group quantization during matrix multiplication
          per-channel:

          $C\_high_{i} = (X\_high\_4bits_{i} \cdot W_{i}) \odot w\_scale_{i}$

          $C\_low_{i} = (X\_low\_4bits_{i} \cdot W_{i}) \odot w\_scale_{i}$

          per-group:

          $C\_high_{i} = \\ Σ_{k=0}^{K-1}((X\_high\_4bits_{i}[:, k * num\_per\_group : (k+1) * num\_per\_group] \cdot W_{i}[k *   num\_per\_group : (k+1) * num\_per\_group, :]) \odot w\_scale_{i}[k, :] )$

          $C\_low_{i} = \\ Σ_{k=0}^{K-1}((X\_low\_4bits_{i}[:, k * num\_per\_group : (k+1) * num\_per\_group] \cdot W_{i}[k *   num\_per\_group : (k+1) * num\_per\_group, :]) \odot w\_scale_{i}[k, :] )$

        - 3.3. Restore the high-bit and low-bit matrix multiplication results to the overall result.

          $C_{i} = (C\_high_{i} * 16 + C\_low_{i} + weightAssistMatrix_{i}) \odot x\_scale_{i}$

          $C_{i,act}, gate_{i} = split(C_{i})$

          $S_{i}=Swish(C_{i,act})\odot gate_{i}$  &nbsp;&nbsp;where $Swish(x)=\frac{x}{1+e^{-x}}$

      - Quantize the output result

        $Q\_scale_{i} = \frac{max(|S_{i}|)}{127}$

        $Q_{i} = \left\lfloor \frac{S_{i}}{Q\_scale_{i}} \right\rceil$
    </details>

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). You must first call "aclnnGroupedMatmulSwigluQuantWeightNZGetWorkspaceSize" to obtain the workspace size required for computation and the executor that includes the operator computation process, and then call "aclnnGroupedMatmulSwigluQuantWeightNZ" to perform the computation.

```Cpp
aclnnStatus aclnnGroupedMatmulSwigluQuantWeightNZGetWorkspaceSize(
  const aclTensor *x,
  const aclTensor *weight,
  const aclTensor *bias,
  const aclTensor *offset,
  const aclTensor *weightScale,
  const aclTensor *xScale,
  const aclTensor *groupList,
  aclTensor       *output,
  aclTensor       *outputScale,
  aclTensor       *outputOffset,
  uint64_t        *workspaceSize,
  aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnGroupedMatmulSwigluQuantWeightNZ(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnGroupedMatmulSwigluQuantWeightNZGetWorkspaceSize

- **Parameters**

  <table style="undefined;table-layout: fixed;width: 1567px"><colgroup>
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
        <th style="white-space: nowrap">Input/Output</th>
        <th>Description</th>
        <th>Instruction</th>
        <th>Data Type</th>
        <th>Data Format</th>
        <th style="white-space: nowrap">Dimension (shape)</th>
        <th>Non-contiguous Tensor</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>x</td>
        <td rowspan="1">Input</td>
        <td>Left matrix, corresponding to X in the formula.</td>
        <td>If the shape is [M,K], K must be less than 65536.</td>
        <td>INT8</td>
        <td>ND</td>
        <td>2</td>
        <td>√</td>
      </tr>
      <tr>
        <td>weight</td>
        <td rowspan="1">Input</td>
        <td>Weight matrix, corresponding to W in the formula.</td>
        <td>Note that this API ignores the data format of weight and forcibly treats it as the FRACTAL_NZ format.</td>
        <td>INT8, INT4, INT32 (INT32 is for adaptation purposes; actually, 1 INT32 is interpreted as 8 INT4 data items.)</td>
        <td>FRACTAL_NZ</td>
        <td>5</td>
        <td>√</td>
      </tr>
      <tr>
        <td>bias</td>
        <td rowspan="1">input</td>
        <td>Auxiliary matrix for matrix multiplication, corresponding to weightAssistMatrix in the formula.</td>
        <td>Takes effect only in the A8W4 use case. A null pointer must be passed in the A8W8 use case.</td>
        <td>FLOAT</td>
        <td>ND</td>
        <td>2</td>
        <td>√</td>
      </tr>
      <tr>
        <td>offset</td>
        <td rowspan="1">Input</td>
        <td>Offset for per-channel asymmetric dequantization, corresponding to offset in the formula.</td>
        <td>Reserved input, not supported currently. Pass a null pointer.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>weightScale</td>
        <td rowspan="1">input</td>
        <td>Quantization scaling factor of the right matrix, corresponding to w_scale in the formula.</td>
        <td>The first axis length must be equal to the first axis dimension of weight, and the last axis length must be the same as that of weight restored to ND format.</td>
        <td>FLOAT, FLOAT16, BFLOAT16</td>
        <td>ND</td>
        <td>2 (per-channel) or 3 (per-group)</td>
        <td>√</td>
      </tr>
      <tr>
        <td>xScale</td>
        <td rowspan="1">input</td>
        <td>Quantization scaling factor of the left matrix, corresponding to x_scale in the formula.</td>
        <td>The length must be equal to the first axis dimension of x.</td>
        <td>FLOAT</td>
        <td>ND</td>
        <td>1</td>
        <td>√</td>
      </tr>
      <tr>
        <td>groupList</td>
        <td rowspan="1">input</td>
        <td>Indicates the number of tokens participating in computation in each group, corresponding to groupList in the formula.</td>
        <td><ul>
          <li>The length must be equal to the first axis dimension of weight.</li>
          <li>The last value in groupList constrains the valid part of the output data. For details, see the computation process section in Function.</li>
        </ul></td>
        <td>INT64</td>
        <td>ND</td>
        <td>1</td>
        <td>√</td>
      </tr>
      <tr>
        <td>Output</td>
        <td rowspan="1">Output</td>
        <td>Quantization result of the output, corresponding to Q in the formula.</td>
        <td>-</td>
        <td>INT8</td>
        <td>ND</td>
        <td>2</td>
        <td>√</td>
      </tr>
      <tr>
        <td>outputScale</td>
        <td rowspan="1">output</td>
        <td>Quantization scaling factor of the output, corresponding to Q_scale in the formula.</td>
        <td>-</td>
        <td>FLOAT</td>
        <td>ND</td>
        <td>1</td>
        <td>√</td>
      </tr>
      <tr>
        <td>outputOffset</td>
        <td rowspan="1">output</td>
        <td>Asymmetric quantization offset of the output, corresponding to Q_offset in the formula.</td>
        <td>Reserved input, not supported currently. Pass a null pointer.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>workspaceSize</td>
        <td rowspan="1">output</td>
        <td>Returns the workspace size that users need to allocate on the NPU device.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>executor</td>
        <td rowspan="1">output</td>
        <td>Returns the op executor containing the computation process of the operator.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
    </tbody>
  </table>

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table style="undefined;table-layout: fixed;width: 1150px"><colgroup>
  <col style="width: 167px">
  <col style="width: 123px">
  <col style="width: 860px">
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
      <td>The input x, weight, weightScale, xScale, groupList, output, or outputScale is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="7">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="7">161002</td>
      <td>The dimensions of the input x, weight, weightScale, xScale, groupList, output, or outputScale do not meet the constraints.</td>
    </tr>
    <tr>
      <td>The shapes of the input x, weight, weightScale, xScale, groupList, output, or outputScale do not meet the constraints.</td>
    </tr>
    <tr>
      <td>The formats of the input x, weight, weightScale, xScale, groupList, output, or outputScale do not meet the constraints.</td>
    </tr>
    <tr>
      <td>The number of elements in groupList is greater than the first axis length of weight.</td>
    </tr>
    <tr>
      <td>The length of the N axis exceeds 10240.</td>
    </tr>
    <tr>
      <td>In the A8W8 scenario, the last axis length of x is greater than or equal to 65536.</td>
    </tr>
    <tr>
      <td>A8W4 use case, where the last axis length of x is greater than or equal to 20000.</td>
    </tr>
  </tbody>
  </table>

## aclnnGroupedMatmulSwigluQuantWeightNZ

- **Parameters**
  <table style="undefined;table-layout: fixed;width: 1150px">
  <colgroup>
    <col style="width: 167px">
    <col style="width: 123px">
    <col style="width: 860px">
  </colgroup>
  <thead>
    <tr><th>Parameter</th>
    <th>Input/Output</th>
    <th>Description</th></tr>
  </thead>
  <tbody>
    <tr><td>workspace</td><td>Input</td><td>Workspace memory address requested on the Device.</td></tr>
    <tr><td>workspaceSize</td><td>Input</td><td>Workspace size requested on the Device, obtained by the first-phase API aclnnGroupedMatmulSwigluQuantWeightNZGetWorkspaceSize.</td></tr>
    <tr><td>executor</td><td>Input</td><td>Op executor that contains the operator computation process.</td></tr>
    <tr><td>stream</td><td>Input</td><td>Specifies the Stream for executing tasks.</td></tr>
  </tbody>
  </table>

- **Return value**

    aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnGroupedMatmulSwigluQuantWeightNZ defaults to a deterministic implementation.

- A8W8 scenario (`A` refers to the activation matrix (left matrix), `W` refers to the weight matrix (right matrix), and `8` refers to the `INT8` data type)

  - The last axis length of x must not be greater than or equal to 65536.
  - The N axis length cannot exceed 10240.
  <!-- npu="A3,910b" id7 -->
  - <term>Atlas A3 products</term> and <term>Atlas A2 products</term>: bias does not participate in the computation. A non-null pointer is accepted, and the output matches a null pointer.
  <!-- end id7 -->

- A8W4 scenario (`A` refers to the activation matrix (left matrix), `W` refers to the weight matrix (right matrix), and `4` refers to the `INT4` data type)

  - The last axis length of x cannot be greater than or equal to 20000.
  - The N axis length cannot exceed 10240.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```c++
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_grouped_matmul_swiglu_quant_weight_nz.h"

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
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream) {
    // Boilerplate code for resource initialization.
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape,
                    void** deviceAddr, aclDataType dataType, aclFormat formatType, aclTensor** tensor) {
    auto size = GetShapeSize(shape) * sizeof(T);
    // Call aclrtMalloc to allocate memory on the device.
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // Call aclrtMemcpy to copy data from the host to the device memory.
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // Calculate the strides of the contiguous tensor.
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
    }

    // Call the aclCreateTensor API to create an aclTensor.
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, formatType,
                            shape.data(), shape.size(), *deviceAddr);
    return 0;
}

int main() {
    // 1. (Boilerplate) Initialize device/stream. For details, see the acl API Reference.
    // Set deviceId based on the actual device.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // Handle the check as needed.
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. Construct the input and output, which requires custom construction based on the API.
    int64_t E = 4;
    int64_t M = 8192;
    int64_t N = 4096;
    int64_t K = 7168;
    std::vector<int64_t> xShape = {M, K};
    std::vector<int64_t> weightShape = {E, N / 32 ,K / 16, 16, 32};
    std::vector<int64_t> weightScaleShape = {E, N};
    std::vector<int64_t> xScaleShape = {M};
    std::vector<int64_t> groupListShape = {E};
    std::vector<int64_t> outputShape = {M, N / 2};
    std::vector<int64_t> outputScaleShape = {M};

    void* xDeviceAddr = nullptr;
    void* weightDeviceAddr = nullptr;
    void* weightScaleDeviceAddr = nullptr;
    void* xScaleDeviceAddr = nullptr;
    void* groupListDeviceAddr = nullptr;
    void* outputDeviceAddr = nullptr;
    void* outputScaleDeviceAddr = nullptr;

    aclTensor* x = nullptr;
    aclTensor* weight = nullptr;
    aclTensor* weightScale = nullptr;
    aclTensor* xScale = nullptr;
    aclTensor* groupList = nullptr;
    aclTensor* output = nullptr;
    aclTensor* outputScale = nullptr;

    std::vector<int8_t> xHostData(M * K, 0);
    std::vector<int8_t> weightHostData(E * N * K, 0);
    std::vector<float> weightScaleHostData(E * N, 0);
    std::vector<float> xScaleHostData(M, 0);
    std::vector<int64_t> groupListHostData(E, 0);
    std::vector<int8_t> outputHostData(M * N / 2, 0);
    std::vector<float> outputScaleHostData(M, 0);

    // Create the x aclTensor.
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr,  aclDataType::ACL_INT8, aclFormat::ACL_FORMAT_ND, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the weight aclTensor.
    ret = CreateAclTensor(weightHostData, weightShape, &weightDeviceAddr,  aclDataType::ACL_INT8, aclFormat::ACL_FORMAT_FRACTAL_NZ, &weight);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the weightScale aclTensor.
    ret = CreateAclTensor(weightScaleHostData, weightScaleShape, &weightScaleDeviceAddr, aclDataType::ACL_FLOAT,  aclFormat::ACL_FORMAT_ND, &weightScale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the xScale aclTensor.
    ret = CreateAclTensor(xScaleHostData, xScaleShape, &xScaleDeviceAddr, aclDataType::ACL_FLOAT,  aclFormat::ACL_FORMAT_ND, &xScale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the groupList aclTensor.
    ret = CreateAclTensor(groupListHostData, groupListShape, &groupListDeviceAddr, aclDataType::ACL_INT64, aclFormat::ACL_FORMAT_ND, &groupList);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the output aclTensor.
    ret = CreateAclTensor(outputHostData, outputShape, &outputDeviceAddr, aclDataType::ACL_INT8, aclFormat::ACL_FORMAT_ND, &output);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the outputScale aclTensor.
    ret = CreateAclTensor(outputScaleHostData, outputScaleShape, &outputScaleDeviceAddr, aclDataType::ACL_FLOAT, aclFormat::ACL_FORMAT_ND, &outputScale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;

    // 3. Call the CANN operator library API.
    // Call the first-phase API of aclnnGroupedMatmulSwigluQuantWeightNZ.
    ret = aclnnGroupedMatmulSwigluQuantWeightNZGetWorkspaceSize(x, weight, nullptr, nullptr, weightScale, xScale,
                                                        groupList, output, outputScale, nullptr,
                                                        &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS,
    LOG_PRINT("aclnnGroupedMatmulSwigluQuantWeightNZGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // Request device memory based on the workspaceSize computed by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // Call the second-phase API of aclnnGroupedMatmulSwigluQuantWeightNZ.
    ret = aclnnGroupedMatmulSwigluQuantWeightNZ(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS,
    LOG_PRINT("aclnnGroupedMatmulSwigluQuantWeightNZ failed. ERROR: %d\n", ret); return ret);

    // 4. (boilerplate) Wait for the task execution to complete synchronously.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. Obtain the output values, copy the result from Device memory to the Host side. Modify it based on the specific API definition.
    auto size = GetShapeSize(outputShape);
    std::vector<int8_t> out1Data(size, 0);
    ret = aclrtMemcpy(out1Data.data(), out1Data.size() * sizeof(out1Data[0]), outputDeviceAddr,
                        size * sizeof(out1Data[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t j = 0; j < size; j++) {
        LOG_PRINT("result[%ld] is: %d\n", j, out1Data[j]);
    }
    size = GetShapeSize(outputScaleShape);
    std::vector<float> out2Data(size, 0);
    ret = aclrtMemcpy(out2Data.data(), out2Data.size() * sizeof(out2Data[0]), outputScaleDeviceAddr,
                        size * sizeof(out2Data[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t j = 0; j < size; j++) {
        LOG_PRINT("result[%ld] is: %f\n", j, out2Data[j]);
    }
    // 6. Release the aclTensor and aclScalar. Modify it based on the specific API definition.
    aclDestroyTensor(x);
    aclDestroyTensor(weight);
    aclDestroyTensor(weightScale);
    aclDestroyTensor(xScale);
    aclDestroyTensor(groupList);
    aclDestroyTensor(output);
    aclDestroyTensor(outputScale);

    // 7. Release the device resources. Modify it based on the specific API definition.
    aclrtFree(xDeviceAddr);
    aclrtFree(weightDeviceAddr);
    aclrtFree(weightScaleDeviceAddr);
    aclrtFree(xScaleDeviceAddr);
    aclrtFree(groupListDeviceAddr);
    aclrtFree(outputDeviceAddr);
    aclrtFree(outputScaleDeviceAddr);
    if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
