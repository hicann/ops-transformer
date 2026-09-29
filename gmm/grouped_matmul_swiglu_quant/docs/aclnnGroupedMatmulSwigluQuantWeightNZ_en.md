# aclnnGroupedMatmulSwigluQuantWeightNZ

[📄 View Source Code](https://gitcode.com/cann/ops-transformer/tree/master/gmm/grouped_matmul_swiglu_quant)

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| Ascend 950PR/Ascend 950DT                            |    ×     |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √     |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference products</term>                            |    ×     |
| <term>Atlas training products</term>                             |    ×     |

## Function

- Description: Fuses `GroupedMatmul`, `dquant`, `swiglu`, and `quant`. For details, see the formulas. This API is the weightNZ specialization version of [aclnnGroupedMatmulSwigluQuant](./aclnnGroupedMatmulSwigluQuant.md).

- Formulas:

  <details>
    <summary>Quantization scenario A8W8 (A: activation matrix; W: weight matrix; 8: INT8)</summary>
    <a id="quantization-scenario-a8w8"></a>

    - **Definition**

      * **⋅** indicates matrix multiplication.
      * **⊙** indicates element-wise multiplication.
      * $\left \lfloor x\right \rceil$ indicates rounding `x` to the nearest integer.
      * $\mathbb{Z_8} = \{ x \in \mathbb{Z} | −128≤x≤127 \}$
      * $\mathbb{Z_{32}} = \{ x \in \mathbb{Z} | -2147483648≤x≤2147483647 \}$
    - **Input**

      * $X∈\mathbb{Z_8}^{M \times K}$: activation matrix (left matrix), where $M$ indicates the total number of tokens and $K$ indicates the feature dimension.
      * $W∈\mathbb{Z_8}^{E \times K \times N}$: grouped weight matrix (right matrix), where `E` indicates the number of experts, `K` indicates the feature dimension, and `N` indicates the output dimension.
      * $w\_scale∈\mathbb{R}^{E \times N}$: per-channel scale factor for the grouped weight matrix (right matrix), where `E` indicates the number of experts and `N` indicates the output dimension.
      * $x\_scale∈\mathbb{R}^{M}$: per-token scale factor for the activation matrix (left matrix), where `M` indicates the total number of tokens.
      * $groupList∈\mathbb{N}^{E}$: Group index list of cumsum.
    - **Output**

      * $Q∈\mathbb{Z_8}^{M \times N / 2}$: quantized output matrix.
      * $Q\_scale∈\mathbb{R}^{M}$: quantization scale factor.

    - **Computation process**

       1. Determine the tokens of the current group based on `groupList[i]`, where $i \in [0,Len(groupList)]$.

          >Example: groupList = [3, 4, 4, 6].
          >
          >Zero-th right matrix `W[0,:,:]`, corresponding to tokens `x[0:3]` (3-0=3 tokens) at index positions [0, 3), corresponding to `x_scale[0:3]`, `w_scale[0]`, `Q[0:3]`, and `Q_scale[0:3]`.
          >
          >First right matrix `W[1,:,:]`, corresponding to token `x[3:4]` (4-3=1 token) at index position [3, 4), corresponding to `x_scale[3:4]`, `w_scale[1]`, `Q[3:4]`, and `Q_scale[3:4]`
          >
          >Second right matrix `W[2,:,:]`, corresponding to token `x[4:4]` (4-4=0 token) at index position [4, 4), corresponding to `x_scale[4:4]`, `w_scale[2]`, `Q[4:4]`, and `Q_scale[4:4]`
          >
          >Third right matrix `W[3,:,:]`, corresponding to tokens `x[4:6]` (6-4=2 tokens) at index positions [4, 6), corresponding to `x_scale[4:6]`, `w_scale[3]`, `Q[4:6]`, and `Q_scale[4:6]`
          >
          >Note: The parts that are not specified in groupList will not be updated.
          >For example, when groupList = [12, 14, 18] and the shape of x is [30, N/2].
          >
          >The shape of the first output Q is [30, N/2]. The part of Q[18:, :] will not be updated or initialized, and the data is the original data when the GPU memory is allocated.
          >
          >Similarly, the shape of the second output Q_scale is [30]. The part of Q_scale[18:] is not updated or initialized, and the data is the original data when the video RAM space is allocated.
          >
          >That is, the output Q[:groupList[-1],:] and Q_scale[:groupList[-1]] are valid data.

       2. Perform the following computation based on the input parameters determined by grouping:

          $C_{i} = (X_{i}\cdot W_{i} )\odot x\_scale_{i\,\text{Broadcast}} \odot w\_scale_{i\,\text{Broadcast}}$

          $C_{i,act}, gate_{i} = split(C_{i})$

          $S_{i}=Swish(C_{i,act})\odot gate_{i}$ &nbsp;&nbsp; where $Swish(x)=\frac{x}{1+e^{-x}}$

       3. Quantize the output.

        $Q\_scale_{i} = \frac{max(|S_{i}|)}{127}$

        $Q_{i} = \left\lfloor \frac{S_{i}}{Q\_scale_{i}} \right\rceil$
    </details>

    <details>
    <summary>MSD scenario A8W4 (A: activation matrix; W: weight matrix; 8: INT8; 4: INT4)</summary>
    <a id="msd-scenario-a8w4"></a>
    
    - **Definition**
      * **⋅** indicates matrix multiplication.
      * **⊙** indicates element-wise multiplication.
      * $\left \lfloor x\right \rceil$ indicates rounding `x` to the nearest integer.
      * $\mathbb{Z_8} = \{ x \in \mathbb{Z} | −128≤x≤127 \}$
      * $\mathbb{Z_4} = \{ x \in \mathbb{Z} | −8≤x≤7 \}$
      * $\mathbb{Z_{32}} = \{ x \in \mathbb{Z} | -2147483648≤x≤2147483647 \}$
    - **Input**
      * $X∈\mathbb{Z_8}^{M \times K}$: activation matrix (left matrix), where $M$ indicates the total number of tokens and $K$ indicates the feature dimension.
      * $W∈\mathbb{Z_4}^{E \times K \times N}$: grouped weight matrix (right matrix), where `E` indicates the number of experts, `K` indicates the feature dimension, and `N` indicates the output dimension.
      * $weightAssistMatrix∈\mathbb{R}^{E \times N}$: Auxiliary matrix used for matrix multiplication. For details about how to generate the auxiliary matrix, see the following description.
      * $w\_scale∈\mathbb{R}^{E \times K\_group\_num \times N}$: per-channel scale factor for the grouped weight matrix (right matrix), where `E` indicates the number of experts, `K_group_num` indicates the number of groups along the K-axis, and `N` indicates the output dimension.
      * $x\_scale∈\mathbb{R}^{M}$: per-token scale factor for the activation matrix (left matrix), where `M` indicates the total number of tokens.
      * $groupList∈\mathbb{N}^{E}$: Group index list of cumsum.
    - **Output**
      * $Q∈\mathbb{Z_8}^{M \times N / 2}$: quantized output matrix.
      * $Q\_scale∈\mathbb{R}^{M}$: quantization scale factor.
    - **Computation process**
       1. Determine the tokens of the current group based on `groupList[i]`, where $i \in [0,Len(groupList)]$.
          - The grouping logic is the same as that of A8W8.
       2. The computation process of generating the auxiliary matrix (weightAssistMatrix) is as follows. (Note that the computation of weightAssistMatrix is performed offline and is not completed inside the operator.)
          - For per-channel quantization ($w\_scale$ is 2D):

            $weightAssistMatrix_{i} = 8 × w\_scale × Σ_{k=0}^{K-1} weight[:,k,:]$

          - For per-group quantization ($w\_scale$ is 3D):

            $weightAssistMatrix_{i} = 8 × Σ_{k=0}^{K-1} (weight[:,k,:] × w\_scale[:, ⌊k/num\_per\_group⌋, :])$

            Note: $num\_per\_group = K // K\_group\_num$

       3. Perform the following computation based on the input parameters determined by grouping:

          - 3.1. Convert the left matrix $\mathbb{Z_8}$ into two $\mathbb{Z_4}$ components that represent the high and low bits.
            $X\_high\_4bits_{i} = \lfloor \frac{X_{i}}{16} \rfloor$
            $X\_low\_4bits_{i} = X_{i} \& 0x0f - 8$
          - 3.2. Enable per-channel or per-group quantization during matrix multiplication.
            Per-channel:

            $C\_high_{i} = (X\_high\_4bits_{i} \cdot W_{i}) \odot w\_scale_{i}$

            $C\_low_{i} = (X\_low\_4bits_{i} \cdot W_{i}) \odot w\_scale_{i}$

            Per-group:

            $C\_high_{i} = \\ Σ_{k=0}^{K-1}((X\_high\_4bits_{i}[:, k * num\_per\_group : (k+1) * num\_per\_group] \cdot W_{i}[k *   num\_per\_group : (k+1) * num\_per\_group, :]) \odot w\_scale_{i}[k, :] )$

            $C\_low_{i} = \\ Σ_{k=0}^{K-1}((X\_low\_4bits_{i}[:, k * num\_per\_group : (k+1) * num\_per\_group] \cdot W_{i}[k *   num\_per\_group : (k+1) * num\_per\_group, :]) \odot w\_scale_{i}[k, :] )$

          - 3.3. Restore the matrix multiplication results of the high and low bits into the overall result.

            $C_{i} = (C\_high_{i} * 16 + C\_low_{i} + weightAssistMatrix_{i}) \odot x\_scale_{i}$

            $C_{i,act}, gate_{i} = split(C_{i})$

            $S_{i}=Swish(C_{i,act})\odot gate_{i}$ &nbsp;&nbsp; where $Swish(x)=\frac{x}{1+e^{-x}}$

       4. Quantize the output.

        $Q\_scale_{i} = \frac{max(|S_{i}|)}{127}$

        $Q_{i} = \left\lfloor \frac{S_{i}}{Q\_scale_{i}} \right\rceil$
    </details>

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnGroupedMatmulSwigluQuantWeightNZGetWorkspaceSize` is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, `aclnnGroupedMatmulSwigluQuantWeightNZ` is called to perform computation.

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
        <th>Name</th>
        <th style="white-space: nowrap">Input/Output</th>
        <th>Description</th>
        <th>Usage</th>
        <th>Data Type</th>
        <th>Data Format</th>
        <th style="white-space: nowrap">Dimension (Shape)</th>
        <th>Non-contiguous Tensor</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>x</td>
        <td rowspan="1">Input</td>
        <td>Left matrix, corresponding to X in the formula.</td>
        <td>If the shape is [M, K], K must be less than 65536.</td>
        <td>INT8</td>
        <td>ND</td>
        <td>2</td>
        <td>√</td>
      </tr>
      <tr>
        <td>weight</td>
        <td rowspan="1">Input</td>
        <td>Weight matrix, corresponding to W in the formula.</td>
        <td>Note that this API ignores the data format of the weight and forcibly considers it as the FRACTAL_NZ format.</td>
        <td>INT8, INT4, INT32 (INT32 is used for adaptation. Actually, one INT32 data record is interpreted as eight INT4 data records.)</td>
        <td>FRACTAL_NZ</td>
        <td>5</td>
        <td>√</td>
      </tr>
      <tr>
        <td>bias</td>
        <td rowspan="1">Input</td>
        <td>Auxiliary matrix for matrix multiplication, corresponding to weightAssistMatrix in the formula.</td>
        <td>This parameter takes effect only in the A8W4 scenario. In the A8W8 scenario, a null pointer needs to be passed.</td>
        <td>FLOAT</td>
        <td>ND</td>
        <td>2</td>
        <td>√</td>
      </tr>
      <tr>
        <td>offset</td>
        <td rowspan="1">Input</td>
        <td>Offset of per-channel asymmetric dequantization, corresponding to offset in the formula.</td>
        <td>This input is reserved and is not supported currently. You need to pass a null pointer.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>weightScale</td>
        <td rowspan="1">Input</td>
        <td>Quantization factor of the right matrix, corresponding to w_scale in the formula.</td>
        <td>The length of the first axis must be the same as that of the first axis of the weight, and the length of the last axis must be the same as that of the last axis of the weight restored to the ND format.</td>
        <td>FLOAT, FLOAT16, BFLOAT16</td>
        <td>ND</td>
        <td>2 (per-channel) or 3 (per-group)</td>
        <td>√</td>
      </tr>
      <tr>
        <td>xScale</td>
        <td rowspan="1">Input</td>
        <td>Quantization factor of the left matrix, corresponding to x_scale in the formula.</td>
        <td>The length must be the same as the first axis dimension of x.</td>
        <td>FLOAT</td>
        <td>ND</td>
        <td>1</td>
        <td>√</td>
      </tr>
      <tr>
        <td>groupList</td>
        <td rowspan="1">Input</td>
        <td>Cumulative token indices for each group, corresponding to groupList in the formula.</td>
        <td><ul>
          <li>The length must be the same as the first axis of weight.</li>
          <li>The last value in groupList restricts the valid part of the output data. For details, see the computation process in the function description.</li>
        </ul></td>
        <td>INT64</td>
        <td>ND</td>
        <td>1</td>
        <td>√</td>
      </tr>
      <tr>
        <td>output</td>
        <td rowspan="1">Output</td>
        <td>Quantization result, corresponding to Q in the formula.</td>
        <td>-</td>
        <td>INT8</td>
        <td>ND</td>
        <td>2</td>
        <td>√</td>
      </tr>
      <tr>
        <td>outputScale</td>
        <td rowspan="1">Output</td>
        <td>Quantization factor of the output, corresponding to Q_scale in the formula.</td>
        <td>-</td>
        <td>FLOAT</td>
        <td>ND</td>
        <td>1</td>
        <td>√</td>
      </tr>
      <tr>
        <td>outputOffset</td>
        <td rowspan="1">Output</td>
        <td>Asymmetric quantization offset of the output, corresponding to Q_offset in the formula.</td>
        <td>This input is reserved and is not supported currently. You need to pass a null pointer.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>workspaceSize</td>
        <td rowspan="1">Output</td>
        <td>Returns the size of the workspace that needs to be allocated on the NPU device.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>executor</td>
        <td rowspan="1">Output</td>
        <td>Operator executor, containing the operator computation process.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
    </tbody>
  </table>

- **Return**
  
  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API implements input parameter validation. The following errors may be thrown:

  <table style="undefined;table-layout: fixed;width: 1150px"><colgroup>
  <col style="width: 167px">
  <col style="width: 123px">
  <col style="width: 860px">
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
      <td>The input x, weight, weightScale, xScale, groupList, output, or outputScale is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="7">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="7">161002</td>
      <td>The data dimensions of the input x, weight, weightScale, xScale, groupList, output, or outputScale do not comply with the constraints.</td>
    </tr>
    <tr>
      <td>The shape of the input x, weight, weightScale, xScale, groupList, output, or outputScale does not comply with the constraints.</td>
    </tr>
    <tr>
      <td>The format of the input x, weight, weightScale, xScale, groupList, output, or outputScale does not comply with the constraints.</td>
    </tr>
    <tr>
      <td>The number of elements in groupList is greater than the length of the first axis of weight.</td>
    </tr>
    <tr>
      <td>The length of the N axis exceeds 10240.</td>
    </tr>
    <tr>
      <td>In the A8W8 scenario, the length of the last axis of x is greater than or equal to 65536.</td>
    </tr>
    <tr>
      <td>In the A8W4 scenario, the length of the last axis of x is greater than or equal to 20000.</td>
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
    <tr><td>workspace</td><td>Input</td><td>Address of the workspace to be allocated on the device.</td></tr>
    <tr><td>workspaceSize</td><td>Input</td><td>Size of the workspace to be allocated on the device, which is obtained by the first-phase API of aclnnGroupedMatmulSwigluQuantWeightNZGetWorkspaceSize.</td></tr>
    <tr><td>executor</td><td>Input</td><td>Operator executor, containing the operator computation process.</td></tr>
    <tr><td>stream</td><td>Input</td><td>Stream for executing a task.</td></tr>
  </tbody>
  </table>
  
- **Return**

    `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - `aclnnGroupedMatmulSwigluQuantWeightNZ` defaults to a deterministic implementation.

- A8W8 scenario (A: activation matrix (left matrix); W: weight matrix (right matrix); 8: INT8)

   1. The length of the last axis of `x` cannot be greater than or equal to 65536.
   2. The length of the N-axis cannot exceed 10240.

- A8W4 scenario (A: activation matrix (left matrix); W: weight matrix (right matrix); 4: INT4)

   1. The length of the last axis of `x` cannot be greater than or equal to 20000.
   2. The length of the N-axis cannot exceed 10240.

## Calling Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

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
    // (Boilerplate) Initialize resources.
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
    // Call aclrtMalloc to allocate device memory.
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // Call aclrtMemcpy to copy data from the host to the device memory.
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // Compute the strides of the contiguous tensor.
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
    }

    // Call aclCreateTensor to create an aclTensor.
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, formatType,
                            shape.data(), shape.size(), *deviceAddr);
    return 0;
}

int main() {
    // 1. (Boilerplate) Initialize the device and stream. For details, see the ACL API manual.
    // Set the device ID in use.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // Customize error handling based on your requirements.
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. Construct inputs and outputs based on API definitions.
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

    // Create an x aclTensor.
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr,  aclDataType::ACL_INT8, aclFormat::ACL_FORMAT_ND, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create a weight aclTensor.
    ret = CreateAclTensor(weightHostData, weightShape, &weightDeviceAddr,  aclDataType::ACL_INT8, aclFormat::ACL_FORMAT_FRACTAL_NZ, &weight);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create a weightScale aclTensor.
    ret = CreateAclTensor(weightScaleHostData, weightScaleShape, &weightScaleDeviceAddr, aclDataType::ACL_FLOAT,  aclFormat::ACL_FORMAT_ND, &weightScale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create an xScale aclTensor.
    ret = CreateAclTensor(xScaleHostData, xScaleShape, &xScaleDeviceAddr, aclDataType::ACL_FLOAT,  aclFormat::ACL_FORMAT_ND, &xScale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create a groupList aclTensor.
    ret = CreateAclTensor(groupListHostData, groupListShape, &groupListDeviceAddr, aclDataType::ACL_INT64, aclFormat::ACL_FORMAT_ND, &groupList);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create an output aclTensor.
    ret = CreateAclTensor(outputHostData, outputShape, &outputDeviceAddr, aclDataType::ACL_INT8, aclFormat::ACL_FORMAT_ND, &output);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create an outputScale aclTensor.
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
    // Allocate device memory based on workspaceSize computed by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // Call the second-phase API of aclnnGroupedMatmulSwigluQuantWeightNZ.
    ret = aclnnGroupedMatmulSwigluQuantWeightNZ(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, 
    LOG_PRINT("aclnnGroupedMatmulSwigluQuantWeightNZ failed. ERROR: %d\n", ret); return ret);

    // 4. (Boilerplate) Wait until the task execution is complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. Obtain the output value and copy the result from the device memory to the host. Modify the code based on the API definition.
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
    // 6. Release aclTensors and aclScalars. Modify the code based on the API definition.
    aclDestroyTensor(x);
    aclDestroyTensor(weight);
    aclDestroyTensor(weightScale);
    aclDestroyTensor(xScale);
    aclDestroyTensor(groupList);
    aclDestroyTensor(output);
    aclDestroyTensor(outputScale);

    // 7. Release device resources. Modify the code based on the API definition.
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
