# aclnnGroupedMatmulSwigluQuantWeightNZ

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √     |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference products</term>                            |    ×     |
| <term>Atlas training products</term>                             |    ×     |

## Function

- Description: Fuses `GroupedMatmul`, `dquant`, `swiglu`, and `quant`. For details, see the formulas. This API is the weightNZ specialization version of [aclnnGroupedMatmulSwigluQuant](../../grouped_matmul_swiglu_quant/docs/aclnnGroupedMatmulSwigluQuant_en.md).
- Formulas:
  - Quantization scenario (A8W8):
    - **Definition**

      * **⋅** indicates matrix multiplication.
      * **⊙** indicates element-wise multiplication.
      * $\left \lfloor x\right \rceil$ indicates rounding `x` to the nearest integer.
      * $\mathbb{Z_8} = \{ x \in \mathbb{Z} | −128≤x≤127 \}$
      * $\mathbb{Z_{32}} = \{ x \in \mathbb{Z} | -2147483648≤x≤2147483647 \}$
    - **Input**

      * $X∈\mathbb{Z_8}^{M \times K}$: input matrix (left matrix), where `M` indicates the total number of tokens and `K` indicates the feature dimension.
      * $W∈\mathbb{Z_8}^{E \times K \times N}$: grouped weight matrix (right matrix), where `E` indicates the number of experts, `K` indicates the feature dimension, and `N` indicates the output dimension.
      * $w\_scale∈\mathbb{R}^{E \times N}$: per-channel scale factor for the grouped weight matrix (right matrix), where `E` indicates the number of experts and `N` indicates the output dimension.
      * $x\_scale∈\mathbb{R}^{M}$: per-token scale factor for the input matrix (left matrix), where `M` indicates the total number of tokens.
      * $grouplist∈\mathbb{N}^{E}$: grouped index list of prefix sums.
    - **Output**

      * $Q∈\mathbb{Z_8}^{M \times N / 2}$: quantized output matrix.
      * $Q\_scale∈\mathbb{R}^{M}$: quantization scale factor.

    - **Computation process**

      1. Determine the tokens of the current group based on `groupList[i]`, where $i \in [0,Len(groupList)]$.
      
          >Example: Assume that `groupList=[3,4,4,6]`. The counting starts from 0.
          >
          >Zero-th right matrix `W[0,:,:]`, corresponding to tokens `x[0:3]` (3-0=3 tokens) at index positions [0,3), corresponding to `x_scale[0:3]`, `w_scale[0]`, `bias[0]`, `offset[0]`, `Q[0:3]`, `Q_scale[0:3]`, and `Q_offset[0:3]`
          >
          >First right matrix `W[1,:,:]`, corresponding to token `x[3:4]` (4-3=1 token) at index position [3,4), corresponding to `x_scale[3:4]`, `w_scale[1]`, `bias[1]`, `offset[1]`, `Q[3:4]`, `Q_scale[3:4]`, and `Q_offset[3:4]`
          >
          >Second right matrix `W[2,:,:]`, corresponding to token `x[4:4]` (4-4=0 token) at index position [4,4), corresponding to `x_scale[4:4]`, `w_scale[2]`, `bias[2]`, `offset[2]`, `Q[4:4]`, `Q_scale[4:4]`, and `Q_offset[4:4]`
          >
          >Third right matrix `W[3,:,:]`, corresponding to tokens `x[4:6]` (6-4=2 tokens) at index positions [4,6), corresponding to `x_scale[4:6]`, `w_scale[3]`, `bias[3]`, `offset[3]`, `Q[4:6]`, `Q_scale[4:6]`, and `Q_offset[4:6]`
          >
          >Note: Any portion not specified in `groupList` will not be updated.
          >Assume `groupList=[12,14,18]` and the shape of `X` is [30,:].
          >
          >The shape of the first output `Q` will be [30,:]. The portion `Q[18:,:]` will not be updated or initialized, and the data therein is consistent with the original data when the device memory is allocated.
          >
          >Similarly, the second output `Q` has a shape of [30]. The portion `Q_scale[18:]` will not be updated or initialized, and the data therein is consistent with the original data when the device memory is allocated.
          >
          >In other words, only `Q[:grouplist[-1],:]` and `Q_scale[:grouplist[-1]]` constitute the valid data portions.

      2. Perform the following computation based on the input parameters determined by grouping:

          $C_{i} = (X_{i}\cdot W_{i} )\odot x\_scale_{i\ BroadCast} \odot w\_scale_{i\ BroadCast}$

          $C_{i,act}, gate_{i} = split(C_{i})$

          $S_{i}=Swish(C_{i,act})\odot gate_{i}$ &nbsp;&nbsp; where $Swish(x)=\frac{x}{1+e^{-x}}$

      3. Quantize the output.

          $Q\_scale_{i} = \frac{max(|S_{i}|)}{127}$

          $Q_{i} = \lfloor \frac{S_{i}}{Q\_scale_{i}} \rceil$

  ----
  - MSD scenario (A8W4):
    - **Definition**
      * **⋅** indicates matrix multiplication.
      * **⊙** indicates element-wise multiplication.
      * $\left \lfloor x\right \rceil$ indicates rounding `x` to the nearest integer.
      * $\mathbb{Z_8} = \{ x \in \mathbb{Z} | −128≤x≤127 \}$
      * $\mathbb{Z_4} = \{ x \in \mathbb{Z} | −8≤x≤7 \}$
      * $\mathbb{Z_{32}} = \{ x \in \mathbb{Z} | -2147483648≤x≤2147483647 \}$
    - **Input**
      * $X∈\mathbb{Z_8}^{M \times K}$: input matrix (left matrix), where `M` indicates the total number of tokens and `K` indicates the feature dimension.
      * $W∈\mathbb{Z_4}^{E \times K \times N}$: grouped weight matrix (right matrix), where `E` indicates the number of experts, `K` indicates the feature dimension, and `N` indicates the output dimension.
      * $bias∈\mathbb{R}^{E \times N}$: auxiliary matrix for matrix multiplication (the computation process for generating the auxiliary matrix is described below).
      * $w\_scale∈\mathbb{R}^{E \times K\_group\_num \times N}$: per-channel scale factor for the grouped weight matrix (right matrix), where `E` indicates the number of experts, `K_group_num` indicates the number of groups along the K-axis, and `N` indicates the output dimension.
      * $x\_scale∈\mathbb{R}^{M}$: per-token scale factor for the input matrix (left matrix), where `M` indicates the total number of tokens.
      * $grouplist∈\mathbb{N}^{E}$: grouped index list of prefix sums.
    - **Output**
      * $Q∈\mathbb{Z_8}^{M \times N / 2}$: quantized output matrix.
      * $Q\_scale∈\mathbb{R}^{M}$: quantization scale factor.
    - **Computation process**
      1. Determine the tokens of the current group based on `groupList[i]`, where $i \in [0,Len(groupList)]$.
  
      2. Compute the auxiliary matrix (`bias`). (Note that the bias computation is performed offline and provided as an input, rather than being executed within the operator.)
          - For per-channel quantization ($w\_scale$ is 2D):

            $bias_{i} = 8 × weightScale × Σ_{k=0}^{K-1} weight[:,k,:]$

          - For per-group quantization ($w\_scale$ is 3D):

            $bias_{i} = 8 × Σ_{k=0}^{K-1} (weight[:,k,:] × weightScale[:, ⌊k/num\_per\_group⌋, :])$

            Note: $num\_per\_group = K // K\_group\_num$

      3. Perform the following computation based on the input parameters determined by grouping:

          - 3.1. Convert the left matrix $\mathbb{Z_8}$ into two $\mathbb{Z_4}$ components that represent the high and low bits.
            $X\_high\_4bits_{i} = \lfloor \frac{X_{i}}{16} \rfloor $
            $X\_low\_4bits_{i} = X_{i} \& 0x0f - 8$
          - 3.2. Enable per-channel or per-group quantization during matrix multiplication.
            
            Per-channel:
            
            $C\_high_{i} = (X\_high\_4bits_{i} \cdot W_{i}) \odot w\_scale_{i}$

            $C\_low_{i} = (X\_low\_4bits_{i} \cdot W_{i}) \odot w\_scale_{i}$

            Per-group:

            $C\_high_{i} = \\ Σ_{k=0}^{K-1}((X\_high\_4bits_{i}[:, k * num\_per\_group : (k+1) * num\_per\_group] \cdot W_{i}[k * num\_per\_group : (k+1) * num\_per\_group, :]) \odot w\_scale_{i}[k, :] )$

            $C\_low_{i} = \\ Σ_{k=0}^{K-1}((X\_low\_4bits_{i}[:, k * num\_per\_group : (k+1) * num\_per\_group] \cdot W_{i}[k * num\_per\_group : (k+1) * num\_per\_group, :]) \odot w\_scale_{i}[k, :] )$

          - 3.3. Restore the matrix multiplication results of the high and low bits into the overall result.

            $C_{i} = (C\_high_{i} * 16 + C\_low_{i} + bias_{i}) \odot x\_scale_{i}$

            $C_{i,act}, gate_{i} = split(C_{i})$

            $S_{i}=Swish(C_{i,act})\odot gate_{i}$ &nbsp;&nbsp; where $Swish(x)=\frac{x}{1+e^{-x}}$

      4. Quantize the output.

        $Q\_scale_{i} = \frac{max(|S_{i}|)}{127}$

        $Q_{i} = \lfloor \frac{S_{i}}{Q\_scale_{i}} \rceil$

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnGroupedMatmulSwigluQuantWeightNZGetWorkspaceSize` is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, `aclnnGroupedMatmulSwigluQuantWeightNZ` is called to perform computation.

- `aclnnStatus aclnnGroupedMatmulSwigluQuantWeightNZGetWorkspaceSize(const aclTensor* x, const aclTensor* weight, const aclTensor* bias, const aclTensor* offset,  const aclTensor* weightScale, const aclTensor* xScale, const aclTensor* groupList, aclTensor* output, aclTensor* outputScale, aclTensor* outputOffset, uint64_t* workspaceSize, aclOpExecutor** executor)`
- `aclnnStatus aclnnGroupedMatmulSwigluQuantWeightNZ(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)`

## aclnnGroupedMatmulSwigluQuantWeightNZGetWorkspaceSize

- **Parameters**
  
  - x (aclTensor*, computation input): left matrix, $X$ in the formula, aclTensor on the device. The shape supports 2D. Assuming the shape is [M,K], K must be less than 65536. The data type can be INT8, and the [data format](../../../docs/en/context/data_format.md) can be ND. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported.

  - weight (aclTensor*, computation input): weight matrix, $W$ in the formula, aclTensor on the device. The shape supports 5D. The data type can be INT8, INT4, or INT32 (INT32 is used for adaptation. Actually, one INT32 data record is interpreted as eight INT4 data records). The [data format](../../../docs/en/context/data_format.md) can be FRACTAL\_NZ. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. Note that this API ignores the weight data format and considers it as the FRACTAL\_NZ format.

  - bias (aclTensor*, computation input): matrix multiplication bias, $bias$ in the formula. The shape supports 2D. The data type can be INT32. This input is reserved and is not supported currently. You need to pass a null pointer.

  - offset (aclTensor*, computation input): offset of per-channel asymmetric dequantization, $offset$ in the formula. The shape supports 2D. The data type is float. This input is reserved and is not supported currently. You need to pass a null pointer.

  - weightScale (aclTensor*, computation input): quantization factor of the right matrix, $w\_scale$ in the formula, aclTensor on the device. The shape supports 2D. The length of the first axis must be the same as the first axis of `weight`. The length of the last axis must be the same as the last axis of `weight` restored to the ND format. The data type can be FLOAT, FLOAT16, or BFLOAT16. The [data format](../../../docs/en/context/data_format.md) can be ND. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported.

  - xScale (aclTensor*, computation input): quantization factor of the left matrix, $x\_scale$ in the formula, aclTensor on the device. The shape supports 1D. The length must be the same as the first axis of `x`. The data type can be FLOAT. The [data format](../../../docs/en/context/data_format.md) can be ND. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported.

  - groupList (aclTensor*, computation input): number of tokens involved in the computation of each group, $grouplist$ in the formula, aclTensor on the device. The shape supports 1D. The length must be the same as the first axis of `weight`. The data type can be INT64. The [data format](../../../docs/en/context/data_format.md) can be ND. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The last value in `grouplist` constrains the valid portion of the output data. For details, see the computation process.

  - output (aclTensor*, computation output): quantization result, $Q$ in the formula, aclTensor on the device. The data type can be INT8, and the shape can be 2D. The [data format](../../../docs/en/context/data_format.md) can be ND. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported.

  - outputScale (aclTensor*, computation output): quantization factor, $Q\_scale$ in the formula, aclTensor on the device. The data type can be FLOAT, and the shape can be 1D. The [data format](../../../docs/en/context/data_format.md) can be ND. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported.

  - outputOffset (aclTensor*, computation output): offset of asymmetric quantization, $Q\_offset$ in the formula, aclTensor on the device. The shape can be 1D, and the data type can be FLOAT. This input is reserved and is not supported currently. You need to pass a null pointer.

  - workspaceSize (uint64_t*, output parameter): size of the workspace to be allocated on the NPU device.

  - executor (aclOpExecutor**, computation output): operator executor, containing the operator computation process.
- **Return**
  
  `aclnnStatus` status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  ```cpp
  The first-phase API implements input parameter validation. The following errors may be thrown:
  161001 (ACLNN_ERR_PARAM_NULLPTR): 1. The input x, weight, weightScale, xScale, groupList, output, or outputScale is a null pointer.
  161002 (ACLNN_ERR_PARAM_INVALID): 1. The data dimensions of the input x, weight, weightScale, xScale, groupList, output, or outputScale do not comply with the constraints.
                                         2. The shape of the input x, weight, weightScale, xScale, groupList, output, or outputScale does not comply with the constraints.
                                         3. The format of the input x, weight, weightScale, xScale, groupList, output, or outputScale does not comply with the constraints.
                                         4. The number of elements in groupList is greater than the length of the first axis of weight.
                                         5. The length of the N-axis exceeds 10240.
                                         6. In the A8W8 scenario, the length of the last axis of x is greater than or equal to 65536.
                                         7. In the A8W4 scenario, the length of the last axis of x is greater than or equal to 20000.
  ```

## aclnnGroupedMatmulSwigluQuantWeightNZ

- **Parameters**
    - workspace (void*, input parameter): address of the workspace to be allocated on the device.
    - workspaceSize (uint64_t, input parameter): workspace size to be allocated on the device, which is obtained by the first-phase API `aclnnGroupedMatmulSwigluQuantWeightNZGetWorkspaceSize`.
    - executor (aclOpExecutor*, input parameter): operator executor, containing the operator computation process.
    - stream (aclrtStream, input parameter): stream for executing the task.
- **Return**

    `aclnnStatus` status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - `aclnnGroupedMatmulSwigluQuantWeightNZ` defaults to a deterministic implementation.

<details>
<summary>A8W8 scenario (A: activation matrix (left matrix); W: weight matrix (right matrix); 8: INT8)</summary>

  1. The length of the last axis of `x` cannot be greater than or equal to 65536.
  2. The length of the N-axis cannot exceed 10240.

</details>

<details>
<summary>A8W4 scenario (A: activation matrix (left matrix); W: weight matrix (right matrix); 8: INT8; 4: INT4)</summary>

  1. The length of the last axis of `x` cannot be greater than or equal to 20000.
  2. The length of the N-axis cannot exceed 10240.

</details>

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
        LOG_PRINT("result[%d] is: %d\n", j, out1Data[j]);
    }
    size = GetShapeSize(outputScaleShape);
    std::vector<float> out2Data(size, 0);
    ret = aclrtMemcpy(out2Data.data(), out2Data.size() * sizeof(out2Data[0]), outputScaleDeviceAddr,
                        size * sizeof(out2Data[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t j = 0; j < size; j++) {
        LOG_PRINT("result[%d] is: %f\n", j, out2Data[j]);
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
