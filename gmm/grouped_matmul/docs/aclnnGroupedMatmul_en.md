# aclnnGroupedMatmul

## Supported Products

|Product     | Supported|
|:----------------------------|:-----------:|
|<term>Atlas A3 training products/Atlas A3 inference products</term>|      ×     |
|<term>Atlas A2 training products/Atlas A2 inference products</term>|      √     |
|<term>Atlas 200I/500 A2 inference products</term>|      ×     |
|<term>Atlas inference products</term>|      ×     |
|<term>Atlas training products</term>|      ×     |

## Function

- Description: Implements grouped matrix multiplication, supporting non-uniform matrix dimension sizes across multiple groups. The basic function is matrix multiplication, for example, $y_i[m_i,n_i]=x_i[m_i,k_i] \times weight_i[k_i,n_i], i=1...g$, where $g$ indicates the number of groups and $m_i$, $k_i$, and $n_i$ define the shapes for each group. The following four scenarios are supported based on the tensor count of $x$, $weight$, and $y$:

    - Multi-tensor $x$, $weight$, and $y$. That is, the tensors of each group are independent.
    - Single-tensor $x$, multi-tensor $weight$ and $y$. In this case, use the optional parameter `group_list` to define the row-wise grouping of $x$. For example, `group_list[0]=10` indicates that the first 10 rows of $x$ participate in the multiplication of the first group of matrices.
    - Multi-tensor $x$ and $weight$, single-tensor $y$. In this case, products of each matrix group multiplication are stored contiguously within a single tensor.
    - Single-tensor $x$ and $y$, multi-tensor $weight$. This is a hybrid configuration combining the preceding two cases.

    **Note**: "Single-tensor" means that tensors of all groups in a tensor list are concatenated into one tensor along the M-axis.
- Formula:
    - **Non-quantization scenario:**

    $$
      y_i=x_i\times weight_i + bias_i
    $$

    - **Quantization scenario:**

    $$
      y_i=(x_i\times weight_i + bias_i) * scale_i + offset_i
    $$

    - **Dequantization scenario:**

    $$
      y_i=(x_i\times weight_i + bias_i) * scale_i
    $$

    - **Fake-quantization scenario:**

    $$
      y_i=x_i\times (weight_i + antiquant\_offset_i) * antiquant\_scale_i + bias_i
    $$

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnGroupedMatmulGetWorkspaceSize` is called to obtain the input parameters and compute the required workspace size based on the process. Then, `aclnnGroupedMatmul` is called to perform computation.

* `aclnnStatus aclnnGroupedMatmulGetWorkspaceSize(const aclTensorList* x, const aclTensorList* weight, const aclTensorList* biasOptional, const aclTensorList* scaleOptional, const aclTensorList* offsetOptional, const aclTensorList* antiquantScaleOptional, const aclTensorList* antiquantOffsetOptional, const aclIntArray* groupListOptional, int64_t splitItem, const aclTensorList* y, uint64_t* workspaceSize, aclOpExecutor** executor)`
* `aclnnStatus aclnnGroupedMatmul(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)`

## aclnnGroupedMatmulGetWorkspaceSize

- **Parameters**
  - x (aclTensorList\*, computation input): required parameter, aclTensorList on the device, $x$ in the formula. The [data format](../../../docs/en/context/data_format.md) can be ND, and the maximum length is 128.
      - <term>Atlas A2 training products/Atlas A2 inference products</term>: The data type can be FLOAT16, BFLOAT16, or INT8.
  - weight (aclTensorList\*, computation input): required parameter, aclTensorList on the device, $weight$ in the formula. The [data format](../../../docs/en/context/data_format.md) can be ND, and the maximum length is 128.
      - <term>Atlas A2 training products/Atlas A2 inference products</term>: The data type can be FLOAT16, BFLOAT16, or INT8.
  - biasOptional (aclTensorList\*, computation input): optional parameter, aclTensorList on the device, $bias$ in the formula. The [data format](../../../docs/en/context/data_format.md) can be ND, and the length is the same as that of `weight`.
      - <term>Atlas A2 training products/Atlas A2 inference products</term>: The data type can be FLOAT16, FLOAT32, or INT32.
  - scaleOptional (aclTensorList\*, computation input): optional parameter, aclTensorList on the device, indicating the scale factor for quantization parameters. The [data format](../../../docs/en/context/data_format.md) can be ND, and the length is the same as that of `weight`.
      - <term>Atlas A2 training products/Atlas A2 inference products</term>: The data type can be UINT64.
  - offsetOptional (aclTensorList\*, computation input): optional parameter, aclTensorList on the device, indicating the offset for quantization parameters. The [data format](../../../docs/en/context/data_format.md) can be ND, and the length is the same as that of `weight`.
      - <term>Atlas A2 training products/Atlas A2 inference products</term>: The data type can be FLOAT32.
  - antiquantScaleOptional (aclTensorList\*, computation input): optional parameter, aclTensorList on the device, indicating the scale factor for fake-quantization parameters. The [data format](../../../docs/en/context/data_format.md) can be ND, the length is the same as that of `weight`, and the data type can be FLOAT16 or BFLOAT16.
  - antiquantOffsetOptional (aclTensorList\*, computation input): optional parameter, aclTensorList on the device, indicating the offset for fake-quantization parameters. The [data format](../../../docs/en/context/data_format.md) can be ND, the length is the same as that of `weight`, and the data type can be FLOAT16 or BFLOAT16.
  - groupListOptional (aclIntArray\*, computation input): optional parameter, aclIntArray type on the host, indicating the Matmul indexing for the input and output along the M-axis. The data type can be INT64, the [data format](../../../docs/en/context/data_format.md) can be ND, and the length is the same as that of `weight`. Note that when the length of the TensorList in the output is 1, the last value in `groupListOptional` constrains the valid portion of the output data. Any portion not specified in `groupListOptional` will not be updated.
  - splitItem (int64\_t, computation input): integer type, indicating whether tensor splitting is required for the output. `0` or `1` indicates multi-tensor, and `2` or `3` indicates single-tensor.
  - y (aclTensorList\*, computation output): aclTensorList on the device, $y$ in the formula. The [data format](../../../docs/en/context/data_format.md) can be ND, and the maximum length is 128.
      - <term>Atlas A2 training products/Atlas A2 inference products</term>: The data type can be FLOAT16, BFLOAT16, INT8, or FLOAT32.
  - workspaceSize (uint64\_t\*, output): size of the workspace to be allocated on the device.
  - executor (aclOpExecutor\*\*, output): operator executor, containing the operator computation process.

- **Return**

  `aclnnStatus` status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  ```cpp
  The first-phase API implements input parameter validation. The following errors may be thrown:
  - 161001 (ACLNN_ERR_PARAM_NULLPTR):
  1. The required input, output, or attribute is passed as a null pointer.
  2. The input weight contains elements that are null pointers.
  3. The input x contains elements that are null pointers, while the corresponding elements in the output y are non-null pointers.
  4. The input x contains elements that are non-null pointers, while the corresponding elements in the output y are null pointers.
  - 161002 (ACLNN_ERR_PARAM_INVALID):
  1. The data type or data format of x, weight, biasOptional, scaleOptional, offsetOptional, antiquantScaleOptional, antiquantOffsetOptional, groupListOptional, splitItem, or y is not supported.
  2. The weight length is greater than 128.
  3. The bias is not null, but the bias length is not equal to the weight length.
  4. When splitItem is set to 2 or 3, the length of y is not 1.
  5. When splitItem is set to 0 or 1, the length of y is not equal to the length of weight, and the length of groupListOptional is not equal to the length of weight.
  ```

## aclnnGroupedMatmul

- **Parameters**
    - workspace (void\*, input): address of the workspace to be allocated on the device.
    - workspaceSize (uint64\_t, input): size of the workspace to be allocated on the device, which is obtained by calling `aclnnGroupedMatmulGetWorkspaceSize`.
    - executor (aclOpExecutor\*, input): operator executor, containing the operator computation process.
    - stream (aclrtStream, input): stream for executing the task.

- **Return**

    `aclnnStatus` status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

  - Deterministic computation:
    - `aclnnGroupedMatmul` defaults to a deterministic implementation.
  - <term>Atlas A2 training products/Atlas A2 inference products</term>:
    - The following input types are supported in non-quantization scenarios:
      - `x`: FLOAT16; `weight`: FLOAT16; `biasOptional`: FLOAT16; `scaleOptional`: null; `offsetOptional`: null; `antiquantScaleOptional`: null; `antiquantOffsetOptional`: null; `y`: FLOAT16
      - `x`: BFLOAT16; `weight`: BFLOAT16; `biasOptional`: FLOAT32; `scaleOptional`: null; `offsetOptional`: null; `antiquantScaleOptional`: null; `antiquantOffsetOptional`: null; `y`: BFLOAT16
      - `x`: FLOAT32; `weight`: FLOAT32; `biasOptional`: FLOAT32; `scaleOptional`: null; `offsetOptional`: null; `antiquantScaleOptional`: null; `antiquantOffsetOptional`: null; `y`: FLOAT32
    - The following input types are supported in quantization scenarios:

      - `x`: INT8; `weight`: INT8; `biasOptional`: INT32; `scaleOptional`: UINT64; `offsetOptional`: null; `antiquantScaleOptional`: null; `antiquantOffsetOptional`: null; `y`: INT8
    - The following input types are supported in fake-quantization scenarios:
      - `x`: FLOAT16; `weight`: INT8; `biasOptional`: FLOAT16; `scaleOptional`: null; `offsetOptional`: null; `antiquantScaleOptional`: FLOAT16; `antiquantOffsetOptional`: FLOAT16; `y`: FLOAT16
      - `x`: BFLOAT16; `weight`: INT8; `biasOptional`: FLOAT32; `scaleOptional`: null; `offsetOptional`: null; `antiquantScaleOptional`: BFLOAT16; `antiquantOffsetOptional`: BFLOAT16; `y`: BFLOAT16
    - If `groupListOptional` is passed, it must be a non-negative ascending array, and its length cannot be 1.
    - The following scenarios are supported:
      "S" stands for single-tensor, and "M" stands for multi-tensor, expressed in the sequence of `x`, `weight`, `y`. For example, "SMS" indicates single-tensor `x`, multi-tensor `weight`, and single-tensor `y`.

      | Supported Scenario| Scenario Restrictions|
      |:-------:| :-------|
      | MMM|(1) `splitItem` can only be set to `0` or `1`.<br>(2) The tensors in `x` must have the same dimensionality, which can be 2D to 6D. The tensors in `weight` must be 2D. The dimensionality of tensors in `y` must match that of `x`.<br>(3) If any tensor in `x` has more than 2 dimensions, pass `groupListOptional` as null.<br>(4) If the tensors in `x` are 2D and `groupListOptional` is passed, the differences of `groupListOptional` values must be in one-to-one mapping with the first dimension of the tensors in `x`.|
      | SMS|(1) `splitItem` can only be set to `2` or `3`.<br>(2) `groupListOptional` must be passed, and its last value must be the same as the first dimension of the tensor in `x`.<br>(3) Tensors in `x`, `weight`, and `y` must be 2D.<br>(4) The N-axis of each tensor in `weight` must be the same.|
      | SMM|(1) `splitItem` can only be set to `0` or `1`.<br>(2) `groupListOptional` must be passed. The differences of `groupListOptional` values must be in one-to-one mapping with the first dimension of the tensors in `y`.<br>(3) Tensors in `x`, `weight`, and `y` must be 2D.|
      | MMS|(1) `splitItem` can only be set to `2` or `3`.<br>(2) Tensors in `x`, `weight`, and `y` must be 2D.<br>(3) The N-axis of each tensor in `weight` must be the same.<br>(4) If `groupListOptional` is passed, the differences of `groupListOptional` values must be in one-to-one mapping with the first dimension of the tensors in `x`.|

    - The size of the last dimension for each tensor in `x` and `weight` should be less than 65536. The last dimension of $x_i$ refers to the K-axis when `transpose_x` is false or the M-axis when `transpose_x` is true.  The last dimension of $weight_i$ refers to the N-axis when `transpose_weight` is false or the K-axis when `transpose_weight` is true.
    - The size of each dimension for every tensor in `x` and `weight`, after 32-byte alignment, should be less than the maximum value of INT32 (2147483647).

## Calling Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```c++
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_grouped_matmul.h"

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
int CreateAclTensor(const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor) {
    auto size = GetShapeSize(shape) * sizeof(T);
    // Call aclrtMalloc to allocate memory on the device.
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    // Call aclrtMemcpy to copy the data on the host to the memory on the device.
    std::vector<T> hostData(size, 0);
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


int CreateAclTensorList(const std::vector<std::vector<int64_t>>& shapes, void** deviceAddr,
                        aclDataType dataType, aclTensorList** tensor) {
    int size = shapes.size();
    aclTensor* tensors[size];
    for (int i = 0; i < size; i++) {
        int ret = CreateAclTensor<uint16_t>(shapes[i], deviceAddr + i, dataType, tensors + i);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
    }
    *tensor = aclCreateTensorList(tensors, size);
    return ACL_SUCCESS;
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
    std::vector<std::vector<int64_t>> xShape = {{1, 16}, {4, 32}};
    std::vector<std::vector<int64_t>> weightShape= {{16, 24}, {32, 16}};
    std::vector<std::vector<int64_t>> biasShape = {{24}, {16}};
    std::vector<std::vector<int64_t>> yShape = {{1, 24}, {4, 16}};
    void* xDeviceAddr[2];
    void* weightDeviceAddr[2];
    void* biasDeviceAddr[2];
    void* yDeviceAddr[2];
    aclTensorList* x = nullptr;
    aclTensorList* weight = nullptr;
    aclTensorList* bias = nullptr;
    aclIntArray* groupedList = nullptr;
    aclTensorList* scale = nullptr;
    aclTensorList* offset = nullptr;
    aclTensorList* antiquantScale = nullptr;
    aclTensorList* antiquantOffset = nullptr;
    aclTensorList* y = nullptr;
    int64_t splitItem = 0;

    // Create an x aclTensorList.
    ret = CreateAclTensorList(xShape, xDeviceAddr, aclDataType::ACL_FLOAT16, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create a weight aclTensorList.
    ret = CreateAclTensorList(weightShape, weightDeviceAddr, aclDataType::ACL_FLOAT16, &weight);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create a bias aclTensorList.
    ret = CreateAclTensorList(biasShape, biasDeviceAddr, aclDataType::ACL_FLOAT16, &bias);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create a y aclTensorList.
    ret = CreateAclTensorList(yShape, yDeviceAddr, aclDataType::ACL_FLOAT16, &y);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;

    // 3. Call the CANN operator library API.
    // Call the first-phase API of aclnnGroupedMatmul.
    ret = aclnnGroupedMatmulGetWorkspaceSize(x, weight, bias, scale, offset, antiquantScale, antiquantOffset, groupedList, splitItem, y, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupedMatmulGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // Allocate device memory based on workspaceSize computed by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // Call the second-phase API of aclnnGroupedMatmul.
    ret = aclnnGroupedMatmul(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupedMatmul failed. ERROR: %d\n", ret); return ret);

    // 4. (Boilerplate) Wait until the task execution is complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. Obtain the output value and copy the result from the device memory to the host. Modify the code based on the API definition.
    for (int i = 0; i < 2; i++) {
        auto size = GetShapeSize(yShape[i]);
        std::vector<uint16_t> resultData(size, 0);
        ret = aclrtMemcpy(resultData.data(), size * sizeof(resultData[0]), yDeviceAddr[i],
                          size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
        for (int64_t j = 0; j < size; j++) {
            LOG_PRINT("result[%ld] is: %f\n", j, resultData[j]);
        }
    }

    // 6. Release aclTensors and aclScalars. Modify the code based on the API definition.
    aclDestroyTensorList(x);
    aclDestroyTensorList(weight);
    aclDestroyTensorList(bias);
    aclDestroyTensorList(y);

    // 7. Release device resources. Modify the code based on the API definition.
    for (int i = 0; i < 2; i++) {
        aclrtFree(xDeviceAddr[i]);
        aclrtFree(weightDeviceAddr[i]);
        aclrtFree(biasDeviceAddr[i]);
        aclrtFree(yDeviceAddr[i]);
    }
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
  ```
