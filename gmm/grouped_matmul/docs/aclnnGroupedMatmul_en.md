# aclnnGroupedMatmul

**Note: This API will be deprecated in later versions. Use the latest aclnnGroupedMatmulV5 API.**

## Supported Products

|Product     | Supported|
|:----------------------------|:-----------:|
|Ascend 950PR/Ascend 950DT|      √     |
|<term>Atlas A3 training products/Atlas A3 inference products</term>|      √     |
|<term>Atlas A2 training products/Atlas A2 inference products</term>|      √     |

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

```cpp
aclnnStatus aclnnGroupedMatmulGetWorkspaceSize(
    const aclTensorList  *x,
    const aclTensorList  *weight,
    const aclTensorList  *biasOptional,
    const aclTensorList  *scaleOptional,
    const aclTensorList  *offsetOptional,
    const aclTensorList  *antiquantScaleOptional,
    const aclTensorList  *antiquantOffsetOptional,
    const aclIntArray    *groupListOptional,
    int64_t               splitItem,
    const aclTensorList  *y,
    uint64_t             *workspaceSize,
    aclOpExecutor        **executor)
```

```cpp
aclnnStatus aclnnGroupedMatmul(
    void            *workspace,
    uint64_t         workspaceSize,
    aclOpExecutor   *executor,
    aclrtStream      stream)
```

## aclnnGroupedMatmulGetWorkspaceSize

- **Parameters**

  <table style="undefined;table-layout: fixed;width: 1540px"><colgroup>
    <col style="width: 170px">
    <col style="width: 120px">
    <col style="width: 300px">
    <col style="width: 330px">
    <col style="width: 212px">
    <col style="width: 100px">
    <col style="width: 190px">
    <col style="width: 118px">
    </colgroup>
    <thead>
      <tr>
        <th>Name</th>
        <th style="white-space: nowrap">Input/Output</th>
        <th>Description</th>
        <th>Usage</th>
        <th>Data Type</th>
        <th><a href="../../../docs/en/context/data_format.md" target="_blank"> Data Format</a></th>
        <th style="white-space: nowrap">Dimension (Shape)</th>
        <th><a href="../../../docs/en/context/non_contiguous_tensor.md" target="_blank">Non-Contiguous Tensor</a></th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>x (aclTensorList)</td>
        <td>Input</td>
        <td>x in the formula.</td>
        <td>
          <ul>
            <li>The maximum length is 128 characters.</li>
          </ul>
        </td>
        <td>FLOAT16, BFLOAT16, FLOAT32, INT8</td>
        <td>ND</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>weight (aclTensorList)</td>
        <td>Input</td>
        <td>weight in the formula.</td>
        <td>
          <ul>
            <li>The maximum length is 128 characters.</li>
          </ul>
        </td>
        <td>FLOAT16, BFLOAT16, FLOAT32, INT8</td>
        <td>ND</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>biasOptional (aclTensorList)</td>
        <td>Optional input</td>
        <td>Bias in the formula.</td>
        <td>
          <ul>
            <li>The length is the same as that of weight.</li>
          </ul>
        </td>
        <td>FLOAT16, FLOAT32, INT32, BFLOAT16</td>
        <td>ND</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>scaleOptional (aclTensorList)</td>
        <td>Optional input</td>
        <td>Scaling factor in the quantization parameters.</td>
        <td>
          <ul>
            <li>The length is the same as that of weight.</li>
          </ul>
        </td>
        <td>UINT64</td>
        <td>ND</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>offsetOptional (aclTensorList)</td>
        <td>Optional input</td>
        <td>Offset in the quantization parameters.</td>
        <td>
          <ul>
            <li>The length is the same as that of weight.</li>
          </ul>
        </td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>antiquantScaleOptional (aclTensorList)</td>
        <td>Optional input</td>
        <td>Indicates the scaling factor in the fake-quantization parameter.</td>
        <td>
          <ul>
            <li>The length is the same as that of weight.</li>
          </ul>
        </td>
        <td>FLOAT16, BFLOAT16</td>
        <td>ND</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>antiquantOffsetOptional (aclTensorList)</td>
        <td>Optional input</td>
        <td>Indicates the offset in the fake-quantization parameter.</td>
        <td>
          <ul>
            <li>The length is the same as that of weight.</li>
          </ul>
        </td>
        <td>FLOAT16, BFLOAT16</td>
        <td>ND</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>groupListOptional (aclTensorList)</td>
        <td>Optional input</td>
        <td>Indicates the Matmul index of the input and output in the M direction.</td>
        <td>
          <ul>
            <li>The length is the same as that of weight.</li>
            <li>When the length of the TensorList output is 1, the last value in groupListOptional specifies the valid part of the output data. The parts that are not specified in groupListOptional will not be updated.</li>
          </ul>
        </td>
        <td>INT64</td>
        <td>ND</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>splitItem (int64_t)</td>
        <td>Input</td>
        <td>Integer, indicating whether to split the output tensor.</td>
        <td>
          <ul>
            <li>0 or 1 indicates that the output is a multi-tensor. 2 or 3 indicates that the output is a single tensor.</li>
          </ul>
        </td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>y (aclTensorList)</td>
        <td>Output</td>
        <td>y in the formula.</td>
        <td>
          <ul>
            <li>The maximum length is 128 characters.</li>
          </ul>
        </td>
        <td>FLOAT16, BFLOAT16, INT8, FLOAT32</td>
        <td>ND</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>workspaceSize (uint64_t)</td>
        <td>Output</td>
        <td>Size of the workspace to be allocated on the device.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>executor (aclOpExecutor)</td>
        <td>Output</td>
        <td>Operator executor, containing the operator computation process.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
    </tbody></table>

  - <term>Atlas A2 training products/Atlas A2 inference products</term>:
    - x and weight support FLOAT16, BFLOAT16, and INT8.
    - y supports FLOAT16, BFLOAT16, INT8, and FLOAT32.
  - Ascend 950PR/Ascend 950DT:
    - x supports FLOAT16, BFLOAT16, and FLOAT32.
    - weight supports FLOAT16, BFLOAT16, FLOAT32, and INT8.
    - y supports FLOAT16, BFLOAT16, and FLOAT32.
    - scaleOptional and offsetOptional are not supported.

- **Return**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API implements input parameter validation. The following errors may be thrown.

  <table style="undefined;table-layout: fixed;width: 1030px"><colgroup>
  <col style="width: 250px">
  <col style="width: 130px">
  <col style="width: 650px">
  </colgroup>
  <thead>
    <tr>
      <th>Return</th>
      <th>Error Code</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>The required input, output, or required attribute is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="5">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="5">161002</td>
      <td>x, weight, biasOptional, scaleOptional, offsetOptional, antiquantScaleOptional, antiquantOffsetOptional, groupListOptional, splitItem The data type and format of y are not supported.</td>
    </tr>
    <tr>
      <td>The length of weight is not supported.</td>
    </tr>
    <tr>
      <td>If bias is not null, the length of bias is not equal to that of weight.</td>
    </tr>
    <tr>
      <td>When splitItem is 0 or 1, the length of y is not equal to that of weight, and the length of groupListOptional is not equal to that of weight.</td>
    </tr>
    <tr>
      <td>When splitItem is 2 or 3, the length of y is not equal to 1.</td>
    </tr>
  </tbody></table>

## aclnnGroupedMatmul

- **Parameters**
    <table>
    <thead>
      <tr><th>Parameter</th><th>Input/Output</th><th>Description</th></tr>
    </thead>
    <tbody>
      <tr><td>workspace</td><td>Input</td><td>Address of the workspace to be allocated on the device.</td></tr>
      <tr><td>workspaceSize</td><td>Input</td><td>Size of the workspace allocated on the device, which is obtained by the first API aclnnGroupedMatmulGetWorkspaceSize.</td></tr>
      <tr><td>executor</td><td>Input</td><td>Operator executor, containing the operator computation process.</td></tr>
      <tr><td>stream</td><td>Input</td><td>Stream for executing a task.</td></tr>
    </tbody>
    </table>

- **Return**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - `aclnnGroupedMatmul` defaults to a deterministic implementation.
- <term>Atlas A2 training products/Atlas A2 inference products</term>:
  - The following input types are supported in non-quantization scenarios:
    - `x`: FLOAT16; `weight`: FLOAT16; `biasOptional`: FLOAT16; `scaleOptional`: null; `offsetOptional`: null; `antiquantScaleOptional`: null; `antiquantOffsetOptional`: null; `y`: FLOAT16
    - `x`: BFLOAT16; `weight`: BFLOAT16; `biasOptional`: FLOAT32; `scaleOptional`: null; `offsetOptional`: null; `antiquantScaleOptional`: null; `antiquantOffsetOptional`: null; `y`: BFLOAT16
  - The following input type is supported in quantization scenarios:

    - `x`: INT8; `weight`: INT8; `biasOptional`: INT32; `scaleOptional`: UINT64; `offsetOptional`: null; `antiquantScaleOptional`: null; `antiquantOffsetOptional`: null; `y`: INT8
  - The following input types are supported in fake-quantization scenarios:
    - `x`: FLOAT16; `weight`: INT8; `biasOptional`: FLOAT16; `scaleOptional`: null; `offsetOptional`: null; `antiquantScaleOptional`: FLOAT16; `antiquantOffsetOptional`: FLOAT16; `y`: FLOAT16
    - `x`: BFLOAT16; `weight`: INT8; `biasOptional`: FLOAT32; `scaleOptional`: null; `offsetOptional`: null; `antiquantScaleOptional`: BFLOAT16; `antiquantOffsetOptional`: BFLOAT16; `y`: BFLOAT16
  - If `groupListOptional` is passed, it must be a non-negative ascending array, and its length cannot be 1.
  - The following scenarios are supported:
      "S" stands for single-tensor, and "M" stands for multi-tensor, expressed in the sequence of x, weight, y. For example, "SMS" indicates single-tensor x, multi-tensor weight, and single-tensor y.

      | Supported Scenario| Scenario Restrictions|
      |:-------:| :-------|
      | MMM|(1) `splitItem` can only be set to `0` or `1`.<br>(2) The tensors in `x` must have the same dimensionality, which can be 2D to 6D. The tensors in `weight` must be 2D. The dimensionality of tensors in `y` must match that of `x`.<br>(3) If any tensor in `x` has more than 2 dimensions, pass `groupListOptional` as null.<br>(4) If the tensors in `x` are 2D and `groupListOptional` is passed, the differences of `groupListOptional` values must be in one-to-one mapping with the first dimension of the tensors in `x`.|
      | SMS|(1) `splitItem` can only be set to `2` or `3`.<br>(2) `groupListOptional` must be passed, and its last value must be the same as the first dimension of the tensor in `x`.<br>(3) Tensors in `x`, `weight`, and `y` must be 2D.<br>(4) The N-axis of each tensor in `weight` must be the same.|
      | SMM|(1) `splitItem` can only be set to `0` or `1`.<br>(2) `groupListOptional` must be passed. The differences of `groupListOptional` values must be in one-to-one mapping with the first dimension of the tensors in `y`.<br>(3) Tensors in `x`, `weight`, and `y` must be 2D.|
      | MMS|(1) `splitItem` can only be set to `2` or `3`.<br>(2) Tensors in `x`, `weight`, and `y` must be 2D.<br>(3) The N-axis of each tensor in `weight` must be the same.<br>(4) If `groupListOptional` is passed, the differences of `groupListOptional` values must be in one-to-one mapping with the first dimension of the tensors in `x`.|

  - The size of the last dimension for each tensor in `x` and `weight` should be less than 65536. The last dimension of $x_i$ refers to the K-axis when `transpose_x` is false or the M-axis when `transpose_x` is true.  The last dimension of $weight_i$ refers to the N-axis when `transpose_weight` is false or the K-axis when `transpose_weight` is true.
  - The size of each dimension for every tensor in `x` and `weight`, after 32-byte alignment, should be less than the maximum value of INT32 (2147483647).

- Ascend 950PR/Ascend 950DT:

  <details>
    <summary>Constraints for the non-quantization scenario</summary>
      <a id="constraints-for-non-quantization-scenario"></a>

  - The following data types are supported in non-quantization scenarios:
    - If `groupListOptional` is passed, it must be a non-negative ascending array, and its length cannot be 1.
    - The following input parameters are empty: scaleOptional, offsetOptional, antiquantScaleOptional, and antiquantOffsetOptional.
    - The data type combinations supported by the parameters that are not empty must meet the requirements in the following table:

      | x       | weight  | biasOptional | y     |
      |:-------:|:-------:| :------      |:------ |
      |BFLOAT16     |BFLOAT16     |BFLOAT16/FLOAT32/null    | BFLOAT16|
      |FLOAT16     |FLOAT16     |FLOAT16/FLOAT32/null    | FLOAT16|

  </details>

    <details>
    <summary>Restrictions on the fake-quantization scenario</summary>
      <a id="constraints-for-fake-quantization-scenario"></a>

    - The fake-quantization scenario supports the following data types:
      - The following input parameters are empty: scaleOptional and offsetOptional.
      - The combinations of data types supported by non-empty parameters must meet the requirements listed in the following table.

          | x       | weight  | biasOptional | antiquantScaleOptional | antiquantOffsetOptional | y     |
          |:-------:|:-------:| :------      |:------ |:------ |:------ |
          |BFLOAT16    |INT8     |BFLOAT16/FLOAT32/null    | BFLOAT16 | BFLOAT16 | BFLOAT16 |
          |FLOAT16     |INT8     |FLOAT16/null             | FLOAT16  | FLOAT16  | FLOAT16  |

      - The antiquantScaleOptional, non-empty biasOptional, and antiquantOffsetOptional must meet the requirements listed in the following table.

        | Application Scenario| Shape Restriction|
        |:---------:| :------ |
        |Weight multi-tensor|Each tensor is one-dimensional, and the shape is ($n_i$). It is not allowed that some tensors in a tensor list have the shape ($n_i$) and some tensors are empty.|

    - Only the MMM scenario is supported.
    </details>

    <details>
    <summary>Constraints on supported scenarios</summary>
      <a id="constraints-on-supported-scenarios"></a>

      - "S" stands for single-tensor and "M" stands for multi-tensor, expressed in the sequence of `x`, `weight`, `y`. For example, "SMS" indicates single-tensor `x`, multi-tensor `weight`, and single-tensor `y`.

        | Supported Scenario| Scenario Restrictions|
        |:-------:| :-------|
        | MMM|(1) `splitItem` can only be set to `0` or `1`.<br>(2) In the fake-quantization scenario, the tensor in x must have the same dimension as that in y, and the dimension can be 2 to 6 dimensions. In the non-quantization scenario, the tensor in x and y must be 2-dimensional, with the shape being ($m_i$, $k_i$) and ($m_i$, $n_i$), respectively. The tensor in weight must be 2-dimensional, with the shape being ($n_i$, $k_i$) or ($k_i$, $n_i$). The tensor in bias must be 1-dimensional, with the shape being ($n_i$).<br>(3) If any tensor in x has more than 2 dimensions, pass groupListOptional as null.<br>(4) If the tensor in x is 2-dimensional and groupListOptional is passed, the difference between the values in groupListOptional must correspond to the first dimension of the tensor in x, and the maximum length is 128.<br>(5) Only ND input and ND output are supported.<br>(6) x transpose and weight transpose are not supported.|
        | SMS|(1) `splitItem` can only be set to `2` or `3`.<br>(2) groupListOptional must be passed. The last value in groupListOptional must be the same as the first dimension of the tensor in x, and the maximum length is 128.<br>(3) The tensor in x and y must be 2-dimensional, with the shape being (M, K) and (M, N), respectively. The tensor in weight must be 2-dimensional, with the shape being (N, K) or (K, N). The tensor in bias must be 1-dimensional, with the shape being (N).<br>(4) The N-axis of each tensor in `weight` must be the same.<br>(5) Only ND input and ND output are supported.<br>(6) Only non-quantization is supported.<br>(7) x transpose is not supported, but weight transpose is supported. If weight is a multi-tensor, the transpose status of each tensor must be the same.|
        | SMM|(1) `splitItem` can only be set to `0` or `1`.<br>(2) groupListOptional must be passed. The last value in groupListOptional must be the same as the first dimension of the tensor in x, and the maximum length is 128.<br>(3) The tensor in x and y must be 2-dimensional, with the shape being (M, K) and (M, N), respectively. The tensor in weight must be 2-dimensional, with the shape being (N, K) or (K, N). The tensor in bias must be 1-dimensional, with the shape being (N).<br>(4) Only ND input and ND output are supported.<br>(5) Only non-quantization is supported.<br>(6) x transpose is not supported, but weight transpose is supported. If weight is a multi-tensor, the transpose status of each tensor must be the same.|
        | MMS|(1) `splitItem` can only be set to `2` or `3`.<br>(2) The tensor in x and y must be 2-dimensional, with the shape being (M, K) and (M, N), respectively. The tensor in weight must be 2-dimensional, with the shape being (N, K) or (K, N). The tensor in bias must be 1-dimensional, with the shape being (N).<br>(3) The N-axis of each tensor in `weight` must be the same.<br>(4) If groupListOptional is passed, the difference between groupListOptional must correspond to the first dimension of the tensor in x, and the maximum length is 128.<br>(5) Only ND input and ND output are supported.<br>(6) Only non-quantization is supported.<br>(7) x transposition is not supported, but weight transposition is supported. If weight contains multiple tensors, the transposition status of each tensor must be the same.|

    </details>

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
    std::vector<T> hostData(GetShapeSize(shape), 0);
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
    std::vector<aclTensor*> tensors(size);
    for (int i = 0; i < size; i++) {
        int ret = CreateAclTensor<uint16_t>(shapes[i], deviceAddr + i, dataType, tensors.data() + i);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
    }
    *tensor = aclCreateTensorList(tensors.data(), size);
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
            LOG_PRINT("result[%ld] is: %hu\n", j, resultData[j]);
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
