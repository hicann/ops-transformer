# aclnnGroupedMatmulAdd

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √     |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference products</term>                            |    ×     |
| <term>Atlas training products</term>                             |    ×     |

## Function

- Description: Implements grouped matrix multiplication, supporting non-uniform matrix dimension sizes across multiple groups. The basic function is matrix multiplication, for example, $y_i[m_i,n_i]=x_i[m_i,k_i] \times weight_i[k_i,n_i]+y_i[m_i,n_i], i=1...g$, where $g$ indicates the number of groups and $m_i$, $k_i$, and $n_i$ define the shapes for each group. The input and output data types are both aclTensor, with K-axis grouping.

  - K-axis grouping: $k_i$ can vary, while $m_i$ and $n_i$ remain the same for each group.
- Formula:

  $$
  yRef_i=x_i\times weight_i + y_i
  $$

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnGroupedMatmulAddGetWorkspaceSize` is called to obtain the input parameters and compute the required workspace size based on the process. Then, `aclnnGroupedMatmulAdd` is called to perform computation.

```c++
aclnnStatus aclnnGroupedMatmulAddGetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *weight,
    const aclTensor *groupList,
    aclTensor       *yRef,
    bool             transposeX,
    bool             transposeWeight,
    int64_t          groupType,
    uint64_t        *workspaceSize,
    aclOpExecutor  **executor)
```

```c++
aclnnStatus aclnnGroupedMatmulAdd(
    void            *workspace,
    uint64_t         workspaceSize,
    aclOpExecutor   *executor,
    aclrtStream      stream
)
```

## aclnnGroupedMatmulAddGetWorkspaceSize

- **Parameters**

    <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
      <col style="width: 170px">
      <col style="width: 120px">
      <col style="width: 300px">  
      <col style="width: 300px">  
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
        <th>Usage Notes</th>
        <th>Data Type</th>
        <th>Data Format</th>
        <th>Dimension (Shape)</th>
        <th>Non-contiguous Tensor</th>
      </tr></thead>
    <tbody>
      <tr>
        <td>x</td>
        <td>Input</td>
        <td>x in the formula.</td>
        <td>-</td>
        <td>FLOAT16, BFLOAT16</td>
        <td>ND</td>
        <td>2</td>
        <td>√</td>
      </tr>
      <tr>
        <td rowspan="2">weight</td>
        <td rowspan="2">Input</td>
        <td>weight in the formula.</td>
        <td>-</td>
        <td>FLOAT16, BFLOAT16</td>
        <td>ND</td>
        <td>2</td>
        <td>√</td>
      </tr>
      <tr>
        <td>Distribution of Matmul sizes along the input K-axis.</td>
        <td>Only the cumulative sum (cumsum) mode is supported.</td>
        <td>INT64</td>
        <td>ND</td>
        <td>1</td>
        <td>√</td>
      </tr>
      <tr>
        <td>y</td>
        <td>Input</td>
        <td>Output matrix for in-place accumulation.</td>
        <td>-</td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>2</td>
        <td>×</td>
      </tr>
      <tr>
        <td>transposeX</td>
        <td>Attribute</td>
        <td>Specifies whether to transpose the x matrix.</td>
        <td>The value can only be True.</td>
        <td>BOOL</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>transposeWeight</td>
        <td>Attribute</td>
        <td>Specifies whether to transpose the weight matrix.</td>
        <td>The value can only be False.</td>
        <td>BOOL</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>groupType</td>
        <td>Attribute</td>
        <td>Grouping type.</td>
        <td>The value can only be 2 (K-axis grouping).</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>groupListType</td>
        <td>Attribute</td>
        <td>groupList format.</td>
        <td>The value can only be 0 (cumsum mode).</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>yRef</td>
        <td>Output</td>
        <td>Output matrix for in-place accumulation.</td>
        <td>-</td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>2</td>
        <td>×</td>
      </tr>
    </tbody></table>

- **Return**

    `aclnnStatus` status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

    The first-phase API implements input parameter validation. The following errors may be thrown:
    <table style="undefined;table-layout: fixed; width: 1150px"> <colgroup>
    <col style="width: 280px">
    <col style="width: 100px">
    <col style="width: 900px"> 
      </colgroup><thead>
      <tr>
        <th>Return</th>
        <th>Error Code</th>
        <th>Description</th>
      </tr></thead>
    <tbody>
      <tr>
        <td>ACLNN_ERR_PARAM_NULLPTR</td>
        <td>161001</td>
        <td>The passed x, weight, groupList, y, or yRef is a null pointer.</td>
      </tr>
      <tr>
        <td rowspan="2">ACLNN_ERR_PARAM_INVALID</td>
        <td rowspan="2">161002</td>
        <td>The data type or format of x, weight, groupList, y, or yRef is not supported.</td>
      </tr>
      <tr>
        <td>The data types of x and weight are inconsistent.</td>
      </tr>
      <tr>
        <td>ACLNN_ERR_INNER_TILING_ERROR</td>
        <td>561002</td>
        <td>The shapes of x, weight, y, and yRef do not meet the matrix multiplication requirements.</td>
      </tr>
    </tbody>
    </table>

## aclnnGroupedMatmulAdd

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1150px"> =
    <col style="width: 150px">
    <col style="width: 100px">
    <col style="width: 900px"> 
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
        <td>Address of the workspace to be allocated on the device.</td>
    </tr>
    <tr>
        <td>workspaceSize</td>
        <td>Input</td>
        <td>Size of the workspace allocated on the device, which is obtained by calling the first-phase API aclnnGroupedMatmulAddGetWorkspaceSize.</td>
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

- **Return**

    `aclnnStatus` status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- `aclnnGroupedMatmulAdd` defaults to a deterministic implementation.
- The size of each dimension for every tensor in `x` and `weight`, after 32-byte alignment, should be less than the maximum value of INT32 (2147483647).
- Supported input types:
  - `x`: FLOAT16; `weight`: FLOAT16; `y`: FLOAT32
  - `x`: BFLOAT16; `weight`: BFLOAT16; `y`: FLOAT32

## Calling Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```c++
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_grouped_matmul_add.h"

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
int CreateAclTensor_New(const std::vector<int64_t>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                        aclDataType dataType, aclTensor** tensor) {
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
  std::vector<int64_t> xShape = {512, 256};
  std::vector<int64_t> weightShape= {512, 256};
  std::vector<int64_t> yShape = {512, 256};
  std::vector<int64_t> groupListShape = {2};
  std::vector<int64_t> groupListData = {256, 512};
  void* xDeviceAddr;
  void* weightDeviceAddr;
  void* yDeviceAddr;

  void* groupListDeviceAddr;
  aclTensor* x = nullptr;
  aclTensor* weight = nullptr;
  aclTensor* groupedList = nullptr;
  aclTensor* y = nullptr;
  aclTensor* yRef = nullptr;

  bool transpose_x = true;
  bool transpose_weight = false;
  int group_type = 2;

  // Create an x aclTensorList.
  ret = CreateAclTensor<uint16_t>(xShape, &xDeviceAddr, aclDataType::ACL_FLOAT16, &x);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a weight aclTensorList.
  ret = CreateAclTensor<uint16_t>(weightShape, &weightDeviceAddr, aclDataType::ACL_FLOAT16, &weight);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a y aclTensorList.
  ret = CreateAclTensor<float>(yShape, &yDeviceAddr, aclDataType::ACL_FLOAT, &y);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a group_list aclTensor.
  ret = CreateAclTensor_New<int64_t>(groupListData, groupListShape, &groupListDeviceAddr, aclDataType::ACL_INT64, &groupedList);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  yRef = y;

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  // 3. Call the CANN operator library API.
  // Call the first-phase API of aclnnGroupedMatmulAdd.
  ret = aclnnGroupedMatmulAddGetWorkspaceSize(x, weight, groupedList, yRef, transpose_x, transpose_weight, group_type, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupedMatmulGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnGroupedMatmulAdd.
  ret = aclnnGroupedMatmulAdd(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupedMatmul failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device memory to the host. Modify the code based on the API definition.
  auto size = GetShapeSize(yShape);
  std::vector<uint16_t> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), size * sizeof(resultData[0]), yDeviceAddr,
                        size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t j = 0; j < size; j++) {
      LOG_PRINT("result[%ld] is: %d\n", j, resultData[j]);
  }


  // 6. Release aclTensors and aclScalars. Modify the code based on the API definition.
  aclDestroyTensor(x);
  aclDestroyTensor(weight);
  aclDestroyTensor(y);

  // 7. Release device resources. Modify the code based on the API definition.
  aclrtFree(xDeviceAddr);
  aclrtFree(weightDeviceAddr);
  aclrtFree(yDeviceAddr);

  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
