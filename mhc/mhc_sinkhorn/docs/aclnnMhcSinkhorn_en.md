# aclnnMhcSinkhorn

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T14:57:52.490Z -->

## Applicable Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 products</term>: Not supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 products</term>: Not supported
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

- Description: Performs the Sinkhorn iterative normalization transformation on the matrix $\mathbf{H}'_{\text{res}}$ in the mHC architecture to finally obtain a doubly stochastic matrix $\mathbf{H}_{\text{res}}$. Supports outputting the intermediate normalization results (norm_out) and summation results (sum_out) generated during the iteration process, used for backward gradient computation.

## Formula

The Sinkhorn transformation performs $\mathbf{num\_iters}$ iterations in total, generating the intermediate normalization result $\mathbf{norm\_out}[k]$ and the summation result $\mathbf{sum\_out}[k]$ during the iteration process, and finally outputs the $\mathbf{norm\_out}$ of the last iteration as the transformation result.

The first iteration (initialization):

$$
\begin{aligned}
    \mathbf{norm\_out}[0] &= \text{softmax}(\mathbf{x}, \dim=-1) + \epsilon, \\
    \mathbf{sum\_out}[1] &= \sum_{\dim=-2,\text{keepdim}=\text{True}} \mathbf{norm\_out}[0] + \epsilon, \\
    \mathbf{norm\_out}[1] &= \frac{\mathbf{norm\_out}[0]}{\mathbf{sum\_out}[1]}, \\
\end{aligned}
$$

Iteration $i$ ($i = 1, 2, \dots, \mathbf{num\_iters}-1$):

$$
\begin{aligned}
    \mathbf{sum\_out}[2i] &= \sum_{\dim=-1,\text{keepdim}=\text{True}} \mathbf{norm\_out}[2i-1] + \epsilon, \\
    \mathbf{norm\_out}[2i] &= \frac{\mathbf{norm\_out}[2i-1]}{\mathbf{sum\_out}[2i]}, \\
    \mathbf{sum\_out}[2i+1] &= \sum_{\dim=-2,\text{keepdim}=\text{True}} \mathbf{norm\_out}[2i] + \epsilon, \\
    \mathbf{norm\_out}[2i+1] &= \frac{\mathbf{norm\_out}[2i]}{\mathbf{sum\_out}[2i+1]}, \\
\end{aligned}
$$

### Final Output

$$
\text{output} = \mathbf{norm\_out}[2 \times \mathbf{num\_iters} - 1]
$$

### 🔍 Symbol Description

| Symbol | Meaning |
|------|------|
| $\mathbf{x}$ | Input tensor (the $\mathbf{H}'_{\text{res}}$ matrix of the mHC layer). |
| $\epsilon$ | Parameter to prevent division by zero (corresponding to input parameter `eps`). |
| $\text{softmax}(\cdot, \dim=-1)$ | Performs softmax normalization along the last dimension. |
| $\sum_{\dim=d,\text{keepdim}=\text{True}}$ | Computes the sum along the specified dimension $d$ and keeps the dimension. |
| $\mathbf{norm\_out}[k]$ | Intermediate normalization result at step $k$. |
| $\mathbf{sum\_out}[k]$ | Intermediate summation result at step $k$. |
| $\mathbf{num\_iters}$ | Iteration count (input parameter). |

## Function Prototype

The operator is called via the two-phase API: first call `aclnnMhcSinkhornGetWorkspaceSize` to obtain the Device-side memory size required for computation, and then call `aclnnMhcSinkhorn` to perform the actual computation.

```c++
aclnnStatus aclnnMhcSinkhornGetWorkspaceSize(
    const aclTensor *x,
    float eps,
    int64_t numIters,
    aclTensor *output,
    aclTensor *normOut,
    aclTensor *sumOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
```

```c++
aclnnStatus aclnnMhcSinkhorn(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream)
```

## aclnnMhcSinkhornGetWorkspaceSize

- **Parameters**

  | Name | Input/Output | Description | Usage Instructions | Data Type | Data Format | Dimension (Shape) | Non-contiguous Tensor |
  |:--- |:--- |:--- |:--- |:--- |:--- |:--- |:--- |
  | x | Input | Data to be computed, the $\mathbf{H}'_{\text{res}}$ matrix of the mHC layer. | Required parameter. Cannot be an empty tensor. | FLOAT32 | ND | (B,S,n,n), (T,n,n) | √ |
  | eps | Input | Parameter to prevent division by zero in normalization. | Recommended value: 1e-6. | FLOAT32 | - | - | - |
  | numIters| Input | Iteration count. | Recommended value: 20; range: 1-100. | INT64 | - | - | - |
  | output | Output | Final result of the MhcSinkhorn transformation (doubly stochastic matrix $\mathbf{H}_{\text{res}}$). | Required output. Its dimension is the same as that of input x. | FLOAT32 | ND | (B,S,n,n), (T,n,n) | √ |
  | normOut | Output | Normalized intermediate result of the iteration process, used for backward. | Optional parameter. No output when a null pointer is passed. | FLOAT32 | ND | [2\*num_iters\*n\*n_align\*B\*S], [2\*num_iters\*n\*n_align\*T], where n_align is n aligned to 8 | √ |
  | sumOut | Output | Sum intermediate result of the iteration process, used for backward. | Optional parameter. No output when a null pointer is passed. | FLOAT32 | ND | [2\*num_iters\*n_align\*B\*S], [2\*num_iters\*n_align\*T], where n_align is n aligned to 8 | √ |
  | workspaceSize | Output | Device-side workspace memory size (in bytes) required for computation. | Calculated internally by the operator, used to allocate memory later. | UINT64 | - | - | - |
  | executor | Output | Operator executor, containing the computation flow and parameter information. | Must be passed to the second-phase API. | aclOpExecutor* | - | - | - |

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  | Return Value | Error Code | Description |
  |:--- |:--- |:--- |
  | ACLNN_ERR_PARAM_NULLPTR | 161001 | A required parameter (x/output) or output parameter (workspaceSize/executor) is a null pointer. |
  | ACLNN_ERR_PARAM_INVALID | 161002 | 1. The data type/format of x is not FLOAT32/ND;<br>2. numIters is out of the range 1-100;<br>3. The value of n is not 4/6/8;<br>4. When normOut/sumOut is not empty, the total number of elements or the format does not meet the requirements. |
  | ACLNN_ERR_RUNTIME_ERROR | 361001 | Failed to call the NPU Runtime API to allocate memory or create a tensor. |

## aclnnMhcSinkhorn

- **Parameters**

  | Name | Input/Output | Description |
  |:--- |:--- |:--- |
  | workspace | Input | Workspace memory address allocated on the Device side. It must match the workspaceSize returned by the first-phase API. |
  | workspaceSize | Input | Workspace size on the Device side, returned by the `aclnnMhcSinkhornGetWorkspaceSize` API. |
  | executor | Input | Operator executor, created by the first-phase API. It contains the computation process and parameter information. |
  | stream | Input | Stream for executing computation tasks. It must be created and bound to a Device in advance. |

- **Return Value**

  Returns an `aclnnStatus` status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic Computation

  - aclnnMhcSinkhorn defaults to a deterministic implementation. The same input produces the same result across multiple calls.

- General Constraints

  1. Input constraints:
     - If the input tensor `x` is empty, the error code `ACLNN_ERR_PARAM_NULLPTR` is returned;
     - Only `ACL_FORMAT_ND` is supported as the data format of all input/output tensors;
     - Only the `FLOAT32` data type is supported. Other precisions (such as FLOAT16/DOUBLE) are not supported.
     - normOut and sumOut are optional parameters and can be set to null pointers; no corresponding output is generated when null pointers are passed. If they are not null, the total number of elements in normOut must be `2*numIters*T*n*8`, and the total number of elements in sumOut must be `2*numIters*T*8` (where `T` is the batch count after merging the inputs: `T=x.shape[0]` for 3 dimensions and `T=x.shape[0]*x.shape[1]` for 4 dimensions; `n` is the matrix dimension, which can be 4/6/8; `8` is the value of n after alignment to 8), and the format must be `ACL_FORMAT_ND`; otherwise, the error code `ACLNN_ERR_PARAM_INVALID` is returned.
     - For input -inf/inf/nan/, the output is nan/nan/nan.
  2. Memory constraints:
     - Workspace memory must be allocated on the Device side, and its size must strictly match the return value of the first-phase API;

- Specification constraints

  | Specification Item | Specification | Specification Description                              |
  | :------- | :------ | :---------------------------------------------------- |
  | numIters | 1-100   | If the iteration count exceeds this range, an invalid parameter error is returned.                |
  | n        | 4, 6, 8 | Only these three values are supported for the size of the last two dimensions (matrix dimension) of the input tensor.  |
  | Number of dimensions | 3/4     | The input x supports 3D (T,n,n) or 4D (B,S,n,n). Other dimension counts are not supported. |

## Example

The following is a C++ call example, which must be compiled and run in the AscendCL environment. For the detailed procedure, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```c++
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_mhc_sinkhorn.h"

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

// Calculate the total number of elements based on the tensor shape.
int64_t GetShapeSize(const std::vector<int64_t>& shape) {
  int64_t size = 1;
  for (int64_t dim : shape) {
    size *= dim;
  }
  return size;
}

// Copy the tensor data on the Device side to the Host side and print it.
void PrintTensorData(const std::vector<int64_t>& shape, void* device_addr) {
  int64_t size = GetShapeSize(shape);
  std::vector<float> host_data(size, 0.0f);

  // Copy data from Device to Host.
  aclError ret = aclrtMemcpy(
      host_data.data(), size * sizeof(float),
      device_addr, size * sizeof(float),
      ACL_MEMCPY_DEVICE_TO_HOST
  );
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("Memcpy device to host failed, error: %d\n", ret);
            return);

  // Print the first 10 elements (example).
  LOG_PRINT("Tensor data (first 10 elements): ");
  for (int i = 0; i < std::min((int64_t)10, size); ++i) {
    LOG_PRINT("%f ", host_data[i]);
  }
  LOG_PRINT("\n");
}

// Initialize the AscendCL environment (Device/Context/Stream).
int InitAcl(int32_t device_id, aclrtContext& context, aclrtStream& stream) {
  // 1. Initialize ACL
  aclError ret = aclInit(nullptr);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclInit failed, error: %d\n", ret);
            return -1);

  // 2. Set Device
  ret = aclrtSetDevice(device_id);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtSetDevice failed, error: %d\n", ret);
            return -1);

  // 3. Create Context
  ret = aclrtCreateContext(&context, device_id);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtCreateContext failed, error: %d\n", ret);
            return -1);

  // 4. Set the current Context
  ret = aclrtSetCurrentContext(context);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtSetCurrentContext failed, error: %d\n", ret);
            return -1);

  // 5. Create Stream
  ret = aclrtCreateStream(&stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtCreateStream failed, error: %d\n", ret);
            return -1);

  return 0;
}

// Create a device-side aclTensor (including data copy)
int CreateAclTensor(
    const std::vector<float>& host_data,
    const std::vector<int64_t>& shape,
    void*& device_addr,
    aclTensor*& tensor) {
  // 1. Calculate the memory size
  int64_t size = GetShapeSize(shape) * sizeof(float);

  // 2. Allocate device-side memory
  aclError ret = aclrtMalloc(&device_addr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtMalloc failed, error: %d\n", ret);
            return -1);

  // 3. Host -> Device data copy
  ret = aclrtMemcpy(
      device_addr, size,
      host_data.data(), size,
      ACL_MEMCPY_HOST_TO_DEVICE
  );
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtMemcpy failed, error: %d\n", ret);
            return -1);

  // 4. Calculate the strides of the Tensor (contiguous Tensor)
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; --i) {
    strides[i] = strides[i + 1] * shape[i + 1];
  }

  // 5. Create an aclTensor.
  tensor = aclCreateTensor(
      shape.data(), shape.size(),
      ACL_FLOAT, strides.data(), 0,
      ACL_FORMAT_ND, shape.data(), shape.size(),
      device_addr
  );
  CHECK_RET(tensor != nullptr,
            LOG_PRINT("aclCreateTensor failed\n");
            return -1);

  return 0;
}

int main() {
  // ========== 1. Initialize the environment ==========
  int32_t device_id = 0;  // Adjust it based on the actual Device ID.
  aclrtContext context = nullptr;
  aclrtStream stream = nullptr;

  int ret = InitAcl(device_id, context, stream);
  CHECK_RET(ret == 0,
            LOG_PRINT("InitAcl failed, error: %d\n", ret);
            return -1);

  // ========== 2. Construct input/output parameters ==========
  // The shape of input x: B=1, S=1024, n=4 → (1024,4,4) (B*S merged into T=1024)
  std::vector<int64_t> x_shape = {1024, 4, 4};
  int64_t x_size = GetShapeSize(x_shape);
  std::vector<float> x_host_data(x_size, 1.0f);  // Initialize the input data to 1.0.

  // The output has the same shape as x.
  std::vector<int64_t> output_shape = x_shape;
  void* output_device_addr = nullptr;
  aclTensor* output_tensor = nullptr;

  std::vector<int64_t> norm_out_shape = {40 * 4 * 8 * 1024};  // 2*num_iters=40, n=4, n_align=8, T=1024
  void* norm_out_device_addr = nullptr;
  aclTensor* norm_out_tensor = nullptr;

  std::vector<int64_t> sum_out_shape = {40 * 8 * 1024};  // 2*num_iters=40, n_align=8, T=1024
  void* sum_out_device_addr = nullptr;
  aclTensor* sum_out_tensor = nullptr;

  // Device tensor of input x
  void* x_device_addr = nullptr;
  aclTensor* x_tensor = nullptr;
  ret = CreateAclTensor(x_host_data, x_shape, x_device_addr, x_tensor);
  CHECK_RET(ret == 0,
            LOG_PRINT("Create x_tensor failed\n");
            return -1);

  // Device tensor of the output (memory allocated only, no initial data)
  ret = aclrtMalloc(&output_device_addr, GetShapeSize(output_shape)*sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("Malloc output failed, error: %d\n", ret);
            return -1);
  output_tensor = aclCreateTensor(
      output_shape.data(), output_shape.size(),
      ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND,
      output_shape.data(), output_shape.size(),
      output_device_addr
  );

  ret = aclrtMalloc(&norm_out_device_addr, GetShapeSize(norm_out_shape)*sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("Malloc norm_out failed, error: %d\n", ret);
            return -1);
  norm_out_tensor = aclCreateTensor(
      norm_out_shape.data(), norm_out_shape.size(),
      ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND,
      norm_out_shape.data(), norm_out_shape.size(),
      norm_out_device_addr
  );

  ret = aclrtMalloc(&sum_out_device_addr, GetShapeSize(sum_out_shape)*sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("Malloc sum_out failed, error: %d\n", ret);
            return -1);
  sum_out_tensor = aclCreateTensor(
      sum_out_shape.data(), sum_out_shape.size(),
      ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND,
      sum_out_shape.data(), sum_out_shape.size(),
      sum_out_device_addr
  );

  // MhcSinkhorn operator parameters
  float eps = 1e-6f;       // Parameter for preventing division by zero.
  int64_t num_iters = 20;  // Iteration count.

  // ========== 3. Call the first-phase API to obtain the workspace size. ==========
  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;

  aclnnStatus aclnn_ret = aclnnMhcSinkhornGetWorkspaceSize(
      x_tensor,
      eps,
      num_iters,
      output_tensor,
      norm_out_tensor,
      sum_out_tensor,
      &workspace_size,
      &executor
  );
  CHECK_RET(aclnn_ret == ACL_SUCCESS,
            LOG_PRINT("aclnnMhcSinkhornGetWorkspaceSize failed, error: %d\n", aclnn_ret);
            return -1);

  // ========== 4. Allocate workspace memory. ==========
  void* workspace_addr = nullptr;
  if (workspace_size > 0) {
    ret = aclrtMalloc(&workspace_addr, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclrtMalloc workspace failed, error: %d\n", ret);
              return -1);
  }

  // ========== 5. Call the second-phase API to perform the MhcSinkhorn computation. ==========
  aclnn_ret = aclnnMhcSinkhorn(
      workspace_addr,
      workspace_size,
      executor,
      stream
  );
  CHECK_RET(aclnn_ret == ACL_SUCCESS,
            LOG_PRINT("aclnnMhcSinkhorn failed, error: %d\n", aclnn_ret);
            return -1);

  // ========== 6. Synchronize the Stream and print the result. ==========
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtSynchronizeStream failed, error: %d\n", ret);
            return -1);

  LOG_PRINT("MhcSinkhorn compute success!\n");
  LOG_PRINT("Output tensor data: ");
  PrintTensorData(output_shape, output_device_addr);

  // ========== 7. Release resources. ==========
  // Destroy the Tensor.
  aclDestroyTensor(x_tensor);
  aclDestroyTensor(output_tensor);
  aclDestroyTensor(norm_out_tensor);
  aclDestroyTensor(sum_out_tensor);

  // Release Device memory.
  aclrtFree(x_device_addr);
  aclrtFree(output_device_addr);
  aclrtFree(norm_out_device_addr);
  aclrtFree(sum_out_device_addr);
  if (workspace_size > 0) {
    aclrtFree(workspace_addr);
  }

  // Destroy the Stream/Context, and reset the Device.
  aclrtDestroyStream(stream);
  aclrtDestroyContext(context);
  aclrtResetDevice(device_id);
  aclFinalize();

  LOG_PRINT("All resources released successfully!\n");
  return 0;
}
```
