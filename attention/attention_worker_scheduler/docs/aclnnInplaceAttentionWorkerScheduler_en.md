# aclnnInplaceAttentionWorkerScheduler

## Product Support

|Product            |  Supported |
|:-------------------------|:----------:|
|  <term>Atlas A3 training products/Atlas A3 inference products</term>  |     √    |
|  <term>Atlas A2 training products/Atlas A2 inference products</term>    |     ×    |
|  <term>Atlas 200I/500 A2 inference products</term>   |    ×     |
|  <term>Atlas inference products</term>   |     ×    |
|  <term>Atlas training products</term>   |     ×    |

## Function

- Description: Scans data in Attention when Attention and FFN are deployed separately. This operator receives the output data from the FFNToAttention operator and scans the data step by step to ensure that the data is ready.

  **It is recommended that this operator be used together with the FFNToAttention and AttentionWorkerCombine operators to form a complete workflow.**

    1. Receives data sent by the FFNToAttention operator. The data is stored in the ScheduleContext structure. For details, see [Example](#example). This structure contains CommonArea, ControlArea, AttentionArea, and FfnArea. This API involves CommonArea (used to store configuration information, such as `session_num`, `micro_batch_num`, `micro_batch_size`, and `selected_expert_num`), ControlArea (used to determine whether the upper-layer control process exits), and AttentionArea (used to manage the core data buffer and status information required during operator computation. The `token_info_buf` parameter stores the data related to the input).

    2. Reads the flag information stored in `token_info_buf` in ScheduleContext.AttentionArea to check whether the communication data is ready.

    3. Allows data to be used by the AttentionWorkerCombine operator after all data is ready.

- Formula:

  $$
  \text{Initialize:} \quad \text{ready\_count} = 0, \quad \text{flag\_num} = \text{micro\_batch\_size} \times \text{selected\_expert\_num}
  $$

  $$
  \text{Check if run\_flag is 0:}
    \quad \text{if run\_flag} = 0, \quad \text{exit and log}
  $$

  $$
  \text{Loop:} \quad \text{while run\_flag} \neq 0:
    \quad \text{ready\_count} = \sum_{i=1}^{\text{flag\_num}} \mathbf{1}_{\{ \text{flag}[i] = 1 \}}; \quad \text{if ready\_count} = \text{flag\_num}, \quad \text{break}
  $$

  $$
  \text{Reset flags:}
    \quad \text{flag}[i] = 0 \quad \text{for} \quad i = 1, 2, \dots, \text{flag\_num}
  $$

  $$
  \text{Set micro\_batch\_id:} \quad \text{micro\_batch\_id} = (\text{micro\_batch\_id} + 1) \% \text{micro\_batch\_num}
  $$

  Note: `micro_batch_size`, `selected_expert_num`, `run_flag`, and `micro_batch_id` are parameters of the ScheduleContext structure, which is described in [Example](#example).

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnInplaceAttentionWorkerSchedulerGetWorkspaceSize` is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, `aclnnInplaceAttentionWorkerScheduler` is called to perform computation.

```Cpp
aclnnStatus aclnnInplaceAttentionWorkerSchedulerGetWorkspaceSize(
    aclTensor* scheduleContextRef,
    uint64_t* workspaceSize,
    aclOpExecutor** executor)
```

```Cpp
aclnnStatus aclnnInplaceAttentionWorkerScheduler(
    void* workspace,
    uint64_t workspaceSize,
    aclOpExecutor* executor,
    aclrtStream stream)
```

## aclnnInplaceAttentionWorkerSchedulerGetWorkspaceSize

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1565px"><colgroup>
  <col style="width: 146px">
  <col style="width: 135px">
  <col style="width: 326px">
  <col style="width: 246px">
  <col style="width: 275px">
  <col style="width: 101px">
  <col style="width: 190px">
  <col style="width: 146px">
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
      <td>scheduleContextRef</td>
      <td>Input/Output</td>
      <td>To-be-processed data received by Attention, indicating the input <code>scheduleContext</code> information. For details about the structure, see the calling example.</td>
      <td>Empty tensors are not supported.</td>
      <td>INT8</td>
      <td>ND</td>
      <td>1D with the shape fixed to (1024).</td>
      <td>×</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td rowspan="1">Output</td>
      <td>Size of the workspace to be allocated on the device.</td>
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

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API implements input parameter verification. The following errors may be thrown.

  <table style="undefined;table-layout: fixed; width: 1134px"><colgroup>
  <col style="width: 319px">
  <col style="width: 144px">
  <col style="width: 671px">
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
      <td>The parameter <code>scheduleContextRef</code> is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
      <td>161002</td>
      <td>The dimension of parameter <code>scheduleContextRef</code> is not <code>1</code>.</td>
    </tr>
    <tr>
      <td>161002</td>
      <td>Parameter <code>scheduleContextRef</code> is an empty tensor.</td>
    </tr>
  </tbody>
  </table>

## aclnnInplaceAttentionWorkerScheduler

- **Parameters:**
  <table style="undefined;table-layout: fixed; width: 598px"><colgroup>
  <col style="width: 144px">
  <col style="width: 125px">
  <col style="width: 700px">
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
      <td>Size of the workspace to be allocated on the device, obtained by calling the first-phase API <code>aclnnInplaceAttentionWorkerSchedulerGetWorkspaceSize</code>.</td>
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
  </tbody>
  </table>

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

  - By default, `aclnnInplaceAttentionWorkerScheduler` defaults to deterministic implementation . The non-deterministic implementation is not supported, and the deterministic computation configuration does not take effect.

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include "acl/acl.h"
#include "aclnnop/aclnn_attention_worker_scheduler.h"
#define CHECK_RET(cond, return_expr) \
  do {                               \
    if (!(cond)) {                   \
      return_expr;                   \
    }                                \
  } while (0)

#define CHECK_FREE_RET(cond, return_expr)  \
  do {                                     \
      if (!(cond)) {                       \
          Finalize(deviceId, stream);      \
          return_expr;                     \
      }                                    \
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
  // (Boilerplate) Perform initialization.
  auto ret = aclInit(nullptr);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
  ret = aclrtSetDevice(deviceId);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
  ret = aclrtCreateStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
  return 0;
}

int CreateAclTensor(const void *hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor) {
  auto size = GetShapeSize(shape);
  // Call aclrtMalloc to allocate memory on the device.
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  // Call aclrtMemcpy to copy the data on the host to the memory on the device.
  ret = aclrtMemcpy(*deviceAddr, size, hostData, size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // Compute the stride of the contiguous tensor.
  std::vector<int64_t> stride(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    stride[i] = shape[i + 1] * stride[i + 1];
  }

  // Call aclCreateTensor to create an aclTensor.
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, stride.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

void Finalize(int32_t deviceId, aclrtStream stream)
{
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
}

#pragma pack(push, 1)
struct AttentionDataDesc {
  int32_t flag[0];
};

struct ScheduleContext {
  struct CommonArea {
    uint32_t session_num;  // Number of attention nodes
    uint32_t micro_batch_num;
    uint32_t micro_batch_size;
    uint32_t selected_expert_num;
    uint32_t expert_num; // Number of experts per layer, including routing experts and shared experts.
    uint32_t attn_to_ffn_token_size;  // Each token in the Ffn window data area has a space size aligned to 512 bytes.
    uint32_t ffn_to_attn_token_size;  // Each token in the Attention window data area has a space size aligned to 512 bytes.
    int32_t schedule_mode;  // 0: Ffn only 1: Attention only
    int8_t reserve0[96];
  };
  struct ControlArea {
    int32_t run_flag;  // 0 : exited  1 : running
    int8_t reserve2[124];
  };
  struct FfnArea {
    // ffn area
    uint64_t token_info_buf;  // Points to device memory.
    uint64_t token_info_buf_size;
    uint64_t token_data_buf;  // Points to device memory.
    uint64_t token_data_buf_size;
    uint64_t polling_index;  // For synchronous computation only:records the micro-batch ids to be processed internally by the ffn worker scheduler.
    int8_t reserve3[88];

    // ffn out area
    uint64_t layer_ids_buf;  // Points to a device memory region that stores the organized layer ids, with an array size of session_num.
    uint64_t layer_ids_buf_size;  // Total size: session_num * sizeof(int32_t)
    uint64_t session_ids_buf;  //  Points to a device memory region that stores the organized session ids. The array size is session_num.
    uint64_t session_ids_buf_size;  // Total size: session_num * sizeof(int32_t)
    uint64_t micro_batch_ids_buf;  // Points to a device memory region that stores the organized micro batch ids. The array size is session_num.
    uint64_t micro_batch_ids_buf_size;  // Total size: session_num * sizeof(int32_t)
    uint64_t expert_ids_buf;  //  Points to a device memory region that stores the organized expert ids. The tensor dimensions are [session_num, batch_size, selected_expert_num].
    uint64_t expert_ids_buf_size;  // Total memory size: session_num * batch_size * selected_expert_num * sizeof(int32_t)
    uint32_t out_num;  // Indicates the number of sessions that have been processed.
    int8_t reserve4[60];
  };

  struct AttentionArea {
    // attention area
    uint64_t token_info_buf;  // Points to device memory.
    uint64_t token_info_buf_size;
    uint64_t token_data_buf;  // Points to device memory.
    uint64_t token_data_buf_size;
    uint32_t micro_batch_id; // Records the latest ready micro batch id.
    int8_t reserve5[92];
  };

  // common area
  CommonArea common;
  ControlArea control;
  AttentionArea attention;
  FfnArea ffn;
  // reserve area
  int8_t reserve6[384];  // Padding to 1024 bytes.
};

static_assert(sizeof(ScheduleContext) == 1024, "ScheduleContext size must be 1024 bytes");

#pragma pack(pop)

int aclnnAttentionWorkerSchedulerTest(int32_t deviceId, aclrtStream &stream) {
  auto ret = Init(deviceId, &stream);
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct inputs and outputs based on the API definition.
  ScheduleContext hostScheduleContext = {};

  // Initialize  CommonArea
  hostScheduleContext.common.session_num = 1;
  hostScheduleContext.common.micro_batch_num = 2;
  hostScheduleContext.common.micro_batch_size = 48;
  hostScheduleContext.common.selected_expert_num = 9;
  hostScheduleContext.common.expert_num = 16;
  hostScheduleContext.common.attn_to_ffn_token_size = 512;
  hostScheduleContext.common.ffn_to_attn_token_size = 512;
  hostScheduleContext.common.schedule_mode = 1;  // attention

  // Initialize ControlArea
  hostScheduleContext.control.run_flag = 1;  // running

  // Initialize AttentionArea
  hostScheduleContext.attention.micro_batch_id = 1;
  size_t per_data_desc_size = sizeof(AttentionDataDesc) + sizeof(int32_t) * hostScheduleContext.common.micro_batch_size * hostScheduleContext.common.selected_expert_num;
  size_t expect_token_info_buf_size = static_cast<size_t>(hostScheduleContext.common.micro_batch_num) * per_data_desc_size;
  void* tokenBufDeviceAddr = nullptr;
  ret = aclrtMalloc(&tokenBufDeviceAddr, expect_token_info_buf_size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  hostScheduleContext.attention.token_info_buf = reinterpret_cast<uint64_t>(tokenBufDeviceAddr);
  hostScheduleContext.attention.token_info_buf_size = expect_token_info_buf_size;
  int target_micro_batch_id = 0;
  auto data_desc_ptr = reinterpret_cast<AttentionDataDesc *>(
      reinterpret_cast<uint8_t *>(hostScheduleContext.attention.token_info_buf) + per_data_desc_size * target_micro_batch_id);
  size_t flag_num =
      static_cast<size_t>(hostScheduleContext.common.micro_batch_size) * hostScheduleContext.common.selected_expert_num;

  // Set all flags to 1.
  std::vector<int32_t> host_flags(flag_num, 1);
  ret = aclrtMemcpy(data_desc_ptr->flag, flag_num * sizeof(int32_t),
                    host_flags.data(), host_flags.size() * sizeof(int32_t),
                    ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy flags from host to device failed. ERROR: %d\n", ret); return ret);

  uint64_t token_data_buf_size = 100;
  void* tokenDataDeviceAddr = nullptr;
  ret = aclrtMalloc(&tokenDataDeviceAddr, token_data_buf_size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  hostScheduleContext.attention.token_data_buf = reinterpret_cast<uint64_t>(tokenDataDeviceAddr);

  // Create a scheduleContext aclTensor.
  std::vector<int64_t> scheduleContextShape = {1024};
  void* scheduleContextDeviceAddr = nullptr;
  aclTensor* scheduleContextRef = nullptr;
  ret = CreateAclTensor(&hostScheduleContext, scheduleContextShape, &scheduleContextDeviceAddr, aclDataType::ACL_INT8, &scheduleContextRef);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnInplaceAttentionWorkerScheduler.
  ret = aclnnInplaceAttentionWorkerSchedulerGetWorkspaceSize(scheduleContextRef, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceAttentionWorkerSchedulerGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
  }

  // Call the second-phase API of aclnnInplaceAttentionWorkerScheduler.
  ret = aclnnInplaceAttentionWorkerScheduler(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceAttentionWorkerScheduler failed. ERROR: %d\n", ret); return ret);
  // 4. (Boilerplate) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
  // 5. Obtain the output value and copy the result from the device memory to the host. Modify the code based on the API definition.
  auto size = GetShapeSize(scheduleContextShape);
  std::vector<int8_t> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), scheduleContextDeviceAddr, size * sizeof(int8_t), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  // Print the output result.
  ScheduleContext *out_schedule_context = reinterpret_cast<ScheduleContext *>(resultData.data());
  LOG_PRINT("micro_batch_id = %u.\n", out_schedule_context->attention.micro_batch_id);

  // 6. Release aclTensor. Modify the code based on the API definition.
  aclDestroyTensor(scheduleContextRef);

  // 7. Release device resources. Modify the code based on the API definition.
  aclrtFree(scheduleContextDeviceAddr);
  aclrtFree(tokenBufDeviceAddr);
  aclrtFree(tokenDataDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }

  return ACL_SUCCESS;
}

int main() {
  // 1. (Boilerplate) Initialize the device and stream. For details, see the external API list.
  // Set the device ID in use.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = aclnnAttentionWorkerSchedulerTest(deviceId, stream);
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAttentionWorkerSchedulerTest failed. ERROR: %d\n", ret); return ret);

  Finalize(deviceId, stream);
  return 0;
}
```
