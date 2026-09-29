# aclnnInplaceFfnWorkerScheduler

## Product Support

|Product            |  Supported |
|:-------------------------|:----------:|
|Ascend 950PR/Ascend 950DT|      √     |
|  <term>Atlas A3 training products/Atlas A3 inference products</term>  |     √    |
|  <term>Atlas A2 training products/Atlas A2 inference products</term>    |     ×    |
|  <term>Atlas 200I/500 A2 inference products</term>   |    ×     |
|  <term>Atlas inference products</term>   |     ×    |
|  <term>Atlas training products</term>   |     ×    |

## Function

- Description: Scans data in FFN when Attention and FFN are separately deployed. This operator receives data from the AttentionToFFN operator for data scanning and sorting.

  **It is recommended that this operator be used with AttentionToFFN and FFNWorkerBatching.**

    1. Receives data from the AttentionToFFN operator. The data is stored in the memory layout of the ScheduleContext structure. For details, see [Example](#example). This structure contains CommonArea, ControlArea, AttentionArea, and FfnArea. This API involves the CommonArea (used to store configuration information, such as session_num, micro_batch_num, micro_batch_size, and selected_expert_num), ControlArea (used to control whether the upper-layer process exits), and FfnArea (used to manage the input and output buffers required for the computation of this operator. The token_info_buf field is used to store the input information of this operator).

    2. Scans the information stored in `token_info_buf`. When the communication data is ready, this operator starts data sorting. As shown in the following figure, the information of `layer id`, `session id`, `micro batch id`, and `expert ids` are written to the `layer_id_buf`, `session_id_buf`, `micro_batch_id_buf`, and `expert_ids_buf`, respectively, in the device memory.

      ```mermaid
      graph TB
          %% Input buffer
          A [token_info_buf input]

          %% Session-level structure
          A --> Session0
          A --> Session1

          %% Internal structure of session 0
          subgraph Session0[session 0]
              direction TB
              S0_M1[micro batch id 0]:::micro
              S0_L1[layer id 0]:::layer
              S0_S1[session id 0]:::session0
              S0_E1[expert ids 0]:::expert
          end

          %% Internal structure of session 1
          subgraph Session1[session 1]
              direction TB
              S1_M1[micro batch id 0]:::micro
              S1_L1[layer id 0]:::layer
              S1_S1[session id 1]:::session1
              S1_E1[expert ids 0]:::expert
          end

          %% Output buffer index area
          subgraph Output [Output area]
              direction TB
              O1[layer_ids_buf]:::layer
              O2[session_ids_buf]:::output
              O3[micro_batch_ids_buf]:::micro
              O4[expert_ids_buf]:::expert
          end

          %% Data flow direction
          S0_L1 -.-> O1
          S0_S1 -.-> O2
          S0_M1 -.-> O3
          S0_E1 -.-> O4

          S1_L1 -.-> O1
          S1_S1 -.-> O2
          S1_M1 -.-> O3
          S1_E1 -.-> O4

          classDef layer fill:#c8e6c9
          classDef session0 fill:#ffcdd2
          classDef session1 fill:#ffccbc
          classDef output fill:#e3f2fd
          classDef micro fill:#e1f5fe
          classDef expert fill:#bbdefd
          
          %% Adding the background color style of the subgraph
          style Session0 fill:#fff3e0,stroke:#ff9800,stroke-width:2px
          style Session1 fill:#fce4ec,stroke:#e91e63,stroke-width:2px
          style Output fill:#e8f5e8,stroke:#4caf50,stroke-width:2px
      ```

    3. Allows data to be used by the FFNWorkerBatching operator after the data is sorted.

- Formula:
    1. Perform initialization. The number of groups is computed based on the values of `session_num` and `sync_group_size` of the input parameter `ScheduleContext`.
    2. If the number of groups is 1, all data is processed synchronously. After all session data is ready, the data is sorted.
    3. If the number of groups is not 1, data is not processed synchronously. After the session data in the groups is ready, the data is sorted.

$$
\text{Initialize:} \quad\text{group\_num} = \frac{\text{session\_num}}{\text{sync\_group\_size}}
$$

$$
\text{Process} = 
\begin{cases}
\text{check\_all\_session\_ready()} \quad \text{data\_reorganization()} & \text{if } \text{group\_num} = 1 \\
\text{check\_all\_sessions\_of\_group\_ready()} \quad \text{data\_reorganization()} & \text{otherwise}
\end{cases} 
$$ 
  
## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnInplaceFfnWorkerSchedulerGetWorkspaceSize` is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, `aclnnInplaceFfnWorkerScheduler` is called to perform computation.

```Cpp
aclnnStatus aclnnInplaceFfnWorkerSchedulerGetWorkspaceSize(
    aclTensor*      scheduleContextRef,
    int32_t         syncGroupSize,
    int32_t         executeMode,
    uint64_t*       workspaceSize,
    aclOpExecutor** executor)
```

```Cpp
aclnnStatus aclnnInplaceFfnWorkerScheduler(
    void*          workspace,
    uint64_t       workspaceSize,
    aclOpExecutor* executor,
    aclrtStream    stream)
```

## aclnnInplaceFfnWorkerSchedulerGetWorkspaceSize

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1540px"><colgroup>
  <col style="width: 210px">
  <col style="width: 135px">
  <col style="width: 320px">
  <col style="width: 320px">
  <col style="width: 130px">
  <col style="width: 120px">
  <col style="width: 160px">
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
      <td>scheduleContextRef (aclTensor*)</td>
      <td>Input/Output</td>
      <td>To-be-processed data received by FFN, indicating the <code>ScheduleContext</code> information. For details about the structure, see the calling example.</td>
      <td>Empty tensors are not supported.</td>
      <td>INT8</td>
      <td>ND</td>
      <td>1D, (1024)</td>
      <td>×</td>
    </tr>
    <tr>
      <td>syncGroupSize (int32_t)</td>
      <td>Input</td>
      <td>Number of sessions processed by each synchronization group.</td>
      <td>The value range is (0, session_num], where <code>session_num</code> indicates the maximum number of sessions to be processed, that is, the value of the <code>session_num</code> field in the CommonArea of the `ScheduleContext` structure in the calling example.</td>
      <td>INT32</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executeMode (int32_t)</td>
      <td>Input</td>
      <td>Execution mode.</td>
      <td>Only mode <code>0</code> is supported, indicating that the system exits after an execution.</td>
      <td>INT32</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>workspaceSize (uint64_t*)</td>
      <td>Output</td>
      <td>Size of the workspace to be allocated on the device.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor (aclOpExecutor**)</td>
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
  <col style="width: 291px">
  <col style="width: 135px">
  <col style="width: 723px">
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
      <td><code>scheduleContextRef</code> is a null pointer.</td>
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
    <tr>
      <td>161002</td>
      <td>The value of <code>executeMode</code> is not <code>0</code>.</td>
    </tr>
  </tbody>
  </table>

## aclnnInplaceFfnWorkerScheduler

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1151px"><colgroup>
  <col style="width: 184px">
  <col style="width: 134px">
  <col style="width: 833px">
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
      <td>Size of the workspace to be allocated on the device, obtained by calling the first-phase API <code>aclnnInplaceFfnWorkerSchedulerGetWorkspaceSize</code>.</td>
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

  - By default, `aclnnInplaceFfnWorkerScheduler` defaults to deterministic implementation . The non-deterministic implementation is not supported, and the deterministic computation configuration does not take effect.

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <memory>
#include <vector>
#include <limits>
#include "acl/acl.h"
#include "aclnnop/aclnn_ffn_worker_scheduler.h"

#define CHECK_RET(cond, return_expr) \
  do {                               \
    if (!(cond)) {                   \
      return_expr;                   \
    }                                \
  } while (0)

#define CHECK_FREE_RET(cond, return_expr) \
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

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor) {
  auto size = GetShapeSize(shape) * sizeof(T);
  // Call aclrtMalloc to allocate memory on the device.
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  // Call aclrtMemcpy to copy the data on the host to the memory on the device.
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
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

constexpr uint32_t kSuccess = 0;
constexpr uint32_t kFailure = 1;
constexpr uint64_t kBufAlignSize = 512;

inline uint64_t AlignUp(uint64_t num, uint64_t align)
{
  return ((num + align - 1) / align) * align;
}

#pragma pack(push, 1)
struct FfnDataDesc {
  volatile int32_t flag;
  volatile int32_t layer_id;
  volatile int32_t expert_ids[0];
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
    uint64_t polling_index;  // For synchronous computation only: records the micro-batch ids to be processed internally by the ffn worker scheduler.
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

uint64_t CalcFfnTokenInfoSize(ScheduleContext& schedule_context)
{
  // token_info_size = (sizeof(FfnDataDesc) + selected_expert_num * micro_batch_size) * micro_batch_num * session_num
  uint64_t flag_and_layer_id_size = sizeof(FfnDataDesc);
  uint64_t token_info_size =(sizeof(int32_t) * static_cast<uint64_t>(schedule_context.common.selected_expert_num) * schedule_context.common.micro_batch_size + flag_and_layer_id_size) * static_cast<uint64_t>(schedule_context.common.micro_batch_num) * static_cast<uint64_t>(schedule_context.common.session_num);
  return token_info_size;
}

uint32_t InitFfnTokenInfoBuf(ScheduleContext& schedule_context)
{
  std::unique_ptr<uint8_t[]> tmp_buf(new (std::nothrow) uint8_t[schedule_context.ffn.token_info_buf_size]);
  if (tmp_buf == nullptr) {
    LOG_PRINT("alloc token info host tmp buf failed, buf_size=   %lu\n", schedule_context.ffn.token_info_buf_size);
    return kFailure;
  }
  auto buf_int = reinterpret_cast<int32_t*>(tmp_buf.get());
  auto token_buf_int = buf_int;
  for (uint32_t session_id = 0; session_id < schedule_context.common.session_num; ++session_id) {
    for (uint32_t micro_batch_id = 0; micro_batch_id < schedule_context.common.micro_batch_num; ++micro_batch_id) {
      // flag
      *buf_int++ = 1;
      // layer_id
      *buf_int++ = 55;
      for (uint32_t idx = 0;
            idx < schedule_context.common.micro_batch_size * schedule_context.common.selected_expert_num; ++idx) {
        // expert_id
        *buf_int++ = idx;
      }
    }
  }
  void *token_info_buf = nullptr;
  size_t token_info_buf_size = sizeof(uint8_t) * schedule_context.ffn.token_info_buf_size;
  if (token_info_buf_size > 0) {
    auto ret = aclrtMalloc(&token_info_buf, token_info_buf_size,  ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate token info buf failed. ERROR: %d\n", ret); return ret;);
  }
  schedule_context.ffn.token_info_buf = reinterpret_cast<uint64_t>(token_info_buf);
  auto token_info_buf_tmp = reinterpret_cast<void*>(static_cast<uintptr_t>(schedule_context.ffn.token_info_buf));
  auto ret = aclrtMemcpy(token_info_buf_tmp, token_info_buf_size, token_buf_int, token_info_buf_size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("memory copy token info buf failed, size_=  %lu\n", token_info_buf_size); return ret;);
  return ACL_SUCCESS;
}

uint32_t InitFfn(ScheduleContext& schedule_context)
{
  uint64_t token_info_size = CalcFfnTokenInfoSize(schedule_context);
  if (token_info_size == 0U) {
    return ACL_ERROR_INVALID_PARAM;
  }

  uint64_t token_info_aligned_size = AlignUp(token_info_size, kBufAlignSize);
  if (token_info_aligned_size < token_info_size) {
     LOG_PRINT("token_info_size[%lu]  overflow after align with %lu\n", token_info_size, kBufAlignSize);
    return ACL_ERROR_INVALID_PARAM;
  }

  schedule_context.ffn.token_info_buf_size = token_info_size;

  // Allocate a buffer of 1024 bytes as the token data buffer.
  schedule_context.ffn.token_data_buf_size = 1024;
  void *token_data_buf = nullptr;
  size_t token_data_buf_size = sizeof(uint8_t) * schedule_context.ffn.token_data_buf_size;
  auto ret = aclrtMalloc(&token_data_buf, token_data_buf_size,  ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate token data buf failed. ERROR: %d\n", ret); return ret;);
  schedule_context.ffn.token_data_buf = reinterpret_cast<uint64_t>(token_data_buf);

  ret = InitFfnTokenInfoBuf(schedule_context);
  if (ret != ACL_SUCCESS) {
    return ret;
  }

  // calc output size
  schedule_context.ffn.layer_ids_buf_size = sizeof(int32_t) * schedule_context.common.session_num;
  schedule_context.ffn.session_ids_buf_size = sizeof(int32_t) * schedule_context.common.session_num;
  schedule_context.ffn.micro_batch_ids_buf_size = sizeof(int32_t) * schedule_context.common.session_num;
  schedule_context.ffn.expert_ids_buf_size = sizeof(int32_t) * schedule_context.common.session_num *
                                             schedule_context.common.micro_batch_size *
                                             schedule_context.common.selected_expert_num;

  void *layer_ids_buf = nullptr;
  size_t layer_ids_buf_size = sizeof(uint8_t) * schedule_context.ffn.layer_ids_buf_size;
  ret = aclrtMalloc(&layer_ids_buf, layer_ids_buf_size,  ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate layer ids buf failed. ERROR: %d\n", ret); return ret;);
  schedule_context.ffn.layer_ids_buf = reinterpret_cast<uint64_t>(layer_ids_buf);

  void *session_ids_buf = nullptr;
  size_t session_ids_buf_size = sizeof(uint8_t) * schedule_context.ffn.session_ids_buf_size;
  ret = aclrtMalloc(&session_ids_buf, session_ids_buf_size,  ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate session ids buf failed. ERROR: %d\n", ret); return ret;);
  schedule_context.ffn.session_ids_buf = reinterpret_cast<uint64_t>(session_ids_buf);

  void *micro_batch_ids_buf = nullptr;
  size_t micro_batch_ids_buf_size = sizeof(uint8_t) * schedule_context.ffn.micro_batch_ids_buf_size;
  ret = aclrtMalloc(&micro_batch_ids_buf, micro_batch_ids_buf_size,  ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate micro batch ids buf failed. ERROR: %d\n", ret); return ret;);
  schedule_context.ffn.micro_batch_ids_buf = reinterpret_cast<uint64_t>(micro_batch_ids_buf);

  void *expert_ids_buf = nullptr;
  size_t expert_ids_buf_size = sizeof(uint8_t) * schedule_context.ffn.expert_ids_buf_size;
  ret = aclrtMalloc(&expert_ids_buf, expert_ids_buf_size,  ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate expert ids buf failed. ERROR: %d\n", ret); return ret;);
  schedule_context.ffn.expert_ids_buf = reinterpret_cast<uint64_t>(expert_ids_buf);

  LOG_PRINT("Init ffn success, token_info_buf_size=%lu,token_data_buf_size= %lu.\n", schedule_context.ffn.token_info_buf_size, schedule_context.ffn.token_data_buf_size);
  return ACL_SUCCESS;
}

uint32_t UninitFfn(ScheduleContext& schedule_context)
{
  if (schedule_context.ffn.token_info_buf != 0) {
    aclrtFree(reinterpret_cast<void*>(static_cast<uintptr_t>(schedule_context.ffn.token_info_buf)));
    schedule_context.ffn.token_info_buf = 0;
  }
  if (schedule_context.ffn.token_data_buf != 0) {
    aclrtFree(reinterpret_cast<void*>(static_cast<uintptr_t>(schedule_context.ffn.token_data_buf)));
    schedule_context.ffn.token_data_buf = 0;
  }
  if (schedule_context.ffn.layer_ids_buf != 0) {
    aclrtFree(reinterpret_cast<void*>(static_cast<uintptr_t>(schedule_context.ffn.layer_ids_buf)));
    schedule_context.ffn.layer_ids_buf = 0;
  }
  if (schedule_context.ffn.session_ids_buf != 0) {
    aclrtFree(reinterpret_cast<void*>(static_cast<uintptr_t>(schedule_context.ffn.session_ids_buf)));
    schedule_context.ffn.session_ids_buf = 0;
  }
  if (schedule_context.ffn.micro_batch_ids_buf != 0) {
    aclrtFree(reinterpret_cast<void*>(static_cast<uintptr_t>(schedule_context.ffn.micro_batch_ids_buf)));
    schedule_context.ffn.micro_batch_ids_buf = 0;
  }
  if (schedule_context.ffn.expert_ids_buf != 0) {
    aclrtFree(reinterpret_cast<void*>(static_cast<uintptr_t>(schedule_context.ffn.expert_ids_buf)));
    schedule_context.ffn.expert_ids_buf = 0;
  }
  return ACL_SUCCESS;
}

int main() {
  // 1. (Boilerplate) Initialize the device and stream. For details, see the ACL API manual.
  // Set the device ID in use.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  // Handle the check as required.
  CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
  // 2. Construct inputs and outputs based on the API definition.
  std::vector<int64_t> selfShape = {1024};
  void* selfDeviceAddr = nullptr;
  aclTensor* scheduleContextRef = nullptr;
  std::vector<int8_t> selfHostData(1024);
  ScheduleContext schedule_context = {};
  schedule_context.common.session_num = 2;
  schedule_context.common.micro_batch_num = 2;
  schedule_context.common.micro_batch_size = 2;
  schedule_context.common.selected_expert_num = 5;
  schedule_context.control.run_flag = 1;
  schedule_context.common.schedule_mode = 0;
  schedule_context.ffn.polling_index = 1;
  InitFfn(schedule_context); 
  ret = aclrtMemcpy(selfHostData.data(), sizeof(ScheduleContext), &schedule_context, sizeof(ScheduleContext), ACL_MEMCPY_DEVICE_TO_HOST);
  if (ret != ACL_SUCCESS) {
    LOG_PRINT("copy schedule context to host failed. ERROR: %d\n", ret);
    UninitFfn(schedule_context);
    return ret;
  }

  int32_t syncGroupSize = 1;
  int32_t executeMode = 0;
  // Create a scheduleContext aclTensor.
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_INT8, &scheduleContextRef);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnInplaceFfnWorkerScheduler.
  ret = aclnnInplaceFfnWorkerSchedulerGetWorkspaceSize(scheduleContextRef, syncGroupSize, executeMode, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceFfnWorkerSchedulerGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
  }
  // Call the second-phase API of aclnnInplaceFfnWorkerScheduler.
  ret = aclnnInplaceFfnWorkerScheduler(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceFfnWorkerScheduler failed. ERROR: %d\n", ret); return ret);
  // 4. (Boilerplate) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
  // 5. Obtain the output value and copy the result from the device memory to the host. Modify the code based on the API definition.
  auto size = GetShapeSize(selfShape);
  std::vector<int8_t> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(int8_t), selfDeviceAddr, size * sizeof(int8_t),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  
  // Print the output result.
  ScheduleContext *out_schedule_context = reinterpret_cast<ScheduleContext *>(resultData.data());
  LOG_PRINT("layer_ids_buf_size = %lu.\n", out_schedule_context->ffn.layer_ids_buf_size);
  LOG_PRINT("session_ids_buf_size = %lu.\n", out_schedule_context->ffn.session_ids_buf_size);
  LOG_PRINT("micro_batch_ids_buf_size = %lu.\n", out_schedule_context->ffn.micro_batch_ids_buf_size);
  LOG_PRINT("expert_ids_buf_size = %lu.\n", out_schedule_context->ffn.expert_ids_buf_size);

  //Print layer_ids information.
  std::vector<int32_t> layer_ids_buf(out_schedule_context->ffn.layer_ids_buf_size / sizeof(int32_t), 0);
  ret = aclrtMemcpy(layer_ids_buf.data(), out_schedule_context->ffn.layer_ids_buf_size, reinterpret_cast<void *>(
    static_cast<uintptr_t>(out_schedule_context->ffn.layer_ids_buf)), out_schedule_context->ffn.layer_ids_buf_size,
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy layer ids buf from device to host failed. ERROR: %d\n", ret); return ret);
  for (int i = 0; i < out_schedule_context->ffn.layer_ids_buf_size / sizeof(int32_t); i++) {
    LOG_PRINT("layer_ids[%d] is: %d\n", i, layer_ids_buf[i]);
  }

  //Print session_ids information.
  std::vector<int32_t> session_ids_buf(out_schedule_context->ffn.session_ids_buf_size / sizeof(int32_t), 0);
  ret = aclrtMemcpy(session_ids_buf.data(), out_schedule_context->ffn.session_ids_buf_size, reinterpret_cast<void *>(
    static_cast<uintptr_t>(out_schedule_context->ffn.session_ids_buf)), out_schedule_context->ffn.session_ids_buf_size,
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy session ids buf from device to host failed. ERROR: %d\n", ret); return ret);
  for (int i = 0; i < out_schedule_context->ffn.session_ids_buf_size / sizeof(int32_t); i++) {
    LOG_PRINT("session_ids[%d] is: %d\n", i, session_ids_buf[i]);
  }

  //Print micro_batch_ids information.
  std::vector<int32_t> micro_batch_ids_buf(out_schedule_context->ffn.micro_batch_ids_buf_size / sizeof(int32_t), 0);
  ret = aclrtMemcpy(micro_batch_ids_buf.data(), out_schedule_context->ffn.micro_batch_ids_buf_size, reinterpret_cast<void *>(
    static_cast<uintptr_t>(out_schedule_context->ffn.micro_batch_ids_buf)), out_schedule_context->ffn.micro_batch_ids_buf_size,
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy micro batch ids buf from device to host failed. ERROR: %d\n", ret); return ret);
  for (int i = 0; i < out_schedule_context->ffn.micro_batch_ids_buf_size / sizeof(int32_t); i++) {
    LOG_PRINT("micro_batch_ids[%d] is: %d\n", i, micro_batch_ids_buf[i]);
  }

  //Print expert_ids information.
  std::vector<int32_t> expert_ids_buf(out_schedule_context->ffn.expert_ids_buf_size / sizeof(int32_t), 0);
  ret = aclrtMemcpy(expert_ids_buf.data(), out_schedule_context->ffn.expert_ids_buf_size, reinterpret_cast<void *>(
    static_cast<uintptr_t>(out_schedule_context->ffn.expert_ids_buf)), out_schedule_context->ffn.expert_ids_buf_size,
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy expert ids buf from device to host failed. ERROR: %d\n", ret); return ret);
  for (int i = 0; i < out_schedule_context->ffn.expert_ids_buf_size / sizeof(int32_t); i++) {
    LOG_PRINT("expert_ids[%d] is: %d\n", i, expert_ids_buf[i]);
  }

  // 6. Release aclTensor. Modify the code based on the API definition.
  aclDestroyTensor(scheduleContextRef);

  // 7. Release device resources. Modify the code based on the API definition.
  UninitFfn(schedule_context);
  aclrtFree(selfDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
