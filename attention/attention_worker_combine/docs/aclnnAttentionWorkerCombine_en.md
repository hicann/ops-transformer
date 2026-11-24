# aclnnAttentionWorkerCombine

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-21T08:01:05.445Z -->

## Applicable Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Supported
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

- Description: Data fusion operator on the Attention side for use cases where Attention and FFN are deployed separately. The operator fuses attention tokens processed by multiple compute units, weights the results based on expert weights, outputs the final fused attention result, and updates the layer ID.

  **This operator is not recommended for standalone use. Use it with operators such as AttentionWorkerScheduler to form a complete workflow.**

    1. Receives the data returned from the FFN side. The data is stored in the memory layout of the ScheduleContext structure. For details, see [Example](#example). The structure contains the CommonArea, ControlArea, AttentionArea, and FfnArea fields. This API reads token data from token_data_buf in AttentionArea.

    2. Performs weighted fusion on token data with expertScales and outputs the final attention result y.

    3. Computes and outputs nextLayerId based on the input layerId, indicating the ID of the next layer to process.

- Formula:

  Let $i$ index tokens, $h$ index the hidden dimension, and $K$ be the number of routed experts per token. The last entry along the expert axis of `token_data` contains the shared expert result.

  - In the non-quantized case (`tokenDtype` is 0 or 1):

    $$
    y[i,h] = \sum_{k=0}^{K-1} \text{expertScales}[i,k] \times \text{token\_data}[i,k,h] + \text{token\_data}[i,K,h]
    $$

  - In the MXFP dequantization case (`tokenDtype` is 2, 3, or 4), `dequant(x, s)` decodes the token value according to its data type and multiplies it by the decoded E8M0 scale.

    $$
    y[i,h] = \sum_{k=0}^{K} \operatorname{dequant}\!\left(\text{token\_data}[i,k,h],\ \text{expertScales}[i,k,\lfloor h/32 \rfloor]\right)
    $$

    Each scale covers a group of up to 32 elements along the hidden dimension. Any extra scale slot added for alignment is padding.

  $$
  \text{nextLayerId} = \text{layerId} + 1
  $$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). First call the "aclnnAttentionWorkerCombineGetWorkspaceSize" API to obtain the workspace size required for the computation and the executor containing the operator computation process. Then call the "aclnnAttentionWorkerCombine" API to execute the computation.

```Cpp
aclnnStatus aclnnAttentionWorkerCombineGetWorkspaceSize(
    const aclTensor *scheduleContext,
    const aclTensor *expertScales,
    const aclTensor *layerId,
    int64_t          hiddenSize,
    int64_t          tokenDtype,
    int64_t          needSchedule,
    const aclTensor *y,
    const aclTensor *nextLayerId,
    uint64_t        *workspaceSize,
    aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnAttentionWorkerCombine(
    void*          workspace,
    uint64_t       workspaceSize,
    aclOpExecutor* executor,
    const aclrtStream stream)
```

## aclnnAttentionWorkerCombineGetWorkspaceSize

- **Parameters**

  `R` is the number of tokens, `K` is the number of routed experts per token (excluding the shared expert), and `H` is the hidden dimension size specified by `hiddenSize`.

  <table style="undefined;table-layout: fixed; width: 1540px"><colgroup>
  <col style="width: 210px">
  <col style="width: 135px">
  <col style="width: 360px">
  <col style="width: 280px">
  <col style="width: 130px">
  <col style="width: 110px">
  <col style="width: 170px">
  <col style="width: 145px">
  </colgroup>
  <thead>
    <tr>
      <th>Parameter</th>
      <th>Input/Output</th>
      <th>Description</th>
      <th>Instruction</th>
      <th>Data Type</th>
      <th>Data Format</th>
      <th>Dimension (shape)</th>
      <th>Non-contiguous Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>scheduleContext</td>
      <td>Input</td>
      <td>Scheduling context received on the Attention side. It contains CommonArea, ControlArea, AttentionArea, and FfnArea. The operator reads token data from token_data_buf in AttentionArea.</td>
      <td>Empty tensors are not supported.</td>
      <td>INT8</td>
      <td>ND</td>
      <td>1-D, with a fixed shape of (1024)</td>
      <td>×</td>
    </tr>
    <tr>
      <td>expertScales</td>
      <td>Input</td>
      <td>Expert weights per token in the non-quantized case. In the MXFP dequantization case, each E8M0 scale covers a group of up to 32 elements along the hidden dimension.</td>
      <td>Empty tensors are not supported. The type is FLOAT when tokenDtype is 0 or 1 and FLOAT8_E8M0 when it is 2, 3, or 4.</td>
      <td>FLOAT, FLOAT8_E8M0</td>
      <td>ND</td>
      <td>2D, (R, K) in the non-quantized case. 3D, (R, K + 1, S) in the MXFP dequantization case. The last entry along the expert axis corresponds to the shared expert, and S is ceil(H / 32) rounded up to an even number.</td>
      <td>×</td>
    </tr>
    <tr>
      <td>layerId</td>
      <td>Input</td>
      <td>Current model layer ID.</td>
      <td>Empty tensors are not supported.</td>
      <td>INT32</td>
      <td>ND</td>
      <td>1-D, (1)</td>
      <td>×</td>
    </tr>
    <tr>
      <td>hiddenSize</td>
      <td>Input</td>
      <td>Size of the hidden dimension of token_data, used to determine the size of the second dimension of output y.</td>
      <td>-</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>tokenDtype</td>
      <td>Input</td>
      <td>Specifies the original data type of the tokens in scheduleContext. The values 0, 1, 2, 3, and 4 denote FLOAT16, BFLOAT16, MXFP8_E5M2, MXFP8_E4M3FN, and MXFP4_E2M1, respectively.</td>
      <td>Use 0 or 1 in the non-quantized case and 2, 3, or 4 in the MXFP dequantization case.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>needSchedule</td>
      <td>Input</td>
      <td>Specifies whether to wait until token data is filled before execution. 0 indicates no wait; 1 indicates wait.</td>
      <td>0 or 1</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Final attention merge result.</td>
      <td>Empty tensors are not supported. The type is FLOAT16 when tokenDtype is 0 and BFLOAT16 for all other supported values.</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>2D, (R, H)</td>
      <td>×</td>
    </tr>
    <tr>
      <td>nextLayerId</td>
      <td>Output</td>
      <td>ID of the next layer to process.</td>
      <td>Empty tensors are not supported.</td>
      <td>INT32</td>
      <td>ND</td>
      <td>1D, (1)</td>
      <td>×</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Output</td>
      <td>Returns the workspace size to be allocated on the Device side.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Output</td>
      <td>Returns an op executor that contains the operator computation flow.</td>
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

    <table style="undefined;table-layout: fixed;width: 1155px"><colgroup>
    <col style="width: 319px">
    <col style="width: 144px">
    <col style="width: 671px">
    </colgroup>
        <thead>
            <th>Return value</th>
            <th>Error code</th>
            <th>Description</th>
        </thead>
        <tbody>
            <tr>
                <td>ACLNN_ERR_PARAM_NULLPTR</td>
                <td>161001</td>
                <td>The input is a null pointer.</td>
            </tr>
            <tr>
                <td>ACLNN_ERR_PARAM_INVALID</td>
                <td>161002</td>
                <td>The input data type is not within the supported range.</td>
            </tr>
        </tbody>
    </table>

## aclnnAttentionWorkerCombine

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1151px"><colgroup>
  <col style="width: 184px">
  <col style="width: 134px">
  <col style="width: 833px">
  </colgroup>
  <thead>
    <tr>
      <th>Parameter</th>
      <th>Input/Output</th>
      <th>Description</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>workspace</td>
      <td>Input</td>
      <td>Workspace memory address allocated on the Device side.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Workspace size allocated on the Device side, obtained by the first-phase API aclnnAttentionWorkerCombineGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Executor for the op. Contains the operator computation flow.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Specifies the Stream for task execution.</td>
    </tr>
  </tbody>
  </table>

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

<!-- npu="A3,910b" id7 -->
- <term>Atlas A2 products</term> and <term>Atlas A3 products</term> support `tokenDtype` values 0 and 1 with FLOAT `expertScales`.
<!-- end id7 -->
<!-- npu="950" id8 -->
- <term>Ascend 950PR&950DT products</term> support `tokenDtype` values 0 through 4. Use FLOAT `expertScales` for 0 or 1 and FLOAT8_E8M0 `expertScales` for 2, 3, or 4.
<!-- end id8 -->

- `R` and `H` must be positive integers, and `K` must be in `[1, 64]`.
- Deterministic computation:
  - aclnnAttentionWorkerCombine defaults to a deterministic implementation.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include <cstring>
#include "acl/acl.h"
#include "aclnnop/aclnn_attention_worker_combine.h"

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
  auto ret = aclInit(nullptr);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
  ret = aclrtSetDevice(deviceId);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
  ret = aclrtCreateStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
  return 0;
}

void Finalize(int32_t deviceId, aclrtStream stream) {
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor) {
  auto size = GetShapeSize(shape) * sizeof(T);
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  std::vector<int64_t> stride(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    stride[i] = shape[i + 1] * stride[i + 1];
  }
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, stride.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

int CreateAclTensorNoData(const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType,
                          aclTensor** tensor) {
  uint64_t elemSize = sizeof(int8_t);
  if (dataType == ACL_INT32) { elemSize = sizeof(int32_t); }
  if (dataType == ACL_FLOAT16) { elemSize = sizeof(int16_t); }
  if (dataType == ACL_BF16) { elemSize = sizeof(int16_t); }
  auto size = GetShapeSize(shape) * elemSize;
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

  std::vector<int64_t> stride(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    stride[i] = shape[i + 1] * stride[i + 1];
  }
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, stride.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

#pragma pack(push, 1)
struct AttentionDataDesc {
  int32_t flag[0];
};

struct ScheduleContext {
  struct CommonArea {
    uint32_t session_num;           // Number of attention nodes
    uint32_t micro_batch_num;
    uint32_t micro_batch_size;
    uint32_t selected_expert_num;   // topK + 1
    uint32_t expert_num;            // experts per layer
    uint32_t attn_to_ffn_token_size;
    uint32_t ffn_to_attn_token_size;
    int32_t  schedule_mode;         // 0: Ffn only 1: Attention only
    int8_t   reserve0[96];
  };
  struct ControlArea {
    int32_t run_flag;               // 0 : exited  1 : running
    int8_t reserve2[124];
  };
  struct AttentionArea {
    uint64_t token_info_buf;        // Points to device memory.
    uint64_t token_info_buf_size;
    uint64_t token_data_buf;        // Points to device memory.
    uint64_t token_data_buf_size;
    uint32_t micro_batch_id;        // Records the latest ready micro batch id.
    int8_t   reserve5[92];
  };
  struct FfnArea {
    uint64_t token_info_buf;
    uint64_t token_info_buf_size;
    uint64_t token_data_buf;
    uint64_t token_data_buf_size;
    uint64_t polling_index;
    int8_t   reserve3[88];
    uint64_t layer_ids_buf;
    uint64_t layer_ids_buf_size;
    uint64_t session_ids_buf;
    uint64_t session_ids_buf_size;
    uint64_t micro_batch_ids_buf;
    uint64_t micro_batch_ids_buf_size;
    uint64_t expert_ids_buf;
    uint64_t expert_ids_buf_size;
    uint32_t out_num;
    int8_t   reserve4[60];
  };
  CommonArea    common;
  ControlArea   control;
  AttentionArea attention;
  FfnArea       ffn;
  int8_t        reserve6[384];      // Padding to 1024 bytes.
};
static_assert(sizeof(ScheduleContext) == 1024, "ScheduleContext size must be 1024 bytes");
#pragma pack(pop)

int main() {
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the input and output based on the API as required.
  int64_t R = 48;       // token count
  int64_t K = 8;         // routed experts per token
  const int64_t expertRows = K + 1;  // includes the shared expert
  int64_t H = 20480;
  int64_t tokenDtype = 1;    // BF16
  int64_t needSchedule = 0;

  // Initialize ScheduleContext.
  ScheduleContext scheduleContext = {};
  scheduleContext.common.session_num = 1;
  scheduleContext.common.micro_batch_num = 1;
  scheduleContext.common.micro_batch_size = R;
  scheduleContext.common.selected_expert_num = expertRows;
  scheduleContext.common.expert_num = 16;
  scheduleContext.common.attn_to_ffn_token_size = 512;
  scheduleContext.common.ffn_to_attn_token_size = 512;
  scheduleContext.common.schedule_mode = 1;   // Attention only
  scheduleContext.control.run_flag = 1;       // running
  scheduleContext.attention.micro_batch_id = 0;

  // Initialize the Attention token_info_buf (flag set to 1 indicates that the data is ready).
  size_t perDataDescSize = sizeof(AttentionDataDesc) + sizeof(int32_t) * R * expertRows;
  size_t tokenInfoBufSize = static_cast<size_t>(scheduleContext.common.micro_batch_num) * perDataDescSize;
  void* tokenInfoBuf = nullptr;
  ret = aclrtMalloc(&tokenInfoBuf, tokenInfoBufSize, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("malloc token info buf failed. ERROR: %d\n", ret); return ret);
  scheduleContext.attention.token_info_buf = reinterpret_cast<uint64_t>(tokenInfoBuf);
  scheduleContext.attention.token_info_buf_size = tokenInfoBufSize;
  std::vector<int32_t> hostFlags(static_cast<size_t>(R) * expertRows, 1);
  ret = aclrtMemcpy(tokenInfoBuf, tokenInfoBufSize, hostFlags.data(),
                    static_cast<size_t>(R) * expertRows * sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("cpy token info buf failed. ERROR: %d\n", ret); return ret);

  // Initialize the Attention token_data_buf.
  uint64_t tokenDataSize = static_cast<uint64_t>(R) * expertRows * H * sizeof(int16_t);
  void* tokenDataBuf = nullptr;
  ret = aclrtMalloc(&tokenDataBuf, tokenDataSize, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("malloc token data buf failed. ERROR: %d\n", ret); return ret);
  scheduleContext.attention.token_data_buf = reinterpret_cast<uint64_t>(tokenDataBuf);
  scheduleContext.attention.token_data_buf_size = tokenDataSize;
  std::vector<int16_t> hostTokenData(static_cast<size_t>(R) * expertRows * H, 1);
  ret = aclrtMemcpy(tokenDataBuf, tokenDataSize, hostTokenData.data(), tokenDataSize, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("cpy token data buf failed. ERROR: %d\n", ret); return ret);

  // Create the scheduleContext aclTensor.
  std::vector<int64_t> scheduleContextShape = {1024};
  void* scheduleContextDeviceAddr = nullptr;
  aclTensor* scheduleContextRef = nullptr;
  std::vector<int8_t> hostCtxData(1024, 0);
  std::memcpy(hostCtxData.data(), &scheduleContext, sizeof(ScheduleContext));
  ret = CreateAclTensor(hostCtxData, scheduleContextShape, &scheduleContextDeviceAddr, aclDataType::ACL_INT8,
                        &scheduleContextRef);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // Create the expert_scales aclTensor, shape (R, K).
  std::vector<int64_t> expertScalesShape = {R, K};
  std::vector<float> hostExpertScales(static_cast<size_t>(R) * K, 0.125f);
  void* expertScalesDeviceAddr = nullptr;
  aclTensor* expertScalesRef = nullptr;
  ret = CreateAclTensor(hostExpertScales, expertScalesShape, &expertScalesDeviceAddr, aclDataType::ACL_FLOAT,
                        &expertScalesRef);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // Create the layer_id aclTensor, shape (1).
  std::vector<int64_t> layerIdShape = {1};
  std::vector<int32_t> hostLayerId = {0};
  void* layerIdDeviceAddr = nullptr;
  aclTensor* layerIdRef = nullptr;
  ret = CreateAclTensor(hostLayerId, layerIdShape, &layerIdDeviceAddr, aclDataType::ACL_INT32, &layerIdRef);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // Create the output y aclTensor, shape (R, H).
  std::vector<int64_t> yShape = {R, H};
  void* yDeviceAddr = nullptr;
  aclTensor* yRef = nullptr;
  ret = CreateAclTensorNoData(yShape, &yDeviceAddr, aclDataType::ACL_BF16, &yRef);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // Create the output next_layer_id aclTensor, shape (1).
  std::vector<int64_t> nextLayerIdShape = {1};
  void* nextLayerIdDeviceAddr = nullptr;
  aclTensor* nextLayerIdRef = nullptr;
  ret = CreateAclTensorNoData(nextLayerIdShape, &nextLayerIdDeviceAddr, aclDataType::ACL_INT32, &nextLayerIdRef);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  ret = aclnnAttentionWorkerCombineGetWorkspaceSize(scheduleContextRef, expertScalesRef, layerIdRef, H,
                                                    tokenDtype, needSchedule, yRef, nextLayerIdRef, &workspaceSize,
                                                    &executor);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclnnAttentionWorkerCombineGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }

  ret = aclnnAttentionWorkerCombine(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAttentionWorkerCombine failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate) Synchronously wait for the task execution to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output result and copy it from device memory to host memory.
  int32_t nextLayerId = 0;
  ret = aclrtMemcpy(&nextLayerId, sizeof(int32_t), nextLayerIdDeviceAddr, sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy next_layer_id failed. ERROR: %d\n", ret); return ret);
  LOG_PRINT("next_layer_id = %d.\n", nextLayerId);

  // 6. Release the aclTensor.
  aclDestroyTensor(scheduleContextRef);
  aclDestroyTensor(expertScalesRef);
  aclDestroyTensor(layerIdRef);
  aclDestroyTensor(yRef);
  aclDestroyTensor(nextLayerIdRef);

  // 7. Release device resources.
  aclrtFree(scheduleContextDeviceAddr);
  aclrtFree(expertScalesDeviceAddr);
  aclrtFree(layerIdDeviceAddr);
  aclrtFree(yDeviceAddr);
  aclrtFree(nextLayerIdDeviceAddr);
  aclrtFree(tokenInfoBuf);
  aclrtFree(tokenDataBuf);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }

  Finalize(deviceId, stream);
  return 0;
}
```
