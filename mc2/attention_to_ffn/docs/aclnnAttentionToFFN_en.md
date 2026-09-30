# aclnnAttentionToFFN

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| Ascend 950PR/Ascend 950DT                            |    ×     |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √     |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    ×     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference products</term>                            |    ×     |
| <term>Atlas training products</term>                             |    ×     |

## Function

Sends token data from the Attention node to the FFN node.

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnAttentionToFFNGetWorkspaceSize` is called to obtain the input parameters and compute the required workspace size based on the process. Then, `aclnnAttentionToFFN` is called to perform computation.

```c++
aclnnStatus aclnnAttentionToFFNGetWorkspaceSize(
    const aclTensor    *x,
    const aclTensor    *sessionId,
    const aclTensor    *microBatchId,
    const aclTensor    *layerId,
    const aclTensor    *expertIds,
    const aclTensor    *expertRankTable,
    const aclTensor    *scalesOptional,
    const aclTensor    *activeMaskOptional,
    const char         *group,
    int64_t             worldSize,
    const aclIntArray  *ffnTokenInfoTableShape,
    const aclIntArray  *ffnTokenDataShape,
    const aclIntArray  *attnTokenInfoTableShape,
    int64_t             moeExpertNum,
    int64_t             quantMode,
    int64_t             syncFlag,
    int64_t             ffnStartRankId,
    uint64_t           *workspaceSize,
    aclOpExecutor     **executor)
```

```c++
aclnnStatus aclnnAttentionToFFN(
    void            *workspace,
    uint64_t        workspaceSize,
    aclOpExecutor   *executor,
    aclrtStream     stream)
```

## aclnnAttentionToFFNGetWorkspaceSize

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
        <th>Usage</th>
        <th>Data Type</th>
        <th>Data Format</th>
        <th>Dimension (Shape)</th>
        <th>Non-contiguous Tensor</th>
      </tr></thead>
    <tbody>
      <tr>
        <td>x</td>
        <td>Input</td>
        <td>Token data sent by the current rank.</td>
        <td>The shape is (X, Bs, H).</td>
        <td>FLOAT16, BFLOAT16</td>
        <td>ND</td>
        <td>3</td>
        <td>√</td>
      </tr>
      <tr>
        <td>sessionId</td>
        <td>Input</td>
        <td>ID of the current Attention Worker node.</td>
        <td>The shape is (X,). The value range is [0, attentionWorkerNum).</td>
        <td>INT32</td>
        <td>ND</td>
        <td>1</td>
        <td>√</td>
      </tr>
      <tr>
        <td>microBatchId</td>
        <td>Input</td>
        <td>ID of the microBatch.</td>
        <td>The shape is (X,).</td>
        <td>INT32</td>
        <td>ND</td>
        <td>1</td>
        <td>√</td>
      </tr>
      <tr>
        <td>layerId</td>
        <td>Input</td>
        <td>ID of the current model layer.</td>
        <td>The shape is (X,).</td>
        <td>INT32</td>
        <td>ND</td>
        <td>1</td>
        <td>√</td>
      </tr>
      <tr>
        <td>expertIds</td>
        <td>Input</td>
        <td>Top K expert indices of each token in each micro batch group.</td>
        <td>The shape is (X, Bs, K), and the value range is [0, moeExpertNum).</td>
        <td>INT32</td>
        <td>ND</td>
        <td>3</td>
        <td>√</td>
      </tr>
      <tr>
        <td>expertRankTable</td>
        <td>Input</td>
        <td>Mapping table from expert IDs to FFN expert deployment in each micro batch group. (The value must be correct externally.)</td>
        <td>The shape is (L, moeExpertNum + sharedExpertNum, M).</td>
        <td>INT32</td>
        <td>ND</td>
        <td>3</td>
        <td>√</td>
      </tr>
      <tr>
        <td>scalesOptional</td>
        <td>Input</td>
        <td>Quantization smoothing parameter for each expert.</td>
        <td>In the non-quantization scenario, a null pointer must be passed. In the dynamic quantization scenario, valid data or a null pointer can be passed. The shape is (L, moeExpertNum + sharedExpertNum, H).</td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>3</td>
        <td>√</td>
      </tr>
      <tr>
        <td>activeMaskOptional</td>
        <td>Input</td>
        <td>Indicates whether a token participates in communication. Valid data or null pointer can be transferred.</td>
        <td>If a null pointer is transferred, all tokens participate in communication by default. If a value is transferred, the shape is (X, Bs). The value true indicates that the token participates in communication, and true must be placed before false.</td>
        <td>BOOL</td>
        <td>ND</td>
        <td>2</td>
        <td>√</td>
      </tr>
      <tr>
        <td>group</td>
        <td>Input</td>
        <td>Name of the communicator (expert parallelism).</td>
        <td>The string length range is [1, 128).</td>
        <td>STRING</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>worldSize</td>
        <td>Input</td>
        <td>Indicates the size of the communicator.</td>
        <td>The value range is [2, 768].</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>ffnTokenInfoTableShape</td>
        <td>Input</td>
        <td>Indicates the list of token information table shape sizes on the FFN node.</td>
        <td>The length is 3, including the number of attention nodes, the size of microBatchSize, and the size of the shape of the related sending status information corresponding to each token.</td>
        <td>INT32</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>ffnTokenDataShape</td>
        <td>Input</td>
        <td>List of token data table shape sizes on the FFN node.</td>
        <td>The length is 5, including the number of attention nodes, microBatchSize, batchSize, number of experts to be sent for each token (including shared experts), and length of a single token.</td>
        <td>INT32</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>attnTokenInfoTableShape</td>
        <td>Input</td>
        <td>List of token information table shape sizes on the attention node.</td>
        <td>The length is 3, including the microBatchSize, batchSize, and number of experts to be sent for each token (including shared experts).</td>
        <td>INT32</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>moeExpertNum</td>
        <td>Input</td>
        <td>Number of MoE experts.</td>
        <td>Value range: (0, 1024].</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>quantMode</td>
        <td>Input</td>
        <td>Quantization mode.</td>
        <td>Only 0 (non-quantization) and 2 (dynamic quantization) are supported.</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>syncFlag</td>
        <td>Input</td>
        <td>Synchronization mode of the FFN node.</td>
        <td>Only 0 (synchronous) and 1 (asynchronous) are supported.</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>ffnStartRankId</td>
        <td>Input</td>
        <td>Start ID of the FFN node.</td>
        <td>The value range is [0, worldSize).</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>workspaceSize</td>
        <td>Output</td>
        <td>Size of the workspace to be allocated on the device.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>executor</td>
        <td>Output</td>
        <td>Returns the operator executor that contains the operator computation process.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
    </tbody></table>

- **Returns:**

    `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

    The first-phase API implements input parameter validation. The following error codes may be returned.

    <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
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
        <td>Mandatory input and output tensors are null pointers.</td>
      </tr>
      <tr>
        <td>ACLNN_ERR_PARAM_INVALID</td>
        <td>161002</td>
        <td>The input and output data types are not supported.</td>
      </tr>
      <tr>
        <td rowspan="2">ACLNN_ERR_INNER_TILING_ERROR</td>
        <td rowspan="2">561002</td>
        <td>The input and output shapes are not supported.</td>
      </tr>
      <tr>
        <td>The parameter value is not supported.</td>
      </tr>
    </tbody>
    </table>

## aclnnAttentionToFFN

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
    <col style="width: 150px">
    <col style="width: 100px">
    <col style="width: 900px">
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
        <td>Size of the workspace to be allocated on the device, which is obtained by calling the first-phase API `aclnnAttentionToFFNGetWorkspaceSize`.</td>
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

- **Returns**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- **Deterministic constraints:**
  - The default deterministic implementation is used for `aclnnAttentionToFFN`.

- **Parameter consistency constraints:**
  - The `group`, `worldSize`, `moeExpertNum`, `ffnTokenInfoTableShape`, `ffnTokenDataShape`, `ffnStartRankId`, and `HCCL_BUFFSIZE` parameters used during API calls must be consistent across all cards, at different layers of the network, and with the parameters of the operators in the separated scenario.

- **Product constraints**:
  - <term>Atlas A3 training products/Atlas A3 inference products</term>: In this scenario, a single rank contains dual dies. Therefore, the "rank" in the parameter description indicates a single die.

- **Shape variable constraints**:

  | Variable        | Definition and Value Range                                                                |
  | :----------- | :----------------------------------------------------------------------------- |
  | X            | Indicates the micro batch sequence size (number of token groups). In the current version, only `X = 1` is supported.|
  | H (hidden size)| Indicates the hidden size. The value range is `[1024, 8192]`.|
  | Bs           | Indicates the batch sequence size (number of tokens output by the card). The value range is `0 < Bs ≤ 512`.|
  | K            | Indicates the number of selected top K experts. The value range is `0 < K ≤ 16` and `0 < K ≤ moeExpertNum`.|
  | L            | Number of model layers. The current version supports only `L = 1`.|
  | M            | Length of the last dimension of expertRankTable, which is the length of the list of the most expert deployment information deployed on the FFN node. The value range is `1 < M ≤ FFNWorkerNum * 2 + 1`.|
  | moeExpertNum | Number of MoE experts. The value range is `(0, 1024]`.                  |
  | sharedExpertNum | Number of shared experts (a shared expert can be replicated and deployed on multiple ffnRank cards). The value range is `[0, 4]`.                  |
  | moeExpertRankNum | Number of FFN nodes where MoE experts are deployed. The value range is `0 < moeExpertRankNum < FFNWorkerNum`.                  |
  | sharedExpertRankNum | Number of FFN nodes where shared experts are deployed. The value range is `0 < sharedExpertRankNum ≤ FFNWorkerNum`.                |
  | FFNWorkerNum | Number of FFN nodes. The value range is `0 < FFNWorkerNum < worldSize`, and the following condition must be met: `FFNWorkerNum = moeExpertRankNum + sharedExpertRankNum`.                  |
  | AttentionWorkerNum | Number of Attention nodes. The value range is `0 < AttentionWorkerNum < worldSize`.

- **Environment variables constraints**:
  - **HCCL_BUFFSIZE**: Before calling this API, check whether the value of the HCCL_BUFFSIZE environment variable is proper. This environment variable indicates the buffer size occupied by a single communication domain, in MB. If this environment variable is not set, the default value 200 MB is used.

- **Constraints on the use of the communication domains**:
  - The communicator of the AttentionToFFN operator cannot contain other operators.

## Example

Preparing files:

1. Create a `AttentionToFFNDemo` directory. Follow the instructions to create `aclnnAttentionToFFNDemo.cpp` and `AttentionToFFN.sh` files in the `AttentionToFFNDemo` directory, and modify them according to the code.

2. Install the CANN package and compile and run AttentionToFFNDemo according to the following instructions.

AttentionToFFN.sh compilation script

```bash
#!/bin/bash
cann_path="/path/to/cann_env" # Change the path to the CANN package environment.
g++ "aclnnAttentionToFFNDemo.cpp" -o AttentionToFFNDemo -I"$cann_path/latest/include/" -I"$cann_path/latest/include/aclnnop/" \
                    -L="$cann_path/latest/lib64/" -lascendcl -lnnopbase -lopapi_math -lop_common -lpthread -lhccl
```

Compilation and execution:

```bash
# Source CANN environment
source /path/to/cann_env/latest/bin/setenv.bash

# Compile aclnnAttentionToFFNDemo.cpp.
bash AttentionToFFN.sh

./AttentionToFFNDemo
```

The sample code is as follows:

```c++
#include <thread>
#include <iostream>
#include <string>
#include <cstring>
#include <vector>
#include "acl/acl.h"
#include "hccl/hccl.h"
#include "aclnn/opdev/fp16_t.h"
#include "aclnnop/aclnn_attention_to_ffn.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while(0)

struct Args {
    uint32_t rankId;
    HcclComm hcclComm;
    aclrtStream attentionToFFNStream;
    aclrtContext context;
};

constexpr uint32_t WORLD_SIZE = 16;
constexpr uint32_t FFN_WORKER_NUM = 5;
constexpr uint32_t ATTENTION_WORKER_NUM = WORLD_SIZE - FFN_WORKER_NUM;

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shape_size = 1;
    for (auto i : shape) {
        shape_size *= i;
    }
    return shape_size;
}

template<typename T>
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
    aclDataType dataType, aclTensor **tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc failed. ret: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMemcpy failed. ret: %d\n", ret); return ret);
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i +1] * strides[i + 1];
    }
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0,
        aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), *deviceAddr);
    return 0;
}

int LaunchOneProcessAttentionToFFN(Args &args)
{
    int ret = aclrtSetCurrentContext(args.context);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetCurrentContext failed, ret %d\n", ret); return ret);

    char hcomName[128] = {0};
    ret = HcclGetCommName(args.hcclComm, hcomName);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclGetCommName failed, ret %d\n", ret); return -1);
    LOG_PRINT("[INFO] rank = %d, hcomName = %s, attentionToFFNStream = %p, context = %p\n", 
              args.rankId, hcomName, args.attentionToFFNStream, args.context);

    int64_t X = 1;
    int64_t L = 1;
    int64_t Bs = 8;
    int64_t H = 7168;
    int64_t K = 4;
    int64_t sharedExpertNum = 1;
    int64_t sharedExpertRankNum = 1;
    int64_t moeExpertNum = 8;
    int64_t expertNumPerToken = K + sharedExpertNum;
    int64_t M = 2 *(FFN_WORKER_NUM - sharedExpertNum * sharedExpertRankNum) + 1;
    int64_t quantMode = 0;
    int64_t syncFlag = 0;
    int64_t ffnStartRankId = 0;
    int64_t ffnTokenInfoTableShapeData[] = {ATTENTION_WORKER_NUM , X, 2 + Bs * expertNumPerToken};
    int64_t ffnTokenDataShapeData[] = {ATTENTION_WORKER_NUM, X, Bs, expertNumPerToken, H};
    int64_t attnTokenInfoTableShapeData[] = {X, Bs, expertNumPerToken};

    /* Construct the input and output variables on the device based on the current scenario. */
    // Declare the input and output variables on the device.
    void *xDeviceAddr = nullptr;
    void *sessionIdDeviceAddr = nullptr;
    void *microBatchIdDeviceAddr = nullptr;
    void *layerIdDeviceAddr = nullptr;
    void *expertIdsDeviceAddr = nullptr;
    void *expertRankTableDeviceAddr = nullptr;
    void *scalesDeviceAddr = nullptr;

    aclTensor *x = nullptr;
    aclTensor *sessionId = nullptr;
    aclTensor *microBatchId = nullptr;
    aclTensor *layerId = nullptr;
    aclTensor *expertIds = nullptr;
    aclTensor *expertRankTable = nullptr;
    aclTensor *scales = nullptr;
    aclIntArray *ffnTokenInfoTableShape = aclCreateIntArray(ffnTokenInfoTableShapeData, 3);
    aclIntArray *ffnTokenDataShape = aclCreateIntArray(ffnTokenDataShapeData, 5);
    aclIntArray *attnTokenInfoTableShape = aclCreateIntArray(attnTokenInfoTableShapeData, 3);

    // Define the dimensions of variables in the current scenario.
    std::vector<int64_t> xShape{X, Bs, H};
    std::vector<int64_t> sessionIdShape{X};
    std::vector<int64_t> microBatchIdShape{X};
    std::vector<int64_t> layerIdShape{X};
    std::vector<int64_t> expertIdsShape{X, Bs, K};
    std::vector<int64_t> expertRankTableShape{L, moeExpertNum + sharedExpertNum, M};
    std::vector<int64_t> scalesShape{L, moeExpertNum + sharedExpertNum, H};

    int64_t xShapeSize = GetShapeSize(xShape);
    int64_t sessionIdShapeSize = GetShapeSize(sessionIdShape);
    int64_t microBatchIdShapeSize = GetShapeSize(microBatchIdShape);
    int64_t layerIdShapeSize = GetShapeSize(layerIdShape);
    int64_t expertIdsShapeSize = GetShapeSize(expertIdsShape);
    int64_t expertRankTableShapeSize = GetShapeSize(expertRankTableShape);
    int64_t scalesShapeSize = GetShapeSize(scalesShape);

    // Construct variables on the host.
    std::vector<int16_t> xHostData(xShapeSize, 1);
    std::vector<int16_t> sessionIdHostData(sessionIdShapeSize, args.rankId - FFN_WORKER_NUM);
    std::vector<int16_t> microBatchIdHostData(microBatchIdShapeSize, 0);
    std::vector<int16_t> layerIdHostData(layerIdShapeSize, 0);
    std::vector<int32_t> expertIdsHostData;
    for (int32_t micro_batch_id = 0; micro_batch_id < expertIdsShape[0]; micro_batch_id++) {
        for (int32_t token_id = 0; token_id < expertIdsShape[1]; token_id++) {
            for (int32_t k_id = 0; k_id < expertIdsShape[2]; k_id++) {
                expertIdsHostData.push_back(k_id);
            }
        }
    } 

    std::vector<int32_t> expertRankTableHostData = {4, 2, 4, 3, 7, 1, 3, 2, 5, 2, 2, 5, 1, 2, 0, 0, 0, 0, 
                                                    3, 2, 5, 0, 0, 3, 7, 0, 0, 4, 1, 3, 0, 1, 2, 4, 3, 7, 
                                                    4, 0, 0, 3, 6, 1, 3, 2, 5, 3, 3, 7, 2, 4, 1, 2, 0, 0,
                                                    2, 2, 5, 0, 0, 0, 0, 0, 0, 3, 3, 6, 2, 5, 3, 7, 0, 0,
                                                    1, 4, 8, 0, 0, 0, 0, 0, 0};

    std::vector<float> scalesHostData(scalesShapeSize, 0.1);

    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_BF16, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(sessionIdHostData, sessionIdShape, &sessionIdDeviceAddr, aclDataType::ACL_INT32, &sessionId);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(microBatchIdHostData, microBatchIdShape, &microBatchIdDeviceAddr, aclDataType::ACL_INT32, &microBatchId);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(layerIdHostData, layerIdShape, &layerIdDeviceAddr, aclDataType::ACL_INT32, &layerId);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(expertIdsHostData, expertIdsShape, &expertIdsDeviceAddr, aclDataType::ACL_INT32, &expertIds);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(expertRankTableHostData, expertRankTableShape, &expertRankTableDeviceAddr, aclDataType::ACL_INT32, &expertRankTable);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(scalesHostData, scalesShape, &scalesDeviceAddr, aclDataType::ACL_FLOAT, &scales);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    uint64_t attentionToFFNWorkspaceSize = 0;
    aclOpExecutor *attentionToFFNExecutor = nullptr;
    void *attentionToFFNWorkspaceAddr = nullptr;

    /**************************************** calls AttentionToFFN ********************************************/.
    // Call the first-phase API.
    ret = aclnnAttentionToFFNGetWorkspaceSize(x, sessionId, microBatchId, layerId, expertIds, expertRankTable, (quantMode > 0 ? scales : nullptr), 
                                              nullptr, hcomName, WORLD_SIZE, ffnTokenInfoTableShape, ffnTokenDataShape, attnTokenInfoTableShape,
                                              moeExpertNum, quantMode, syncFlag, ffnStartRankId, &attentionToFFNWorkspaceSize, &attentionToFFNExecutor);

    CHECK_RET(ret == ACL_SUCCESS,
        LOG_PRINT("[ERROR] aclnnAttentionToFFNGetWorkspaceSize failed. ret = %d \n", ret); return ret);

    // Allocate device memory based on the computed workspaceSize.
    if (attentionToFFNWorkspaceSize > 0) {
        ret = aclrtMalloc(&attentionToFFNWorkspaceAddr, attentionToFFNWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc workspace failed. ret = %d \n", ret); return ret);
    }
    
    if (args.rankId < FFN_WORKER_NUM) {  // FFN Worker
        // Wait until the Attention Worker task is complete.
        LOG_PRINT("[INFO] device_%d is FFN worker, skipping aclnnAttentionToFFN execute.\n", args.rankId);
        std::this_thread::sleep_for(std::chrono::seconds(30));
    } else {    // Attention Worker
        // Call the second-phase API.
        ret = aclnnAttentionToFFN(attentionToFFNWorkspaceAddr, attentionToFFNWorkspaceSize,
                                    attentionToFFNExecutor, args.attentionToFFNStream);

        // (Boilerplate) Wait until the task execution is complete.
        ret = aclrtSynchronizeStreamWithTimeout(args.attentionToFFNStream, 10000);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnAttentionToFFN failed. ret = %d \n", ret);  \
            return ret);

        LOG_PRINT("[INFO] device_%d aclnnAttentionToFFN execute successfully.\n", args.rankId);
    }

    // Free device resources.
    if (attentionToFFNWorkspaceSize > 0) {
        aclrtFree(attentionToFFNWorkspaceAddr);
    }

    if (x != nullptr) {
        aclDestroyTensor(x);
    }
    if (sessionId != nullptr) {
        aclDestroyTensor(sessionId);
    }
    if (microBatchId != nullptr) {
        aclDestroyTensor(microBatchId);
    }
    if (layerId != nullptr) {
        aclDestroyTensor(layerId);
    }
    if (expertIds != nullptr) {
        aclDestroyTensor(expertIds);
    }
    if (expertRankTable != nullptr) {
        aclDestroyTensor(expertRankTable);
    }
    if (scales != nullptr) {
        aclDestroyTensor(scales);
    }
    if (ffnTokenInfoTableShape != nullptr) {
        aclDestroyIntArray(ffnTokenInfoTableShape);
    }
    if (ffnTokenDataShape != nullptr) {
        aclDestroyIntArray(ffnTokenDataShape);
    }
    if (attnTokenInfoTableShape != nullptr) {
        aclDestroyIntArray(attnTokenInfoTableShape);
    }  

    if (xDeviceAddr != nullptr) {
        aclrtFree(xDeviceAddr);
    }
    if (sessionIdDeviceAddr != nullptr) {
        aclrtFree(sessionIdDeviceAddr);
    }
    if (microBatchIdDeviceAddr != nullptr) {
        aclrtFree(microBatchIdDeviceAddr);
    }
    if (layerIdDeviceAddr != nullptr) {
        aclrtFree(layerIdDeviceAddr);
    }
    if (expertIdsDeviceAddr != nullptr) {
        aclrtFree(expertIdsDeviceAddr);
    }
    if (expertRankTableDeviceAddr != nullptr) {
        aclrtFree(expertRankTableDeviceAddr);
    }
    if (scalesDeviceAddr != nullptr) {
        aclrtFree(scalesDeviceAddr);
    }

    HcclCommDestroy(args.hcclComm);
    aclrtDestroyStream(args.attentionToFFNStream);
    aclrtDestroyContext(args.context);
    aclrtResetDevice(args.rankId);

    return 0;
}

int main(int argc, char *argv[])
{
    // This example is implemented based on Atlas A3 and can only run on Atlas A3.
    int ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtInit failed, ret = %d\n", ret); return ret);

    aclrtStream attentionToFFNStream[WORLD_SIZE];
    aclrtContext context[WORLD_SIZE];

    for (uint32_t rankId = 0; rankId < WORLD_SIZE; rankId++) {
        ret = aclrtSetDevice(rankId);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetDevice failed, ret = %d\n", ret); return ret);
        ret = aclrtCreateContext(&context[rankId], rankId);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateContext failed, ret = %d\n", ret); return ret);
        ret = aclrtCreateStream(&attentionToFFNStream[rankId]);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateStream failed, ret = %d\n", ret); return ret);
    }

    int32_t devices[WORLD_SIZE];
    for (int32_t id = 0; id < WORLD_SIZE; id++) {
        devices[id] = id;
    }

    HcclComm comms[WORLD_SIZE];
    ret = HcclCommInitAll(WORLD_SIZE, devices, comms);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("[ERROR] HcclCommInitAll failed, ret %d\n", ret); return ret);

    Args args[WORLD_SIZE];
    std::vector<std::unique_ptr<std::thread>> threads(WORLD_SIZE);
    for (uint32_t rankId = 0; rankId < WORLD_SIZE; rankId++) {
        args[rankId].rankId = rankId;
        args[rankId].hcclComm = comms[rankId];
        args[rankId].attentionToFFNStream = attentionToFFNStream[rankId];
        args[rankId].context = context[rankId];
        threads[rankId].reset(new(std::nothrow) std::thread(&LaunchOneProcessAttentionToFFN, std::ref(args[rankId])));
    }

    for(uint32_t rankId = 0; rankId < WORLD_SIZE; rankId++) {
        threads[rankId]->join();
    }

    aclFinalize();
    LOG_PRINT("[INFO] aclFinalize success\n");

    return 0;
}
```
