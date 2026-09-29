# aclnnFFNToAttention

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

Sends token data on the FFN node to the Attention node.

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnFFNToAttentionGetWorkspaceSize` is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, `aclnnFFNToAttention` is called to perform computation.

```cpp
aclnnStatus aclnnFFNToAttentionGetWorkspaceSize(
    const aclTensor   *x,
    const aclTensor   *sessionIds,
    const aclTensor   *microBatchIds,
    const aclTensor   *tokenIds,
    const aclTensor   *expertOffsets,
    const aclTensor   *actualTokenNum,
    const aclTensor   *attnRankTableOptional,
    const char        *group,
    int64_t            worldSize,
    const aclIntArray *tokenInfoTableShape,
    const aclIntArray *tokenDataShape,
    uint64_t          *workspaceSize,
    aclOpExecutor    **executor)
```

```cpp
aclnnStatus aclnnFFNToAttention(
    void           *workspace,
    uint64_t        workspaceSize,
    aclOpExecutor  *executor,
    aclrtStream     stream)
```

## aclnnFFNToAttentionGetWorkspaceSize

- **Parameters:**

    <table style="undefined;table-layout: fixed; width: 1484px"><colgroup>
    <col style="width: 186px">
    <col style="width: 123px">
    <col style="width: 283px">
    <col style="width: 295px">
    <col style="width: 181px">
    <col style="width: 122px">
    <col style="width: 147px">
    <col style="width: 147px">
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>x</td>
      <td>Input</td>
      <td>Token data sent by the current rank.</td>
      <td>The shape is <code>(Y, H)</code>.</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>sessionIds</td>
      <td>Input</td>
      <td>Index of the Attention Worker node for each token.</td>
      <td>The shape is <code>(Y,)</code>, and the value range is [0, attnRankNum-1].</td>
      <td>INT32</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>microBatchIds</td>
      <td>Input</td>
      <td>Index of the microBatch for each token.</td>
      <td>The shape is <code>(Y,)</code>, and the value range is [0, MicroBatchNum-1].</td>
      <td>INT32</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>tokenIds</td>
      <td>Input</td>
      <td>Index of each token in the microBatch.</td>
      <td>The shape is <code>(Y,)</code>, and the value range is [0, Bs-1].</td>
      <td>INT32</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>expertOffsets</td>
      <td>Input</td>
      <td>Index of each token in the PerTokenExpertNum field in tokenInfoTableShape.</td>
      <td>The shape is <code>(Y,)</code>, and the value range is [0, ExpertNumPerToken – 1].</td>
      <td>INT32</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>actualTokenNum</td>
      <td>Input</td>
      <td>Total number of tokens sent by the card, which is a 1D tensor.</td>
      <td>The shape is <code>(1,)</code>.</td>
      <td>INT64</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>attnRankTableOptional</td>
      <td>Optional input</td>
      <td>ID of the card corresponding to each Attention Worker.</td>
      <td>Attention Workers must be deployed continuously starting from card 0. If a null pointer is passed, the default policy is used: The ID of each card is used as the ID of the corresponding Attention Worker. The value range is [0, attnRankNum – 1].</td>
      <td>INT32</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>group</td>
      <td>Input</td>
      <td>Name of the communicator (expert parallelism).</td>
      <td>The string length is [1, 128).</td>
      <td>STRING</td>
      <td>-</td>
      <td>-</td>
      <td>×</td>
    </tr>
    <tr>
      <td>worldSize</td>
      <td>Input</td>
      <td>Communication domain size.</td>
      <td>The value range of worldSize is [2, 768].</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>×</td>
    </tr>
    <tr>
      <td>tokenInfoTableShape</td>
      <td>Input</td>
      <td>Size of the token information list.</td>
      <td>It contains the size of microBatch (MicroBatchNum), BatchSize (Bs), and the number of experts corresponding to each token (ExpertNumPerToken).</td>
      <td>INT32</td>
      <td>-</td>
      <td>-</td>
      <td>×</td>
    </tr>
    <tr>
      <td>tokenDataShape</td>
      <td>Input</td>
      <td>Size of the token data list.</td>
      <td>It contains the size of microBatch (MicroBatchNum), BatchSize (Bs), the number of experts corresponding to each token (ExpertNumPerToken), and the token and scale length (HS).</td>
      <td>INT32</td>
      <td>-</td>
      <td>-</td>
      <td>×</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Output</td>
      <td>Size of the workspace to be allocated on the device.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>×</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Output</td>
      <td>Operator executor that contains the operator execution process.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>×</td>
    </tr>
  </tbody></table>

- **Returns**

    `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

    The first-phase API implements input parameter validation. The following error codes may be returned.

    <table style="undefined;table-layout: fixed; width: 1166px">
    <colgroup>
    <col style="width: 166px">
    <col style="width: 100px">
    <col style="width: 900px">
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
    <td>Mandatory input and output tensors are null pointers.</td>
    </tr>
    <tr>
    <td>ACLNN_ERR_PARAM_INVALID</td>
    <td>161002</td>
    <td>The input and output data types are not supported.</td>
    </tr>
    </tbody>
    </table>

## aclnnFFNToAttention

- **Parameters:**

    <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
    <col style="width: 168px">
    <col style="width: 128px">
    <col style="width: 854px">
    </colgroup>
    <thead>
    <tr>
    <th>Name</th>
    <th>Input/Output</th>
    <th>Description</th>
    </tr>
    </thead>
    <tbody>
    <tr>
    <td>workspace</td>
    <td>Input</td>
    <td>Memory address of the workspace to be allocated on the device.</td>
    </tr>
    <tr>
    <td>workspaceSize</td>
    <td>Input</td>
    <td>Size of the workspace to be allocated on the device, which is obtained by calling the first-phase API `aclnnFFNToAttentionGetWorkspaceSize`.</td>
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

- **Returns**

  aclnnStatus status code. For details, see aclnn Return Code.

## Constraints

- Deterministic constraints:
  - `aclnnFFNToAttention` defaults to a deterministic implementation.

- **Parameter consistency constraints:**
  - The group, worldSize, tokenInfoTableShape, tokenDataShape, and HCCL_BUFFSIZE parameters of all devices must be the same.

- **Product constraints**:
  - <term>Atlas A3 training products/Atlas A3 inference products</term>: In this scenario, a single rank contains dual dies. Therefore, the "rank" in the parameter description indicates a single die.

- **Shape variable constraints**:

  | Variable        | Definition and Value Range                                                                          |
  | :----------- | :------------------------------------------------------------------------------------- |
  | Y            | Maximum number of tokens to be distributed on the current card.|
  | Bs           | Number of tokens sent on each attention node. <ul><li><term>Atlas A3 training products/Atlas A3 inference products</term>: <code>0 < Bs ≤ 512</code></li></ul> |
  | H (hidden size)| Size of the hidden layer. <ul><li><term>Atlas A3 training products/Atlas A3 inference products</term>: <code>1024 ≤ H ≤ 8192</code></li></ul> |
  | HS (hidden and scale size)| Size of the hidden layer and scale layer. <ul><li><term>Atlas A3 training products/Atlas A3 inference products</term>: <code>1152 ≤ HS ≤ 8320</code>.</li></ul>|
  | MicroBatchNum    | Size of the microBatch. Currently, only <code>MicroBatchNum = 1</code> is supported.|  
  | ExpertNumPerToken    | Number of experts sent for each token. The value is <code>ExpertNumPerToken = K + sharedExpertNum</code>.|  
  | K    | Number of top K experts to be selected. The value range is <code>0 < K ≤ 16</code>.|  
  | ffnRankNum    | Number of cards selected as FFnWorkers. The value range is <code>0 < ffnRankNum < worldSize</code>.| 
  | attnRankNum    | Number of cards selected as AttnWorkers. The value range is <code>0 < attnRankNum < worldSize</code>.| 
  | sharedExpertNum    | Number of shared experts (a shared expert can be replicated and deployed on multiple FFnRank cards). The value range is [0, 4].|  

- **Constraints on the use of the communication domains**:
  - No other operators are allowed in the communication domain of the `FFNToAttention` operator.

## Examples

- Preparing files:
  
  1. Create the FFNtoAttentionDemo directory, create the aclnnFFNtoAttentionDemo.cpp and FFNtoAttention.sh files in the FFNtoAttentionDemo directory, and modify the files by referring to the following code.

  2. Install the CANN package and compile and run the FFNtoAttentionDemo according to the following instructions.

- FFNtoAttention.sh compilation script

    ```bash
    #!/bin/bash
    cann_path="/path/to/cann_env" # Change the path to the CANN package environment.
    g++ "aclnnFFNtoAttentionDemo.cpp" -o FFNtoAttentionDemo -I"$cann_path/latest/include/" -I"$cann_path/latest/include/aclnnop/" \
                        -L="$cann_path/latest/lib64/" -lascendcl -lnnopbase -lopapi_math -lop_common -lpthread -lhccl
    ```

- Compilation and execution:

    ```bash
    # Source CANN environment
    source /path/to/cann_env/latest/bin/setenv.bash

    # Compiling aclnnFFNtoAttentionDemo.cpp
    bash FFNtoAttention.sh

    ./FFNtoAttentionDemo
    ```

- The sample code is as follows:

    ```Cpp
    #include <thread>
    #include <iostream>
    #include <string>
    #include <vector>
    #include <unordered_set>
    #include "acl/acl.h"
    #include "hccl/hccl.h"
    #include "aclnnop/aclnn_ffn_to_attention.h"

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
        aclrtStream FFN2AttentionStream;
        aclrtContext context;
    };

    constexpr uint32_t WORLD_SIZE = 16;
    constexpr uint32_t ATTN_NUM = 8;
    constexpr uint32_t DEV_NUM = WORLD_SIZE;

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
        *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
            shape.data(), shape.size(), *deviceAddr);
        return 0;
    }

    int LaunchOneProcessFFN2Attention(Args &args)
    {
        int ret = aclrtSetCurrentContext(args.context);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetCurrentContext failed, ret %d\n", ret); return ret);

        char hcomName[128] = {0};
        ret = HcclGetCommName(args.hcclComm, hcomName);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclGetCommName failed, ret %d\n", ret); return -1);
        LOG_PRINT("[INFO] rank = %d, hcomName = %s, FFN2AttentionStream = %p, \
                    context = %p\n", args.rankId, hcomName, args.FFN2AttentionStream, \
                    args.context);

        int64_t micro_batch_num = 1;
        int64_t Y = 8;
        int64_t H = 7168;
        int64_t K = 7;
        int64_t attention_worker_num = ATTN_NUM;
        int64_t sharedExpertNum = 1;
        int64_t expert_num_per_token = K + sharedExpertNum;
        int64_t Token_info_shape[] = {micro_batch_num, Y, expert_num_per_token};
        int64_t Token_data_shape[] = {micro_batch_num, Y, expert_num_per_token, H};

        void *xDeviceAddr = nullptr;
        void *sessionIdsDeviceAddr = nullptr;
        void *microBatchIdsDeviceAddr = nullptr;
        void *tokenIdsDeviceAddr = nullptr;
        void *expertOffsetsDeviceAddr = nullptr;
        void *actualTokenNumDeviceAddr = nullptr;
        void *attnRankTableDeviceAddr = nullptr;

        aclTensor *x = nullptr;
        aclTensor *sessionIds = nullptr;
        aclTensor *microBatchIds = nullptr;
        aclTensor *tokenIds = nullptr;
        aclTensor *expertOffsets = nullptr;
        aclTensor *actualTokenNum = nullptr;
        aclTensor *attnRankTable = nullptr;
        aclIntArray *tokenInfoTableShape = aclCreateIntArray(Token_info_shape, 3);   
        aclIntArray *tokenDataShape = aclCreateIntArray(Token_data_shape, 4);   

        
        // Define the dimensions of variables in the current scenario.
        std::vector<int64_t> xShape{Y, H};
        std::vector<int64_t> sessionIdsShape{Y};
        std::vector<int64_t> microBatchIdsShape{Y};
        std::vector<int64_t> tokenIdsShape{Y};
        std::vector<int64_t> expertOffsetsShape{Y};
        std::vector<int64_t> actualTokenNumShape{1};
        std::vector<int64_t> attnRankTableShape{attention_worker_num};


        int64_t xShapeSize = GetShapeSize(xShape);
        int64_t sessionIdsShapeSize = GetShapeSize(sessionIdsShape);
        int64_t microBatchIdsShapeSize = GetShapeSize(microBatchIdsShape);
        int64_t tokenIdsShapeSize = GetShapeSize(tokenIdsShape);
        int64_t expertOffsetsShapeSize = GetShapeSize(expertOffsetsShape);
        int64_t actualTokenNumShapeSize = GetShapeSize(actualTokenNumShape);
        int64_t attnRankTableShapeSize = GetShapeSize(attnRankTableShape);
        

        std::vector<int16_t> xHostData(xShapeSize, 1);
        std::vector<int32_t> sessionIdsHostData(sessionIdsShapeSize, 0);
        std::vector<int32_t> microBatchIdsHostData(microBatchIdsShapeSize, 0);
        std::vector<int32_t> tokenIdsHostData(tokenIdsShapeSize, 0);
        std::vector<int32_t> expertOffsetsHostData(expertOffsetsShapeSize, 0);
        std::vector<int64_t> actualTokenNumHostData(actualTokenNumShapeSize, 8);
        std::vector<int32_t> attnRankTableHostData(attnRankTableShapeSize);
        for (int32_t i = 0; i < Y; i++) {
            sessionIdsHostData[i] = i % attention_worker_num;
            tokenIdsHostData[i] = i % Y;
            expertOffsetsHostData[i] = i % expert_num_per_token;
        }
        for (int32_t i = 0; i < attention_worker_num; i++) {
            attnRankTableHostData[i] = static_cast<int32_t>(i);
        }


        ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_BF16, &x);
        CHECK_RET(ret == ACL_SUCCESS, return ret);  
        ret = CreateAclTensor(sessionIdsHostData, sessionIdsShape, &sessionIdsDeviceAddr, aclDataType::ACL_INT32, &sessionIds);
        CHECK_RET(ret == ACL_SUCCESS, return ret);  
        ret = CreateAclTensor(microBatchIdsHostData, microBatchIdsShape, &microBatchIdsDeviceAddr, aclDataType::ACL_INT32, &microBatchIds);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(tokenIdsHostData, tokenIdsShape, &tokenIdsDeviceAddr, aclDataType::ACL_INT32, &tokenIds);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(expertOffsetsHostData, expertOffsetsShape, &expertOffsetsDeviceAddr, aclDataType::ACL_INT32, &expertOffsets);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(actualTokenNumHostData, actualTokenNumShape, &actualTokenNumDeviceAddr,  aclDataType::ACL_INT64, &actualTokenNum);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(attnRankTableHostData, attnRankTableShape, &attnRankTableDeviceAddr, aclDataType::ACL_INT32, &attnRankTable);
        CHECK_RET(ret == ACL_SUCCESS, return ret);


        uint64_t FFN2AttentionWorkspaceSize = 0;
        aclOpExecutor *FFN2AttentionExecutor = nullptr;
        void *FFN2AttentionWorkspaceAddr = nullptr;
        
        /**************************************** Call FFN2Attention ********************************************/
        // Call the first-phase API.
        ret = aclnnFFNToAttentionGetWorkspaceSize(x, sessionIds, microBatchIds, tokenIds,
                                                            expertOffsets, actualTokenNum, attnRankTable,
                                                            hcomName, WORLD_SIZE, tokenInfoTableShape, tokenDataShape,
                                                            &FFN2AttentionWorkspaceSize, &FFN2AttentionExecutor);
        CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("[ERROR] aclnnFFNToAttentionGetWorkspaceSize failed. ret = %d \n", ret); return ret);
        // Allocate device memory based on the computed workspaceSize.
        if (FFN2AttentionWorkspaceSize > 0) {
            ret = aclrtMalloc(&FFN2AttentionWorkspaceAddr, FFN2AttentionWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc workspace failed. ret = %d \n", ret); return ret);
        }
        // Call the second-phase API.
        ret = aclnnFFNToAttention(FFN2AttentionWorkspaceAddr, FFN2AttentionWorkspaceSize, FFN2AttentionExecutor, args.FFN2AttentionStream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnFFNToAttention failed. ret = %d \n", ret);
            return ret);
        // (Boilerplate) Wait until the task execution is complete.
        if (args.rankId >= ATTN_NUM) {
            ret = aclrtSynchronizeStreamWithTimeout(args.FFN2AttentionStream, 10000);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSynchronizeStreamWithTimeout failed. ret = %d \n", ret);
            return ret);
            LOG_PRINT("[INFO] device_%d FFNToAttention execute successfully.\n", args.rankId);
        } else {
            std::this_thread::sleep_for(std::chrono::seconds(10));
            LOG_PRINT("[INFO] device_%d is AttentionWorker, sleeping 10 seconds...\n", args.rankId);
        }

        
        // Free device resources.
        if (FFN2AttentionWorkspaceSize > 0) {
            aclrtFree(FFN2AttentionWorkspaceAddr);
        }
        if (x != nullptr) {
            aclDestroyTensor(x);
        }
        if (sessionIds != nullptr) {
            aclDestroyTensor(sessionIds);
        }
        if (microBatchIds != nullptr) {
            aclDestroyTensor(microBatchIds);
        }
        if (tokenIds != nullptr) {
            aclDestroyTensor(tokenIds);
        }

        if (expertOffsets != nullptr) {
            aclDestroyTensor(expertOffsets);
        }
        if (actualTokenNum != nullptr) {
            aclDestroyTensor(actualTokenNum);
        }
        if (attnRankTable != nullptr) {
            aclDestroyTensor(attnRankTable);
        }
        if (tokenInfoTableShape != nullptr) {
            aclDestroyIntArray(tokenInfoTableShape);
        }
        if (tokenDataShape != nullptr) {
            aclDestroyIntArray(tokenDataShape);
        }


        if (xDeviceAddr != nullptr) {
            aclrtFree(xDeviceAddr);
        }
        if (sessionIdsDeviceAddr != nullptr) {
            aclrtFree(sessionIdsDeviceAddr);
        }
        if (microBatchIdsDeviceAddr != nullptr) {
            aclrtFree(microBatchIdsDeviceAddr);
        }
        if (tokenIdsDeviceAddr != nullptr) {
            aclrtFree(tokenIdsDeviceAddr);
        }
        if (expertOffsetsDeviceAddr != nullptr) {
            aclrtFree(expertOffsetsDeviceAddr);
        }
        if (actualTokenNumDeviceAddr != nullptr) {
            aclrtFree(actualTokenNumDeviceAddr);
        }
        if (attnRankTableDeviceAddr != nullptr) {
            aclrtFree(attnRankTableDeviceAddr);
        }

        HcclCommDestroy(args.hcclComm);
        aclrtDestroyStream(args.FFN2AttentionStream);
        aclrtDestroyContext(args.context);
        aclrtResetDevice(args.rankId);

        return 0;
    }

    int main(int argc, char *argv[])
    {
        int ret = aclInit(nullptr);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclInit failed, ret = %d\n", ret); return ret);

        aclrtStream FFN2AttentionStream[DEV_NUM];
        aclrtContext context[DEV_NUM];
        for (uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
            ret = aclrtSetDevice(rankId);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetDevice failed, ret = %d\n", ret); return ret);
            ret = aclrtCreateContext(&context[rankId], rankId);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateContext failed, ret = %d\n", ret); return ret);
            ret = aclrtCreateStream(&FFN2AttentionStream[rankId]);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateStream failed, ret = %d\n", ret); return ret);
        }

        int32_t devices[WORLD_SIZE];
        for (int32_t deviceId = 0; deviceId < WORLD_SIZE; deviceId++) {
            devices[deviceId] = deviceId ;
        }

        HcclComm comms[WORLD_SIZE];
        ret = HcclCommInitAll(WORLD_SIZE, devices, comms);
        CHECK_RET(ret == ACL_SUCCESS,
                    LOG_PRINT("[ERROR] HcclCommInitAll failed, ret %d\n", ret); return ret);


        Args args[DEV_NUM];
        std::vector<std::unique_ptr<std::thread>> threads(DEV_NUM);
        for (uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
            args[rankId].rankId = rankId;
            args[rankId].hcclComm = comms[rankId];
            args[rankId].FFN2AttentionStream = FFN2AttentionStream[rankId];
            args[rankId].context = context[rankId];
            threads[rankId].reset(new(std::nothrow) std::thread(&LaunchOneProcessFFN2Attention, std::ref(args[rankId])));
        }

        for(uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
            threads[rankId]->join();
        }

        aclFinalize();
        LOG_PRINT("[INFO] aclFinalize success\n");

        return 0;
    }
    ```
