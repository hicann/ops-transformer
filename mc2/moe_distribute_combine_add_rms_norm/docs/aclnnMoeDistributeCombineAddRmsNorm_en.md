# aclnnMoeDistributeCombineAddRmsNorm

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 training products / Atlas A3 inference products</term>    |    √     |
| <term>Atlas A2 training products / Atlas A2 inference products</term>|    ×     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference products</term>                            |    ×     |
| <term>Atlas training products</term>                             |    ×     |

## Function Description

- Operator functions: When TP domain communication is involved, ReduceScatterV is performed first, followed by AlltoAllV, and finally the received data is aggregated (multiplied by weights and then summed). When TP domain communication is not involved, AlltoAllV is performed, after which the received data is aggregated (multiplied by weights and then summed), followed by fused Add + RMSNorm.

- Formula:

$$
rsOut = ReduceScatterV(expandX)\\
ataOut = AllToAllV(rsOut)\\
combineOut = Sum(expertScales * ataOut + expertScales * sharedExpertX)\\
x = combineOut + residualX\\
y = \frac{x}{RMS(x)} * gamma,\quad\text{where}\ RMS(x) = \sqrt{\frac{1}{H}\sum_{i=1}^{H}x_{i}^{2}+normEps}
$$

> Note that this API must be used together with `aclnnMoeDistributeDispatchV2`, and it performs a reverse data flow along the same path used by the `MoeDistributeDispatchV2` operator for data dispatch.

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnMoeDistributeCombineAddRmsNormGetWorkspaceSize` is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, `aclnnMoeDistributeCombineAddRmsNorm` is called to perform computation.

```cpp
aclnnStatus aclnnMoeDistributeCombineAddRmsNormGetWorkspaceSize(
    const aclTensor* expandX,
    const aclTensor* expertIds,
    const aclTensor* assistInfoForCombine,
    const aclTensor* epSendCounts,
    const aclTensor* expertScales,
    const aclTensor* residualX,
    const aclTensor* gamma,
    const aclTensor* tpSendCountsOptional,
    const aclTensor* xActiveMaskOptional,
    const aclTensor* activationScaleOptional,
    const aclTensor* weightScaleOptional,
    const aclTensor* groupListOptional,
    const aclTensor* expandScalesOptional,
    const aclTensor* sharedExpertXOptional,
    const char*      groupEp,
    int64_t          epWorldSize,
    int64_t          epRankId,
    int64_t          moeExpertNum,
    const char*      groupTp,
    int64_t          tpWorldSize,
    int64_t          tpRankId,
    int64_t          expertShardType,
    int64_t          sharedExpertNum,
    int64_t          sharedExpertRankNum,
    int64_t          globalBS,
    int64_t          outDtype,
    int64_t          commQuantMode,
    int64_t          groupListType,
    const char*      commAlg,
    float            normEps,
    aclTensor*       yOut,
    aclTensor*       rstdOut,
    aclTensor*       xOut,
    uint64_t*        workspaceSize,
    aclOpExecutor**  executor)
```

```cpp
aclnnStatus aclnnMoeDistributeCombineAddRmsNorm(
    void            *workspace,
    uint64_t        workspaceSize,
    aclOpExecutor   *executor,
    aclrtStream     stream)
```

## aclnnMoeDistributeCombineAddRmsNormGetWorkspaceSize

- **Parameters**

    <table style="undefined;table-layout: fixed; width: 1237px"><colgroup>
    <col style="width: 252px">
    <col style="width: 121px">
    <col style="width: 523px">
    <col style="width: 216px">
    <col style="width: 125px">
    </colgroup>
    <thead>
    <tr>
    <th>Name</th>
    <th>Input/Output</th>
    <th>Description</th>
    <th>Data Type</th>
    <th>Data Format</th>
    </tr>
    </thead>
    <tbody>
    <tr>
    <td>expandX</td>
    <td>Input</td>
    <td>Token features expanded according to expertIds. It must be a 2D tensor with shape (max(tpWorldSize, 1) * A, H). non-contiguous tensor are supported.</td>
    <td>BFLOAT16</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>expertIds</td>
    <td>Input</td>
    <td>Top-K expert indexes for each token. It must be a 2D tensor with shape (Bs, K). non-contiguous tensor are supported.</td>
    <td>INT32</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>assistInfoForCombine</td>
    <td>Input</td>
    <td>Corresponds to the assistInfoForCombineOut output of aclnnMoeDistributeDispatchV2. It must be a 1D tensor with shape (A * 128, ). non-contiguous tensor are supported.</td>
    <td>INT32</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>epSendCounts</td>
    <td>Input</td>
    <td>Corresponds to the epRecvCounts output of aclnnMoeDistributeDispatchV2. It must be a 1D tensor with shape (epWorldSize * max(tpWorldSize, 1) * localExpertNum, ). non-contiguous tensor are supported.</td>
    <td>INT32</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>expertScales</td>
    <td>Input</td>
    <td>Top-K expert weights for each token. It must be a 2D tensor with shape (Bs, K). non-contiguous tensor are supported.</td>
    <td>FLOAT32</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>residualX</td>
    <td>Input</td>
    <td>Right-hand matrix of Add in AddRmsNorm. It must be a 3D tensor with shape (Bs, 1, H). non-contiguous tensor are supported.</td>
    <td>BFLOAT16</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>gamma</td>
    <td>Input</td>
    <td>Gamma input of RmsNorm. It must be a 1D tensor with shape (H, ). non-contiguous tensor are supported.</td>
    <td>BFLOAT16</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>tpSendCountsOptional</td>
    <td>Input</td>
    <td>Corresponds to the tpRecvCounts output of aclnnMoeDistributeDispatchV2. It is a device-side aclTensor. If TP domain communication is used, this parameter must be provided; otherwise, a null pointer should be passed. When TP domain communication is enabled, it must be a 1D tensor with shape (tpWorldSize, ). non-contiguous tensor are supported.</td>
    <td>INT32</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>xActiveMaskOptional</td>
    <td>Input</td>
    <td>Indicates whether each token participates in communication. This is a device-side aclTensor. It must be a 1D or 2D tensor: shape (BS, ) for 1D and (BS, K) for 2D. Valid data or a null pointer may be provided. For 1D tensors, all <code>true</code> values must precede false values (for example, <code>{true, false, true}</code> is invalid). For 2D tensors, if all K values corresponding to a token are <code>false</code>, the token does not participate in communication. By default, all tokens participate in communication. If BS differs across ranks, all tokens must be valid. non-contiguous tensor are supported.</td>
    <td>BOOL</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>activationScaleOptional</td>
    <td>Input</td>
    <td>Reserved parameter, which is not supported in the current version. Pass a null pointer.</td>
    <td>-</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>weightScaleOptional</td>
    <td>Input</td>
    <td>Reserved parameter, which is not supported in the current version. Pass a null pointer.</td>
    <td>-</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>groupListOptional</td>
    <td>Input</td>
    <td>Reserved parameter, which is not supported in the current version. Pass a null pointer.</td>
    <td>-</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>expandScalesOptional</td>
    <td>Input</td>
    <td>Corresponds to the output of expandScales in aclnnMoeDistributeDispatchV2. This is a device-side aclTensor. This parameter is reserved and not supported in the current version. Pass a null pointer.</td>
    <td>-</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>sharedExpertXOptional</td>
    <td>Input</td>
    <td>Tokens calculated by shared experts. They are device-side aclTensors. They must be 2D or 3D tensors: shape (Bs, H) for 2D or (Bs, 1, H) for 3D. They must have the same data type as <code>expandX</code>. The tensors may be provided or omitted. non-contiguous tensor are supported.</td>
    <td>BFLOAT16</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>groupEp</td>
    <td>Input</td>
    <td>EP communication domain name (expert parallel communication domain). It must be a string with length in the range [1, 128) and different from groupTp.</td>
    <td>STRING</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>epWorldSize</td>
    <td>Input</td>
    <td>EP communication domain size, which must be in the range (1, 768].</td>
    <td>INT64</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>epRankId</td>
    <td>Input</td>
    <td>ID of the current rank in the EP communication domain, which must be in the range [0, epWorldSize). The value of epRankId for each rank in the same EP communication domain must be unique.</td>
    <td>INT64</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>moeExpertNum</td>
    <td>Input</td>
    <td>Number of MoE experts, which must be in the range (0, 1024] and satisfy moeExpertNum % (epWorldSize - sharedExpertRankNum) = 0.</td>
    <td>INT64</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>groupTp</td>
    <td>Input</td>
    <td>TP communication domain name (data parallel communication domain). It must be a string with length in the range [1, 128) and different from groupEp.</td>
    <td>STRING</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>tpWorldSize</td>
    <td>Input</td>
    <td>TP communication domain size, which must be in the range [0, 2]. 0 and 1 indicate no TP domain communication. 2 is required when TP domain communication is enabled.</td>
    <td>INT64</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>tpRankId</td>
    <td>Input</td>
    <td>ID of the current rank in the TP domain, which must be in the range [0, 1]. tpRankId of each rank in the same TP domain must be unique. If TP domain communication is not used, pass 0.</td>
    <td>INT64</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>expertShardType</td>
    <td>Input</td>
    <td>Distribution type of shared expert ranks, which must be 0. 0 indicates that shared expert ranks are placed before MoE expert ranks.</td>
    <td>INT64</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>sharedExpertNum</td>
    <td>Input</td>
    <td>Number of shared experts, which is not supported in the current version. Pass 0.</td>
    <td>INT64</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>sharedExpertRankNum</td>
    <td>Input</td>
    <td>Number of shared expert ranks, which is not supported in the current version. Pass 0.</td>
    <td>INT64</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>globalBS</td>
    <td>Input</td>
    <td>Global batch size in the EP domain. If all ranks have the same BS, globalBS = Bs * epWorldSize or 0. If BS differs across ranks, globalBS = maxBs * epWorldSize, where maxBs is the maximum BS of a single rank.</td>
    <td>INT64</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>outDtype</td>
    <td>Input</td>
    <td>Data type of output x. This parameter is reserved and not supported in the current version. Pass 0.</td>
    <td>INT64</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>commQuantMode</td>
    <td>Input</td>
    <td>Communication quantization type, which is not supported in the current version. Pass 0.</td>
    <td>INT64</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>groupListType</td>
    <td>Input</td>
    <td>Group list format. This parameter is reserved and not supported in the current version. Pass 0.</td>
    <td>INT64</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>commAlg</td>
    <td>Input</td>
    <td>Communication affinity memory layout algorithm. The value is of the string type. This parameter is reserved and not supported in the current version. Pass a null pointer.</td>
    <td>STRING</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>normEps</td>
    <td>Input</td>
    <td>Used to prevent division by zero in AddRmsNorm. It can be set to 1e-6.</td>
    <td>FLOAT</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>yOut</td>
    <td>Output</td>
    <td>Output of RmsNorm, which must be a 3D tensor with shape (Bs, 1, H). Its data type and format must match residualX.</td>
    <td>BFLOAT16</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>rstdOut</td>
    <td>Output</td>
    <td>Output of RmsNorm, which must be a 3D tensor with shape (Bs, 1, 1). non-contiguous tensor are supported.</td>
    <td>FLOAT32</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>xOut</td>
    <td>Output</td>
    <td>Output of Add, which must be a 3D tensor with shape (Bs, 1, H). Its data type and format must match residualX.</td>
    <td>BFLOAT16</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>workspaceSize</td>
    <td>Output</td>
    <td>Size of the workspace required to be allocated on the device.</td>
    <td>UINT64</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>executor</td>
    <td>Output</td>
    <td>Operator executor, containing the operator computation process.</td>
    <td>aclOpExecutor*</td>
    <td>ND</td>
    </tr>
    </tbody>
    </table>

- **Returns**

    `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

    The first-phase API implements input parameter verification. The following errors may be thrown.

    <table style="undefined;table-layout: fixed; width: 1149px"><colgroup>
    <col style="width: 281px">
    <col style="width: 119px">
    <col style="width: 749px">
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

## aclnnMoeDistributeCombineAddRmsNorm

- **Parameters**

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
    <td>Address of the workspace to be allocated on the device.</td>
    </tr>
    <tr>
    <td>workspaceSize</td>
    <td>Input</td>
    <td>Size of the workspace to be allocated on the device, which is obtained by calling <code>aclnnMoeDistributeCombineAddRmsNormGetWorkspaceSize</code>.</td>
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

    aclnnStatus status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

1. Deterministic computing:
     - `aclnnMoeDistributeCombineAddRmsNorm` defaults to a deterministic implementation.

2. `aclnnMoeDistributeDispatchV2` and `aclnnMoeDistributeCombineAddRmsNorm` must be used together. For details, see the example.

3. The values of `groupEp`, `epWorldSize`, `moeExpertNum`, `groupTp`, `tpWorldSize`, `expertShardType`, `sharedExpertNum`, `sharedExpertRankNum`, and `globalBS` used during API calling must be the same for all ranks, at all network layers, and the same as those of `aclnnMoeDistributeDispatchV2`.

4. The shape format is described as follows:
    - **A**: Maximum number of tokens to be distributed by the current rank. The value range is as follows:
      - For shared experts, A = `Bs * epWorldSize * sharedExpertNum / sharedExpertRankNum`.
      - For MoE experts, when `globalBS` is 0, A >= `Bs * epWorldSize * min(localExpertNum, K)`. When `globalBS` is not 0, A >= globalBS * min(localExpertNum, K).
    - **H**: Hidden layer size, which must be in the range [1024, 8192].
    - **Bs**: Batch sequence size (number of tokens output by the rank), which must be in the range 0 < Bs ≤ 512.
    - **K**: Number of top K experts, which must be in the ranges 0 < K ≤ 16 and 0 < K ≤ moeExpertNum.
    - **localExpertNum**: Number of experts on the current rank.
      - For shared expert ranks, localExpertNum = 1.
      - For MoE expert ranks, localExpertNum = moeExpertNum/(epWorldSize - sharedExpertRankNum). If localExpertNum > 1, TP domain communication is not supported.

5. **HCCL_BUFFSIZE**:
   Before calling this API, check whether the value of the `HCCL_BUFFSIZE` environment variable is proper. This environment variable indicates the buffer size occupied by a single communication domain, in MB. If this environment variable is not set, the default value 200 MB is used.
    - Within an EP communication domain: The value must be greater than or equal to 2 and satisfy `(1024^2 * (HCCL_BUFFSIZE - 2) / 2 ≥ Bs * 2 * (H + 128) * (epWorldSize * localExpertNum + K + 1))`. `localExpertNum` must be the number of experts assigned to the current rank when using MoE.
    - Within a TP communication domain: The value must satisfy \>= (A \* Align512(Align32(h \* 2) + 44) + A \* Align512(h \* 2)) \* 2.

6. Constraints on the use of communication domains:
   - `aclnnMoeDistributeCombineAddRmsNorm` and `aclnnMoeDistributeDispatchV2` in a model support only the same EP communication domain, and no other operators are allowed in the communication domain.
   - `aclnnMoeDistributeCombineAddRmsNorm` and `aclnnMoeDistributeDispatchV2` in a model support only the same TP communication domain or both do not support a TP communication domain. If a TP communication domain is supported, no other operators are allowed in the communication domain.
   - <term>Atlas A3 training products / Atlas A3 inference products</term>: Nodes in a communication domain must be in the same SuperPoD. Cross-SuperPoD nodes are not supported.

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

- <term>Atlas A3 training products / Atlas A3 inference products</term>:

    ```Cpp
    #include <thread>
    #include <iostream>
    #include <string>
    #include <vector>
    #include "acl/acl.h"
    #include "hccl/hccl.h"
    #include "aclnnop/aclnn_moe_distribute_dispatch_v2.h"
    #include "aclnnop/aclnn_moe_distribute_combine_add_rms_norm.h"

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
        uint32_t epRankId;
        uint32_t tpRankId;
        HcclComm hcclEpComm;
        HcclComm hcclTpComm;
        aclrtStream dispatchStream;
        aclrtStream combineStream;
        aclrtContext context;
    };

    constexpr uint32_t EP_WORLD_SIZE = 8;
    constexpr uint32_t TP_WORLD_SIZE = 2;
    constexpr uint32_t DEV_NUM = EP_WORLD_SIZE * TP_WORLD_SIZE;

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
            strides[i] = shape[i + 1] * strides[i + 1];
        }
        *tensor = aclCreateTensor(
            shape.data(), shape.size(), dataType, strides.data(), 0,
            aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), *deviceAddr);
        return 0;
    }

    int LaunchOneProcessDispatchAndCombine(Args &args)
    {
        int ret = aclrtSetCurrentContext(args.context);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetCurrentContext failed, ret %d\n", ret); return ret);

        char hcomEpName[128] = {0};
        ret = HcclGetCommName(args.hcclEpComm, hcomEpName);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclGetEpCommName failed, ret %d\n", ret); return -1);
        char hcomTpName[128] = {0};
        ret = HcclGetCommName(args.hcclTpComm, hcomTpName);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclGetTpCommName failed, ret %d\n", ret); return -1);
        LOG_PRINT(
            "[INFO] rank = %d, hcomEpName = %s, hcomTpName = %s, dispatchStream = %p, combineStream = %p, context = %p\n",
            args.rankId, hcomEpName, hcomTpName, args.dispatchStream, args.combineStream, args.context
        );

        int64_t BS = 8;
        int64_t H = 7168;
        int64_t K = 3;
        int64_t expertShardType = 0;
        int64_t sharedExpertNum = 0;
        int64_t sharedExpertRankNum = 0;
        int64_t moeExpertNum = 8;
        int64_t quantMode = 0;
        int64_t globalBS = BS * EP_WORLD_SIZE;
        int64_t expertTokenNumsType = 1;
        int64_t outDtype = 0;
        int64_t commQuantMode = 0;
        int64_t groupList_type = 1;
        int64_t localExpertNum;
        int64_t A;
        if (args.epRankId < sharedExpertRankNum) {
            // Shared expert ranks
            localExpertNum = 1;
            A = globalBS / sharedExpertRankNum;
        } else {
            // MoE expert ranks
            localExpertNum = moeExpertNum / (EP_WORLD_SIZE - sharedExpertRankNum);
            A = globalBS * (localExpertNum < K ? localExpertNum : K);
        }

        /* Construct the input and output variables on the device based on the current scenario. */
        // Declare the input and output variables on the device.
        void *xDeviceAddr = nullptr;
        void *expertIdsDeviceAddr = nullptr;
        void *scalesDeviceAddr = nullptr;
        void *expertScalesDeviceAddr = nullptr;
        void *expandXDeviceAddr = nullptr;
        void *dynamicScalesDeviceAddr = nullptr;
        void *expandIdxDeviceAddr = nullptr;
        void *expertTokenNumsDeviceAddr = nullptr;
        void *epRecvCountsDeviceAddr = nullptr;
        void *tpRecvCountsDeviceAddr = nullptr;
        void *expandScalesDeviceAddr = nullptr;
        void *residualXDeviceAddr = nullptr;
        void *sharedExpertXDeviceAddr = nullptr;
        void *gammaDeviceAddr = nullptr;
        void *yOutDeviceAddr = nullptr;
        void *rstdOutDeviceAddr = nullptr;
        void *xOutDeviceAddr = nullptr;

        aclTensor *x = nullptr;
        aclTensor *expertIds = nullptr;
        aclTensor *scales = nullptr;
        aclTensor *expertScales = nullptr;
        aclTensor *expandX = nullptr;
        aclTensor *dynamicScales = nullptr;
        aclTensor *expandIdx = nullptr;
        aclTensor *expertTokenNums = nullptr;
        aclTensor *epRecvCounts = nullptr;
        aclTensor *tpRecvCounts = nullptr;
        aclTensor *expandScales = nullptr;
        aclTensor *residualX = nullptr;
        aclTensor *sharedExpertX = nullptr;
        aclTensor *gamma = nullptr;
        aclTensor *yOut = nullptr;
        aclTensor *rstdOut = nullptr;
        aclTensor *xOut = nullptr;

        // Define the dimensions of variables in the current scenario.
        std::vector<int64_t> xShape{BS, H};
        std::vector<int64_t> expertIdsShape{BS, K};
        std::vector<int64_t> scalesShape{(sharedExpertRankNum > 0) ? moeExpertNum + 1 : moeExpertNum, H};
        std::vector<int64_t> expertScalesShape{BS, K};
        std::vector<int64_t> expandXShape{TP_WORLD_SIZE * A, H};
        std::vector<int64_t> dynamicScalesShape{TP_WORLD_SIZE * A};
        std::vector<int64_t> expandIdxShape{A * 128};
        std::vector<int64_t> expertTokenNumsShape{localExpertNum};
        std::vector<int64_t> epRecvCountsShape{TP_WORLD_SIZE * localExpertNum * EP_WORLD_SIZE};
        std::vector<int64_t> tpRecvCountsShape{TP_WORLD_SIZE};
        std::vector<int64_t> expandScalesShape{A};
        std::vector<int64_t> residualXShape{BS, 1, H};
        std::vector<int64_t> sharedExpertXShape{BS, 1, H};
        std::vector<int64_t> gammaShape{H};
        std::vector<int64_t> yOutShape{BS, 1, H};
        std::vector<int64_t> rstdOutShape{BS, 1, 1};
        std::vector<int64_t> xOutShape{BS, 1, H};

        int64_t xShapeSize = GetShapeSize(xShape);
        int64_t expertIdsShapeSize = GetShapeSize(expertIdsShape);
        int64_t scalesShapeSize = GetShapeSize(scalesShape);
        int64_t expertScalesShapeSize = GetShapeSize(expertScalesShape);
        int64_t expandXShapeSize = GetShapeSize(expandXShape);
        int64_t dynamicScalesShapeSize = GetShapeSize(dynamicScalesShape);
        int64_t expandIdxShapeSize = GetShapeSize(expandIdxShape);
        int64_t expertTokenNumsShapeSize = GetShapeSize(expertTokenNumsShape);
        int64_t epRecvCountsShapeSize = GetShapeSize(epRecvCountsShape);
        int64_t tpRecvCountsShapeSize = GetShapeSize(tpRecvCountsShape);
        int64_t expandScalesShapeSize = GetShapeSize(expandScalesShape);
        int64_t residualXShapeSize = GetShapeSize(residualXShape);
        int64_t sharedExpertXShapeSize = GetShapeSize(sharedExpertXShape);
        int64_t gammaShapeSize = GetShapeSize(gammaShape);
        int64_t yOutShapeSize = GetShapeSize(yOutShape);
        int64_t rstdOutShapeSize = GetShapeSize(rstdOutShape);
        int64_t xOutShapeSize = GetShapeSize(xOutShape);

        // Construct variables on the host.
        std::vector<int16_t> xHostData(xShapeSize, 1);
        std::vector<int32_t> expertIdsHostData;
        for (int32_t token_id = 0; token_id < expertIdsShape[0]; token_id++) {
            // Each token is sent to the MoE experts {0, 1, ... k - 1}.
            for (int32_t k_id = 0; k_id < expertIdsShape[1]; k_id++) {
                expertIdsHostData.push_back(k_id);
            }
        }

        std::vector<float> scalesHostData(scalesShapeSize, 0.1);
        std::vector<float> expertScalesHostData(expertScalesShapeSize, 0.1);
        std::vector<int16_t> expandXHostData(expandXShapeSize, 0);
        std::vector<float> dynamicScalesHostData(dynamicScalesShapeSize, 0);
        std::vector<int32_t> expandIdxHostData(expandIdxShapeSize, 0);
        std::vector<int64_t> expertTokenNumsHostData(expertTokenNumsShapeSize, 0);
        std::vector<int32_t> epRecvCountsHostData(epRecvCountsShapeSize, 0);
        std::vector<int32_t> tpRecvCountsHostData(tpRecvCountsShapeSize, 0);
        std::vector<float> expandScalesHostData(expandScalesShapeSize, 0);
        std::vector<int16_t> residualXHostData(residualXShapeSize, 1);
        std::vector<int16_t> sharedExpertXHostData(sharedExpertXShapeSize, 1);
        std::vector<int16_t> gammaHostData(gammaShapeSize, 1);
        std::vector<int16_t> yOutHostData(yOutShapeSize, 0);
        std::vector<float> rstdOutHostData(rstdOutShapeSize, 0);
        std::vector<int16_t> xOutHostData(xOutShapeSize, 0);

        // Construct variables on the device.
        ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_BF16, &x);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(expertIdsHostData, expertIdsShape, &expertIdsDeviceAddr, aclDataType::ACL_INT32, &expertIds);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(scalesHostData, scalesShape, &scalesDeviceAddr, aclDataType::ACL_FLOAT, &scales);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(expertScalesHostData, expertScalesShape, &expertScalesDeviceAddr, aclDataType::ACL_FLOAT, &expertScales);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(expandXHostData, expandXShape, &expandXDeviceAddr, (quantMode > 0) ? aclDataType::ACL_INT8 : aclDataType::ACL_BF16, &expandX);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(dynamicScalesHostData, dynamicScalesShape, &dynamicScalesDeviceAddr, aclDataType::ACL_FLOAT, &dynamicScales);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(expandIdxHostData, expandIdxShape, &expandIdxDeviceAddr, aclDataType::ACL_INT32, &expandIdx);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(expertTokenNumsHostData, expertTokenNumsShape, &expertTokenNumsDeviceAddr, aclDataType::ACL_INT64, &expertTokenNums);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(epRecvCountsHostData, epRecvCountsShape, &epRecvCountsDeviceAddr, aclDataType::ACL_INT32, &epRecvCounts);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(tpRecvCountsHostData, tpRecvCountsShape, &tpRecvCountsDeviceAddr, aclDataType::ACL_INT32, &tpRecvCounts);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(expandScalesHostData, expandScalesShape, &expandScalesDeviceAddr, aclDataType::ACL_FLOAT, &expandScales);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(residualXHostData, residualXShape, &residualXDeviceAddr, aclDataType::ACL_BF16, &residualX);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(sharedExpertXHostData, sharedExpertXShape, &sharedExpertXDeviceAddr, aclDataType::ACL_BF16, &sharedExpertX);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(gammaHostData, gammaShape, &gammaDeviceAddr, aclDataType::ACL_BF16, &gamma);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(yOutHostData, yOutShape, &yOutDeviceAddr, aclDataType::ACL_BF16, &yOut);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(rstdOutHostData, rstdOutShape, &rstdOutDeviceAddr, aclDataType::ACL_FLOAT, &rstdOut);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(xOutHostData, xOutShape, &xOutDeviceAddr, aclDataType::ACL_BF16, &xOut);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        
        /* Declare the variables required for operator execution. */
        uint64_t dispatchWorkspaceSize = 0;
        aclOpExecutor *dispatchExecutor = nullptr;
        void *dispatchWorkspaceAddr = nullptr;

        uint64_t combineAddRmsNormWorkspaceSize = 0;
        aclOpExecutor *combineAddRmsNormExecutor = nullptr;
        void *combineWorkspaceAddr = nullptr;

        /**************************************** Call dispatch. ********************************************/

        ret = aclnnMoeDistributeDispatchV2GetWorkspaceSize(x, expertIds, (quantMode > 0 ? scales : nullptr), nullptr, 
                expertScales, hcomEpName, EP_WORLD_SIZE, args.epRankId, moeExpertNum, hcomTpName, TP_WORLD_SIZE,
                args.tpRankId, expertShardType, sharedExpertNum,sharedExpertRankNum, quantMode, globalBS,
                expertTokenNumsType, nullptr, expandX, dynamicScales, expandIdx, expertTokenNums, epRecvCounts,
                tpRecvCounts, expandScales, &dispatchWorkspaceSize, &dispatchExecutor);
        
        CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("[ERROR] aclnnMoeDistributeDispatchV2GetWorkspaceSize failed. ret = %d \n", ret); return ret);

        if (dispatchWorkspaceSize > 0) {
            ret = aclrtMalloc(&dispatchWorkspaceAddr, dispatchWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc workspace failed. ret = %d \n", ret); return ret);
        }
        // Call the second-phase API.
        ret = aclnnMoeDistributeDispatchV2(dispatchWorkspaceAddr, dispatchWorkspaceSize,
                                            dispatchExecutor, args.dispatchStream);
        ret = aclrtSynchronizeStreamWithTimeout(args.dispatchStream, 10000);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnMoeDistributeDispatchV2 failed. ret = %d \n", ret);  \
            return ret);

        /**************************************** Call combineAddRmsNorm. ********************************************/
        // Call the first-phase API.
        ret = aclnnMoeDistributeCombineAddRmsNormGetWorkspaceSize(
            expandX, expertIds, expandIdx, epRecvCounts, expertScales, residualX, gamma, tpRecvCounts, nullptr, nullptr,
            nullptr, nullptr, nullptr, sharedExpertX, hcomEpName, EP_WORLD_SIZE, args.epRankId, moeExpertNum, hcomTpName, TP_WORLD_SIZE,
            args.tpRankId, expertShardType, sharedExpertNum, sharedExpertRankNum, globalBS, outDtype, commQuantMode,
            groupList_type, nullptr, 1e-6, yOut, rstdOut, xOut, &combineAddRmsNormWorkspaceSize, &combineAddRmsNormExecutor);
        CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("[ERROR] aclnnMoeDistributeCombineAddRmsNormGetWorkspaceSize failed. ret = %d \n", ret); return ret);
        // Allocate device memory based on the computed workspaceSize.
        if (combineAddRmsNormWorkspaceSize > 0) {
            ret = aclrtMalloc(&combineWorkspaceAddr, combineAddRmsNormWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc workspace failed. ret = %d \n", ret); return ret);
        }

        // Call the second-phase API.
        ret = aclnnMoeDistributeCombineAddRmsNorm(combineWorkspaceAddr, combineAddRmsNormWorkspaceSize, combineAddRmsNormExecutor, args.combineStream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnMoeDistributeCombineAddRmsNorm failed. ret = %d \n", ret);
            return ret);
        // (Fixed writing) Wait until the task execution is complete.
        ret = aclrtSynchronizeStreamWithTimeout(args.combineStream, 10000);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSynchronizeStreamWithTimeout failed. ret = %d \n", ret);
            return ret);
        LOG_PRINT("[INFO] device_%d aclnnMoeDistributeDispatchV2 and aclnnMoeDistributeCombineAddRmsNorm                      \
                    execute successfully.\n", args.rankId);

        // Release device resources.
        if (dispatchWorkspaceSize > 0) {
            aclrtFree(dispatchWorkspaceAddr);
        }
        if (combineAddRmsNormWorkspaceSize > 0) {
            aclrtFree(combineWorkspaceAddr);
        }
        if (x != nullptr) {
            aclDestroyTensor(x);
        }
        if (expertIds != nullptr) {
            aclDestroyTensor(expertIds);
        }
        if (scales != nullptr) {
            aclDestroyTensor(scales);
        }
        if (expertScales != nullptr) {
            aclDestroyTensor(expertScales);
        }
        if (expandX != nullptr) {
            aclDestroyTensor(expandX);
        }
        if (dynamicScales != nullptr) {
            aclDestroyTensor(dynamicScales);
        }
        if (expandIdx != nullptr) {
            aclDestroyTensor(expandIdx);
        }
        if (expertTokenNums != nullptr) {
            aclDestroyTensor(expertTokenNums);
        }
        if (epRecvCounts != nullptr) {
            aclDestroyTensor(epRecvCounts);
        }
        if (tpRecvCounts != nullptr) {
            aclDestroyTensor(tpRecvCounts);
        }
        if (expandScales != nullptr) {
            aclDestroyTensor(expandScales);
        }
        if (residualX != nullptr) {
            aclDestroyTensor(residualX);
        }
        if (sharedExpertX != nullptr) {
            aclDestroyTensor(sharedExpertX);
        }
        if (gamma != nullptr) {
            aclDestroyTensor(gamma);
        }
        if (yOut != nullptr) {
            aclDestroyTensor(yOut);
        }
        if (rstdOut != nullptr) {
            aclDestroyTensor(rstdOut);
        }
        if (xOut != nullptr) {
            aclDestroyTensor(xOut);
        }
        if (xDeviceAddr != nullptr) {
            aclrtFree(xDeviceAddr);
        }
        if (expertIdsDeviceAddr != nullptr) {
            aclrtFree(expertIdsDeviceAddr);
        }
        if (scalesDeviceAddr != nullptr) {
            aclrtFree(scalesDeviceAddr);
        }
        if (expertScalesDeviceAddr != nullptr) {
            aclrtFree(expertScalesDeviceAddr);
        }
        if (expandXDeviceAddr != nullptr) {
            aclrtFree(expandXDeviceAddr);
        }
        if (dynamicScalesDeviceAddr != nullptr) {
            aclrtFree(dynamicScalesDeviceAddr);
        }
        if (expandIdxDeviceAddr != nullptr) {
            aclrtFree(expandIdxDeviceAddr);
        }
        if (expertTokenNumsDeviceAddr != nullptr) {
            aclrtFree(expertTokenNumsDeviceAddr);
        }
        if (epRecvCountsDeviceAddr != nullptr) {
            aclrtFree(epRecvCountsDeviceAddr);
        }
        if (expandScalesDeviceAddr != nullptr) {
            aclrtFree(expandScalesDeviceAddr);
        }
        if (tpRecvCountsDeviceAddr != nullptr) {
            aclrtFree(tpRecvCountsDeviceAddr);
        }
        if (residualXDeviceAddr != nullptr) {
            aclrtFree(residualXDeviceAddr);
        }
        if (sharedExpertXDeviceAddr != nullptr) {
            aclrtFree(sharedExpertXDeviceAddr);
        }
        if (gammaDeviceAddr != nullptr) {
            aclrtFree(gammaDeviceAddr);
        }
        if (yOutDeviceAddr != nullptr) {
            aclrtFree(yOutDeviceAddr);
        }
        if (rstdOutDeviceAddr != nullptr) {
            aclrtFree(rstdOutDeviceAddr);
        }
        if (xOutDeviceAddr != nullptr) {
            aclrtFree(xOutDeviceAddr);
        }
        HcclCommDestroy(args.hcclEpComm);
        HcclCommDestroy(args.hcclTpComm);
        aclrtDestroyStream(args.dispatchStream);
        aclrtDestroyStream(args.combineStream);
        aclrtDestroyContext(args.context);
        aclrtResetDevice(args.rankId);

        return 0;
    }

    int main(int argc, char *argv[])
    {
        // This sample is based on and can only run on Atlas A3.
        int ret = aclInit(nullptr);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclInit failed, ret = %d\n", ret); return ret);

        aclrtStream dispatchStream[DEV_NUM];
        aclrtStream combineStream[DEV_NUM];
        aclrtContext context[DEV_NUM];
        for (uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
            ret = aclrtSetDevice(rankId);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetDevice failed, ret = %d\n", ret); return ret);
            ret = aclrtCreateContext(&context[rankId], rankId);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateContext failed, ret = %d\n", ret); return ret);
            ret = aclrtCreateStream(&dispatchStream[rankId]);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateStream failed, ret = %d\n", ret); return ret);
            ret = aclrtCreateStream(&combineStream[rankId]);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateStream failed, ret = %d\n", ret); return ret);
        }

        int32_t devicesEp[TP_WORLD_SIZE][EP_WORLD_SIZE];
        for (int32_t tpId = 0; tpId < TP_WORLD_SIZE; tpId++) {
            for (int32_t epId = 0; epId < EP_WORLD_SIZE; epId++) {
                devicesEp[tpId][epId] = epId * TP_WORLD_SIZE + tpId;
            }
        }

        HcclComm commsEp[TP_WORLD_SIZE][EP_WORLD_SIZE];
        for (int32_t tpId = 0; tpId < TP_WORLD_SIZE; tpId++) {
            ret = HcclCommInitAll(EP_WORLD_SIZE, devicesEp[tpId], commsEp[tpId]);
            CHECK_RET(ret == ACL_SUCCESS,
                        LOG_PRINT("[ERROR] HcclCommInitAll ep %d failed, ret %d\n", tpId, ret); return ret);
        }

        int32_t devicesTp[EP_WORLD_SIZE][TP_WORLD_SIZE];
        for (int32_t epId = 0; epId < EP_WORLD_SIZE; epId++) {
            for (int32_t tpId = 0; tpId < TP_WORLD_SIZE; tpId++) {
                devicesTp[epId][tpId] = epId * TP_WORLD_SIZE + tpId;
            }
        }

        HcclComm commsTp[EP_WORLD_SIZE][TP_WORLD_SIZE];
        for (int32_t epId = 0; epId < EP_WORLD_SIZE; epId++) {
            ret = HcclCommInitAll(TP_WORLD_SIZE, devicesTp[epId], commsTp[epId]);
            CHECK_RET(ret == ACL_SUCCESS,
                        LOG_PRINT("[ERROR] HcclCommInitAll tp %d failed, ret %d\n", epId, ret); return ret);
        }

        Args args[DEV_NUM];
        // Each thread calls each rank to execute the operators.
        std::vector<std::unique_ptr<std::thread>> threads(DEV_NUM);
        for (uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
            uint32_t epRankId = rankId / TP_WORLD_SIZE;
            uint32_t tpRankId = rankId % TP_WORLD_SIZE;

            args[rankId].rankId = rankId;
            args[rankId].epRankId = epRankId;
            args[rankId].tpRankId = tpRankId;
            args[rankId].hcclEpComm = commsEp[tpRankId][epRankId];
            args[rankId].hcclTpComm = commsTp[epRankId][tpRankId];
            args[rankId].dispatchStream = dispatchStream[rankId];
            args[rankId].combineStream = combineStream[rankId];
            args[rankId].context = context[rankId];
            threads[rankId].reset(new(std::nothrow) std::thread(&LaunchOneProcessDispatchAndCombine, std::ref(args[rankId])));
        }

        for (uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
            threads[rankId]->join();
        }

        aclFinalize();
        LOG_PRINT("[INFO] aclFinalize success\n");

        return 0;
    }
    ```
