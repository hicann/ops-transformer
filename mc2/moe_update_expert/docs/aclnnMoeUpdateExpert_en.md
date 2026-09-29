# aclnnMoeUpdateExpert

[📄 View Source Code](https://gitcode.com/cann/ops-transformer/tree/master/mc2/moe_update_expert)

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| Ascend 950PR/Ascend 950DT                            |    √     |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √     |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    ×     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference products</term>                            |    ×     |
| <term>Atlas training products</term>                             |    ×     |

## Function

This API supports load balancing and expert pruning. The mapped expert table and mask can be transferred to the mixture of experts (MoE) layer for data distribution and processing.

* Load balancing: To address load imbalance, this operator can map the top K logical expert IDs to the rank IDs for each token. The computation method is as follows:

  The following describes how to compute the rank to which the *i*th token is sent.

    ```python
    place_num = 8
    F = 2
    new_expert_id = eplb_table[table_offset + 1]
    expert_id = expert_ids[i]
    table_offset = expert_id * F
    if (eplb_table[table_offset] == 1):
        new_expert_id = eplb_table[table_offset + 1]
    else:
        if (balance_mode == 0):
            mode_value = ceil(world_size / eplb_table[table_offset])
            place_idx = local_rank_id / mode_value + 1
        else:
            place_idx = i % place_num
    new_expert_id = eplb_table[table_offset + place_idx]
    ```

* Expert pruning: The top K experts to which tokens are sent can be pruned based on the threshold. The computation method is as follows:

  `active_mask` with shape `(BS,)` is broadcasted to become `active_mask_tensor` with shape `(BS,K)`, where experts corresponding to `False` in the `BS` dimension will be directly pruned. `True` elements in `active_mask_tensor` will also be pruned if they meet conditions.

    ```python
    active_mask_tensor = broadcast(active_mask, (BS, K))
    for i in range(BS):
        expert_scales_vec[:] = sum(expert_scales[i, :] * pruning_threshold[:])
        balanced_active_mask[i, :] = (expert_scales[i, :] < expert_scales_vec[:]) && active_mask_tensor[i, :]
    ```

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnMoeUpdateExpertGetWorkspaceSize` is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, `aclnnMoeUpdateExpert` is called to perform computation.

```cpp
aclnnStatus aclnnMoeUpdateExpertGetWorkspaceSize(
    const aclTensor* expertIds,
    const aclTensor* eplbTable,
    const aclTensor* expertScalesOptional,
    const aclTensor* pruningThresholdOptional,
    const aclTensor* activeMaskOptional,
    int64_t          localRankId,
    int64_t          worldSize,
    int64_t          balanceMode,
    aclTensor*       balancedExpertIds,
    aclTensor*       balancedActiveMask,
    uint64_t*        workspaceSize,
    aclOpExecutor**  executor)
```

```cpp
aclnnStatus aclnnMoeUpdateExpert(
    void*           workspace,
    uint64_t        workspaceSize,
    aclOpExecutor*  executor,
    aclrtStream     stream)
```

## aclnnMoeUpdateExpertGetWorkspaceSize

- **Parameters**

    <table style="undefined;table-layout: fixed; width: 1390px"> <colgroup>
    <col style="width: 150px">
    <col style="width: 160px">
    <col style="width: 200px">
    <col style="width: 200px">
    <col style="width: 130px">
    <col style="width: 180px">
    <col style="width: 180px">
    <col style="width: 190px">
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
    <td>expertIds(aclTensor*)</td>
    <td>Input</td>
    <td>Top K expert indexes of each token.</td>
    <td>The value must be a 2D tensor with shape (BS, K).</td>
    <td>INT32, INT64</td>
    <td>ND</td>
    <td>2</td>
    <td>×</td>
    </tr>
    <tr>
    <td>eplbTable(aclTensor*)</td>
    <td>Input</td>
    <td>Mapping table from logical experts to physical experts (ensure that the input is correct externally):</td>
    <td><ul><li>world_size * place_per_rank expert instances in total. </li><li>The first column of each row indicates the number of deployed instances of the corresponding logic expert (value range: [1, world_size]), and the last [1, count] column indicates the instance ID (value range: [0, world_size*place_per_rank), which must be unique). </li><li>The value must be a 2D Tensor with the shape of (moeExperNum, F).</li></ul></td>
    <td>INT32</td>
    <td>ND</td>
    <td>2</td>
    <td>×</td>
    </tr>
    <tr>
    <td>expertScalesOptional (aclTensor*)</td>
    <td>Input</td>
    <td>Scale weight of top K experts for each token.</td>
    <td><ul><li>Ensure that tokens are sorted in descending order. </li><li>Valid data or a null pointer can be transferred. When valid data is transferred, pruningThresholdOptional must also be transferred. </li><li>The value must be a 2D tensor with the shape of (BS, K).</li></ul></td>
    <td>FLOAT16, BFLOAT16, FLOAT</td>
    <td>ND</td>
    <td>2</td>
    <td>×</td>
    </tr>
    <tr>
    <td>pruningThresholdOptional (aclTensor*)</td>
    <td>Input</td>
    <td>Minimum threshold of the expert scale weight. (If the expert scale corresponding to the token is less than the threshold, the expert is pruned.)</td>
    <td><ul><li>Valid data or a null pointer can be transferred. When valid data is transferred, expertScalesOptional must also be transferred. </li><li>The value must be a 1D or 2D tensor with the shape of (K,) or (1, K).</li></ul></td>
    <td>FLOAT</td>
    <td>ND</td>
    <td>1/2</td>
    <td>×</td>
    </tr>
    <tr>
    <td>activeMaskOptional (aclTensor*)</td>
    <td>Input</td>
    <td>Indicates whether the tokens participate in communication.</td>
    <td><ul><li> Valid data or a null pointer can be transferred. When valid data is transferred, both expertScalesOptional and pruningThresholdOptional must be transferred. </li><li>true indicates that the communication is involved and true must be placed before false. For example, {true, false, true} is invalid. </li><li>When a null pointer is passed, all tokens are used for communication by default. </li><li>The value must be a 1D tensor with the shape of (BS,).</li></ul></td>
    <td>BOOL</td>
    <td>ND</td>
    <td>1</td>
    <td>×</td>
    </tr>
    <tr>
    <td>localRankId (int64_t)</td>
    <td>Input</td>
    <td>ID of the card.</td>
    <td><ul><li>When balanceMode is set to 0, the value range is [0, worldSize). </li><li>The localRankId of each card in the same communicator is unique.</li></ul></td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>worldSize (int64_t)</td>
    <td>Input</td>
    <td>Communication domain size.</td>
    <td>When balanceMode is set to 0, the value range is [2, 768].</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>balanceMode (int64_t)</td>
    <td>Input</td>
    <td>Balancing rule. The default value is <code>0</code>.</td>
    <td><ul><li>0: by rank </li><li>1: by token </li><li>The value range is [0, 1].</li></ul></td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>balancedExpertIds (aclTensor*)</td>
    <td>Output</td>
    <td>Instance ID of the physical expert to which the top K experts of each token after mapping belong.</td>
    <td><ul><li>The value must be a 2D tensor with the shape of (BS, K). </li><li>The data type and format must be the same as those of expertIds.</li></ul></td>
    <td>INT32, INT64</td>
    <td>ND</td>
    <td>2</td>
    <td>×</td>
    </tr>
    <tr>
    <td>balancedActiveMask (aclTensor*)</td>
    <td>Output</td>
    <td>Balanced <code>activeMask</code> after pruning. This parameter is valid only when <code>expertScalesOptional</code> and <code>pruningThresholdOptional</code> are passed with valid data.</td>
    <td>The value must be a 2D tensor with the shape of (BS, K).</td>
    <td>BOOL</td>
    <td>ND</td>
    <td>2</td>
    <td>×</td>
    </tr>
    <tr>
    <td>workspaceSize (uint64_t*)</td>
    <td>Output</td>
    <td>Size of the workspace required to be allocated on the device.</td>
    <td>-</td>
    <td>UINT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>executor(aclOpExecutor**)</td>
    <td>Output</td>
    <td>Operator executor, containing the operator computation process.</td>
    <td>-</td>
    <td>aclOpExecutor*</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    </tbody>
    </table>

- **Returns**

    `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md). 

    The first-phase API implements input parameter verification. The following errors may be thrown.

    <table style="undefined;table-layout: fixed; width: 1149px"><colgroup>
    <col style="width: 282px">
    <col style="width: 120px">
    <col style="width: 747px">
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
    <tr>
    <td class="merged-cell" rowspan="2">ACLNN_ERR_INNER_TILING_ERROR</td>
    <td class="merged-cell" rowspan="2">561002</td>
    <td>1. The input and output shapes are not supported.</td>
    </tr>
    <tr>
    <td>2. The parameter value is not supported.</td>
    </tr>
    </tbody>
    </table>

## aclnnMoeUpdateExpert

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
    <td>Size of the workspace to be allocated on the device, which is obtained by calling the first-phase API <code>aclnnMoeUpdateExpertGetWorkspaceSize</code>.</td>
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

    `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

1. Deterministic computation:
     - `aclnnMoeUpdateExpert` defaults to a deterministic implementation.

2. **API mapping and calling sequence:** 
    This API must be used together either with `aclnnMoeDistributeDispatchV2` and `aclnnMoeDistributeCombineV2`/`aclnnMoeDistributeCombineAddRmsNorm`, with **a fixed calling sequence** 
    (`aclnnMoeUpdateExpert` → `aclnnMoeDistributeDispatchV2` → `aclnnMoeDistributeCombineV2`/`aclnnMoeDistributeCombineAddRmsNorm`);

    Or with `aclnnMoeDistributeDispatchV3` and `aclnnMoeDistributeCombineV3`/`aclnnMoeDistributeCombineAddRmsNormV2`, with **a fixed calling sequence** 
    (`aclnnMoeUpdateExpert` → `aclnnMoeDistributeDispatchV3` → `aclnnMoeDistributeCombineV3`/`aclnnMoeDistributeCombineAddRmsNormV2`). For details, see [Example](#example).

3. **Parameter consistency requirements:** 
   The values of the `worldSize` and `moeExpertNum` parameters used during the calling must be consistent for all ranks and at different network layers, and must be consistent with the corresponding parameters of `aclnnMoeDistributeDispatchV2` and `aclnnMoeDistributeCombineV2`/`aclnnMoeDistributeCombineAddRmsNorm`.

4. **Hardware-related definitions:** 
   <term>Atlas A3 training products/Atlas A3 inference products</term>: In this scenario, a single rank contains dual dies. Therefore, the "rank" in the parameter description indicates **a single die**.

5. **Restriction on the shape format:**
   - `BS`: Number of tokens output by the rank, which must be in the range (0, 512].
   - `K`: Number of selected top experts, which must be in the ranges (0, 16] and (0, `moeExpertNum`].
   - `moeExpertNum`: Number of MoE experts, which must be in the range (0, 1024].
   - `F`: Number of columns in the mapping table `eplbTable`. The value range is [2, `worldSize` + 1]. The first column indicates the number of deployed instances of logical experts (value > 0), and the following `F` – 1 columns indicate the corresponding rank IDs.
   - **Restriction on the total number of instances**: The total number of MoE expert instances deployed on all ranks cannot exceed 1024. That is, `place_per_rank*world_size ≤ 1024` (`place_per_rank` indicates the number of instances deployed on a single rank).
   - **Consistency of the number of instances per rank**: The number of expert instances deployed on each rank must be the same.

## Example

The following uses the <term>Atlas A3 training series products/Atlas A3 inference series products</term> and Ascend 950PR/Ascend 950DTas examples to describe how to invoke the MoeUpdateExpert, MoeDistributeDispatchV2, and MoeDistributeCombineAddRmsNorm operators.

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

    ```Cpp
    #include <thread>
    #include <iostream>
    #include <string>
    #include <cstring>
    #include <vector>
    #include "acl/acl.h"
    #include "hccl/hccl.h"
    #include "aclnnop/aclnn_moe_update_expert.h"
    #include "aclnnop/aclnn_moe_distribute_dispatch_v2.h"
    #include "aclnnop/aclnn_moe_distribute_combine_v2.h"

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
        aclrtStream eplbStream;
        aclrtStream dispatchStream;
        aclrtStream combineStream;
        aclrtContext context;
    };

    constexpr uint32_t EP_WORLD_SIZE = 2;
    constexpr uint32_t TP_WORLD_SIZE = 1;
    constexpr uint32_t DEV_NUM = EP_WORLD_SIZE * TP_WORLD_SIZE;

    inline int64_t GetShapeSize(const std::vector<int64_t> &shape)
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

    int LaunchOneProcessUpdateExpertAndDispatchAndCombine(Args &args)
    {
        int ret = aclrtSetCurrentContext(args.context);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetCurrentContext failed, ret %d\n", ret); return ret);

        char hcomEpName[128] = {0};
        ret = HcclGetCommName(args.hcclEpComm, hcomEpName);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclGetEpCommName failed, ret %d\n", ret); return -1);
        char hcomTpName[128] = {0};

        int64_t BS = 8;
        int64_t H = 7168;
        int64_t K = 2;
        int64_t F = 2;
        int64_t expertShardType = 0;
        int64_t sharedExpertNum = 0;
        int64_t sharedExpertRankNum = 0;
        int64_t moeExpertNum = 2;
        int64_t quantMode = 0;
        int64_t globalBS = BS * EP_WORLD_SIZE;
        int64_t balanceMode = 0;
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
        void *eplbTableDeviceAddr = nullptr;
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
        void *balancedExpertIdsDeviceAddr = nullptr;
        void *balancedActiveMaskDeviceAddr = nullptr;

        aclTensor *x = nullptr;
        aclTensor *expertIds = nullptr;
        aclTensor *eplbTable = nullptr;
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
        aclTensor *balancedExpertIds = nullptr;
        aclTensor *balancedActiveMask = nullptr;

        // Define the dimensions of variables in the current scenario.
        std::vector<int64_t> xShape{BS, H};
        std::vector<int64_t> expertIdsShape{BS, K};
        std::vector<int64_t> eplbTableShape{moeExpertNum, F};
        std::vector<int64_t> scalesShape{(sharedExpertRankNum > 0) ? moeExpertNum + 1 : moeExpertNum, H};
        std::vector<int64_t> expertScalesShape{BS, K};
        std::vector<int64_t> expandXShape{TP_WORLD_SIZE * A, H};
        std::vector<int64_t> dynamicScalesShape{TP_WORLD_SIZE * A};
        std::vector<int64_t> expandIdxShape{A * 128};
        std::vector<int64_t> expertTokenNumsShape{localExpertNum};
        std::vector<int64_t> epRecvCountsShape{TP_WORLD_SIZE * localExpertNum * EP_WORLD_SIZE};
        std::vector<int64_t> tpRecvCountsShape{TP_WORLD_SIZE * localExpertNum};
        std::vector<int64_t> expandScalesShape{A};
        std::vector<int64_t> residualXShape{BS, 1, H};
        std::vector<int64_t> sharedExpertXShape{BS, 1, H};
        std::vector<int64_t> gammaShape{H, };
        std::vector<int64_t> yOutShape{BS, 1, H};
        std::vector<int64_t> rstdOutShape{BS, 1, 1};
        std::vector<int64_t> xOutShape{BS, 1, H};
        std::vector<int64_t> balancedExpertIdsShape{BS, K};
        std::vector<int64_t> balancedActiveMaskShape{BS, K};

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
        int64_t balancedExpertIdsShapeSize = GetShapeSize(balancedExpertIdsShape);
        int64_t balancedActiveMaskShapeSize = GetShapeSize(balancedActiveMaskShape);

        // Construct variables on the host.
        std::vector<int16_t> xHostData(xShapeSize, 1);
        std::vector<int32_t> expertIdsHostData;
        for (int32_t token_id = 0; token_id < expertIdsShape[0]; token_id++) {
            // Each token is sent to the MoE experts {0, 1, ... k - 1}.
            for (int32_t k_id = 0; k_id < expertIdsShape[1]; k_id++) {
                expertIdsHostData.push_back(k_id);
            }
        }
        // Construct eplb_table data. There are eight MoE experts, and each expert has one instance. Each rank is deployed with one instance. For example, the first two numbers 1 and 0 indicate that one instance is deployed for the first MoE expert at place0.
        std::vector<int32_t> eplbTableHostData = {1, 0, 1, 1, 1, 2, 1, 3, 1, 4, 1, 5, 1, 6, 1, 7};

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
        std::vector<int32_t> balancedExpertIdsHostData(balancedExpertIdsShapeSize, 0);
        std::vector<int32_t> balancedActiveMaskHostData(balancedActiveMaskShapeSize, 0);

        // Construct variables on the device.
        ret = CreateAclTensor(expertIdsHostData, expertIdsShape, &expertIdsDeviceAddr, aclDataType::ACL_INT32, &expertIds);
        ret = CreateAclTensor(eplbTableHostData, eplbTableShape, &eplbTableDeviceAddr, aclDataType::ACL_INT32, &eplbTable);
        ret = CreateAclTensor(balancedExpertIdsHostData, balancedExpertIdsShape, &balancedExpertIdsDeviceAddr, aclDataType::ACL_INT32, &balancedExpertIds);
        ret = CreateAclTensor(balancedActiveMaskHostData, balancedActiveMaskShape, &balancedActiveMaskDeviceAddr, aclDataType::ACL_BOOL, &balancedActiveMask);
        ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_BF16, &x);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
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
        uint64_t eplbworkspaceSize = 0;
        aclOpExecutor *eplbexecutor = nullptr;
        void *eplbWorkspaceAddr = nullptr;

        uint64_t dispatchWorkspaceSize = 0;
        aclOpExecutor *dispatchExecutor = nullptr;
        void *dispatchWorkspaceAddr = nullptr;

        uint64_t combineWorkspaceSize = 0;
        aclOpExecutor *combineExecutor = nullptr;
        void *combineWorkspaceAddr = nullptr;

        /**************************************** Call eplb. ********************************************/
        ret = aclnnMoeUpdateExpertGetWorkspaceSize(expertIds, eplbTable, nullptr, nullptr, nullptr, args.epRankId,
            EP_WORLD_SIZE, balanceMode, balancedExpertIds, balancedActiveMask, &eplbworkspaceSize, &eplbexecutor);

        CHECK_RET(ret == ACL_SUCCESS,
                LOG_PRINT("[ERROR] aclnnMoeUpdateExpertGetWorkspaceSize failed. ret = %d \n", ret); return ret);

        if (eplbworkspaceSize > 0) {
            ret = aclrtMalloc(&eplbWorkspaceAddr, eplbworkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc workspace failed. ret = %d \n", ret); return ret);
        }
        // Call the second-phase API.
        ret = aclnnMoeUpdateExpert(eplbWorkspaceAddr, eplbworkspaceSize, eplbexecutor, args.eplbStream);
        ret = aclrtSynchronizeStreamWithTimeout(args.eplbStream, 10000);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnMoeUpdateExpert failed. ret = %d \n", ret);  \
            return ret);

        /**************************************** Call dispatch. ********************************************/

        ret = aclnnMoeDistributeDispatchV2GetWorkspaceSize(x, balancedExpertIds, (quantMode > 0 ? scales : nullptr), nullptr, 
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
        
        /**************************************** Call combine. ********************************************/
        
        // Call the first-phase API of combine.
        ret = aclnnMoeDistributeCombineV2GetWorkspaceSize(expandX, expertIds, expandIdx, epRecvCounts, expertScales,
            tpRecvCounts, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
            hcomEpName, EP_WORLD_SIZE, args.epRankId, moeExpertNum, hcomTpName, TP_WORLD_SIZE, args.tpRankId,
            expertShardType, sharedExpertNum, sharedExpertRankNum, globalBS, outDtype, commQuantMode,
            groupList_type, nullptr, x, &combineWorkspaceSize, &combineExecutor);
        CHECK_RET(
            ret == ACL_SUCCESS,
            LOG_PRINT("[ERROR] aclnnMoeDistributeCombineV2GetWorkspaceSize failed. ret = %d \n", ret); return ret
        );
        // Allocate device memory based on the workspaceSize computed by the first-phase API of combine.
        if (combineWorkspaceSize > 0) {
            ret = aclrtMalloc(&combineWorkspaceAddr, combineWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc failed. ret = %d \n", ret);
                    return ret);
        }

        // Call the second-phase API of combine.
        ret = aclnnMoeDistributeCombineV2(combineWorkspaceAddr, combineWorkspaceSize, combineExecutor, args.combineStream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnMoeDistributeCombineV2 failed. ret = %d \n", ret);
                return ret);

        LOG_PRINT("[INFO] device_%d aclnnMoeUpdateExpert, aclnnMoeDistributeDispatchV2 and aclnnMoeDistributeCombineV2    \
                    execute successfully.\n", args.rankId);

        // Release device resources.
        if (dispatchWorkspaceSize > 0) {
            aclrtFree(dispatchWorkspaceAddr);
        }
        if (combineWorkspaceSize > 0) {
            aclrtFree(combineWorkspaceAddr);
        }
        if (x != nullptr) {
            aclDestroyTensor(x);
        }
        if (expertIds != nullptr) {
            aclDestroyTensor(expertIds);
        }
        if (eplbTable != nullptr) {
            aclDestroyTensor(eplbTable);
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
        if (balancedExpertIds != nullptr) {
            aclDestroyTensor(balancedExpertIds);
        }
        if (balancedActiveMask != nullptr) {
            aclDestroyTensor(balancedActiveMask);
        }
        if (xDeviceAddr != nullptr) {
            aclrtFree(xDeviceAddr);
        }
        if (expertIdsDeviceAddr != nullptr) {
            aclrtFree(expertIdsDeviceAddr);
        }
        if (eplbTableDeviceAddr != nullptr) {
            aclrtFree(eplbTableDeviceAddr);
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
        if (balancedExpertIdsDeviceAddr != nullptr) {
            aclrtFree(balancedExpertIdsDeviceAddr);
        }
        if (balancedActiveMaskDeviceAddr != nullptr) {
            aclrtFree(balancedActiveMaskDeviceAddr);
        }
        HcclCommDestroy(args.hcclEpComm);
        HcclCommDestroy(args.hcclTpComm);
        aclrtDestroyStream(args.eplbStream);
        aclrtDestroyStream(args.dispatchStream);
        aclrtDestroyStream(args.combineStream);
        aclrtDestroyContext(args.context);
        aclrtResetDevice(args.rankId);

        return 0;
    }

    int main(int argc, char *argv[])
    {
        int ret = aclInit(nullptr);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclInit failed, ret = %d\n", ret); return ret);

        aclrtStream eplbStream[DEV_NUM];
        aclrtStream dispatchStream[DEV_NUM];
        aclrtStream combineStream[DEV_NUM];
        aclrtContext context[DEV_NUM];
        for (uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
            ret = aclrtSetDevice(rankId);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetDevice failed, ret = %d\n", ret); return ret);
            ret = aclrtCreateContext(&context[rankId], rankId);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateContext failed, ret = %d\n", ret); return ret);
            ret = aclrtCreateStream(&eplbStream[rankId]);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateStream failed, ret = %d\n", ret); return ret);
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
        // Each thread calls a rank to execute the operator.
        std::vector<std::unique_ptr<std::thread>> threads(DEV_NUM);
        for (uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
            uint32_t epRankId = rankId / TP_WORLD_SIZE;
            uint32_t tpRankId = rankId % TP_WORLD_SIZE;

            args[rankId].rankId = rankId;
            args[rankId].epRankId = epRankId;
            args[rankId].tpRankId = tpRankId;
            args[rankId].hcclEpComm = commsEp[tpRankId][epRankId];
            args[rankId].hcclTpComm = commsTp[epRankId][tpRankId];
            args[rankId].eplbStream = eplbStream[rankId];
            args[rankId].dispatchStream = dispatchStream[rankId];
            args[rankId].combineStream = combineStream[rankId];
            args[rankId].context = context[rankId];
            threads[rankId].reset(new(std::nothrow) std::thread(&LaunchOneProcessUpdateExpertAndDispatchAndCombine, std::ref(args[rankId])));
        }

        for (uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
            threads[rankId]->join();
        }

        aclFinalize();
        LOG_PRINT("[INFO] aclFinalize success\n");

        return 0;
    }
    ```
