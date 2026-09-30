# aclnnMoeDistributeDispatchV4

[📄 View Source Code](https://gitcode.com/cann/ops-transformer/tree/master/mc2/moe_distribute_dispatch_v4)

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √     |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference products</term>                            |    ×     |
| <term>Atlas training products</term>                             |    ×     |

## Function

- Description: Quantizes token data (optional). When there is TP domain communication, AllToAllV communication in the EP domain is performed first, and then AllGatherV TP domain communication is performed. When there is no such communication, AllToAllV communication in the EP domain is performed.

    Compared with the `aclnnMoeDistributeDispatchV3` API, this API has the following changes:

    Added the capability to collect the communication duration, that is, record the communication time of each rank. This function is enabled by passing the `performanceInfoOptional` parameter. It is recommended that this function be used together with [DeepXTrace](https://github.com/antgroup/DeepXTrace). The communication duration per rank for each operator call is accumulated in this tensor. Clear it as needed before use.

- Formula:

    $$
    agOut = AllGatherV(X)\\
    expandXOut = AllToAllV(agOut)\\
    $$

    - <term>Atlas A2 training products/Atlas A2 inference products</term>: This API must be used together with `aclnnMoeDistributeCombineV4`.
    - <term>Atlas A3 training products/Atlas A3 inference products</term>: This API must be used together with `aclnnMoeDistributeCombineV4` or `aclnnMoeDistributeCombineAddRmsNormV2`.

  Note: `aclnnMoeDistributeCombineV4` and `aclnnMoeDistributeCombineAddRmsNormV2` operators are collectively referred to as **CombineV4 series operators** in subsequent documents.

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnMoeDistributeDispatchV4GetWorkspaceSize` is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, `aclnnMoeDistributeDispatchV4` is called to perform computation.

```cpp
aclnnStatus aclnnMoeDistributeDispatchV4GetWorkspaceSize(
    const aclTensor* x,
    const aclTensor* expertIds,
    const aclTensor* scalesOptional,
    const aclTensor* xActiveMaskOptional,
    const aclTensor* expertScalesOptional,
    const aclTensor* elasticInfoOptional,
    const aclTensor* performanceInfoOptional,
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
    int64_t          quantMode,
    int64_t          globalBs,
    int64_t          expertTokenNumsType,
    const char*      commAlg,
    int64_t          zeroExpertNum,
    int64_t          copyExpertNum,
    int64_t          constExpertNum,
    aclTensor*       expandXOut,
    aclTensor*       dynamicScalesOut,
    aclTensor*       assistInfoForCombineOut,
    aclTensor*       expertTokenNumsOut,
    aclTensor*       epRecvCountsOut,
    aclTensor*       tpRecvCountsOut,
    aclTensor*       expandScalesOut,
    uint64_t*        workspaceSize,
    aclOpExecutor**  executor)
```

```cpp
aclnnStatus aclnnMoeDistributeDispatchV4(
    void*           workspace,
    uint64_t        workspaceSize,
    aclOpExecutor*  executor,
    aclrtStream     stream)
```

## aclnnMoeDistributeDispatchV4GetWorkspaceSize

- **Parameters**

    <table style="undefined;table-layout: fixed; width: 1567px"> <colgroup>
    <col style="width: 120px">
    <col style="width: 140px">
    <col style="width: 300px">
    <col style="width: 330px">
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
    </tr>
    </thead>
    <tbody>
    <tr>
    <td>x</td>
    <td>Input</td>
    <td>Token data sent by the current rank.</td>
    <td>2D tensor.</td>
    <td>FLOAT16, BFLOAT16</td>
    <td>ND</td>
    <td><code>(Bs, H)</code> (Bs=batch size, H=hidden size) </td>
    <td>√</td>
    </tr>
    <tr>
    <td>expertIds</td>
    <td>Input</td>
    <td>Top K expert indices of each token.</td>
    <td>2D tensor.</td>
    <td>INT32</td>
    <td>ND</td>
    <td><code>(Bs, K)</code></td>
    <td>√</td>
    </tr>
    <tr>
    <td>scalesOptional</td>
    <td>Input</td>
    <td>Quantization smoothing parameter for each expert.</td>
    <td>2D tensor,<br>in non-quantized scenarios, pass a null pointer. In dynamic quantization scenarios, either valid data or a null pointer can be passed.</td>
    <td>FLOAT32</td>
    <td>ND</td>
    <td><code>(sharedExpertNum + moeExpertNum, H)</code></td>
    <td>√</td>
    </tr>
    <tr>
    <td>xActiveMaskOptional</td>
    <td>Input</td>
    <td>Indicates whether the tokens participate in communication.</td>
    <td>You can pass valid data or a null pointer.<br>When the input is a 1D tensor, true indicates that the corresponding tokens participate in communication. true must be placed before false. For example, {true, false, true} is invalid.<br>If the input is a 2D tensor, true indicates that the expert_ids corresponding to the current tokens participate in communication. If the K BOOL values corresponding to the current tokens are all false, the current tokens do not participate in communication. By default, all tokens participate in communication. If the value of Bs on each rank is different, all tokens must be valid.</td>
    <td>BOOL</td>
    <td>ND</td>
    <td><br>When the input is a 1D tensor, the shape is (Bs, ). When the input is a 2D tensor, the shape is (Bs, K).</td>
    <td>√</td>
    </tr>
    <tr>
    <td>expertScalesOptional</td>
    <td>Input</td>
    <td>Top K expert weights of each token.</td>
    <td>-</td>
    <td>FLOAT32</td>
    <td>ND</td>
    <td><code>(Bs, K)</code></td>
    <td>√</td>
    </tr>
    <tr>
    <td>elasticInfoOptional</td>
    <td>Input</td>
    <td>Dynamic scale-in information of the EP communication domain.</td>
    <td>When some communication ranks are removed from the communication domain due to exceptions, the actual number of ranks involved in communication can be obtained from this parameter.</td>
    <td>INT32</td>
    <td>ND</td>
    <td>-</td>
    <td>√</td>
    </tr>
    <tr>
    <td>performanceInfoOptional</td>
    <td>Input</td>
    <td>Represents profiling data for communication duration across ranks. When used with the DeepXTrace tool, it enables dynamic recording of communication time for each rank.</td>
    <td>The communication duration per rank for each operator call is accumulated in this tensor. Clear it as needed before use.</td>
    <td>INT64</td>
    <td>ND</td>
    <td>-</td>
    <td>√</td>
    </tr>
    <tr>
    <td>groupEp</td>
    <td>Input</td>
    <td>Name of the EP communication domain (expert parallelism).</td>
    <td>STRING</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>epWorldSize</td>
    <td>Input</td>
    <td>Size of the EP communication domain.</td>
    <td>-</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>epRankId</td>
    <td>Input</td>
    <td>ID of the current rank in the EP communication domain.</td>
    <td>The value range is [0, epWorldSize). The epRankId of each rank in the same EP communication domain must be unique.</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>moeExpertNum</td>
    <td>Input</td>
    <td>Number of MoE experts.</td>
    <td>It must satisfy moeExpertNum % (epWorldSize - sharedExpertRankNum) = 0.</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>groupTp</td>
    <td>Input</td>
    <td>Name of the TP communication domain (tensor parallelism).</td>
    <td>-</td>
    <td>STRING</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>tpWorldSize</td>
    <td>Input</td>
    <td>Size of the TP communication domain.</td>
    <td>-</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>tpRankId</td>
    <td>Input</td>
    <td>ID of the current rank in the TP domain.</td>
    <td>-</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>expertShardType</td>
    <td>Input</td>
    <td>Distribution type of shared expert ranks.</td>
    <td>-</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>sharedExpertNum</td>
    <td>Input</td>
    <td>Number of shared experts. (A shared expert can be replicated and deployed on multiple ranks.)</td>
    <td>-</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>sharedExpertRankNum</td>
    <td>Input</td>
    <td>Number of shared expert ranks.</td>
    <td>-</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>quantMode</td>
    <td>Input</td>
    <td>Quantization mode.</td>
    <td>0: non-quantization; 2: dynamic quantization.</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>globalBs</td>
    <td>Input</td>
    <td>Global batch size in the EP domain.</td>
    <td><br> <li>If the Bs values of all ranks are the same: globalBs = Bs × epWorldSize or 0; </li> <li>If the Bs values of all ranks are different: globalBs = maxBs × epWorldSize, where maxBs is the maximum Bs value of a single device.</li></td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>expertTokenNumsType</td>
    <td>Input</td>
    <td>Semantic type of the output expertTokenNums.</td>
    <td>0: The output of expertTokenNums is the sum of the prefixes of the tokens processed by each expert. 1: The output of expertTokenNums is the number of tokens processed by each expert.</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>commAlg</td>
    <td>Input</td>
    <td>Communication affinity memory layout algorithm.</td>
    <td>-</td>
    <td>STRING</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>zeroExpertNum</td>
    <td>Input</td>
    <td>Number of zero experts.</td>
    <td>-</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>copyExpertNum</td>
    <td>Input</td>
    <td>Number of copied experts.</td>
    <td>-</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>constExpertNum</td>
    <td>Input</td>
    <td>Number of constant experts.</td>
    <td>-</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>expandXOut</td>
    <td>Output</td>
    <td>Token features extended based on expertIds.</td>
    <td>2D tensor.</td>
    <td>FLOAT16, BFLOAT16, INT8</td>
    <td>-</td>
    <td><code>(max(tpWorldSize, 1) × A, H)</code></td>
    <td>√</td>
    </tr>
    <tr>
    <td>dynamicScalesOut</td>
    <td>Output</td>
    <td>Scaling parameter in dynamic quantization scenarios.</td>
    <td>1D tensor, output only when quantMode=2.</td>
    <td>FLOAT32</td>
    <td>-</td>
    <td><code>(A,)</code></td>
    <td>√</td>
    </tr>
    <tr>
    <td>assistInfoForCombineOut</td>
    <td>Output</td>
    <td>Number of tokens sent to the same expert (corresponding to assistInfoForCombine in aclnnMoeDistributeCombineV4).</td>
    <td>1D Tensor.</td>
    <td>INT32</td>
    <td>-</td>
    <td><code>(A × 128,)</code></td>
    <td>√</td>
    </tr>
    <tr>
    <td>expertTokenNumsOut</td>
    <td>Output</td>
    <td>Number of tokens received by each expert.</td>
    <td>1D Tensor.</td>
    <td>INT64</td>
    <td>-</td>
    <td><code>(localExpertNum,)</code></td>
    <td>√</td>
    </tr>
    <tr>
    <td>epRecvCountsOut</td>
    <td>Output</td>
    <td>Number of tokens received from each rank in the EP communication domain (corresponding to epSendCounts in aclnnMoeDistributeCombineV4).</td>
    <td>1D Tensor.</td>
    <td>INT32</td>
    <td>-</td>
    <td><code>(moeExpertNum + 2 × globalBs × K × serverNum,)</code></td>
    <td>√</td>
    </tr>
    <tr>
    <td>tpRecvCountsOut</td>
    <td>Output</td>
    <td>Number of tokens received from each rank in the TP communication domain (corresponding to tpSendCounts in aclnnMoeDistributeCombineV4).</td>
    <td>-</td>
    <td>INT32</td>
    <td>-</td>
    <td>-</td>
    <td>√</td>
    </tr>
    <tr>
    <td>expandScalesOut</td>
    <td>Output</td>
    <td>Weight of the token output by the rank (`expertScalesOptional` in `aclnnMoeDistributeCombineV4`).</td>
    <td>-</td>
    <td>FLOAT32</td>
    <td>-</td>
    <td>-</td>
    <td>√</td>
    </tr>
    <tr>
    <td>workspaceSize</td>
    <td>Output</td>
    <td>Size of the workspace to be allocated on the device.</td>
    <td>-</td>
    <td>UINT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>executor</td>
    <td>Output</td>
    <td>Returns the operator executor that contains the operator computation process.</td>
    <td>-</td>
    <td>aclOpExecutor*</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    </tbody>
    </table>

    <details>

    <summary><term>Atlas A2 training products/Atlas A2 inference products</term></summary>

    - The value of `commAlg` can be `nullptr`, `""`, `"fullmesh"`, or `"hierarchy"`. It is recommended to use `"hierarchy"` with driver version 25.0.RC1.1 or later. When set to `nullptr` or `""`, the communication algorithm is selected based on HCCL environment variables (not recommended). `"fullmesh"` indicates that tokens are directly transmitted through RDMA. `"hierarchy"` indicates a two-stage communication process: intra-server communication followed by inter-server communication, which reduces cross-server data transmission.
    - `scalesOptional` must be passed as a null pointer when `commAlg` is `hierarchy`, or when HCCL_INTRA_PCIE_ENABLE=1 and HCCL_INTRA_ROCE_ENABLE=0
    - `xActiveMaskOptional` depends on the `commAlg` value: For `"fullmesh"`, it requires a 1D tensor with shape (Bs, ), where `true` must precede `false` (for example, {true, false, true} is invalid); for `"hierarchy"`, it is currently not supported and a null pointer should be passed.
    - The value of `expertScalesOptional` must be a 2D tensor with the shape of (Bs, K).
    - The value of `epWorldSize` depends on the `commAlg` value: For `"fullmesh"`, it supports 16, 32, 64, 128, 192, and 256; for `"hierarchy"`, it supports 16, 32, and 64.
    - The value of `moeExpertNum` must be in the range (0, 512] and satisfy moeExpertNum / (epWorldSize - sharedExpertRankNum) ≤ 24.
    - `groupTp` is not supported in the current version. Pass an empty string.
    - The current version does not support `tpWorldSize`, `tpRankId`, `expertShardType`, `sharedExpertNum`, and `sharedExpertRankNum`. Pass 0 for these parameters.
    - The shape of `epRecvCountsOut` is (moeExpertNum + 2globalBsK × serverNum,). (The first `moeExpertNum` elements indicate the number of received tokens, and the remaining elements indicate the `reduce` information before communication.)
    - Currently, TP domain communication is not supported.
    - `expandScalesOut` must be a 1D tensor with shape (A ).
    - `quantMode` supports 0 (non-quantization) and 2 (dynamic quantization).
    - `elasticInfoOptional` is not supported in the current version. Pass a null pointer.
    - When commAlg is `"fullmesh"`, the value of `zeroExpertNum` must be in the range [0, MAX_INT32), where MAX_INT32 = 2^31 - 1. Valid zero expert IDs must be in the range [`moeExpertNum`, `moeExpertNum + zeroExpertNum`).
    - When commAlg is `"fullmesh"`, the value of `copyExpertNum` must be in the range [0, MAX_INT32), where MAX_INT32 = 2^31 - 1. Valid copy expert IDs must be in the range [`moeExpertNum + zeroExpertNum`, `moeExpertNum + zeroExpertNum + copyExpertNum`).
    - `constExpertNum` is not supported in the current version. Pass 0.
    - You can pass valid data or a null pointer for `performanceInfoOptional`. If you pass a null pointer, the function of recording the communication duration is disabled. If you pass valid data, it must be a 1D tensor with shape (ep\_world\_size,), and its data type and data format must be int64 and ND, respectively.

    </details>

    <details>

    <summary><term>Atlas A3 training products/Atlas A3 inference products</term></summary>
    
    - `commAlg` is not supported in the current version. Pass a null pointer.
    - `xActiveMaskOptional` must be a 1D tensor with shape (Bs, ) or a 2D tensor with shape (Bs, K). If it is a 1D tensor, `true` must be placed before `false`. If it is a 2D tensor and the K values corresponding to tokens are all `false`, the tokens do not participate in communication.
    - `expertScalesOptional` is not supported in the current version. Pass a null pointer.
    - The value of `epWorldSize` must be in the range [2, 768].
    - The value of `moeExpertNum` must be in the range (0, 1024].
    - `groupTp` must be a string of length [0, 128) and cannot be the same as `groupEp`. This parameter can be left empty only when there is no TP domain communication.
    - The value of `tpWorldSize` must be in the range [0, 2]. 0 and 1 indicate no TP domain communication. 2 is required when TP domain communication is used.
    - The value of `tpRankId` must be in the range [0, 1]. `tpRankId` of each rank in the same TP domain must be unique. If TP domain communication is not used, pass 0.
    - The value of `expertShardType` must be 0, indicating that shared expert ranks are placed in front of MoE expert ranks.
    - The value of `sharedExpertNum` must be in the range [0, 4].
    - The value of `sharedExpertRankNum` must be in the range [0, epWorldSize). If the value is 0, `sharedExpertNum` is 0 or 1. If the value is not 0, `sharedExpertRankNum % sharedExpertNum` is 0.
    - The shape of `epRecvCountsOut` is (epWorldSize × max(tpWorldSize, 1) × localExpertNum, ).
    - When there is TP domain communication, `tpRecvCountsOut` is a 1D tensor with shape (tpWorldSize, ).
    - `expandScalesOut` is not supported in the current version.
    - `quantMode` supports 0 (non-quantization) and 2 (dynamic quantization).
    - You can pass valid data or a null pointer for `elasticInfoOptional`. If a null pointer is passed, the dynamic scale-in feature is disabled. If valid data is passed, it must be a 1D tensor with shape `(4 + 2 × epWorldSize, )`. The first four numbers in the tensor indicate: whether scale-in is performed, the actual number of ranks after scale-in, the number of ranks used by shared experts after scale-in, and the number of MoE experts after scale-in. The remaining 2 × `epWorldSize` indicates two rank mapping tables. After scale-in, some ranks on the current device may be removed from the EP communication domain due to failures. The mapping for the first table is Table1[epRankId]=localEpRankId or Table1[epRankId]=-1. Here, `localEpRankId` denotes the rank index in the new EP communication domain, and `-1` indicates that the rank with the corresponding `epRankId` has been removed from the communication domain. The mapping for the second table is Table2[localEpRankId] = epRankId .
    - The value of `zeroExpertNum` must be in the range [0, MAX_INT32), where MAX_INT32 = 2^31 - 1. Valid zero expert IDs must be in the range <code>[moeExpertNum, moeExpertNum + zeroExpertNum)</code>.
    - The value of `copyExpertNum` must be in the range [0, MAX_INT32), where MAX_INT32 = 2^31 - 1. Valid expert IDs must be in the range <code>[moeExpertNum + zeroExpertNum, moeExpertNum + zeroExpertNum + copyExpertNum)</code>.
    - The value of `constExpertNum` must be in the range [0, MAX_INT32), where MAX_INT32 = 2^31 - 1. Valid expert IDs must be in the range `[moeExpertNum + zeroExpertNum + copyExpertNum, moeExpertNum + zeroExpertNum + copyExpertNum + constExpertNum)`.
    - `performanceInfoOptional` is a reserved parameter, which is not supported in the current version. Pass a null pointer.

    </details>

- **Returns**:

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
    <td rowspan="2">ACLNN_ERR_INNER_TILING_ERROR</td>
    <td rowspan="2">561002</td>
    <td>The input and output shapes are not supported.</td>
    </tr>
    <tr>
        <td>The parameter value is not supported.</td>
    </tr>
    </tbody>
    </table>

## aclnnMoeDistributeDispatchV4

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
        <td>Size of the workspace to be allocated on the device, which is obtained by calling the first-phase API `aclnnMoeDistributeDispatchV4GetWorkspaceSize`.</td>
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

    `aclnnStatus` status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computing:
  - `aclnnMoeDistributeDispatchV4` defaults to a deterministic implementation.

- **API constraints**:
  - `aclnnMoeDistributeDispatchV4` and CombineV4 operators must be used together. The `assistInfoForCombineOut`, `epRecvCountsOut`, `tpRecvCountsOut`, and `expandScalesOut` outputs of `aclnnMoeDistributeDispatchV4` must be directly passed to the corresponding parameters of `aclnnMoeDistributeCombineV4`. The service logic cannot depend on the specific values of these tensors.

- **Parameter consistency constraints:**
  - The values of `groupEp`, `epWorldSize`, `moeExpertNum`, `groupTp`, `tpWorldSize`, `expertShardType`, `sharedExpertNum`, `sharedExpertRankNum`, `globalBs`, `commAlg`, and `HCCL_BUFFSIZE` must be consistent across all ranks, and be the same as the value of CombineV4.
  - The deployment information after dynamic scale-in is transferred to the operator through the `elasticInfoOptional` parameter. Other parameters do not need to be modified. The scale-in parameters take effect only when `tpWorldSize` is set to 1. After dynamic scale-in, the number of MoE experts deployed on the current rank must be the same as that before scale-in. Configurations where no MoE expert ranks remain after scale-in are not supported.

- **Product constraints**:
  - <term>Atlas A3 training products / Atlas A3 inference products</term>: In this scenario, a single rank contains dual dies. Therefore, the "rank" in the parameter description indicates a single die.
  - The dynamic scale-in feature cannot be enabled in the tensor parallelism scenario.

- **Shape variable constraints**:

  | Variable        | Definition and Value Range                                                                |
  | :----------- | :----------------------------------------------------------------------------- |
  | A            | Maximum number of tokens that need to be distributed by the current rank. The value range is as follows: <ul><li>When dynamic scale-in is disabled: <ul><li>For shared experts, the value must satisfy `A = Bs × epWorldSize × sharedExpertNum / sharedExpertRankNum`. </li> <li>For MoE experts, when `globalBs` is 0, A ≥ Bs × epWorldSize × min(localExpertNum, K). When globalBs is not 0, A ≥ globalBs × min(localExpertNum, K). </li> </ul> </li> <li> When dynamic scale-in is enabled: <ul><li>When `globalBs` is 0, A ≥ max(Bs × epWorldSize × sharedExpertNum / sharedExpertRankNum, Bs × epWorldSize × min(localExpertNum, K)); </li> <li> When `globalBs` is not 0, A ≥ max(Bs × epWorldSize × sharedExpertNum / sharedExpertRankNum, globalBS × min(localExpertNum, K)); </li> </ul> </li> </ul>|
  | H (hidden size)| Size of the hidden layer. <ul><li> <term>Atlas A2 training products / Atlas A2 inference products</term>:The value varies with `commAlg`. For `"fullmesh"`, the value is in the range (0, 7168] and is an integer multiple of 32. For `"hierarchy"` and the driver version 25.0.RC1.1 or later, the value is in the range (0, 10 × 1024] and is an integer multiple of 32.</li><li><term>Atlas A3 training products / Atlas A3 inference products</term>: [1024, 8192].</li></ul> |
  | Bs           | Number of tokens output by the current rank. <ul><li> <term>Atlas A2 training products / Atlas A2 inference products</term>: 0 < Bs ≤ 256.</li><li><term>Atlas A3 training products / Atlas A3 inference products</term>: 0 < Bs ≤ 512.</li></ul> |
  | topK    | Indicates the number of selected top K experts. The value range is 0 < K ≤16, and `0 < K ≤ moeExpertNum+zeroExpertNum+copyExpertNum+constExpertNum`.|
  | serverNum    | Number of server nodes. Only 2, 4, and 8 are supported.<br>Atlas A2 training products/Atlas A2 inference products: This variable is used only in shapes in this scenario.                                                  |
  | localExpertNum |  Number of experts in this rank: <ul><li>For shared expert ranks, `localExpertNum` = 1.</li><li>For MoE expert ranks, `localExpertNum` = `moeExpertNum/(epWorldSize-sharedExpertRankNum)`. When `localExpertNum` > 1, TP communication is not supported. </li><li><term>Atlas A3 training products / Atlas A3 inference products</term>: The value must satisfy 0 < localExpertNum × epWorldSize ≤ 2048.</li></ul>|

- **Environment variables constraints**:
  - **HCCL_BUFFSIZE**:
      Before calling this API, check whether the value of the `HCCL_BUFFSIZE` environment variable is proper. This environment variable indicates the buffer size occupied by a single communication domain, in MB. If this environment variable is not set, the default value 200 MB is used.
      - <term>Atlas A2 training products/Atlas A2 inference products</term>:
          - If `commAlg` is set to `""` or `nullptr`, select the `"fullmesh"` or `"hierarchy"` formula based on the `HCCL_INTRA_PCIE_ENABLE` and `HCCL_INTRA_ROCE_ENABLE` environment variables.
          - If `commAlg` is set to `"fullmesh"`, the value must satisfy `≥ 2 × (Bs × epWorldSize × min(localExpertNum, K) × H × sizeof(uint16) + 2 MB)`.
          - If commAlg is set to `"hierarchy"`, the value must be `≥ moeExpertNum × Bs × (H × sizeof(dtypeX) + 4 × ((K + 7) / 8 × 8) × sizeof(uint32)) + 4 MB + 100 MB`, and `moeExpertNum/(epWorldSize - sharedExpertRankNum) ≤ 24` is not required.
      - <term>Atlas A3 training products/Atlas A3 inference products</term>:
        - When `commAlg` is `"fullmesh_v1"`, a null character string, or a null pointer, the value must satisfy ≥ 2 × (localExpertNum × maxBs × epWorldSize × Align512(Align32(2 × H) + 64) + (K + sharedExpertNum) × maxBs × Align512(2 × H)).
        - When `commAlg` is `"fullmesh_v2"`, the value must satisfy ≥ 2 × (localExpertNum × maxBs × epWorldSize × 480Align512(Align32(2 × H) + 64) + (K + sharedExpertNum) × maxBs × Align512(2 × H)).
        - `480Align512(x) = ((x + 480 - 1) / 480) × 512`, `Align512(x) = ((x + 512 - 1) / 512) × 512` and `Align32(x) = ((x + 32 - 1) / 32) × 32`.

  - **HCCL_INTRA_PCIE_ENABLE** and **HCCL_INTRA_ROCE_ENABLE**:
      - <term>Atlas A2 training products / Atlas A2 inference products</term>: This environment variable is not recommended. You are advised to set `commAlg` to `"hierarchy"`.

- **Constraints on the use of the communication domains**:
  - `aclnnMoeDistributeCombineV4` and `aclnnMoeDistributeDispatchV4` in a model support only the same EP communication domain, and no other operators are allowed in the communication domain.
  - `aclnnMoeDistributeCombineV4` and `aclnnMoeDistributeDispatchV4` in a model support only the same TP communication domain or both do not support a TP communication domain. If a TP communication domain is supported, no other operators are allowed in the communication domain.
  - <term>Atlas A3 training products / Atlas A3 inference products</term>: Nodes in a communication domain must be in the same SuperPoD. Cross-SuperPoD nodes are not supported.

- **Networking constraints**:
  - <term>Atlas A2 training products / Atlas A2 inference products</term>: In multi-server scenarios, only switch-based networking is supported, and direct point-to-point networking between two servers is not supported.

- **Other constraints**:
  - In the formulas, / denotes integer division.
  - <code>moeExpertNum + zeroExpertNum + copyExpertNum + constExpertNum < MAX_INT32</code>

## Example

<term>Atlas A2 training products / Atlas A2 inference products</term>: Similar to the following example for <term>Atlas A3 training products / Atlas A3 inference products</term>. For the new scenario parameters of V4 compared with V3, set the parameter values based on the preceding parameter description.

<term>Atlas A3 training products / Atlas A3 inference products</term>: The sample code is as follows (for reference only). Call the `aclnnMoeDistributeCombineV4` and `aclnnMoeDistributeDispatchV4` APIs.

- Preparing files:
  1. Create a `dispatchDemo` directory. Follow the instructions to create `aclnnDispatchDemo.cpp` and `buildCombine.sh` files in the `dispatchDemo` directory, and modify them according to the code.

  2. Install the CANN package and compile and run `dispatchDemo`.

- Compilation script:

    ```bash
    #!/bin/bash
    cann_path="/path/to/cann_env" # Change the path to the CANN package environment.
    g++ "aclnnDispatchDemo.cpp" -o dispatchDemo -I"$cann_path/latest/include/" -I"$cann_path/latest/include/aclnnop/" \
                        -L="$cann_path/latest/lib64/" -lascendcl -lnnopbase -lopapi_math -lop_common -lpthread -lhccl
    ```

- Compilation and execution:

    ```bash
    # Source CANN environment
    source /path/to/cann_env/latest/bin/setenv.bash

    # Compile `aclnnDispatchDemo.cpp`.
    bash buildCombine.sh

    ./dispatchDemo
    ```

- The sample code is as follows:

    ```cpp
    #include <thread>
    #include <iostream>
    #include <string>
    #include <vector>
    #include <unordered_set>
    #include "acl/acl.h"
    #include "hccl/hccl.h"
    #include "aclnnop/aclnn_moe_distribute_dispatch_v4.h"
    #include "aclnnop/aclnn_moe_distribute_combine_v4.h"

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
    constexpr uint32_t TP_WORLD_SIZE = 1;
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
            strides[i] = shape[i +1] * strides[i + 1];
        }
        *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
            shape.data(), shape.size(), *deviceAddr);
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
        LOG_PRINT("[INFO] rank = %d, hcomEpName = %s, hcomTpName = %s, dispatchStream = %p, combineStream = %p, \
                    context = %p\n", args.rankId, hcomEpName, hcomTpName, args.dispatchStream, args.combineStream,                 \
                    args.context);

        int64_t Bs = 8;
        int64_t H = 7168;
        int64_t K = 3;
        int64_t expertShardType = 0;
        int64_t sharedExpertNum = 1;
        int64_t sharedExpertRankNum = 1;
        int64_t moeExpertNum = 7;
        int64_t quantMode = 0;
        int64_t globalBs = Bs * EP_WORLD_SIZE;
        int64_t expertTokenNumsType = 1;
        int64_t outDtype = 0;
        int64_t commQuantMode = 0;
        int64_t groupList_type = 1;
        int64_t localExpertNum;
        int64_t A;
        int64_t zeroExpertNum = 1;
        int64_t copyExpertNum = 1;
        int64_t constExpertNum = 1;
        if (args.epRankId < sharedExpertRankNum) {
            localExpertNum = 1;
            A = globalBs / sharedExpertRankNum;
        } else {
            localExpertNum = moeExpertNum / (EP_WORLD_SIZE - sharedExpertRankNum);
            A = globalBs * (localExpertNum < K ? localExpertNum : K);
        }

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

        // Input for dynamic scale-in and zero-expert scenarios
        void *elasticInfoDeviceAddr = nullptr;
        void *oriXDeviceAddr = nullptr;
        void *constExpertAlpha1DeviceAddr = nullptr;
        void *constExpertAlpha2DeviceAddr = nullptr;
        void *constExpertVDeviceAddr = nullptr;

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


        aclTensor *elasticInfo = nullptr;
        aclTensor *oriX = nullptr;
        aclTensor *constExpertAlpha1 = nullptr;
        aclTensor *constExpertAlpha2 = nullptr;
        aclTensor *constExpertV = nullptr;

        aclTensor *xOut = nullptr;

        // Define the dimensions of variables in the current scenario.
        std::vector<int64_t> xShape{Bs, H};
        std::vector<int64_t> expertIdsShape{Bs, K};
        std::vector<int64_t> scalesShape{moeExpertNum + 1, H};
        std::vector<int64_t> expertScalesShape{Bs, K};

        std::vector<int64_t> expandXShape{TP_WORLD_SIZE * A, H};
        std::vector<int64_t> dynamicScalesShape{TP_WORLD_SIZE * A};
        std::vector<int64_t> expandIdxShape{A * 128};
        std::vector<int64_t> expertTokenNumsShape{localExpertNum};
        std::vector<int64_t> epRecvCountsShape{TP_WORLD_SIZE * localExpertNum * EP_WORLD_SIZE};
        std::vector<int64_t> tpRecvCountsShape{TP_WORLD_SIZE};
        std::vector<int64_t> expandScalesShape{A};
        std::vector<int64_t> sharedExpertXShape{Bs, 1, H};


        std::vector<int64_t> elasticInfoShape{4 + EP_WORLD_SIZE * 2};
        std::vector<int64_t> oriXShape{Bs, H};
        std::vector<int64_t> constExpertAlpha1Shape{constExpertNum, H};
        std::vector<int64_t> constExpertAlpha2Shape{constExpertNum, H};
        std::vector<int64_t> constExpertVShape{constExpertNum, H};

        std::vector<int64_t> xOutShape{Bs, H};

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
        int64_t sharedExpertXShapeSize = GetShapeSize(sharedExpertXShape);

        int64_t elasticInfoSize = GetShapeSize(elasticInfoShape);
        int64_t oriXSize = GetShapeSize(oriXShape);
        int64_t constExpertAlpha1Size = GetShapeSize(constExpertAlpha1Shape);
        int64_t constExpertAlpha2Size = GetShapeSize(constExpertAlpha2Shape);
        int64_t constExpertVSize = GetShapeSize(constExpertVShape);

        int64_t xOutShapeSize = GetShapeSize(xOutShape);

        std::vector<int16_t> xHostData(xShapeSize, 1);
        std::vector<int32_t> expertIdsHostData;
        for (int32_t token_id = 0; token_id < expertIdsShape[0]; token_id++) {
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
        std::vector<int16_t> sharedExpertXHostData(sharedExpertXShapeSize, 1);

        int32_t isElastic = 1;
        int32_t rankNumAfterElastic = 4;
        int32_t sharedExpertRankNumAfterElastic = sharedExpertRankNum;
        int32_t moeExpertNumAfterElastic = rankNumAfterElastic - sharedExpertRankNumAfterElastic;
        std::unordered_set<int16_t> availableRank{
            0, 1, /*2, 3, 4, 5,*/ 6, 7
        };
        std::vector<int32_t> elasticInfoHostData{
            isElastic, rankNumAfterElastic, sharedExpertRankNumAfterElastic, moeExpertNumAfterElastic,
            0, 1, -1, -1, -1, -1, 2, 3,
            0, 1, 6, 7, -1, -1, -1, -1
        };
        std::vector<int16_t> oriXHostData(oriXSize, 1);
        std::vector<int16_t> constExpertAlpha1HostData(constExpertAlpha1Size, 0);
        std::vector<int16_t> constExpertAlpha2HostData(constExpertAlpha2Size, 0);
        std::vector<int16_t> constExpertVHostData(constExpertVSize, 0);

        std::vector<int16_t> xOutHostData(xOutShapeSize, 0);


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
        ret = CreateAclTensor(sharedExpertXHostData, sharedExpertXShape, &sharedExpertXDeviceAddr, aclDataType::ACL_BF16, &sharedExpertX);
        CHECK_RET(ret == ACL_SUCCESS, return ret);

        ret = CreateAclTensor(elasticInfoHostData, elasticInfoShape, &elasticInfoDeviceAddr, aclDataType::ACL_INT32, &elasticInfo);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(oriXHostData, oriXShape, &oriXDeviceAddr, aclDataType::ACL_BF16, &oriX);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(constExpertAlpha1HostData, constExpertAlpha1Shape, &constExpertAlpha1DeviceAddr, aclDataType::ACL_BF16, &constExpertAlpha1);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(constExpertAlpha2HostData, constExpertAlpha2Shape, &constExpertAlpha2DeviceAddr, aclDataType::ACL_BF16, &constExpertAlpha2);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(constExpertVHostData, constExpertVShape, &constExpertVDeviceAddr, aclDataType::ACL_BF16, &constExpertV);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(xOutHostData, xOutShape, &xOutDeviceAddr, aclDataType::ACL_BF16, &xOut);
        CHECK_RET(ret == ACL_SUCCESS, return ret);



        uint64_t dispatchWorkspaceSize = 0;
        aclOpExecutor *dispatchExecutor = nullptr;
        void *dispatchWorkspaceAddr = nullptr;

        uint64_t combineWorkspaceSize = 0;
        aclOpExecutor *combineExecutor = nullptr;
        void *combineWorkspaceAddr = nullptr;
        /**************************************** Call dispatch warm up. ********************************************/
        // Simulate the dynamic scale-in scenario. Run a normal case first to establish the communication domain, calling the first-phase API.
        ret = aclnnMoeDistributeDispatchV4GetWorkspaceSize(x, expertIds, (quantMode > 0 ? scales : nullptr), nullptr,
                expertScales, nullptr, nullptr, // performanceInfoOptional
                hcomEpName, EP_WORLD_SIZE, args.epRankId, moeExpertNum, hcomTpName, TP_WORLD_SIZE,
                args.tpRankId, expertShardType, sharedExpertNum,sharedExpertRankNum, quantMode, globalBs,
                expertTokenNumsType, nullptr, zeroExpertNum, copyExpertNum, constExpertNum, expandX, dynamicScales, expandIdx, expertTokenNums, epRecvCounts,
                tpRecvCounts, expandScales, &dispatchWorkspaceSize, &dispatchExecutor);

        CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("[ERROR] warm up aclnnMoeDistributeDispatchV4GetWorkspaceSize failed. ret = %d \n", ret); return ret);

        if (dispatchWorkspaceSize > 0) {
            ret = aclrtMalloc(&dispatchWorkspaceAddr, dispatchWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] warm up aclrtMalloc workspace failed. ret = %d \n", ret); return ret);
        }
        // Call the second-phase API.
        ret = aclnnMoeDistributeDispatchV4(dispatchWorkspaceAddr, dispatchWorkspaceSize,
                                            dispatchExecutor, args.dispatchStream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] warm up aclnnMoeDistributeDispatchV4 failed. ret = %d \n", ret);  \
                return ret);
        ret = aclrtSynchronizeStreamWithTimeout(args.dispatchStream, 10000);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] warm up aclrtSynchronizeStreamWithTimeout failed. ret = %d \n", ret);  \
            return ret);

        /**************************************** Call dispatch. ********************************************/
        if (availableRank.find(args.rankId) != availableRank.end()) {
            // Call the first-phase API.
        ret = aclnnMoeDistributeDispatchV4GetWorkspaceSize(x, expertIds, (quantMode > 0 ? scales : nullptr), nullptr,
                expertScales, elasticInfo, nullptr, // performanceInfoOptional
                hcomEpName, EP_WORLD_SIZE, args.epRankId, moeExpertNum, hcomTpName, TP_WORLD_SIZE,
                args.tpRankId, expertShardType, sharedExpertNum,sharedExpertRankNum, quantMode, globalBs,
                expertTokenNumsType, nullptr, zeroExpertNum, copyExpertNum, constExpertNum, expandX, dynamicScales, expandIdx, expertTokenNums, epRecvCounts,
                tpRecvCounts, expandScales, &dispatchWorkspaceSize, &dispatchExecutor);

        CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("[ERROR] aclnnMoeDistributeDispatchV4GetWorkspaceSize failed. ret = %d \n", ret); return ret);

        if (dispatchWorkspaceSize > 0) {
            ret = aclrtMalloc(&dispatchWorkspaceAddr, dispatchWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc workspace failed. ret = %d \n", ret); return ret);
        }
        // Call the second-phase API.
        ret = aclnnMoeDistributeDispatchV4(dispatchWorkspaceAddr, dispatchWorkspaceSize,
                                            dispatchExecutor, args.dispatchStream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnMoeDistributeDispatchV4 failed. ret = %d \n", ret);  \
                return ret);
        ret = aclrtSynchronizeStreamWithTimeout(args.dispatchStream, 10000);
                    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] dispatch aclrtSynchronizeStreamWithTimeout failed. ret = %d \n", ret);  \
                return ret);
        }
        /**************************************** Call combine. ********************************************/
        // Call the first-phase API.
        if (availableRank.find(args.rankId) != availableRank.end()) {
        ret = aclnnMoeDistributeCombineV4GetWorkspaceSize(expandX, expertIds,
                                                            expandIdx, epRecvCounts,
                                                            expertScales, tpRecvCounts,
                                                            nullptr, nullptr, nullptr,
                                                            nullptr, nullptr, nullptr,
                                                            elasticInfo, oriX, constExpertAlpha1, constExpertAlpha2, constExpertV,
                                                            nullptr, // performanceInfoOptional
                                                            hcomEpName, EP_WORLD_SIZE, args.epRankId, moeExpertNum,
                                                            hcomTpName, TP_WORLD_SIZE, args.tpRankId, expertShardType,
                                                            sharedExpertNum, sharedExpertRankNum, globalBs, outDtype,
                                                            commQuantMode, groupList_type, nullptr, zeroExpertNum, copyExpertNum, constExpertNum, xOut,
                                                            &combineWorkspaceSize, &combineExecutor);
        CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("[ERROR] aclnnMoeDistributeCombineV4GetWorkspaceSize failed. ret = %d \n", ret); return ret);
        // Allocate device memory based on the computed workspaceSize.
        if (combineWorkspaceSize > 0) {
            ret = aclrtMalloc(&combineWorkspaceAddr, combineWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc workspace failed. ret = %d \n", ret); return ret);
        }

        // Call the second-phase API.
        ret = aclnnMoeDistributeCombineV4(combineWorkspaceAddr, combineWorkspaceSize, combineExecutor, args.combineStream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnMoeDistributeCombineV4 failed. ret = %d \n", ret);
            return ret);
        // (Boilerplate) Wait until the task execution is complete.
        ret = aclrtSynchronizeStreamWithTimeout(args.combineStream, 10000);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSynchronizeStreamWithTimeout failed. ret = %d \n", ret);
            return ret);
        LOG_PRINT("[INFO] device_%d aclnnMoeDistributeDispatchV4 and aclnnMoeDistributeCombineV4                      \
                    execute successfully.\n", args.rankId);
        }
        // Free device resources.
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
        if (elasticInfo != nullptr) {
            aclDestroyTensor(elasticInfo);
        }
        if (oriX != nullptr) {
            aclDestroyTensor(oriX);
        }
        if (constExpertAlpha1 != nullptr) {
            aclDestroyTensor(constExpertAlpha1);
        }
        if (constExpertAlpha2 != nullptr) {
            aclDestroyTensor(constExpertAlpha2);
        }
        if (constExpertV != nullptr) {
            aclDestroyTensor(constExpertV);
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
        if (sharedExpertXDeviceAddr != nullptr) {
            aclrtFree(sharedExpertXDeviceAddr);
        }

        if (elasticInfoDeviceAddr != nullptr) {
            aclrtFree(elasticInfoDeviceAddr);
        }
        if (oriXDeviceAddr != nullptr) {
            aclrtFree(oriXDeviceAddr);
        }
        if (constExpertAlpha1DeviceAddr != nullptr) {
            aclrtFree(constExpertAlpha1DeviceAddr);
        }
        if (constExpertAlpha2DeviceAddr != nullptr) {
            aclrtFree(constExpertAlpha2DeviceAddr);
        }
        if (constExpertVDeviceAddr != nullptr) {
            aclrtFree(constExpertVDeviceAddr);
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

        for(uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
            threads[rankId]->join();
        }

        aclFinalize();
        LOG_PRINT("[INFO] aclFinalize success\n");

        return 0;
    }
    ```
