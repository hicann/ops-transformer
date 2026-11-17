# aclnnMoeDistributeDispatchV4

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T14:45:24.011Z -->

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

- Description: Quantizes token data (optional) and performs AllToAllV communication in the EP (Expert Parallelism) domain. Communication in the TP (Tensor Parallelism) domain is no longer supported in the current version. TP-related parameters are reserved parameters.

    Compared with the `aclnnMoeDistributeDispatchV3` API, this API has the following changes:

    Adds the communication overhead collection feature, which records the communication time of each device. The feature is enabled by passing in the `performanceInfoOptional` parameter. It is recommended to use this feature with the [DeepXTrace](https://github.com/antgroup/DeepXTrace) tool. The communication overhead of each device in a single operator call is accumulated on this tensor. Clear the tensor as needed before use.

- Formula:

    $$
    expandXOut = AllToAllV(X)
    $$

    When the data type of `x` is `INT32`, `scalesOptional` is communicated along with the corresponding token:

    $$
    dynamicScalesOut = AllToAllV(scalesOptional)
    $$

    > NOTE: TP domain communication is no longer supported in the current version.

<!-- npu="910b" id7 -->
- <term>Atlas A2 products</term>: This API must be used together with `aclnnMoeDistributeCombineV4`.
<!-- end id7 -->
<!-- npu="950,A3" id8 -->
- <term>Atlas A3 products</term>, Ascend 950PR&950DT products: This API must be used together with `aclnnMoeDistributeCombineV4` or `aclnnMoeDistributeCombineAddRmsNormV2`.

  NOTE: `aclnnMoeDistributeCombineV4` and `aclnnMoeDistributeCombineAddRmsNormV2` are collectively referred to as the **CombineV4 series operator** in subsequent documents.

<!-- end id8 -->

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). You must first call the "aclnnMoeDistributeDispatchV4GetWorkspaceSize" API to obtain the workspace size required for computation and the executor containing the operator computation process, and then call the "aclnnMoeDistributeDispatchV4" API to perform the computation.

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
    int64_t          globalBS,
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
    <th>Parameter</th>
    <th>Input/Output</th>
    <th>Description</th>
    <th>Instruction</th>
    <th>Data Type</th>
    <th>Data Format</th>
    <th>Dimension (shape)</th>
    <th>Non-contiguous Tensor</th>
    </tr>
    </thead>
    <tbody>
    <tr>
    <td>x</td>
    <td>Input</td>
    <td>Token data sent by the current device.</td>
    <td>2D tensor.</td>
    <td>FLOAT16, BFLOAT16, FLOAT8_E5M2, FLOAT8_E4M3FN, HIFLOAT8, FLOAT4_E2M1, FLOAT4_E1M2, INT32</td>
    <td>ND</td>
    <td><code>(BS, H)</code> (BS = batch size, H = hidden size)</td>
    <td>√</td>
    </tr>
    <tr>
    <td>expertIds</td>
    <td>Input</td>
    <td>TopK expert indexes of each token.</td>
    <td>2D Tensor.</td>
    <td>INT32</td>
    <td>ND</td>
    <td><code>(BS, K)</code></td>
    <td>√</td>
    </tr>
    <tr>
    <td>scalesOptional</td>
    <td>Input</td>
    <td>Quantization smoothing parameter of each expert or scaling parameter for INT32 tokens.</td>
    <td>-</td>
    <td>FLOAT32, FLOAT8_E8M0</td>
    <td>ND</td>
    <td><code>(sharedExpertNum + moeExpertNum, H)</code></td>
    <td>√</td>
    </tr>
    <tr>
    <td>xActiveMaskOptional</td>
    <td>Input</td>
    <td>Indicates whether tokens participate in communication.</td>
    <td>Pass valid data or a null pointer.<br>For 1D input, true indicates that the corresponding token participates in communication. true must be placed before false. For example, {true, false, true} is invalid input;<br>For 2D input, true indicates that the expert_ids corresponding to the current token participate in communication. If all K BOOL values corresponding to the current token are false, the current token does not participate in communication. By default, all tokens participate in communication. When the BS count differs across devices, all tokens must be valid.</td>
    <td>BOOL</td>
    <td>ND</td>
    <td><br>For 1D input, the shape is <code>(BS,)</code>; for 2D input, the shape is <code>(BS, K)</code></td>
    <td>√</td>
    </tr>
    <tr>
    <td>expertScalesOptional</td>
    <td>Input</td>
    <td>topK expert weights for each token.</td>
    <td>-</td>
    <td>FLOAT32</td>
    <td>ND</td>
    <td><code>(BS, K)</code></td>
    <td>√</td>
    </tr>
    <tr>
    <td>elasticInfoOptional</td>
    <td>Input</td>
    <td>Dynamic scale-down information of the EP communicator.</td>
    <td>If some devices are removed from the communicator due to exceptions, the actual number of devices participating in communication can be obtained from this parameter.</td>
    <td>INT32</td>
    <td>ND</td>
    <td>-</td>
    <td>√</td>
    </tr>
    <tr>
    <td>performanceInfoOptional</td>
    <td>Input</td>
    <td>Indicates the communication overhead for the current device to wait for data from other devices, in us (microseconds).</td>
    <td>The communication overhead of each device in a single operator call is accumulated to this tensor. The operator does not automatically clear the tensor. Therefore, users need to clear the tensor each time before enabling it to record overhead.</td>
    <td>INT64</td>
    <td>ND</td>
    <td>-</td>
    <td>√</td>
    </tr>
    <tr>
    <td>groupEp</td>
    <td>Input</td>
    <td>Name of the EP communicator (expert parallelism).</td>
    <td>String length [1, 128)</td>
    <td>STRING</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>epWorldSize</td>
    <td>Input</td>
    <td>EP communicator size.</td>
    <td>-</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>epRankId</td>
    <td>Input</td>
    <td>ID of the current device in the EP domain.</td>
    <td>Value range: [0, epWorldSize). The epRankId of each device in the same EP communicator must be unique.</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>moeExpertNum</td>
    <td>Input</td>
    <td>Number of MoE experts.</td>
    <td>Must satisfy <code>moeExpertNum % (epWorldSize - sharedExpertRankNum) = 0</code>.</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>groupTp</td>
    <td>Input</td>
    <td>TP communicator name (reserved parameter, not supported in the current version).</td>
    <td>-</td>
    <td>STRING</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>tpWorldSize</td>
    <td>Input</td>
    <td>TP communicator size (reserved parameter, not supported in the current version, only 0 or 1 is supported).</td>
    <td>-</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>tpRankId</td>
    <td>Input</td>
    <td>Local device ID in the TP domain (reserved parameter, not supported in the current version, pass 0).</td>
    <td>Default value: 0.</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>expertShardType</td>
    <td>Input</td>
    <td>Distribution type of shared expert devices.</td>
    <td>-</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>sharedExpertNum</td>
    <td>Input</td>
    <td>Number of shared experts (a shared expert can be replicated and deployed on multiple devices).</td>
    <td>-</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>sharedExpertRankNum</td>
    <td>Input</td>
    <td>Number of devices hosting shared experts.</td>
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
    <td>-</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>globalBS</td>
    <td>Input</td>
    <td>Global batch size in the EP domain.</td>
    <td><br> <li> If the BS is the same on all devices: <code>globalBS = BS * epWorldSize</code> or 0;</li> <li> If the BS differs across devices: <code>globalBS = maxBS * epWorldSize</code>, where maxBS is the upper limit of the single-rank BS capacity configured for the service and must not be less than the actual BS on any rank; all ranks must pass in the same non-zero globalBS.</li></td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>expertTokenNumsType</td>
    <td>Input</td>
    <td>Semantic type of the output <code>expertTokenNums</code>.</td>
    <td>Supported values: 0: the output in expertTokenNums is the prefix sum of the number of tokens processed by each expert; 1: the output in expertTokenNums is the number of tokens processed by each expert.</td>
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
    <td>Number of copy experts.</td>
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
    <td>Token features expanded based on expertIds.</td>
    <td>2D tensor.</td>
    <td>FLOAT16, BFLOAT16, INT8, FLOAT8_E4M3FN, FLOAT8_E5M2, HIFLOAT8, FLOAT4_E2M1, FLOAT4_E1M2, INT32</td>
    <td>-</td>
    <td><code>(A, H)</code></td>
    <td>√</td>
    </tr>
    <tr>
    <td>dynamicScalesOut</td>
    <td>Output</td>
    <td>Scaling parameter.</td>
    <td>Must be a 1D or 2D Tensor. Has output when quantMode is 2, 3, or 4; also has output when quantMode is 0 and the data type of `x` is `HIFLOAT8`, `FLOAT8_E5M2`, `FLOAT8_E4M3FN`, `FLOAT4_E2M1`, `FLOAT4_E1M2`, or `INT32`.</td>
    <td>FLOAT32, FLOAT8_E8M0</td>
    <td>-</td>
    <td>-</td>
    <td>√</td>
    </tr>
    <tr>
    <td>assistInfoForCombineOut</td>
    <td>Output</td>
    <td>Number of tokens sent to the same expert (corresponding to assistInfoForCombine in aclnnMoeDistributeCombineV4).</td>
    <td>1D tensor.</td>
    <td>INT32</td>
    <td>-</td>
    <td><code>(A * 128,)</code></td>
    <td>√</td>
    </tr>
    <tr>
    <td>expertTokenNumsOut</td>
    <td>Output</td>
    <td>Number of tokens received by each expert.</td>
    <td>1D tensor.</td>
    <td>INT64</td>
    <td>-</td>
    <td><code>(localExpertNum,)</code></td>
    <td>√</td>
    </tr>
    <tr>
    <td>epRecvCountsOut</td>
    <td>Output</td>
    <td>Number of tokens received from each device in the EP communicator (corresponding to epSendCounts in aclnnMoeDistributeCombineV4).</td>
    <td>1D Tensor.</td>
    <td>INT32</td>
    <td>-</td>
    <td><code>(moeExpertNum + 2 * globalBS * K * serverNum,)</code></td>
    <td>√</td>
    </tr>
    <tr>
    <td>tpRecvCountsOut</td>
    <td>Output</td>
    <td>Reserved output, not supported in the current version. Pass a null pointer.</td>
    <td>Reserved parameter, not supported in the current version. Pass a null pointer.</td>
    <td>INT32</td>
    <td>-</td>
    <td>-</td>
    <td>√</td>
    </tr>
    <tr>
    <td>expandScalesOut</td>
    <td>Output</td>
    <td>Weights of the tokens output on this device (corresponding to expertScalesOptional in aclnnMoeDistributeCombineV4).</td>
    <td>-</td>
    <td>FLOAT32</td>
    <td>-</td>
    <td>-</td>
    <td>√</td>
    </tr>
    <tr>
    <td>workspaceSize</td>
    <td>Output</td>
    <td>Returns the size of the workspace to be requested on the Device.</td>
    <td>-</td>
    <td>UINT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>executor</td>
    <td>Output</td>
    <td>Returns the op executor that contains the operator computation flow.</td>
    <td>-</td>
    <td>aclOpExecutor*</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    </tbody>
    </table>

    <!-- npu="910b" id9 -->
    <details>
    <summary><term>Atlas A2 products</term>:</summary>

    - dynamicScalesOut is output when quantMode is set to 2 or the data type of `x` is INT32.
    - commAlg supports nullptr, "", "fullmesh", and "hierarchy". "hierarchy" is recommended together with a driver of version ≥25.0.RC1.1. nullptr and "" select the algorithm based on HCCL environment variables (not recommended). "fullmesh" transfers tokens directly over RDMA. "hierarchy" optimizes communication through two sends: cross-node and intra-node.
    - If commAlg is "hierarchy", or HCCL_INTRA_PCIE_ENABLE=1 and HCCL_INTRA_ROCE_ENABLE=0, pass nullptr for scalesOptional. If commAlg is "fullmesh", scalesOptional can be set to valid data or a null pointer. If the data type of `x` is INT32, only commAlg "fullmesh" is supported, and scalesOptional must be set to valid data.
    - The value of xActiveMaskOptional depends on commAlg. For "fullmesh", xActiveMaskOptional must be a 1D Tensor with shape (BS, ), and true must precede false (for example, {true, false, true} is invalid). "hierarchy" is not supported in the current version; pass a null pointer.
    - expertScalesOptional must be a 2D Tensor with shape (BS, K).
    - The value of epWorldSize depends on commAlg. For "fullmesh", supported values are 2, 3, 4, 5, 6, 7, 8, 16, 32, 64, 128, 192, 256, and 384. For "hierarchy", supported values are 16, 32, and 64.
    - The value of moeExpertNum depends on commAlg. For "fullmesh", the supported range is (0, 1024]. For "hierarchy", the supported range is (0, 512].
    - groupTp is not supported in the current version. Pass an empty string.
    - tpWorldSize, tpRankId, expertShardType, sharedExpertNum, and sharedExpertRankNum are not supported in the current version. Pass 0.
    - The shape of epRecvCountsOut is (moeExpertNum + 2 \* globalBS \* K \* serverNum,). The first moeExpertNum elements are the numbers of received tokens, and the remaining elements are pre-communication reduce-related information.
    - TP domain communication is not supported currently.
    - expandScalesOut must be a 1D Tensor with shape (A,).
    - quantMode supports 0 (non-quantization) and 2 (dynamic quantization).
    - elasticInfoOptional is not supported in the current version. Pass a null pointer.
    - When commAlg="fullmesh", the value range of zeroExpertNum is [0, MAX_INT32), where MAX_INT32 = 2^31 - 1. Valid zero expert IDs are [<code>moeExpertNum</code>, <code>moeExpertNum + zeroExpertNum</code>).
    - When commAlg="fullmesh", the value range of copyExpertNum is [0, MAX_INT32), where MAX_INT32 = 2^31 - 1. Valid copy expert IDs are [<code>moeExpertNum + zeroExpertNum</code>, <code>moeExpertNum + zeroExpertNum + copyExpertNum</code>).
    - constExpertNum is not supported in the current version. Pass 0.
    - For performanceInfoOptional, pass either valid data or a null pointer. Passing a null pointer indicates that communication overhead recording is disabled. If valid data is passed, it must be a 1D Tensor with shape (epWorldSize,). The supported data type is int64, and the required data format is ND.
    </details>
    <!-- end id9 -->

    <!-- npu="A3" id10 -->
    <details>
    <summary><term>Atlas A3 products</term>  :</summary>

    - dynamicScalesOut is output only when quantMode is set to 2.
    - commAlg supports three input values: "", "fullmesh_v1", "fullmesh_v2", "hierarchy". "": Default value; the fullmesh_v2 template is not enabled. "fullmesh_v1": The fullmesh_v2 template is not enabled. "fullmesh_v2": The fullmesh_v2 template is enabled. "hierarchy": The cross-SuperPoD template is enabled. This template supports different BS values across ranks, and supports only the use case where tpWorldSize is 1 and the number of shared experts is 0. It does not support the 2D mask, special expert, or performanceInfo use cases. For the variable BS use case, all ranks must be configured with the same non-zero globalBS capacity, and xActiveMaskOptional must not be passed.
    - xActiveMaskOptional must be a 1D or 2D tensor (1D: shape (BS, ); 2D: shape (BS, K)); in the 1D case, true must be placed before false; in the 2D case, if all K values of a token are false, the token does not participate in communication.
    - expertScalesOptional must be a 2D tensor with shape (BS, K) when commAlg="hierarchy"; when commAlg="", "fullmesh_v1", or "fullmesh_v2", `pass a null pointer.
    - The value range of epWorldSize is [2, 768]; when commAlg="hierarchy", the value range is [16, 256], and the value must be a multiple of 16.
    - The value range of moeExpertNum is (0, 1024]; when commAlg="hierarchy", the value range is (0, 512].
    - groupTp is a reserved parameter and is not supported in the current version. Pass an empty string.
    - tpWorldSize (reserved parameter, not supported in the current version; only 0 or 1 can be passed).
    - tpRankId (reserved parameter, not supported in the current version, only 0 can be passed).
    - expertShardType supports only 0 currently, indicating that shared expert devices are placed before MoE expert devices.
    - The current value range of sharedExpertNum is [0, 4].
    - The value range of sharedExpertRankNum is [0, epWorldSize). If it is 0, sharedExpertNum must be 0 or 1. If it is not 0, sharedExpertRankNum % sharedExpertNum = 0 must be satisfied.
    - The shape of epRecvCountsOut is (epWorldSize \* localExpertNum,).
    - tpRecvCountsOut (reserved output, not supported in the current version, pass a null pointer).
    - expandScalesOut must be a 1D tensor with shape (A,) in the commAlg="hierarchy" use case. This output is not supported in the commAlg="", "fullmesh_v1", or "fullmesh_v2" use cases.
    - quantMode supports 0 (no quantization) and 2 (dynamic quantization).
    - elasticInfoOptional is not supported in the current version. Pass a null pointer.
    - The value range of zeroExpertNum is [0, MAX_INT32), where MAX_INT32 = 2^31 - 1. The valid zero expert ID range is <code>[moeExpertNum, moeExpertNum + zeroExpertNum)</code>.
    - copyExpertNum value range: [0, MAX_INT32), where MAX_INT32 = 2^31 - 1. Valid expert IDs are <code>[moeExpertNum + zeroExpertNum, moeExpertNum + zeroExpertNum + copyExpertNum)</code>.
    - constExpertNum value range: [0, MAX_INT32), where MAX_INT32 = 2^31 - 1. Valid expert IDs are <code>[moeExpertNum + zeroExpertNum + copyExpertNum, moeExpertNum + zeroExpertNum + copyExpertNum + constExpertNum)</code>.
    - performanceInfoOptional can be set to valid data or a null pointer. Passing a null pointer indicates that recording of communication overhead is disabled. If valid data is passed, it must be a 1D tensor with shape (epWorldSize,), data type int64, and data format ND.
    - scalesOptional is a 2D tensor. Pass a null pointer for non-quantization use cases. For dynamic quantization, valid data or a null pointer can be passed.
    </details>
    <!-- end id10 -->

    <!-- npu="950" id11 -->
    <details>
    <summary><term>Ascend 950PR&950DT products</term>:</summary>

    - dynamicScalesOut has output when quantMode is 2, 3, or 4. It also has output when quantMode is 0 and the data type of `x` is `HIFLOAT8`, `FLOAT8_E5M2`, `FLOAT8_E4M3FN`, `FLOAT4_E2M1`, or `FLOAT4_E1M2`.
    - commAlg supports three input values: "", "fullmesh_v1", and "fullmesh_v2". "": default value, the fullmesh_v2 template is not enabled. "fullmesh_v1": the fullmesh_v2 template is not enabled. "fullmesh_v2": the fullmesh_v2 template is enabled.
    - xActiveMaskOptional must be a 1D or 2D tensor (1D: shape (BS, ); 2D: shape (BS, K)). For a 1D tensor, true values must be placed before false values. For a 2D tensor, if all K values of a token are false, the token does not participate in communication.
    - expertScalesOptional can be set to a valid tensor or a null pointer. If a valid tensor is passed, the expertScales expert weight feature is enabled. The valid tensor is a 2D tensor with shape (BS, K). If a null pointer is passed, the expertScales expert weight feature is disabled. Empty tensors are not supported.
    - epWorldSize: value range [2, 1024].
    - moeExpertNum: value range (0, 1024].
    - groupTp: Not supported in the current version. Pass an empty string.
    - tpWorldSize: Not supported in the current version. Pass 0.
    - tpRankId: Not supported in the current version. Pass 0.
    - expertShardType: only 0 is supported currently, indicating that shared expert devices are placed before MoE expert devices.
    - sharedExpertNum: current value range [0, 4].
    - sharedExpertRankNum: value range [0, epWorldSize); if it is 0, sharedExpertNum must be 0 or 1; if it is not 0, sharedExpertRankNum % sharedExpertNum = 0 must be satisfied.
    - epRecvCountsOut: shape (epWorldSize \* localExpertNum,).
    - tpRecvCountsOut (reserved output, not supported in the current version, pass a null pointer).
    - expandScalesOut is valid when the expertScales expert weight function is enabled; otherwise, this output is invalid.
    - quantMode supports 0 (no quantization), 1 (static quantization), 2 (pertoken dynamic quantization), 3 (pergroup dynamic quantization), and 4 (mx dynamic quantization).
    - elasticInfoOptional is not supported in the current version. Pass a null pointer.
    - zeroExpertNum value range: [0, MAX_INT32), where MAX_INT32 = 2^31 - 1. The valid zero expert ID range is <code>[moeExpertNum, moeExpertNum + zeroExpertNum)</code>.
    - copyExpertNum value range: [0, MAX_INT32), where MAX_INT32 = 2^31 - 1. The expert ID range is <code>[moeExpertNum + zeroExpertNum, moeExpertNum + zeroExpertNum + copyExpertNum)</code>.
    - constExpertNum value range: [0, MAX_INT32), where MAX_INT32 = 2^31 - 1. The expert ID range is <code>[moeExpertNum + zeroExpertNum + copyExpertNum, moeExpertNum + zeroExpertNum + copyExpertNum + constExpertNum)</code>.
    - performanceInfoOptional accepts either valid data or a null pointer. Passing a null pointer indicates that the communication overhead recording function is disabled. If valid data is passed, the input must be a 1D tensor with shape (epWorldSize,), the supported data type is int64, and the required data format is ND.
    - scalesOptional If quantMode is 0, pass a null pointer if the data type of `x` is `FLOAT16` or `BFLOAT16`, and pass valid data if the data type of `x` is `HIFLOAT8`, `FLOAT8_E5M2`, `FLOAT8_E4M3FN`, `FLOAT4_E2M1`, or `FLOAT4_E1M2`. If quantMode is 1, valid data is required. If quantMode is 2 or 3, either valid data or a null pointer can be passed. If quantMode is 4, pass a null pointer.
    </details>
    <!-- end id11 -->

- **Return Value**

    aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

    The first-phase API performs input parameter validation and returns an error in the following scenarios:

    <table style="undefined;table-layout: fixed; width: 1149px"><colgroup>
    <col style="width: 282px">
    <col style="width: 120px">
    <col style="width: 747px">
    </colgroup>
    <thead>
    <tr>
    <th>Return Values</th>
    <th>Error Code</th>
    <th>Description</th>
    </tr>
    </thead>
    <tbody>
    <tr>
    <td>ACLNN_ERR_PARAM_NULLPTR</td>
    <td>161001</td>
    <td>The tensor of a required input or output parameter is a null pointer.</td>
    </tr>
    <tr>
    <td>ACLNN_ERR_PARAM_INVALID</td>
    <td>161002</td>
    <td>The data type of an input or output parameter is not in the supported range.</td>
    </tr>
    <tr>
    <td rowspan="2">ACLNN_ERR_INNER_TILING_ERROR</td>
    <td rowspan="2">561002</td>
    <td>The shapes of the input and output are out of the supported range.</td>
    </tr>
    <tr>
        <td>The parameter value is out of the supported range.</td>
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
        <th>Parameter</th>
        <th>Input/Output</th>
        <th>Description</th>
    </tr></thead>
    <tbody>
    <tr>
        <td>workspace</td>
        <td>Input</td>
        <td>Workspace memory address requested on the Device side.</td>
    </tr>
    <tr>
        <td>workspaceSize</td>
        <td>Input</td>
        <td>Workspace size requested on the Device side, obtained by calling the first-phase API aclnnMoeDistributeDispatchV4GetWorkspaceSize.</td>
    </tr>
    <tr>
        <td>executor</td>
        <td>Input</td>
        <td>Op executor that contains the operator computation flow.</td>
    </tr>
    <tr>
        <td>stream</td>
        <td>Input</td>
        <td>Specifies the Stream for task execution.</td>
    </tr>
    </tbody>
    </table>

- **Return Value**

    Returns an aclnnStatus status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- **Deterministic computation**:
  - aclnnMoeDistributeDispatchV4 defaults to a deterministic implementation.

- **Driver Constraints**:
  - The driver versions of all nodes in the operator communication domain must be the same.

- **API Compatibility Constraints**:

  - `aclnnMoeDistributeDispatchV4` must be used together with the CombineV4 series operators. The `assistInfoForCombineOut`, `epRecvCountsOut`, `tpRecvCountsOut`, and `expandScalesOut` output by the former must be directly passed to the corresponding parameters of the latter. The business logic must not depend on the specific values of these tensors.

- **Parameter Consistency Constraints**:

  - The `groupEp`, `epWorldSize`, `moeExpertNum`, `groupTp`, `tpWorldSize`, `expertShardType`, `sharedExpertNum`, `sharedExpertRankNum`, `globalBS`, and `commAlg` parameters, as well as the `HCCL_BUFFSIZE` value, must be consistent across all cards, and must also be consistent with the corresponding parameters of the CombineV4 series operators.

- **Product-Specific Constraints**:

  <!-- npu="A3" id12 -->
  - <term>Atlas A3 products</term>: In this use case, each device contains dual DIEs (also known as "grain" or "bare die"). Therefore, "this device" in Parameters refers to a single DIE.

  <!-- end id12 -->

- **Shape Variable Constraints**:

  | Variable     | Definition and value range                                                                 |
  | :----------- | :----------------------------------------------------------------------------- |
  | A            | Indicates the maximum number of tokens to be dispatched on this device. The value range is as follows:<ul> <li>For shared experts, A = BS \* epWorldSize \* sharedExpertNum / sharedExpertRankNum must be satisfied.</li><li>For MoE experts, if globalBS is 0, A >= BS \* epWorldSize \* min(localExpertNum, K) must be satisfied; if globalBS is not 0, A >= globalBS \* min(localExpertNum, K) must be satisfied.</li></ul>|
  | H (hidden size) | Indicates the hidden size.<ul><li><term>Atlas A2 products</term>: If the data type of `x` is INT32, the value range is (0, 5120] and the value must be a multiple of 32; for other data types, the value range is (0, 10240] and the value must be a multiple of 32.</li> <li><term>Atlas A3 products</term>: Depending on the value of commAlg, if commAlg="hierarchy", the value range of H is [1024, 7168] and the value must be a multiple of 32; in other use cases, the value range is [1024, 8192].</li><li><term>Ascend 950PR&950DT products</term>: The value range is [1024, 8192].</li></ul> |
  | BS           | Indicates the number of tokens finally output on this device.<ul><li><term>Atlas A2 products</term>: Depending on the value of commAlg, the value range is (0 < BS ≤ 256) for "fullmesh"; the value range is (0 < BS ≤ 512) for "hierarchy" when the driver version is ≥ 25.0.RC1.1;</li><li><term>Atlas A3 products</term>: Depending on the value of commAlg, the value range is (0 < BS ≤ 256) for "fullmesh_v2" and "hierarchy", and (0 < BS ≤ 512) for "fullmesh_v1" and "".</li><li><term>Ascend 950PR&950DT products</term>: 0 < BS ≤ 512.</li></ul> |
  | K | Number of selected topK experts. The value range is 0 < K ≤ 16, and <code>0 < K ≤ moeExpertNum + zeroExpertNum + copyExpertNum + constExpertNum</code>. See the product-specific constraints below. |
  | serverNum    | Indicates the number of server nodes. Only 2, 4, and 8 are supported.<br>Atlas A2 products: This variable is used only by shapes in this use case.                                                  |
  | localExpertNum |  Number of experts on this device:<ul><li>For shared expert devices, localExpertNum = 1;</li><li>For MoE expert devices, localExpertNum = <code>moeExpertNum/(epWorldSize-sharedExpertRankNum)</code>. TP domain communication is not supported in the current version. </li><li><term>Atlas A3 products</term> and Ascend 950PR&950DT products: 0 < <code>localExpertNum \* epWorldSize</code> ≤ 2048 must be satisfied. If commAlg="hierarchy", localExpertNum ≤ 24 and <code>localExpertNum \* epWorldSize</code> ≤ 512 must be satisfied.</li><li><term>Ascend 950PR&950DT products</term>: 0 < <code>localExpertNum \* epWorldSize</code> ≤ 2048 must be satisfied.</li></ul>|

  <!-- npu="A3" id28 -->
  - K: <term>Atlas A3 products</term>: When `commAlg` is "fullmesh_v2", the value range is 0 < K ≤ 12.
  <!-- end id28 -->
  <!-- npu="950" id29 -->
  - K: <term>Ascend 950PR&950DT products</term>: The ordinary fullmesh_v2 path (`commAlg` is "fullmesh_v2", non-hierarchical and non-RoCE) supports 0 < K ≤ 32; other paths retain 0 < K ≤ 16. For K > 16, the paired Combine operator must not use AddRmsNorm fusion.
  <!-- end id29 -->

- **quantMode-related constraints**:

  <!-- npu="910b" id13 -->
  - <term>Atlas A2 products</term>:
      - When `quantMode` is 0, indicating the non-quantization use case:
          - When the data type of `x` is `FLOAT16` or `BFLOAT16`, pass a null pointer for the input `scalesOptional`, and the data type of `expandX` is the same as that of `x`.
          - When the data type of `x` is `INT32`, only `commAlg` configured to "fullmesh" is supported. The input `scalesOptional` must be a 1D Tensor with the data type `FLOAT32` and the shape (`BS`, ); the data type of `expandX` is `INT32`, and the data type of the output `dynamicScalesOut` is `FLOAT32` with the shape (`A`, ).
      - When `quantMode` is set to 2, it indicates a per-token dynamic quantization scenario, and the data type of `expandX` supports `INT8`.
          - A null pointer can be passed for the input `scalesOptional`.
          - If valid data is passed for the input `scalesOptional`, its shape is (`moeExpertNum`, `H`).
          - The shape of the output `dynamicScalesOut` is `(A, )`
  <!-- end id13 -->
  <!-- npu="A3" id14 -->
  - <term>Atlas A3 products</term>  :
      - When `quantMode` is 0, indicating the non-quantization use case, pass a null pointer for the input `scalesOptional`; the data type of `expandX` can be `FLOAT16` or `BFLOAT16`.
      - When `quantMode` is set to 2, it indicates a per-token dynamic quantization scenario, and the data type of `expandX` supports `INT8`.
          - The input `scalesOptional` can be a null pointer.
          - If valid data is passed to the input `scalesOptional` and shared expert devices exist, its shape is (`sharedExpertNum` + `moeExpertNum`, `H`).
          - If valid data is passed to the input `scalesOptional` and no shared expert devices exist, its shape is (`moeExpertNum`, `H`).
          - The shape of the output `dynamicScalesOut` is `(A, )`
  <!-- end id14 -->
  <!-- npu="950" id15 -->
  - <term>Ascend 950PR&950DT products</term>:
      - When the value of `quantMode` is 0, it indicates the non-quantization use case.
          - When the data type of `x` is `FLOAT16` or `BFLOAT16`, the data type of `expandX` can be the same as that of `x` or `HIFLOAT8`, and a null pointer must be passed for the input `scalesOptional`.
          - When the data type of `x` is `HIFLOAT8`, `FLOAT8_E5M2`, `FLOAT8_E4M3FN`, `FLOAT4_E2M1`, or `FLOAT4_E1M2`, valid data must be passed to the input `scalesOptional`, and the data type of `expandX` is the same as that of `x`.
              - If the data type of `x` is `HIFLOAT8`, the data type of `scalesOptional` is `FLOAT`.
              - If the data type of `x` is `FLOAT8_E5M2` or `FLOAT8_E4M3FN`, the data type of `scalesOptional` is `FLOAT` or `FLOAT8_E8M0`.
              - When the data type of `x` is `FLOAT4_E2M1` or `FLOAT4_E1M2`, the data type of `scalesOptional` is `FLOAT8_E8M0`, and H must be an even number.
              - The shape of `scalesOptional` is (`BS`, `dim1`), where `dim1` must be less than or equal to `H`.
      - When `quantMode` is 1, it indicates the static quantization use case. `expandX` supports the data types `INT8` and `HIFLOAT8`.
          - When the data type of `expandX` is `INT8`, the following use cases exist:
              - The input `scalesOptional` indicates the quantization coefficient, and the shape is (1, );
              - When the input `scalesOptional` indicates the smoothing weight shared by each expert, the shape is (`H`,);
              - When the input `scalesOptional` indicates the quantization coefficient fused with the smoothing weight of each expert, the shape is (`sharedExpertNum` + `moeExpertNum`, `H`) if there are shared expert devices, and (`moeExpertNum`, `H`) if there are no shared expert devices.
          - When the data type of `expandX` is `HIFLOAT8`, the shape of `scalesOptional` must be (1, ).
      - When `quantMode` is 2, it indicates the pertoken dynamic quantization use case. `expandX` supports the data types `INT8`, `FLOAT8_E4M3FN`, and `FLOAT8_E5M2`.
          - A null pointer can be passed for the input `scalesOptional`.
          - If valid data is passed for the input `scalesOptional` and there are shared expert devices, its shape is (`sharedExpertNum` + `moeExpertNum`, `H`).
          - If valid data is passed for the input `scalesOptional` and there are no shared expert devices, its shape is (`moeExpertNum`, `H`).
          - The shape of the output `dynamicScalesOut` is `(A, )`
      - When `quantMode` is set to 3, it indicates the pergroup dynamic quantization use case. expandX supports the data types `FLOAT8_E4M3FN` and `FLOAT8_E5M2`.
          - A null pointer can be passed for the input `scalesOptional`.
          - If valid data is passed for the input `scalesOptional` and there are shared expert devices, its shape is (`sharedExpertNum` + `moeExpertNum`, `H`).
          - If valid data is passed for the input `scalesOptional` and there are no shared expert devices, its shape is (`moeExpertNum`, `H`).
          - The shape of the output `dynamicScalesOut` is `(A, Ceil(H, 128))`, where `Ceil(H, 128) = (H + 128 - 1) / 128`
      - When `quantMode` is set to 4, it indicates the mx quantization use case. `expandX` supports the data types `FLOAT8_E4M3FN`, `FLOAT8_E5M2`, `FLOAT4_E2M1`, and `FLOAT4_E1M2`. A null pointer must be passed for the input `scalesOptional`. The shape of the output `dynamicScalesOut` is `(A, Ceil(H, 64))`, where `Ceil(H, 64) = (H + 64 - 1) / 64`. If the data type of `expandX` is `FLOAT4_E2M1` or `FLOAT4_E1M2`, H must be an even number.

  <!-- end id15 -->

- **Environment Variable Constraints**:
  - **HCCL_BUFFSIZE**:
      Before calling this API, check whether the value of the HCCL_BUFFSIZE environment variable is reasonable. This environment variable indicates the memory size occupied by a single communication domain, in MB. The default value is 200 MB when not configured.

      <!-- npu="910b" id16 -->
      - <term>Atlas A2 products</term>:
          - commAlg is configured as "" or nullptr: Based on the HCCL_INTRA_PCIE_ENABLE and HCCL_INTRA_ROCE_ENABLE environment variable configuration, selects the "fullmesh" or "hierarchy" formula.
          - When commAlg is set to "fullmesh": if the data type of `x` is `INT32`, the value must be >= 2 \* (BS \* epWorldSize \* min(localExpertNum, K) \* (H \* sizeof(int32) + 32B) + 2MB); for other data types, the value must be >= 2 \* (BS \* epWorldSize \* min(localExpertNum, K) \* H \* sizeof(uint16) + 2MB).
          - commAlg is configured as "hierarchy": Must be >= (`moeExpertNum` + `epWorldSize` / 4) \* Align512(`maxBS` \* (`H` \* 2 + 16 \* Align8(`K`))) \* 1B + 8MB, where Align8(x) = ((x + 8 - 1) / 8) \* 8, Align512(x) = ((x + 512 - 1) / 512) \* 512.
      <!-- end id16 -->
      <!-- npu="A3" id17 -->
      - <term>Atlas A3 products</term>:
        - When commAlg is "fullmesh_v1" or an empty string or a null pointer: The required value must satisfy ≥ 2 \* (localExpertNum \* maxBS \* epWorldSize \* Align512(Align32(2 \* H) + 64) + (K + sharedExpertNum) \* maxBS \* Align512(2 \* H)).
        - When commAlg is set to "fullmesh_v2": the value must satisfy ≥ 2 \* (localExpertNum \* maxBS \* epWorldSize \* 480Align512(Align32(2 \* H) + 64) + (K + sharedExpertNum) \* maxBS \* Align512(2 \* H)).
        - Where `480Align512(x) = ((x + 480 - 1) / 480) * 512`, `Align512(x) = ((x + 512 - 1) / 512) * 512`, and `Align32(x) = ((x + 32 - 1) / 32) * 32`.
        - When commAlg is set to "hierarchy": the value must satisfy (moeExpertNum \* maxBS \* (H \* 2 + (3 \* (K + 7) / 8 \* 8)) \* 4 + 64) + 404 \* 1024 \* 1024.
      <!-- end id17 -->
      <!-- npu="950" id18 -->
      - <term>Ascend 950PR&950DT products</term>:
        - When commAlg is "fullmesh_v1" or an empty string or a null pointer: The required value must satisfy ≥ 2 \* (localExpertNum \* maxBS \* epWorldSize \* Align512(Align32(2 \* H) + 64) + (K + sharedExpertNum) \* maxBS \* Align512(2 \* H)).
        - When commAlg is set to "fullmesh_v2": the value must satisfy ≥ 2 \* (localExpertNum \* maxBS \* epWorldSize \* 480Align512(Align32(2 \* H) + 64) + (K + sharedExpertNum) \* maxBS \* Align512(2 \* H)).
        - Where `480Align512(x) = ((x + 480 - 1) / 480) * 512`, `Align512(x) = ((x + 512 - 1) / 512) * 512`, and `Align32(x) = ((x + 32 - 1) / 32) * 32`.

      <!-- end id18 -->

  - **HCCL_INTRA_PCIE_ENABLE** and **HCCL_INTRA_ROCE_ENABLE**:

      <!-- npu="910b" id19 -->
      - <term>Atlas A2 products</term>: These environment variables are no longer recommended. It is recommended to configure commAlg as "hierarchy".
      <!-- end id19 -->
      <!-- npu="950,A3" id20 -->
      - <term>Atlas A3 products</term>, Ascend 950PR&950DT products: this environment variable is not supported.
      <!-- end id20 -->

  - **HCCL_LOGIC_SUPERPOD_ID**

      <!-- npu="A3" id21 -->
      - <term>Atlas A3 products</term>: when commAlg is set to "hierarchy", this environment variable must be configured based on different logical SuperPoDs. For example, set `export HCCL_LOGIC_SUPERPOD_ID=0` and `export HCCL_LOGIC_SUPERPOD_ID=1` on the two hosts respectively.

      <!-- end id21 -->

- **Communication Domain Usage Constraints**:
  - The CombineV4 series operators and `aclnnMoeDistributeDispatchV4` in a model only support the same EP communication domain, and no other operators are allowed in this communication domain.
  - The CombineV4 series operator and `aclnnMoeDistributeDispatchV4` in a model do not support TP-domain communication in the current version.

  <!-- npu="A3" id22 -->
  - <term>Atlas A3 products</term>: The communicator supports cross-SuperPoD communication only when commAlg is set to "hierarchy". In other use cases, nodes in a communicator must be in the same SuperPoD, and cross-SuperPoD communication is not supported.

  <!-- end id22 -->

- **Communication mode constraints**:

  <!-- npu="950" id23 -->
  - <term>Ascend 950PR&950DT products</term>: Only UB Memory communication is supported.

  <!-- end id23 -->

- **Networking Constraints**:

  <!-- npu="910b" id24 -->
  - <term>Atlas A2 products</term>: In multi-machine scenarios, only switch networking is supported. Direct-connect networking between two machines is not supported.

  <!-- end id24 -->

- **Other Constraints**:
  - In the formula, "/" indicates integer division.
  - <code>moeExpertNum + zeroExpertNum + copyExpertNum + constExpertNum < MAX_INT32</code>.

## Example

<!-- npu="910b" id25 -->
- <term>Atlas A2 products</term>:

    This sample supports running the A2 operator in a single-node environment with a card count in the range [2, 8]. You can set EP_WORLD_SIZE_A2 to the card count in the sample code as needed, and modify moeExpertNum so that moeExpertNum is divisible by EP_WORLD_SIZE_A2.

    - Compile the operator: The operator compilation command is as follows. Both the moe_distribute_dispatch_v2 and moe_distribute_combine_v2 operators need to be compiled, and these two operators must be executed in pairs.

        ```bash
        bash build.sh --pkg --soc=ascend910b --ops=moe_distribute_dispatch_v2,moe_distribute_combine_v2
        ```

    - Create the A2 sample code: After compilation, refer to the existing [test_aclnn_moe_distribute_dispatch_v2.cpp](https://gitcode.com/cann/ops-transformer/tree/9.1.0/mc2/moe_distribute_dispatch_v2/examples/test_aclnn_moe_distribute_dispatch_v2.cpp) file in the operator [examples](https://gitcode.com/cann/ops-transformer/tree/9.1.0/mc2/moe_distribute_dispatch_v2/examples/) directory, and create a new test file test_aclnn_moe_distribute_dispatch_v4.cpp using the A2 sample code.

    - Execute operator samples: The sample operator execution command is as follows. This command executes all sample code files in the operator [examples](https://gitcode.com/cann/ops-transformer/tree/9.1.0/mc2/moe_distribute_dispatch_v2/examples/) directory.

        ```bash
        bash build.sh --run_example --ops=moe_distribute_dispatch_v2 eager cust
        ```

    - A2 sample code:

        ```Cpp
        #include <thread>
        #include <iostream>
        #include <string>
        #include <cstring>
        #include <vector>
        #include <memory>
        #include <cstdio>
        #include "acl/acl.h"
        #include "hccl/hccl.h"
        #include "aclnn/opdev/fp16_t.h"
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
            } while (0)

        struct Args {
            uint32_t rankId;
            uint32_t epRankId;
            HcclComm hcclEpComm;
            aclrtStream dispatchV4Stream;
            aclrtStream combineV4Stream;
            aclrtContext context;
        };

        const uint32_t EP_WORLD_SIZE_A2 = 8;
        const uint32_t DEV_NUM_A2 = EP_WORLD_SIZE_A2;

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
            *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                shape.data(), shape.size(), *deviceAddr);
            return 0;
        }

        void DestroyTensor(aclTensor *tensor) {
            if (tensor != nullptr) {
                aclDestroyTensor(tensor);
            }
        }

        void FreeDeviceAddr(void *deviceAddr) {
            if (deviceAddr != nullptr) {
                aclrtFree(deviceAddr);
            }
        }

        int launchOneThreadDispatchV4AndCombineV4_A2(Args &args)
        {
            int ret = aclrtSetCurrentContext(args.context);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetCurrentContext failed, ret %d\n", ret); return ret);
            char hcomEpName[128] = {0};
            ret = HcclGetCommName(args.hcclEpComm, hcomEpName);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclGetEpCommName failed, ret %d\n", ret); return -1);
            LOG_PRINT("[INFO] rank = %d, hcomEpName = %s, dispatchV4Stream = %p, combineV4Stream = %p, \
                        context = %p\n", args.rankId, hcomEpName, args.dispatchV4Stream, args.combineV4Stream,                 \
                        args.context);

            int64_t BS = 32;
            int64_t H = 7168;
            int64_t K = 8;
            int64_t expertShardType = 0;
            int64_t sharedExpertNum = 0;
            int64_t sharedExpertRankNum = 0;
            int64_t moeExpertNum = 256;
            int64_t quantMode = 0;
            int64_t globalBS = BS * EP_WORLD_SIZE_A2;
            int64_t expertTokenNumsType = 1;
            int64_t outDtype = 0;
            int64_t commQuantMode = 0;
            int64_t groupList_type = 1;
            int64_t localExpertNum;
            int64_t A;
            int64_t zeroExpertNum = 0;
            int64_t copyExpertNum = 0;
            int64_t constExpertNum = 0; // A3 only
            std::string commAlg = "fullmesh";
            if (args.epRankId < sharedExpertRankNum) {
                localExpertNum = 1;
                A = globalBS / sharedExpertRankNum;
            } else {
                localExpertNum = moeExpertNum / (EP_WORLD_SIZE_A2 - sharedExpertRankNum);
                A = globalBS * (localExpertNum < K ? localExpertNum : K);
            }

            void *xDeviceAddr = nullptr;
            void *expertIdsDeviceAddr = nullptr;
            void *scalesDeviceAddr = nullptr;
            void *expertScalesDeviceAddr = nullptr;

            void *expandXDeviceAddr = nullptr;
            void *dynamicScalesDeviceAddr = nullptr;
            void *assistInfoForCombineDeviceAddr = nullptr;
            void *expertTokenNumsDeviceAddr = nullptr;
            void *epRecvCountsDeviceAddr = nullptr;
            void *tpRecvCountsDeviceAddr = nullptr;
            void *expandScalesDeviceAddr = nullptr;

            // Input of the zero-expert use case
            void *oriXDeviceAddr = nullptr;

            // Performance profiling
            void *performanceInfoDeviceAddr = nullptr;

            void *xOutDeviceAddr = nullptr;

            aclTensor *x = nullptr;
            aclTensor *expertIds = nullptr;
            aclTensor *scales = nullptr;
            aclTensor *xActiveMask = nullptr;
            aclTensor *expertScales = nullptr;

            aclTensor *elasticInfo = nullptr; // A3
            aclTensor *performanceInfo = nullptr;
            aclTensor *expandX = nullptr;
            aclTensor *dynamicScales = nullptr;
            aclTensor *assistInfoForCombine = nullptr; // expandIdx
            aclTensor *expertTokenNums = nullptr;
            aclTensor *epRecvCounts = nullptr;
            aclTensor *tpRecvCounts = nullptr;
            aclTensor *expandScales = nullptr;

            aclTensor *activationScale = nullptr; // Reserved parameter
            aclTensor *weightScale = nullptr; // Reserved parameter
            aclTensor *groupList = nullptr; // Reserved parameter

            aclTensor *sharedExpertX = nullptr; // A3

            aclTensor *oriX = nullptr;
            aclTensor *constExpertAlpha1 = nullptr; // A3
            aclTensor *constExpertAlpha2 = nullptr; // A3
            aclTensor *constExpertV = nullptr; // A3

            aclTensor *xOut = nullptr;

            // Define the dimensions of each variable in the current use case.
            std::vector<int64_t> xShape{BS, H};
            std::vector<int64_t> expertIdsShape{BS, K};
            std::vector<int64_t> scalesShape{moeExpertNum + 1, H};
            std::vector<int64_t> expertScalesShape{BS, K};

            std::vector<int64_t> expandXShape{A, H};
            std::vector<int64_t> dynamicScalesShape{A};
            std::vector<int64_t> assistInfoForCombineShape{A * 128};
            std::vector<int64_t> expertTokenNumsShape{localExpertNum};
            std::vector<int64_t> epRecvCountsShape{localExpertNum * EP_WORLD_SIZE_A2}; // Non-hierarchical
            std::vector<int64_t> tpRecvCountsShape{1};
            std::vector<int64_t> expandScalesShape{A};

            std::vector<int64_t> oriXShape{BS, H};
            std::vector<int64_t> xOutShape{BS, H};
            std::vector<int64_t> performanceInfoShape{EP_WORLD_SIZE_A2};

            int64_t xShapeSize = GetShapeSize(xShape);
            int64_t expertIdsShapeSize = GetShapeSize(expertIdsShape);
            int64_t scalesShapeSize = GetShapeSize(scalesShape);
            int64_t expertScalesShapeSize = GetShapeSize(expertScalesShape);

            int64_t expandXShapeSize = GetShapeSize(expandXShape);
            int64_t dynamicScalesShapeSize = GetShapeSize(dynamicScalesShape);
            int64_t assistInfoForCombineShapeSize = GetShapeSize(assistInfoForCombineShape);
            int64_t expertTokenNumsShapeSize = GetShapeSize(expertTokenNumsShape);
            int64_t epRecvCountsShapeSize = GetShapeSize(epRecvCountsShape);
            int64_t tpRecvCountsShapeSize = GetShapeSize(tpRecvCountsShape);
            int64_t expandScalesShapeSize = GetShapeSize(expandScalesShape);

            int64_t performanceInfoShapeSize = GetShapeSize(performanceInfoShape);
            int64_t oriXSize = GetShapeSize(oriXShape);

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
            std::vector<int32_t> assistInfoForCombineHostData(assistInfoForCombineShapeSize, 0);
            std::vector<int64_t> expertTokenNumsHostData(expertTokenNumsShapeSize, 0);
            std::vector<int32_t> epRecvCountsHostData(epRecvCountsShapeSize, 0);
            std::vector<int32_t> tpRecvCountsHostData(tpRecvCountsShapeSize, 0);
            std::vector<float> expandScalesHostData(expandScalesShapeSize, 0);

            std::vector<int16_t> oriXHostData(oriXSize, 1);
            std::vector<int16_t> performanceInfoHostData(performanceInfoShapeSize, 0);
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
            ret = CreateAclTensor(assistInfoForCombineHostData, assistInfoForCombineShape, &assistInfoForCombineDeviceAddr, aclDataType::ACL_INT32, &assistInfoForCombine);
            CHECK_RET(ret == ACL_SUCCESS, return ret);
            ret = CreateAclTensor(expertTokenNumsHostData, expertTokenNumsShape, &expertTokenNumsDeviceAddr, aclDataType::ACL_INT64, &expertTokenNums);
            CHECK_RET(ret == ACL_SUCCESS, return ret);
            ret = CreateAclTensor(epRecvCountsHostData, epRecvCountsShape, &epRecvCountsDeviceAddr, aclDataType::ACL_INT32, &epRecvCounts);
            CHECK_RET(ret == ACL_SUCCESS, return ret);
            ret = CreateAclTensor(tpRecvCountsHostData, tpRecvCountsShape, &tpRecvCountsDeviceAddr, aclDataType::ACL_INT32, &tpRecvCounts);
            CHECK_RET(ret == ACL_SUCCESS, return ret);
            ret = CreateAclTensor(expandScalesHostData, expandScalesShape, &expandScalesDeviceAddr, aclDataType::ACL_FLOAT, &expandScales);
            CHECK_RET(ret == ACL_SUCCESS, return ret);

            ret = CreateAclTensor(oriXHostData, oriXShape, &oriXDeviceAddr, aclDataType::ACL_BF16, &oriX);
            CHECK_RET(ret == ACL_SUCCESS, return ret);
            ret = CreateAclTensor(performanceInfoHostData, performanceInfoShape, &performanceInfoDeviceAddr, aclDataType::ACL_BF16, &performanceInfo);
            CHECK_RET(ret == ACL_SUCCESS, return ret);

            ret = CreateAclTensor(xOutHostData, xOutShape, &xOutDeviceAddr, aclDataType::ACL_BF16, &xOut);
            CHECK_RET(ret == ACL_SUCCESS, return ret);


            uint64_t dispatchWorkspaceSize = 0;
            aclOpExecutor *dispatchExecutor = nullptr;
            void *dispatchWorkspaceAddr = nullptr;

            uint64_t combineWorkspaceSize = 0;
            aclOpExecutor *combineExecutor = nullptr;
            void *combineWorkspaceAddr = nullptr;

            /**************************************** Call dispatch ********************************************/
            // Call the first-phase API.
            ret = aclnnMoeDistributeDispatchV4GetWorkspaceSize(x, expertIds, (quantMode > 0 ? scales : nullptr), xActiveMask,
                    expertScales, elasticInfo, performanceInfo, hcomEpName, EP_WORLD_SIZE_A2, args.epRankId, moeExpertNum, "", 0,
                    0, expertShardType, sharedExpertNum, sharedExpertRankNum, quantMode, globalBS,
                    expertTokenNumsType, commAlg.c_str(), zeroExpertNum, copyExpertNum, constExpertNum, expandX, dynamicScales, assistInfoForCombine, expertTokenNums, epRecvCounts,
                    tpRecvCounts, expandScales, &dispatchWorkspaceSize, &dispatchExecutor);

            CHECK_RET(ret == ACL_SUCCESS,
                LOG_PRINT("[ERROR] aclnnMoeDistributeDispatchV4GetWorkspaceSize failed. ret = %d \n", ret); return ret);

            if (dispatchWorkspaceSize > 0) {
                ret = aclrtMalloc(&dispatchWorkspaceAddr, dispatchWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
                CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc workspace failed. ret = %d \n", ret); return ret);
            }
            // Call the second-phase API.
            ret = aclnnMoeDistributeDispatchV4(dispatchWorkspaceAddr, dispatchWorkspaceSize,
                                                dispatchExecutor, args.dispatchV4Stream);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnMoeDistributeDispatchV4 failed. ret = %d \n", ret);  \
                    return ret);
            ret = aclrtSynchronizeStreamWithTimeout(args.dispatchV4Stream, 10000);
                        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] dispatch aclrtSynchronizeStreamWithTimeout failed. ret = %d \n", ret);  \
                    return ret);
            LOG_PRINT("[INFO] device_%d aclnnMoeDistributeDispatchV4 execute successfully.\n", args.rankId);
            /**************************************** Call combine ********************************************/
            // Call the first-phase API.
            ret = aclnnMoeDistributeCombineV4GetWorkspaceSize(expandX, expertIds,
                                                                assistInfoForCombine, epRecvCounts,
                                                                expertScales, tpRecvCounts,
                                                                xActiveMask, activationScale, weightScale,
                                                                groupList, expandScales, sharedExpertX,
                                                                elasticInfo, oriX, constExpertAlpha1, constExpertAlpha2, constExpertV, performanceInfo,
                                                                hcomEpName, EP_WORLD_SIZE_A2, args.epRankId, moeExpertNum,
                                                                "", 0, 0, expertShardType,
                                                                sharedExpertNum, sharedExpertRankNum, globalBS, outDtype,
                                                                commQuantMode, groupList_type, commAlg.c_str(), zeroExpertNum, copyExpertNum, constExpertNum, xOut,
                                                                &combineWorkspaceSize, &combineExecutor);
            CHECK_RET(ret == ACL_SUCCESS,
                LOG_PRINT("[ERROR] aclnnMoeDistributeCombineV4GetWorkspaceSize failed. ret = %d \n", ret); return ret);
            // Allocate device memory based on the workspaceSize calculated by the first-phase API.
            if (combineWorkspaceSize > 0) {
                ret = aclrtMalloc(&combineWorkspaceAddr, combineWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
                CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc workspace failed. ret = %d \n", ret); return ret);
            }

            // Call the second-phase API.
            ret = aclnnMoeDistributeCombineV4(combineWorkspaceAddr, combineWorkspaceSize, combineExecutor, args.combineV4Stream);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnMoeDistributeCombineV4 failed. ret = %d \n", ret);
                return ret);
            // (Boilerplate) Synchronize and wait for the task execution to complete.
            ret = aclrtSynchronizeStreamWithTimeout(args.combineV4Stream, 10000);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSynchronizeStreamWithTimeout failed. ret = %d \n", ret);
                return ret);
            LOG_PRINT("[INFO] device_%d aclnnMoeDistributeDispatchV4 and aclnnMoeDistributeCombineV4                      \
                        execute successfully.\n", args.rankId);
            // Release device resources.
            if (dispatchWorkspaceSize > 0) {
                aclrtFree(dispatchWorkspaceAddr);
            }
            if (combineWorkspaceSize > 0) {
                aclrtFree(combineWorkspaceAddr);
            }
            DestroyTensor(x);
            DestroyTensor(expertIds);
            DestroyTensor(scales);
            DestroyTensor(xActiveMask);
            DestroyTensor(expertScales);
            DestroyTensor(elasticInfo);
            DestroyTensor(performanceInfo);
            DestroyTensor(expandX);
            DestroyTensor(dynamicScales);
            DestroyTensor(assistInfoForCombine);
            DestroyTensor(expertTokenNums);
            DestroyTensor(epRecvCounts);
            DestroyTensor(tpRecvCounts);
            DestroyTensor(expandScales);
            DestroyTensor(activationScale);
            DestroyTensor(weightScale);
            DestroyTensor(groupList);
            DestroyTensor(sharedExpertX);
            DestroyTensor(oriX);
            DestroyTensor(constExpertAlpha1);
            DestroyTensor(constExpertAlpha2);
            DestroyTensor(constExpertV);
            DestroyTensor(xOut);

            FreeDeviceAddr(xDeviceAddr);
            FreeDeviceAddr(expertIdsDeviceAddr);
            FreeDeviceAddr(scalesDeviceAddr);
            FreeDeviceAddr(expertScalesDeviceAddr);
            FreeDeviceAddr(expandXDeviceAddr);
            FreeDeviceAddr(dynamicScalesDeviceAddr);
            FreeDeviceAddr(assistInfoForCombineDeviceAddr);
            FreeDeviceAddr(expertTokenNumsDeviceAddr);
            FreeDeviceAddr(epRecvCountsDeviceAddr);
            FreeDeviceAddr(tpRecvCountsDeviceAddr);
            FreeDeviceAddr(expandScalesDeviceAddr);
            FreeDeviceAddr(oriXDeviceAddr);
            FreeDeviceAddr(performanceInfoDeviceAddr);
            FreeDeviceAddr(xOutDeviceAddr);

            HcclCommDestroy(args.hcclEpComm);
            aclrtDestroyStream(args.dispatchV4Stream);
            aclrtDestroyStream(args.combineV4Stream);
            aclrtDestroyContext(args.context);
            LOG_PRINT("[INFO] device_%d DeStroy.\n", args.rankId);
            aclrtResetDevice(args.rankId);
            LOG_PRINT("[INFO] device_%d Reset.\n", args.rankId);
            return 0;
        }
        int main(int argc, char *argv[])
        {
            LOG_PRINT("[INFO] run_example_on_A2.\n");
            int ret = aclInit(nullptr);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclInit failed, ret = %d\n", ret); return ret);
            aclrtStream dispatchV4Stream[DEV_NUM_A2];
            aclrtStream combineV4Stream[DEV_NUM_A2];
            aclrtContext context[DEV_NUM_A2];
            for (uint32_t rankId = 0; rankId < DEV_NUM_A2; rankId++) {
                ret = aclrtSetDevice(rankId);
                CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetDevice failed, ret = %d\n", ret); return ret);
                ret = aclrtCreateContext(&context[rankId], rankId);
                CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateContext failed, ret = %d\n", ret); return ret);
                ret = aclrtCreateStream(&dispatchV4Stream[rankId]);
                CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateStream failed, ret = %d\n", ret); return ret);
                ret = aclrtCreateStream(&combineV4Stream[rankId]);
                CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateStream failed, ret = %d\n", ret); return ret);
            }

            int32_t devicesEp[EP_WORLD_SIZE_A2];
            for (int32_t epId = 0; epId < EP_WORLD_SIZE_A2; epId++) {
                devicesEp[epId] = epId;
            }

            HcclComm commsEp[EP_WORLD_SIZE_A2];
            ret = HcclCommInitAll(EP_WORLD_SIZE_A2, devicesEp, commsEp);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclCommInitAll ep failed, ret %d\n", ret); return ret);

            Args args[DEV_NUM_A2];
            std::vector<std::unique_ptr<std::thread>> threads(DEV_NUM_A2);
            for (uint32_t rankId = 0; rankId < DEV_NUM_A2; rankId++) {
                uint32_t epRankId = rankId;

                args[rankId].rankId = rankId;
                args[rankId].epRankId = epRankId;
                args[rankId].hcclEpComm = commsEp[epRankId];
                args[rankId].dispatchV4Stream = dispatchV4Stream[rankId];
                args[rankId].combineV4Stream = combineV4Stream[rankId];
                args[rankId].context = context[rankId];
                threads[rankId].reset(new (std::nothrow) std::thread(&launchOneThreadDispatchV4AndCombineV4_A2, std::ref(args[rankId])));
            }

            for (uint32_t rankId = 0; rankId < DEV_NUM_A2; rankId++) {
                threads[rankId]->join();
            }

            aclFinalize();
            LOG_PRINT("[INFO] aclFinalize success\n");
            return 0;
        }
        ```

<!-- end id25 -->
<!-- npu="950" id26 -->
- <term>Ascend 950PR&950DT products</term>: Refer to the preparation section and example code in the API call of [aclnnMoeDistributeDispatchV2](../docs/aclnnMoeDistributeDispatchV2_en.md), and reset the involved variables according to the preceding constraints. For the task parameters newly added in the V3 API compared with the V2 API, pass values according to the preceding parameter descriptions.

<!-- end id26 -->
<!-- npu="A3" id27 -->
- <term>Atlas A3 products</term>:

    For the compilation and execution procedure, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

<!-- end id27 -->
- The sample code is as follows, for reference only.

    ```Cpp
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
        } while (0)

    struct Args {
        uint32_t rankId;
        uint32_t epRankId;
        HcclComm hcclEpComm;
        aclrtStream dispatchStream;
        aclrtStream combineStream;
        aclrtContext context;
    };

    constexpr uint32_t EP_WORLD_SIZE = 2;
    constexpr uint32_t DEV_NUM = EP_WORLD_SIZE;

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
        LOG_PRINT("[INFO] rank = %d, hcomEpName = %s, dispatchStream = %p, combineStream = %p, \
                    context = %p\n", args.rankId, hcomEpName, args.dispatchStream, args.combineStream,                 \
                    args.context);

        int64_t BS = 8;
        int64_t H = 7168;
        int64_t K = 3;
        int64_t expertShardType = 0;
        int64_t sharedExpertNum = 1;
        int64_t sharedExpertRankNum = 1;
        int64_t moeExpertNum = 7;
        int64_t quantMode = 0;
        int64_t globalBS = BS * EP_WORLD_SIZE;
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
            A = globalBS / sharedExpertRankNum;
        } else {
            localExpertNum = moeExpertNum / (EP_WORLD_SIZE - sharedExpertRankNum);
            A = globalBS * (localExpertNum < K ? localExpertNum : K);
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

        // Define the dimensions of variables in the current use case.
        std::vector<int64_t> xShape{BS, H};
        std::vector<int64_t> expertIdsShape{BS, K};
        std::vector<int64_t> scalesShape{moeExpertNum + 1, H};
        std::vector<int64_t> expertScalesShape{BS, K};

        std::vector<int64_t> expandXShape{A, H};
        std::vector<int64_t> dynamicScalesShape{A};
        std::vector<int64_t> expandIdxShape{A * 128};
        std::vector<int64_t> expertTokenNumsShape{localExpertNum};
        std::vector<int64_t> epRecvCountsShape{localExpertNum * EP_WORLD_SIZE};
        std::vector<int64_t> tpRecvCountsShape{1};
        std::vector<int64_t> expandScalesShape{A};
        std::vector<int64_t> sharedExpertXShape{BS, 1, H};


        std::vector<int64_t> elasticInfoShape{4 + EP_WORLD_SIZE * 2};
        std::vector<int64_t> oriXShape{BS, H};
        std::vector<int64_t> constExpertAlpha1Shape{constExpertNum, H};
        std::vector<int64_t> constExpertAlpha2Shape{constExpertNum, H};
        std::vector<int64_t> constExpertVShape{constExpertNum, H};

        std::vector<int64_t> xOutShape{BS, H};

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

        /**************************************** Call dispatch ********************************************/
        // Call the first-phase API.
        ret = aclnnMoeDistributeDispatchV4GetWorkspaceSize(x, expertIds, (quantMode > 0 ? scales : nullptr), nullptr,
                expertScales, elasticInfo, nullptr,// performanceInfoOptional
                hcomEpName, EP_WORLD_SIZE, args.epRankId, moeExpertNum, "", 0,
                0, expertShardType, sharedExpertNum, sharedExpertRankNum, quantMode, globalBS,
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
        /**************************************** Call combine. ********************************************/
        // Call the first-phase API.
        ret = aclnnMoeDistributeCombineV4GetWorkspaceSize(expandX, expertIds,
                                                            expandIdx, epRecvCounts,
                                                            expertScales, tpRecvCounts,
                                                            nullptr, nullptr, nullptr,
                                                            nullptr, nullptr, nullptr,
                                                            elasticInfo, oriX, constExpertAlpha1, constExpertAlpha2, constExpertV,
                                                            nullptr,// performanceInfoOptional
                                                            hcomEpName, EP_WORLD_SIZE, args.epRankId, moeExpertNum,
                                                            "", 0, 0, expertShardType,
                                                            sharedExpertNum, sharedExpertRankNum, globalBS, outDtype,
                                                            commQuantMode, groupList_type, nullptr, zeroExpertNum, copyExpertNum, constExpertNum, xOut,
                                                            &combineWorkspaceSize, &combineExecutor);
        CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("[ERROR] aclnnMoeDistributeCombineV4GetWorkspaceSize failed. ret = %d \n", ret); return ret);
        // Allocate device memory based on the workspaceSize calculated by the first-phase API.
        if (combineWorkspaceSize > 0) {
            ret = aclrtMalloc(&combineWorkspaceAddr, combineWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc workspace failed. ret = %d \n", ret); return ret);
        }

        // Call the second-phase API.
        ret = aclnnMoeDistributeCombineV4(combineWorkspaceAddr, combineWorkspaceSize, combineExecutor, args.combineStream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnMoeDistributeCombineV4 failed. ret = %d \n", ret);
            return ret);
        // (Boilerplate) Wait synchronously for task execution to complete.
        ret = aclrtSynchronizeStreamWithTimeout(args.combineStream, 10000);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSynchronizeStreamWithTimeout failed. ret = %d \n", ret);
            return ret);
        LOG_PRINT("[INFO] device_%d aclnnMoeDistributeDispatchV4 and aclnnMoeDistributeCombineV4                      \
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
        if (tpRecvCountsDeviceAddr != nullptr) {
            aclrtFree(tpRecvCountsDeviceAddr);
        }
        if (expandScalesDeviceAddr != nullptr) {
            aclrtFree(expandScalesDeviceAddr);
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

        int32_t devicesEp[EP_WORLD_SIZE];
        for (int32_t epId = 0; epId < EP_WORLD_SIZE; epId++) {
            devicesEp[epId] = epId;
        }

        HcclComm commsEp[EP_WORLD_SIZE];
        ret = HcclCommInitAll(EP_WORLD_SIZE, devicesEp, commsEp);
        CHECK_RET(ret == ACL_SUCCESS,
                    LOG_PRINT("[ERROR] HcclCommInitAll ep failed, ret %d\n", ret); return ret);

        Args args[DEV_NUM];
        std::vector<std::unique_ptr<std::thread>> threads(DEV_NUM);
        for (uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
            uint32_t epRankId = rankId;

            args[rankId].rankId = rankId;
            args[rankId].epRankId = epRankId;
            args[rankId].hcclEpComm = commsEp[epRankId];
            args[rankId].dispatchStream = dispatchStream[rankId];
            args[rankId].combineStream = combineStream[rankId];
            args[rankId].context = context[rankId];
            threads[rankId].reset(new (std::nothrow) std::thread(&LaunchOneProcessDispatchAndCombine, std::ref(args[rankId])));
        }

        for (uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
            threads[rankId]->join();
        }

        aclFinalize();
        LOG_PRINT("[INFO] aclFinalize success\n");

        return 0;
    }
    ```
