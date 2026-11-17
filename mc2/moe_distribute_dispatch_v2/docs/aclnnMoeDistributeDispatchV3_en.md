# aclnnMoeDistributeDispatchV3

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T14:34:13.281Z -->

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

- Description: Performs optional quantization on token data and AllToAllV communication in the EP (Expert Parallelism) domain. TP (Tensor Parallelism) domain communication is no longer supported in the current version. TP-related parameters are reserved parameters.

    Compared with the `aclnnMoeDistributeDispatchV2` API, this API has the following changes:
    - Added support for special expert scenarios.
        - zeroExpertNum≠0: Set zeroExpertNum to a value greater than 0 to enable this feature.

            $$Moe(oriXOptional) = 0$$

        - copyExpertNum≠0: To enable this feature, pass in a copyExpertNum parameter greater than 0. Also pass in a valid oriXOptional parameter.

            $$Moe(oriXOptional) = oriXOptional$$

        - constExpertNum≠0: To enable this feature, pass in a constExpertNum parameter greater than 0. Also pass in valid oriXOptional, constExpertAlpha1Optional, constExpertAlpha2Optional, and constExpertVOptional parameters.

            $$Moe(oriXOptional) = constExpertAlpha1Optional * oriXOptional + constExpertAlpha2Optional * constExpertVOptional$$

        For details, see the following parameter descriptions.
        For parameters oriXOptional, constExpertAlpha1Optional, constExpertAlpha2Optional, and constExpertVOptional, see aclnnMoeDistributeCombineV3.md.

- Formula:

    $$
    expandXOut = AllToAllV(X)
    $$

    When the data type of `x` is `INT32`, `scalesOptional` is communicated together with the corresponding token:

    $$
    dynamicScalesOut = AllToAllV(scalesOptional)
    $$

    > NOTE: TP domain communication is no longer supported in the current version.

<!-- npu="910b" id7 -->
- <term>Atlas A2 products</term>: This API must be used together with `aclnnMoeDistributeCombineV3`.
<!-- end id7 -->
<!-- npu="950,A3" id8 -->
- <term>Atlas A3 products</term>, Ascend 950PR&950DT products: This API must be used together with `aclnnMoeDistributeCombineV3` or `aclnnMoeDistributeCombineAddRmsNormV2`.

<!-- end id8 -->

> NOTE
> `aclnnMoeDistributeCombineV3` and `aclnnMoeDistributeCombineAddRmsNormV2` are collectively referred to as **CombineV3 series operators** in subsequent documents.

## Function Prototype

Each operator is implemented as a two-phase API. Call "aclnnMoeDistributeDispatchV3GetWorkspaceSize" first to obtain the workspace size required for computation and the executor that contains the operator computation process, and then call "aclnnMoeDistributeDispatchV3" to execute the computation.

```cpp
aclnnStatus aclnnMoeDistributeDispatchV3GetWorkspaceSize(
    const aclTensor* x,
    const aclTensor* expertIds,
    const aclTensor* scalesOptional,
    const aclTensor* xActiveMaskOptional,
    const aclTensor* expertScalesOptional,
    const aclTensor* elasticInfoOptional,
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
aclnnStatus aclnnMoeDistributeDispatchV3(
    void*           workspace,
    uint64_t        workspaceSize,
    aclOpExecutor*  executor,
    aclrtStream     stream)
```

## aclnnMoeDistributeDispatchV3GetWorkspaceSize

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
    <td>Token data sent by this device.</td>
    <td>Requires a 2D tensor.</td>
    <td>FLOAT16, BFLOAT16, FLOAT8_E5M2, FLOAT8_E4M3FN, HIFLOAT8, FLOAT4_E2M1, FLOAT4_E1M2, INT32</td>
    <td>ND</td>
    <td><code>(BS, H)</code> (BS=batch size, H=hidden size)</td>
    <td>√</td>
    </tr>
    <tr>
    <td>expertIds</td>
    <td>Input</td>
    <td>TopK expert indexes of each token.</td>
    <td>Requires a 2D tensor.</td>
    <td>INT32</td>
    <td>ND</td>
    <td><code>(BS, K)</code></td>
    <td>√</td>
    </tr>
    <tr>
    <td>scalesOptional</td>
    <td>Input</td>
    <td>Quantization smoothing parameters of each expert or scaling parameters of INT32 tokens.</td>
    <td>-</td>
    <td>FLOAT32, FLOAT8_E8M0</td>
    <td>ND</td>
    <td>-</td>
    <td>√</td>
    </tr>
    <tr>
    <td>xActiveMaskOptional</td>
    <td>Input</td>
    <td>Indicates whether tokens participate in communication.</td>
    <td>Valid data or a null pointer can be passed in.<br>If the input is 1D, true indicates that the corresponding token participates in communication. true must precede false. For example, {true, false, true} is invalid input;<br>If the input is 2D, true indicates that the expert_ids corresponding to the current token participate in communication. If all K BOOL values corresponding to the current token are false, the current token does not participate in communication. By default, all tokens participate in communication. When the number of BS differs across devices, all tokens must be valid.</td>
    <td>BOOL</td>
    <td>ND</td>
    <td><br>If the input is 1D, the shape is <code>(BS,)</code>; if the input is 2D, the shape is <code>(BS, K)</code></td>
    <td>√</td>
    </tr>
    <tr>
    <td>expertScalesOptional</td>
    <td>Input</td>
    <td>topK expert weights of each token.</td>
    <td>-</td>
    <td>FLOAT32</td>
    <td>ND</td>
    <td><code>(BS, K)</code></td>
    <td>√</td>
    </tr>
    <tr>
    <td>elasticInfoOptional</td>
    <td>Input</td>
    <td>Dynamic scale-in information of the EP communicator.</td>
    <td>When some devices are removed from the communicator due to faults, the number of devices actually participating in the communication can be obtained from this parameter.</td>
    <td>INT32</td>
    <td>ND</td>
    <td>-</td>
    <td>√</td>
    </tr>
    <tr>
    <td>groupEp</td>
    <td>Input</td>
    <td>Name of the EP communicator (expert parallelism communicator).</td>
    <td>The string length is <code>[1, 128)</code>.</td>
    <td>STRING</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>epWorldSize</td>
    <td>Input</td>
    <td>Size of the EP communicator.</td>
    <td>-</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>epRankId</td>
    <td>Input</td>
    <td>ID of the current device in the EP communicator domain.</td>
    <td><ul><li>value range <code>[0, epWorldSize)</code></li><li>In the same EP communicator domain, each device has a unique <code>epRankId</code>.</li></ul></td>
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
    <td>Default value: "".</td>
    <td>STRING</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>tpWorldSize</td>
    <td>Input</td>
    <td>TP communicator size (reserved parameter, not supported in the current version. Only 0 or 1 can be passed).</td>
    <td>-</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>tpRankId</td>
    <td>Input</td>
    <td>TP domain ID of the current device (reserved parameter, not supported in the current version; pass 0).</td>
    <td>Reserved parameter. Not supported in the current version. Pass 0.</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>expertShardType</td>
    <td>Input</td>
    <td>Distribution type of shared expert cards.</td>
    <td>-</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>sharedExpertNum</td>
    <td>Input</td>
    <td>Number of shared experts (one shared expert can be replicated and deployed on multiple devices).</td>
    <td>-</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>sharedExpertRankNum</td>
    <td>Input</td>
    <td>Number of shared expert cards.</td>
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
    <td><br><li> If BS is the same on all devices: <code>globalBS = BS * epWorldSize</code> or 0;</li> <li> If BS differs across devices: <code>globalBS = maxBS * epWorldSize</code>, where maxBS is the maximum BS on a single device.</li></td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>expertTokenNumsType</td>
    <td>Input</td>
    <td>Semantic type of values in output <code>expertTokenNums</code>.</td>
    <td>Supported values: 0: the output in <code>expertTokenNums</code> is the prefix sum of the number of tokens processed by each expert; 1: the output in <code>expertTokenNums</code> is the number of tokens processed by each expert.</td>
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
    <td>Token features expanded based on <code>expertIds</code>.</td>
    <td>2D Tensor</td>
    <td>FLOAT16, BFLOAT16, INT8, FLOAT8_E4M3FN, FLOAT8_E5M2, HIFLOAT8, FLOAT4_E2M1, FLOAT4_E1M2, INT32</td>
    <td>-</td>
    <td><code>(A, H)</code></td>
    <td>√</td>
    </tr>
    <tr>
    <td>dynamicScalesOut</td>
    <td>Output</td>
    <td>Scaling parameters.</td>
    <td>Must be a 1D or 2D tensor. Output exists when quantMode is 2, 3, or 4. Output also exists when quantMode is 0 and the data type of `x` is `HIFLOAT8`, `FLOAT8_E5M2`, `FLOAT8_E4M3FN`, `FLOAT4_E2M1`, `FLOAT4_E1M2`, or `INT32`.</td>
    <td>FLOAT32, FLOAT8_E8M0</td>
    <td>-</td>
    <td>-</td>
    <td>√</td>
    </tr>
    <tr>
    <td>assistInfoForCombineOut</td>
    <td>Output</td>
    <td>Number of tokens sent to the same expert (corresponding to <code>assistInfoForCombine</code> in <code>aclnnMoeDistributeCombineV3</code>).</td>
    <td>1D Tensor.</td>
    <td>INT32</td>
    <td>-</td>
    <td><code>(A * 128,)</code></td>
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
    <td>Number of tokens received from each device in the EP communicator (corresponding to <code>epSendCounts</code> in <code>aclnnMoeDistributeCombineV3</code>).</td>
    <td>1D Tensor.</td>
    <td>INT32</td>
    <td>-</td>
    <td>-</td>
    <td>√</td>
    </tr>
    <tr>
    <td>tpRecvCountsOut</td>
    <td>Output</td>
    <td>Reserved output. Not supported in the current version. Pass a null pointer.</td>
    <td>Reserved parameter. Not supported in the current version. Pass a null pointer.</td>
    <td>INT32</td>
    <td>-</td>
    <td>-</td>
    <td>√</td>
    </tr>
    <tr>
    <td>expandScalesOut</td>
    <td>Output</td>
    <td>Weight of the output tokens on this device (corresponding to <code>expertScalesOptional</code> in <code>aclnnMoeDistributeCombineV3</code>).</td>
    <td>-</td>
    <td>FLOAT32</td>
    <td>-</td>
    <td>-</td>
    <td>√</td>
    </tr>
    <tr>
    <td>workspaceSize</td>
    <td>Output</td>
    <td>Returns the workspace size to be requested on the Device side.</td>
    <td>-</td>
    <td>UINT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>executor</td>
    <td>Output</td>
    <td>Returns the op executor containing the operator computation flow.</td>
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

    - dynamicScalesOut has an output when quantMode is 2 or the data type of `x` is INT32.
    - commAlg supports nullptr, "", "fullmesh", and "hierarchy"; "hierarchy" is recommended, with driver version ≥ 25.0.RC1.1; nullptr and "" select an algorithm based on HCCL environment variables (not recommended); "fullmesh" directly transfers tokens through RDMA; "hierarchy" sends data twice (across nodes and within a node) to optimize communication.
    - When commAlg is "hierarchy", or HCCL_INTRA_PCIE_ENABLE=1 and HCCL_INTRA_ROCE_ENABLE=0, scalesOptional must be nullptr; when commAlg is "fullmesh", scalesOptional can be valid data or a null pointer. If the data type of `x` is INT32, commAlg must be set to "fullmesh" and valid data must be passed to scalesOptional.
    - xActiveMaskOptional depends on commAlg. "fullmesh" requires a 1D tensor with shape (BS, ), where true must be placed before false (for example, {true, false, true} is invalid); "hierarchy" is not supported in the current version; pass a null pointer.
    - expertScalesOptional must be a 2D tensor with shape (BS, K).
    - epWorldSize depends on commAlg. "fullmesh" supports 2, 3, 4, 5, 6, 7, 8, 16, 32, 64, 128, 192, 256, and 384. "hierarchy" supports 16, 32, and 64.
    - moeExpertNum depends on commAlg. "fullmesh" supports (0, 1024]; "hierarchy" supports (0, 512].
    - groupTp is not supported in the current version. Pass an empty string.
    - tpWorldSize, tpRankId, expertShardType, sharedExpertNum, and sharedExpertRankNum are not supported in the current version. Pass 0.
    - The shape of epRecvCountsOut is (moeExpertNum + 2 \* globalBS \* K \* serverNum,). The first moeExpertNum values are the received token counts, and the rest are information related to reduce before communication.
    - TP domain communication is not supported currently.
    - expandScalesOut must be a 1D tensor with shape (A,).
    - quantMode supports 0 (non-quantized) and 2 (dynamic quantization).
    - elasticInfoOptional is not supported in the current version. Pass a null pointer.
    - zeroExpertNum: when commAlg="fullmesh", the value range is [0, MAX_INT32), where MAX_INT32 = 2^31 - 1. Valid zero expert IDs are [<code>moeExpertNum</code>, <code>moeExpertNum + zeroExpertNum</code>).
    - copyExpertNum: when commAlg="fullmesh", the value range is [0, MAX_INT32), where MAX_INT32 = 2^31 - 1. Valid copy expert IDs are [<code>moeExpertNum + zeroExpertNum</code>, <code>moeExpertNum + zeroExpertNum + copyExpertNum</code>).
    - constExpertNum is not supported in the current version. Pass 0.
    </details>
    <!-- end id9 -->

    <!-- npu="A3" id10 -->
    <details>
    <summary><term>Atlas A3 products</term>  :</summary>

    - dynamicScalesOut has output only when quantMode is set to 2.
    - commAlg supports three input values: "", "fullmesh_v1", and "fullmesh_v2". "": default value, enables the fullmesh_v1 template; "fullmesh_v1": enables the fullmesh_v1 template; "fullmesh_v2": enables the fullmesh_v2 template.
    - xActiveMaskOptional must be a 1D or 2D tensor (for 1D, the shape is (BS, ); for 2D, the shape is (BS, K)); in the 1D case, true must be placed before false; in the 2D case, a token does not participate in communication if all K values corresponding to the token are false.
    - expertScalesOptional is not supported in the current version. Pass a null pointer.
    - epWorldSize: value range [2, 768].
    - moeExpertNum: value range (0, 1024].
    - groupTp is not supported in the current version. Pass an empty string.
    - tpWorldSize (reserved parameter, which is not supported in the current version; only 0 or 1 is supported).
    - tpRankId (reserved parameter, which is not supported in the current version; only 0 is supported).
    - expertShardType: only 0 is supported currently, indicating that shared expert cards are placed before MoE expert cards.
    - sharedExpertNum: the current value range is [0, 4].
    - sharedExpertRankNum: the value range is [0, epWorldSize). If the value is 0, sharedExpertNum must be 0 or 1. If the value is not 0, sharedExpertRankNum % sharedExpertNum = 0 must be met.
    - The shape of epRecvCountsOut is (epWorldSize \* localExpertNum,).
    - tpRecvCountsOut (reserved output, which is not supported in the current version; set it to a null pointer).
    - expandScalesOut is not supported in the current version.
    - quantMode supports 0 (non-quantized) and 2 (dynamic quantization).
    - elasticInfoOptional is not supported in the current version. Set it to a null pointer.
    - zeroExpertNum value range: [0, MAX_INT32). MAX_INT32 = 2^31 - 1. Valid zero expert ID values are in <code>[moeExpertNum, moeExpertNum + zeroExpertNum)</code>.
    - copyExpertNum value range: [0, MAX_INT32). MAX_INT32 = 2^31 - 1. Expert ID range: <code>[moeExpertNum + zeroExpertNum, moeExpertNum + zeroExpertNum + copyExpertNum)</code>.
    - constExpertNum value range: [0, MAX_INT32). MAX_INT32 = 2^31 - 1. Expert ID range: <code>[moeExpertNum + zeroExpertNum + copyExpertNum, moeExpertNum + zeroExpertNum + copyExpertNum + constExpertNum)</code>.
    - scalesOptional 2D Tensor. For non-quantized use cases, pass a null pointer. For dynamic quantization, pass valid data or a null pointer.
    </details>
    <!-- end id10 -->

    <!-- npu="950" id11 -->
    <details>
    <summary><term>Ascend 950PR&950DT products</term>:</summary>

    - dynamicScalesOut has output when quantMode is 2, 3, or 4. It also has output when quantMode is 0 and the data type of `x` is `HIFLOAT8`, `FLOAT8_E5M2`, `FLOAT8_E4M3FN`, `FLOAT4_E2M1`, or `FLOAT4_E1M2`.
    - commAlg supports "", "fullmesh_v1", and "fullmesh_v2". "": default value, enables the fullmesh_v1 template. "fullmesh_v1": enables the fullmesh_v1 template. "fullmesh_v2": enables the fullmesh_v2 template.
    - xActiveMaskOptional must be a 1D or 2D Tensor (1D: shape (BS, ); 2D: shape (BS, K)). For 1D, true must be placed before false. For 2D, a token does not participate in communication if all its K values are false.
    - expertScalesOptional can be a valid Tensor or a null pointer. If a valid Tensor is passed, the expertScales expert weight feature is enabled. The valid Tensor is a 2D Tensor with shape (BS, K). If a null pointer is passed, the expertScales expert weight feature is disabled. Empty Tensors are not supported.
    - epWorldSize: value range [2, 1024].
    - moeExpertNum: value range (0, 1024].
    - groupTp: Not supported in the current version. Pass an empty string.
    - tpWorldSize: Not supported in the current version. Pass 0.
    - tpRankId: Not supported in the current version. Pass 0.
    - expertShardType: currently only 0 is supported, indicating that shared expert cards are placed before MoE expert cards.
    - sharedExpertNum: current value range [0, 4].
    - sharedExpertRankNum: value range [0, epWorldSize). If it is 0, sharedExpertNum must be 0 or 1. If it is not 0, sharedExpertRankNum % sharedExpertNum = 0 must be satisfied.
    - The shape of epRecvCountsOut is (epWorldSize \* localExpertNum,).
    - tpRecvCountsOut (reserved output. Not supported in the current version. Pass a null pointer.)
    - expandScalesOut: Valid when the expertScales expert weight function is enabled; invalid when disabled.
    - quantMode supports 0 (non-quantized), 1 (static quantization), 2 (pertoken dynamic quantization), 3 (pergroup dynamic quantization), and 4 (mx dynamic quantization).
    - elasticInfoOptional: Not supported in the current version. Pass a null pointer.
    - zeroExpertNum: value range [0, MAX_INT32), where MAX_INT32 = 2^31 - 1. Valid zero expert IDs are <code>[moeExpertNum, moeExpertNum + zeroExpertNum)</code>.
    - copyExpertNum: value range [0, MAX_INT32), where MAX_INT32 = 2^31 - 1. Valid expert IDs are <code>[moeExpertNum + zeroExpertNum, moeExpertNum + zeroExpertNum + copyExpertNum)</code>.
    - constExpertNum: value range [0, MAX_INT32), where MAX_INT32 = 2^31 - 1. Valid expert IDs are <code>[moeExpertNum + zeroExpertNum + copyExpertNum, moeExpertNum + zeroExpertNum + copyExpertNum + constExpertNum)</code>.
    - scalesOptional: When quantMode is 0, pass a null pointer if the data type of `x` is `FLOAT16` or `BFLOAT16`; pass valid data if the data type of `x` is `HIFLOAT8`, `FLOAT8_E5M2`, `FLOAT8_E4M3FN`, `FLOAT4_E2M1`, or `FLOAT4_E1M2`. When quantMode is 1, pass valid data. When quantMode is 2 or 3, pass valid data or a null pointer. When quantMode is 4, pass a null pointer.
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
    <th>Return Value</th>
    <th>Error Code</th>
    <th>Description</th>
    </tr>
    </thead>
    <tbody>
    <tr>
    <td>ACLNN_ERR_PARAM_NULLPTR</td>
    <td>161001</td>
    <td>Required input and output tensor parameters are null pointers.</td>
    </tr>
    <tr>
    <td>ACLNN_ERR_PARAM_INVALID</td>
    <td>161002</td>
    <td>Data types of input and output are out of the supported range.</td>
    </tr>
    <tr>
    <td rowspan="2">ACLNN_ERR_INNER_TILING_ERROR</td>
    <td rowspan="2">561002</td>
    <td>The shapes of input and output are out of the supported range.</td>
    </tr>
    <tr>
        <td>The parameter value is out of the supported range.</td>
    </tr>
    </tbody>
    </table>

## aclnnMoeDistributeDispatchV3

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
        <td>Workspace memory address allocated on the Device.</td>
    </tr>
    <tr>
        <td>workspaceSize</td>
        <td>Input</td>
        <td>Workspace size allocated on the Device. Obtained by the first-phase API aclnnMoeDistributeDispatchV3GetWorkspaceSize.</td>
    </tr>
    <tr>
        <td>executor</td>
        <td>Input</td>
        <td>op executor that contains the operator computation flow.</td>
    </tr>
    <tr>
        <td>stream</td>
        <td>Input</td>
        <td>Specifies the Stream used to execute tasks.</td>
    </tr>
    </tbody>
    </table>

- **Return Value**

    aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- **Deterministic computation**:
  - aclnnMoeDistributeDispatchV3 defaults to a deterministic implementation.

- **Driver Constraints**:
  - The driver versions of all nodes in the operator communication domain must be the same.

- **API Compatibility Constraints**:

  - `aclnnMoeDistributeDispatchV3` must be used together with the CombineV3 series operator. The `assistInfoForCombineOut`, `epRecvCountsOut`, `tpRecvCountsOut`, and `expandScalesOut` output by the former must be directly passed to the corresponding parameters of the latter. The business logic must not depend on the specific values of these tensors.

- **Parameter Consistency Constraints**:

  - The `groupEp`, `epWorldSize`, `moeExpertNum`, `groupTp`, `tpWorldSize`, `expertShardType`, `sharedExpertNum`, `sharedExpertRankNum`, `globalBS`, and `commAlg` parameters, as well as the `HCCL_BUFFSIZE` value, must be consistent across all cards, and must also be consistent with the corresponding parameters of the CombineV3 series operator.

- **Product-Specific Constraints**:

  <!-- npu="A3" id12 -->
  - <term>Atlas A3 products</term>: In this use case, each device contains two DIEs (also called "die" or "bare die"). Therefore, "current device" in Parameters refers to a single DIE.

  <!-- end id12 -->

- **Shape Variable Constraints**:

  | Variable     | Definition and Value Range                                                     |
  | :----------- | :----------------------------------------------------------------------------- |
  | A            | Maximum number of tokens to be dispatched by the current device. The value range is as follows: <ul> <li>For shared experts, ensure A = BS \* epWorldSize \* sharedExpertNum / sharedExpertRankNum.</li> <li>For MoE experts, if globalBS is 0, ensure A >= BS \* epWorldSize \* min(localExpertNum, K); if globalBS is not 0, ensure A >= globalBS \* min(localExpertNum, K).</li> </ul>|
  | H (hidden size) | Indicates the hidden size. <ul><li><term>Atlas A2 products</term>: If the data type of `x` is INT32, the value range is (0, 5120] and the value must be a multiple of 32. For other data types, the value range is (0, 10240] and the value must be a multiple of 32.</li> <li><term>Atlas A3 products</term>, Ascend 950PR&950DT products: The value range is [1024, 8192].</li> </ul> |
  | BS           | Number of final output tokens of the current device. <ul><li><term>Atlas A2 products</term>: Depending on the value of commAlg, the value range is (0 < BS ≤ 256) when commAlg is "fullmesh"; the value range is (0 < BS ≤ 512) when commAlg is "hierarchy" and the driver version is ≥25.0.RC1.1;</li><li><term>Atlas A3 products</term>: Depending on the value of commAlg, the value range is (0 < BS ≤ 256) when commAlg is "fullmesh_v2", and (0 < BS ≤ 512) when commAlg is "fullmesh_v1" or "".</li><li><term>Ascend 950PR&950DT products</term>: 0 < BS ≤ 512.</li></ul> |
  | K | Number of selected topK experts. The value range is 0 < K ≤ 16, and <code>0 < K ≤ moeExpertNum + zeroExpertNum + copyExpertNum + constExpertNum</code>. See the product-specific constraints below. |
  | serverNum    | Number of server nodes. Only 2, 4, and 8 are supported.<br>Atlas A2 products: This variable is used only by the shape in this use case.                                                  |
  | localExpertNum |  Number of experts on the current device: <ul><li>For shared expert cards, localExpertNum = 1;</li><li>For MoE expert cards, localExpertNum = <code>moeExpertNum/(epWorldSize-sharedExpertRankNum)</code>. TP domain communication is not supported in the current version. </li><li><term>Atlas A3 products</term>, Ascend 950PR&950DT products: Ensure 0 < <code>localExpertNum * epWorldSize</code> ≤ 2048.</li></ul>|

  <!-- npu="A3" id26 -->
  - K: <term>Atlas A3 products</term>: When `commAlg` is "fullmesh_v2", the value range is 0 < K ≤ 12.
  <!-- end id26 -->
  <!-- npu="950" id27 -->
  - K: <term>Ascend 950PR&950DT products</term>: The ordinary fullmesh_v2 path (`commAlg` is "fullmesh_v2", non-hierarchical and non-RoCE) supports 0 < K ≤ 32; other paths retain 0 < K ≤ 16. For K > 16, the paired Combine operator must not use AddRmsNorm fusion.
  <!-- end id27 -->

- **quantMode-related constraints**:

  <!-- npu="910b" id13 -->
  - <term>Atlas A2 products</term>:
      - When `quantMode` is 0, the scenario is non-quantized:
          - If the data type of `x` is `FLOAT16` or `BFLOAT16`, pass a null pointer for `scalesOptional`. The data type of `expandX` is the same as that of `x`.
          - If the data type of `x` is `INT32`, `commAlg` can only be set to "fullmesh". `scalesOptional` must be a 1D Tensor of the `FLOAT32` data type, with shape (`BS`, ). The data type of `expandX` is `INT32`. The output `dynamicScalesOut` is of the `FLOAT32` data type, with shape (`A`, ).
      - When `quantMode` is set to 2, it indicates a per-token dynamic quantization scenario, and the data type of `expandX` supports `INT8`.
          - The input `scalesOptional` can be a null pointer.
          - If the input `scalesOptional` contains valid data, its shape is (`moeExpertNum`, `H`).
          - The shape of output `dynamicScalesOut` is `(A, )`
  <!-- end id13 -->
  <!-- npu="A3" id14 -->
  - <term>Atlas A3 products</term>  :
      - When `quantMode` is 0, the scenario is non-quantized. Pass a null pointer for `scalesOptional`. The data type of `expandX` supports `FLOAT16` and `BFLOAT16`.
      - When `quantMode` is set to 2, it indicates a per-token dynamic quantization scenario, and the data type of `expandX` supports `INT8`.
          - The input `scalesOptional` can be a null pointer.
          - If valid data is passed to `scalesOptional` and shared expert cards exist, its shape is (`sharedExpertNum` + `moeExpertNum`, `H`).
          - If valid data is passed to `scalesOptional` and no shared expert card exists, its shape is (`moeExpertNum`, `H`).
          - The shape of output `dynamicScalesOut` is `(A, )`
  <!-- end id14 -->
  <!-- npu="950" id15 -->
  - <term>Ascend 950PR&950DT products</term>:
      - If `quantMode` is 0, it indicates the non-quantized use case.
          - When the data type of `x` is `FLOAT16` or `BFLOAT16`, the data type of `expandX` can be the same as that of `x` or `HIFLOAT8`, and a null pointer must be passed to `scalesOptional`.
          - When the data type of `x` is `HIFLOAT8`, `FLOAT8_E5M2`, `FLOAT8_E4M3FN`, `FLOAT4_E2M1`, or `FLOAT4_E1M2`, valid data must be passed to `scalesOptional`, and the data type of `expandX` is the same as that of `x`.
              - When the data type of `x` is `HIFLOAT8`, the data type of `scalesOptional` is `FLOAT`.
              - When the data type of `x` is `FLOAT8_E5M2` or `FLOAT8_E4M3FN`, the data type of `scalesOptional` is `FLOAT` or `FLOAT8_E8M0`.
              - When the data type of `x` is `FLOAT4_E2M1` or `FLOAT4_E1M2`, the data type of `scalesOptional` must be `FLOAT8_E8M0`, and H must be an even number.
              - The shape of `scalesOptional` is (`BS`, `dim1`), where `dim1` must be less than or equal to `H`.
      - When `quantMode` is 1, static quantization is used. The data types supported by `expandX` are `INT8` and `HIFLOAT8`.
          - When the data type of `expandX` is `INT8`, the following use cases apply:
              - The input `scalesOptional` represents quantization coefficients with the shape (1, );
              - When the input `scalesOptional` represents the smoothing weight shared by each expert, the shape is (`H`,);
              - When the input `scalesOptional` represents quantization coefficients fused with the smoothing weight of each expert, the shape is (`sharedExpertNum` + `moeExpertNum`, `H`) if there are shared expert cards, or (`moeExpertNum`, `H`) otherwise.
          - When the data type of `expandX` is `HIFLOAT8`, the shape of `scalesOptional` must be (1, ).
      - When `quantMode` is 2, pertoken dynamic quantization is used. The data types supported by `expandX` are `INT8`, `FLOAT8_E4M3FN`, and `FLOAT8_E5M2`.
          - The input `scalesOptional` can be a null pointer.
          - If input `scalesOptional` is passed with valid data and shared expert cards exist, its shape is (`sharedExpertNum` + `moeExpertNum`, `H`).
          - If input `scalesOptional` is passed with valid data and no shared expert cards exist, its shape is (`moeExpertNum`, `H`).
          - The shape of output `dynamicScalesOut` is `(A, )`
      - When `quantMode` is 3, pergroup dynamic quantization is used. expandX supports the data types `FLOAT8_E4M3FN` and `FLOAT8_E5M2`.
          - Input `scalesOptional` can be a null pointer.
          - If input `scalesOptional` is passed with valid data and shared expert cards exist, its shape is (`sharedExpertNum` + `moeExpertNum`, `H`).
          - If input `scalesOptional` is passed with valid data and no shared expert cards exist, its shape is (`moeExpertNum`, `H`).
          - The shape of output `dynamicScalesOut` is `(A, Ceil(H, 128))`, where `Ceil(H, 128) = (H + 128 - 1) / 128`
      - When `quantMode` is 4, mx quantization is used. `expandX` supports the data types `FLOAT8_E4M3FN`, `FLOAT8_E5M2`, `FLOAT4_E2M1`, and `FLOAT4_E1M2`. Input `scalesOptional` must be a null pointer. The shape of output `dynamicScalesOut` is `(A, Ceil(H, 64))`, where `Ceil(H, 64) = (H + 64 - 1) / 64`. If the data type of `expandX` is `FLOAT4_E2M1` or `FLOAT4_E1M2`, H must be an even number.

  <!-- end id15 -->

- **Environment Variable Constraints**:
  - **HCCL_BUFFSIZE**:

      Before calling this API, check whether the value of the HCCL_BUFFSIZE environment variable is reasonable. This environment variable indicates the memory size occupied by a single communication domain, in MB. The default value is 200 MB when not configured.

      <!-- npu="910b" id16 -->
      - <term>Atlas A2 products</term>:
          - commAlg is configured as "" or nullptr: Based on the HCCL_INTRA_PCIE_ENABLE and HCCL_INTRA_ROCE_ENABLE environment variable configuration, selects the "fullmesh" or "hierarchy" formula.
          - commAlg is set to "fullmesh": when the data type of `x` is `INT32`, the set size must be >= 2 \* (BS \* epWorldSize \* min(localExpertNum, K) \* (H \* sizeof(int32) + 32B) + 2MB); for other data types, the set size must be >= 2 \* (BS \* epWorldSize \* min(localExpertNum, K) \* H \* sizeof(uint16) + 2MB).
          - commAlg configured as "hierarchy": Set size requirement to ≥ (`moeExpertNum` + `epWorldSize` / 4) \* Align512(`maxBS` \* (`H` \* 2 + 16 \* Align8(`K`))) \* 1B + 8MB, where Align8(x) = ((x + 8 - 1) / 8) \* 8, Align512(x) = ((x + 512 - 1) / 512) \* 512.
      <!-- end id16 -->
      <!-- npu="950,A3" id17 -->
      - <term>Atlas A3 products</term>, Ascend 950PR&950DT products:
          - Within the EP communication domain, when commAlg is "fullmesh_v1" or an empty string or a null pointer: Set size requirement to a value that satisfies ≥ 2 \* (localExpertNum \* maxBS \* epWorldSize \* Align512(Align32(2 \* H) + 64) + (K + sharedExpertNum) \* maxBS \* Align512(2 \* H)).
          - Within the ep communicator, when commAlg is "fullmesh_v2": the set size must be ≥ 2 \* (localExpertNum \* maxBS \* epWorldSize \* 480Align512(Align32(2 \* H) + 64) + (K + sharedExpertNum) \* maxBS \* Align512(2 \* H)).
          - Where `480Align512(x) = ((x + 480 - 1) / 480) * 512`, `Align512(x) = ((x + 512 - 1) / 512) * 512`, and `Align32(x) = ((x + 32 - 1) / 32) * 32`.

      <!-- end id17 -->

  - **HCCL_INTRA_PCIE_ENABLE** and **HCCL_INTRA_ROCE_ENABLE**:

      <!-- npu="910b" id18 -->
      - <term>Atlas A2 products</term>: These environment variables are no longer recommended. It is recommended to configure commAlg as "hierarchy".
      <!-- end id18 -->
      <!-- npu="950,A3" id19 -->
      - <term>Atlas A3 products</term>, Ascend 950PR&950DT products: This environment variable is not supported.

      <!-- end id19 -->

- **Communication Domain Usage Constraints**:
  - The CombineV3 series operators and `aclnnMoeDistributeDispatchV3` in a model only support the same EP communication domain, and no other operators are allowed in this communication domain.
  - TP domain communication is not supported currently.

  <!-- npu="A3" id20 -->
  - <term>Atlas A3 products</term>  : Nodes in a communicator must be in one SuperPoD. Cross-SuperPoD communication is not supported.

  <!-- end id20 -->

- **Communication mode constraints**:

  <!-- npu="950" id21 -->
  - <term>Ascend 950PR&950DT products</term>: Only UB Memory communication is supported.

  <!-- end id21 -->

- **Networking Constraints**:

  <!-- npu="910b" id22 -->
  - <term>Atlas A2 products</term>: In multi-machine scenarios, only switch networking is supported. Direct-connect networking between two machines is not supported.

  <!-- end id22 -->

- **Other Constraints**:
  - In the formula, "/" indicates integer division.
  - <code>moeExpertNum + zeroExpertNum + copyExpertNum + constExpertNum < MAX_INT32</code>.

## Example

<!-- npu="910b" id23 -->
- <term>Atlas A2 products</term> :

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
        #include "aclnnop/aclnn_moe_distribute_dispatch_v3.h"
        #include "aclnnop/aclnn_moe_distribute_combine_v3.h"

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
            aclrtStream dispatchV3Stream;
            aclrtStream combineV3Stream;
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

        int launchOneThreadDispatchV3AndCombineV3_A2(Args &args)
        {
            int ret = aclrtSetCurrentContext(args.context);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetCurrentContext failed, ret %d\n", ret); return ret);
            char hcomEpName[128] = {0};
            ret = HcclGetCommName(args.hcclEpComm, hcomEpName);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclGetEpCommName failed, ret %d\n", ret); return -1);
            LOG_PRINT("[INFO] rank = %d, hcomEpName = %s, dispatchV3Stream = %p, combineV3Stream = %p, \
                        context = %p\n", args.rankId, hcomEpName, args.dispatchV3Stream, args.combineV3Stream,                 \
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

            // Input for the zero-expert use case
            void *oriXDeviceAddr = nullptr;

            void *xOutDeviceAddr = nullptr;

            aclTensor *x = nullptr;
            aclTensor *expertIds = nullptr;
            aclTensor *scales = nullptr;
            aclTensor *xActiveMask = nullptr;
            aclTensor *expertScales = nullptr;

            aclTensor *elasticInfo = nullptr; // A3
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

            // Define the dimensions of each variable for the current use case.
            std::vector<int64_t> xShape{BS, H};
            std::vector<int64_t> expertIdsShape{BS, K};
            std::vector<int64_t> scalesShape{moeExpertNum + 1, H};
            std::vector<int64_t> expertScalesShape{BS, K};

            std::vector<int64_t> expandXShape{A, H};
            std::vector<int64_t> dynamicScalesShape{A};
            std::vector<int64_t> assistInfoForCombineShape{A * 128};
            std::vector<int64_t> expertTokenNumsShape{localExpertNum};
            std::vector<int64_t> epRecvCountsShape{localExpertNum * EP_WORLD_SIZE_A2}; // Not layered
            std::vector<int64_t> tpRecvCountsShape{1};
            std::vector<int64_t> expandScalesShape{A};

            std::vector<int64_t> oriXShape{BS, H};
            std::vector<int64_t> xOutShape{BS, H};

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
            ret = aclnnMoeDistributeDispatchV3GetWorkspaceSize(x, expertIds, (quantMode > 0 ? scales : nullptr), xActiveMask,
                    expertScales, elasticInfo, hcomEpName, EP_WORLD_SIZE_A2, args.epRankId, moeExpertNum, "", 0,
                    0, expertShardType, sharedExpertNum, sharedExpertRankNum, quantMode, globalBS,
                    expertTokenNumsType, commAlg.c_str(), zeroExpertNum, copyExpertNum, constExpertNum, expandX, dynamicScales, assistInfoForCombine, expertTokenNums, epRecvCounts,
                    tpRecvCounts, expandScales, &dispatchWorkspaceSize, &dispatchExecutor);

            CHECK_RET(ret == ACL_SUCCESS,
                LOG_PRINT("[ERROR] aclnnMoeDistributeDispatchV3GetWorkspaceSize failed. ret = %d \n", ret); return ret);

            if (dispatchWorkspaceSize > 0) {
                ret = aclrtMalloc(&dispatchWorkspaceAddr, dispatchWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
                CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc workspace failed. ret = %d \n", ret); return ret);
            }
            // Call the second-phase API.
            ret = aclnnMoeDistributeDispatchV3(dispatchWorkspaceAddr, dispatchWorkspaceSize,
                                                dispatchExecutor, args.dispatchV3Stream);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnMoeDistributeDispatchV3 failed. ret = %d \n", ret);  \
                    return ret);
            ret = aclrtSynchronizeStreamWithTimeout(args.dispatchV3Stream, 10000);
                        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] dispatch aclrtSynchronizeStreamWithTimeout failed. ret = %d \n", ret);  \
                    return ret);
            LOG_PRINT("[INFO] device_%d aclnnMoeDistributeDispatchV3 execute successfully.\n", args.rankId);
            /**************************************** Call combine ********************************************/
            // Call the first-phase API.
            ret = aclnnMoeDistributeCombineV3GetWorkspaceSize(expandX, expertIds,
                                                                assistInfoForCombine, epRecvCounts,
                                                                expertScales, tpRecvCounts,
                                                                xActiveMask, activationScale, weightScale,
                                                                groupList, expandScales, sharedExpertX,
                                                                elasticInfo, oriX, constExpertAlpha1, constExpertAlpha2, constExpertV,
                                                                hcomEpName, EP_WORLD_SIZE_A2, args.epRankId, moeExpertNum,
                                                                "", 0, 0, expertShardType,
                                                                sharedExpertNum, sharedExpertRankNum, globalBS, outDtype,
                                                                commQuantMode, groupList_type, commAlg.c_str(), zeroExpertNum, copyExpertNum, constExpertNum, xOut,
                                                                &combineWorkspaceSize, &combineExecutor);
            CHECK_RET(ret == ACL_SUCCESS,
                LOG_PRINT("[ERROR] aclnnMoeDistributeCombineV3GetWorkspaceSize failed. ret = %d \n", ret); return ret);
            // Request device memory based on the workspaceSize calculated by the first-phase API.
            if (combineWorkspaceSize > 0) {
                ret = aclrtMalloc(&combineWorkspaceAddr, combineWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
                CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc workspace failed. ret = %d \n", ret); return ret);
            }

            // Call the second-phase API.
            ret = aclnnMoeDistributeCombineV3(combineWorkspaceAddr, combineWorkspaceSize, combineExecutor, args.combineV3Stream);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnMoeDistributeCombineV3 failed. ret = %d \n", ret);
                return ret);
            // (boilerplate) Synchronize and wait for the task to complete.
            ret = aclrtSynchronizeStreamWithTimeout(args.combineV3Stream, 10000);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSynchronizeStreamWithTimeout failed. ret = %d \n", ret);
                return ret);
            LOG_PRINT("[INFO] device_%d aclnnMoeDistributeDispatchV3 and aclnnMoeDistributeCombineV3                      \
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
            FreeDeviceAddr(xOutDeviceAddr);

            HcclCommDestroy(args.hcclEpComm);
            aclrtDestroyStream(args.dispatchV3Stream);
            aclrtDestroyStream(args.combineV3Stream);
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
            aclrtStream dispatchV3Stream[DEV_NUM_A2];
            aclrtStream combineV3Stream[DEV_NUM_A2];
            aclrtContext context[DEV_NUM_A2];
            for (uint32_t rankId = 0; rankId < DEV_NUM_A2; rankId++) {
                ret = aclrtSetDevice(rankId);
                CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetDevice failed, ret = %d\n", ret); return ret);
                ret = aclrtCreateContext(&context[rankId], rankId);
                CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateContext failed, ret = %d\n", ret); return ret);
                ret = aclrtCreateStream(&dispatchV3Stream[rankId]);
                CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateStream failed, ret = %d\n", ret); return ret);
                ret = aclrtCreateStream(&combineV3Stream[rankId]);
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
                args[rankId].dispatchV3Stream = dispatchV3Stream[rankId];
                args[rankId].combineV3Stream = combineV3Stream[rankId];
                args[rankId].context = context[rankId];
                threads[rankId].reset(new (std::nothrow) std::thread(&launchOneThreadDispatchV3AndCombineV3_A2, std::ref(args[rankId])));
            }

            for (uint32_t rankId = 0; rankId < DEV_NUM_A2; rankId++) {
                threads[rankId]->join();
            }

            aclFinalize();
            LOG_PRINT("[INFO] aclFinalize success\n");
            return 0;
        }
        ```

<!-- end id23 -->
<!-- npu="950" id24 -->
- <term>Ascend 950PR&950DT products</term>: See the preparation and example code in the Example of [aclnnMoeDistributeDispatchV2](../docs/aclnnMoeDistributeDispatchV2_en.md). Reset the involved variables according to the preceding Constraints. For the use case parameters newly added in V3 compared with V2, pass values as described in the preceding Parameters.

<!-- end id24 -->
<!-- npu="A3" id25 -->
- <term>Atlas A3 products</term>:

    For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

<!-- end id25 -->
- The sample code is as follows, for reference only.

    ```Cpp
    #include <thread>
    #include <iostream>
    #include <string>
    #include <vector>
    #include <unordered_set>
    #include "acl/acl.h"
    #include "hccl/hccl.h"
    #include "aclnnop/aclnn_moe_distribute_dispatch_v3.h"
    #include "aclnnop/aclnn_moe_distribute_combine_v3.h"

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
        ret = aclnnMoeDistributeDispatchV3GetWorkspaceSize(x, expertIds, (quantMode > 0 ? scales : nullptr), nullptr,
                expertScales, elasticInfo, hcomEpName, EP_WORLD_SIZE, args.epRankId, moeExpertNum, "", 0,
                0, expertShardType, sharedExpertNum, sharedExpertRankNum, quantMode, globalBS,
                expertTokenNumsType, nullptr, zeroExpertNum, copyExpertNum, constExpertNum, expandX, dynamicScales, expandIdx, expertTokenNums, epRecvCounts,
                tpRecvCounts, expandScales, &dispatchWorkspaceSize, &dispatchExecutor);

        CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("[ERROR] aclnnMoeDistributeDispatchV3GetWorkspaceSize failed. ret = %d \n", ret); return ret);

        if (dispatchWorkspaceSize > 0) {
            ret = aclrtMalloc(&dispatchWorkspaceAddr, dispatchWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc workspace failed. ret = %d \n", ret); return ret);
        }
        // Call the second-phase API.
        ret = aclnnMoeDistributeDispatchV3(dispatchWorkspaceAddr, dispatchWorkspaceSize,
                                            dispatchExecutor, args.dispatchStream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnMoeDistributeDispatchV3 failed. ret = %d \n", ret);  \
                return ret);
        ret = aclrtSynchronizeStreamWithTimeout(args.dispatchStream, 10000);
                    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] dispatch aclrtSynchronizeStreamWithTimeout failed. ret = %d \n", ret);  \
                return ret);
        /**************************************** Call combine ********************************************/
        // Call the first-phase API.
        ret = aclnnMoeDistributeCombineV3GetWorkspaceSize(expandX, expertIds,
                                                            expandIdx, epRecvCounts,
                                                            expertScales, tpRecvCounts,
                                                            nullptr, nullptr, nullptr,
                                                            nullptr, nullptr, nullptr,
                                                            elasticInfo, oriX, constExpertAlpha1, constExpertAlpha2, constExpertV,
                                                            hcomEpName, EP_WORLD_SIZE, args.epRankId, moeExpertNum,
                                                            "", 0, 0, expertShardType,
                                                            sharedExpertNum, sharedExpertRankNum, globalBS, outDtype,
                                                            commQuantMode, groupList_type, nullptr, zeroExpertNum, copyExpertNum, constExpertNum, xOut,
                                                            &combineWorkspaceSize, &combineExecutor);
        CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("[ERROR] aclnnMoeDistributeCombineV3GetWorkspaceSize failed. ret = %d \n", ret); return ret);
        // Allocate device memory based on the workspaceSize calculated by the first-phase API.
        if (combineWorkspaceSize > 0) {
            ret = aclrtMalloc(&combineWorkspaceAddr, combineWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc workspace failed. ret = %d \n", ret); return ret);
        }

        // Call the second-phase API.
        ret = aclnnMoeDistributeCombineV3(combineWorkspaceAddr, combineWorkspaceSize, combineExecutor, args.combineStream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnMoeDistributeCombineV3 failed. ret = %d \n", ret);
            return ret);
        // (boilerplate) Synchronize and wait for the task execution to complete.
        ret = aclrtSynchronizeStreamWithTimeout(args.combineStream, 10000);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSynchronizeStreamWithTimeout failed. ret = %d \n", ret);
            return ret);
        LOG_PRINT("[INFO] device_%d aclnnMoeDistributeDispatchV3 and aclnnMoeDistributeCombineV3                      \
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
