# aclnnMoeDistributeDispatchV2

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T14:32:18.568Z -->

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

- Description: Quantizes (optional) token data and performs AllToAllV communication in the EP (Expert Parallelism) domain. Communication in the TP (Tensor Parallelism) domain is no longer supported in the current version, and TP-related parameters are reserved parameters.

    Compared with the `aclnnMoeDistributeDispatch` API, this API has the following changes:
    1. More detailed token information is output to help the CombineV2 series operator efficiently perform synchronization across all devices. Therefore, the `expandIdx` output parameter with shape `(BS * K,)` in the original API is replaced by the `assistInfoForCombineOut` parameter with shape `(A * 128,)`;
    2. Adds the `commAlg` input parameter, which replaces the `HCCL_INTRA_PCIE_ENABLE` and `HCCL_INTRA_ROCE_ENABLE` environment variables used to control the hierarchical algorithm.

    For details, refer to the following Parameters.
- Formula:

    - Case 1: If quantMode=0 (non-quantization scenario):

    $$
    allToAllXOut = AllToAllV(X)\\
    expandXOut = AllToAllV(X)
    $$

    When the data type of `x` is `INT32`, `scalesOptional` is communicated with the corresponding token:

    $$
    allToAllDynamicScalesOut = AllToAllV(scalesOptional)\\
    dynamicScalesOut = allToAllDynamicScalesOut
    $$

    - Case 2: If quantMode=1 (static quantization scenario):

    $$
    xFp32 = CastToFp32(X) \times scales \\
    quantOut = Cast(xFp32,dstType) \\
    allToAllXOut = AllToAllV(quantOut)\\
    expandXOut = AllToAllV(quantOut)
    $$

    - Case 3: If quantMode=2 (pertoken dynamic quantization scenario):

    $$
    xFp32 = CastToFp32(X) \times scales \\
    dynamicScales = dstTypeMax/Max(Abs(xFp32)) \\
    quantOut = CastToInt8(xFp32 \times dynamicScales) \\
    allToAllXOut = AllToAllV(quantOut) \\
    allToAllDynamicScalesOut = AllToAllV(1.0/dynamicScales) \\
    expandXOut = AllToAllV(quantOut) \\
    dynamicScalesOut = allToAllDynamicScalesOut
    $$

    - Case 4: If quantMode=3 (pertile dynamic quantization scenario):

    $$
    xFp32 = CastToFp32(X) \times scales \\
    dynamicScales = dstTypeMax/Max(Abs(xFp32)) \\
    quantOut = CastToInt8(xFp32 \times dynamicScales) \\
    allToAllXOut = AllToAllV(quantOut) \\
    allToAllDynamicScalesOut = AllToAllV(1.0/dynamicScales) \\
    expandXOut = AllToAllV(quantOut) \\
    dynamicScalesOut = allToAllDynamicScalesOut
    $$

    - Scenario 5: If quantMode=4 (mx quantization scenario):

    $$
    sharedExp = Floor(log_2(max(x))) - emax \\
    dynamicScales = 2^{sharedExp} \\
    quantOut = CastToFp8(X / dynamicScales) \\
    allToAllXOut = AllToAllV(quantOut) \\
    allToAllDynamicScalesOut = AllToAllV(1.0 / dynamicScales) \\
    expandXOut = AllToAllV(quantOut) \\
    dynamicScalesOut = allToAllDynamicScalesOut
    $$

    where $emax$ is the exponent value of the largest normal number of the data type.

<!-- npu="910b" id7 -->
- <term>Atlas A2 products</term>: This API must be used together with `aclnnMoeDistributeCombineV2`.
<!-- end id7 -->
<!-- npu="950,A3" id8 -->
- <term>Atlas A3 products</term>, Ascend 950PR&950DT products: This API must be used together with `aclnnMoeDistributeCombineV2` or `aclnnMoeDistributeCombineAddRmsNorm`.

<!-- end id8 -->

> NOTE:
> `aclnnMoeDistributeCombineV2` and `aclnnMoeDistributeCombineAddRmsNorm` are collectively referred to as **CombineV2 series operators** in subsequent documents.

## Function Prototype

Each operator is a two-phase API. First, call the "aclnnMoeDistributeDispatchV2GetWorkspaceSize" API to obtain the workspace size required for computation and the executor containing the operator computation process. Then, call the "aclnnMoeDistributeDispatchV2" API to perform the computation.

```cpp
aclnnStatus aclnnMoeDistributeDispatchV2GetWorkspaceSize(
    const aclTensor* x,
    const aclTensor* expertIds,
    const aclTensor* scalesOptional,
    const aclTensor* xActiveMaskOptional,
    const aclTensor* expertScalesOptional,
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
aclnnStatus aclnnMoeDistributeDispatchV2(
    void            *workspace,
    uint64_t        workspaceSize,
    aclOpExecutor   *executor,
    aclrtStream     stream)
```

## aclnnMoeDistributeDispatchV2GetWorkspaceSize

- **Parameters**

    <table style="undefined;table-layout: fixed; width: 1567px"> <colgroup>
    <col style="width: 120px">
    <col style="width: 120px">
    <col style="width: 280px">
    <col style="width: 350px">
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
    <td>Token data sent by the local device.</td>
    <td>A 2D tensor is required.</td>
    <td>FLOAT16, BFLOAT16, FLOAT8_E5M2, FLOAT8_E4M3FN, HIFLOAT8, FLOAT4_E2M1, FLOAT4_E1M2, INT32</td>
    <td>ND</td>
    <td><code>(BS, H)</code> (BS = batch size, H = hidden size)</td>
    <td>-</td>
    </tr>
    <tr>
    <td>expertIds</td>
    <td>Input</td>
    <td>TopK expert indexes of each token.</td>
    <td>A 2D tensor is required.</td>
    <td>INT32</td>
    <td>ND</td>
    <td><code>(BS, K)</code></td>
    <td>-</td>
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
    <td>Indicates whether a token participates in communication.</td>
    <td>Either valid data or a null pointer can be passed.<br>For 1D input, true indicates that the corresponding token participates in communication, and true must be placed before false. For example, {true, false, true} is invalid input;<br>For 2D input, true indicates that the expert_ids corresponding to the current token participate in communication. If all K BOOL values corresponding to a token are false, the token does not participate in communication. By default, all tokens participate in communication. When the number of BSs varies across devices, all tokens must be valid.</td>
    <td>BOOL</td>
    <td>ND</td>
    <td><br>For 1D input, the shape is <code>(BS,)</code>; for 2D input, the shape is <code>(BS, K)</code></td>
    <td>√</td>
    </tr>
    <tr>
    <td>expertScalesOptional</td>
    <td>Input</td>
    <td>TopK expert weights for each token.</td>
    <td>-</td>
    <td>FLOAT32</td>
    <td>ND</td>
    <td><code>(BS, K)</code></td>
    <td>√</td>
    </tr>
    <tr>
    <td>groupEp</td>
    <td>Input</td>
    <td>Name of the EP communication domain (expert parallelism communication domain).</td>
    <td>String length range: <code>[1, 128)</code>.</td>
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
    <td>ID of the current device in the EP domain.</td>
    <td><ul><li>Value range <code>[0, epWorldSize)</code></li><li>The <code>epRankId</code> of each device in the same EP communication domain must be unique.</li></ul></td>
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
    <td>Name of the TP communication domain (reserved parameter, not supported in the current version).</td>
    <td>Default value: "".</td>
    <td>STRING</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>tpWorldSize</td>
    <td>Input</td>
    <td>Size of the TP communication domain (reserved parameter; not supported in the current version; only 0 or 1 is supported).</td>
    <td>Default value: 0.</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>tpRankId</td>
    <td>Input</td>
    <td>ID of the current device in the TP domain (reserved parameter; not supported in the current version; only 0 is supported).</td>
    <td>Default value: 0.</td>
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
    <td><ul><li>When BS values are identical across devices, <code>globalBS = BS * epWorldSize</code> or 0.</li><li>When BS values differ across devices, <code>globalBS = maxBS * epWorldSize</code> (maxBS is the maximum BS on a single device).</li></ul></td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>expertTokenNumsType</td>
    <td>Input</td>
    <td>Semantic type of the values in output <code>expertTokenNums</code>.</td>
    <td>Supports 0: the output in <code>expertTokenNums</code> is the prefix sum of tokens processed by each expert, 1: the output in <code>expertTokenNums</code> is the number of tokens processed by each expert.</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>√</td>
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
    <td>expandXOut</td>
    <td>Output</td>
    <td>Token features expanded based on <code>expertIds</code>.</td>
    <td>Must be a 2D tensor.</td>
    <td>FLOAT16, BFLOAT16, INT8, FLOAT8_E4M3FN, FLOAT8_E5M2, HIFLOAT8, FLOAT4_E2M1, FLOAT4_E1M2, INT32</td>
    <td>ND</td>
    <td><code>(A, H)</code></td>
    <td>√</td>
    </tr>
    <tr>
    <td>dynamicScalesOut</td>
    <td>Output</td>
    <td>Scaling parameter.</td>
    <td>Must be a 1D or 2D tensor. Output exists when quantMode is 2, 3, 4, or 5; output also exists when quantMode is 0 and the data type of `x` is `HIFLOAT8`, `FLOAT8_E5M2`, `FLOAT8_E4M3FN`, `FLOAT4_E2M1`, `FLOAT4_E1M2`, or `INT32`.</td>
    <td>FLOAT32, FLOAT8_E8M0</td>
    <td>-</td>
    <td>-</td>
    <td>√</td>
    </tr>
    <tr>
    <td>assistInfoForCombineOut</td>
    <td>Output</td>
    <td>Number of tokens sent to the same expert (corresponding to <code>assistInfoForCombine</code> in <code>aclnnMoeDistributeCombineV2</code>).</td>
    <td>Must be a 1D tensor.</td>
    <td>INT32</td>
    <td>ND</td>
    <td><code>(A*128, )</code></td>
    <td>√</td>
    </tr>
    <tr>
    <td>expertTokenNumsOut</td>
    <td>Output</td>
    <td>Number of tokens received by each expert.</td>
    <td>Must be a 1D tensor.</td>
    <td>INT64</td>
    <td>ND</td>
    <td><code>(localExpertNum, )</code></td>
    <td>√</td>
    </tr>
    <tr>
    <td>epRecvCountsOut</td>
    <td>Output</td>
    <td>Number of tokens received from each device in the EP communication domain (corresponding to <code>epSendCounts</code> in <code>aclnnMoeDistributeCombineV2</code>).</td>
    <td>Must be a 1D Tensor.</td>
    <td>INT32</td>
    <td>ND</td>
    <td>-</td>
    <td>√</td>
    </tr>
    <tr>
    <td>tpRecvCountsOut</td>
    <td>Output</td>
    <td>Number of tokens received from each device in the TP communication domain (reserved output; not supported in the current version; pass a null pointer).</td>
    <td>INT32</td>
    <td>ND</td>
    <td>-</td>
    <td>√</td>
    </tr>
    <tr>
    <td>expandScalesOut</td>
    <td>Output</td>
    <td>Indicates the weights of the tokens output by the current device (corresponding to <code>expertScalesOptional</code> in <code>aclnnMoeDistributeCombineV2</code>).</td>
    <td>-</td>
    <td>FLOAT32</td>
    <td>ND</td>
    <td>-</td>
    <td>√</td>
    </tr>
    <tr>
    <td>workspaceSize</td>
    <td>Output</td>
    <td>Size of the workspace to be requested on the Device.</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>executor</td>
    <td>Output</td>
    <td>Returns an op executor containing the operator computation flow.</td>
    <td>-</td>
    <td>aclOpExecutor*</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    </tbody>
    </table>

    <!-- npu="910b" id9 -->
    - <term>Atlas A2 products</term>:
        - `dynamicScalesOut` has an output when quantMode is set to 2 or the data type of `x` is INT32.
        - `commAlg` supports nullptr, "", "fullmesh", and "hierarchy"; "hierarchy" is recommended together with a driver of version 25.0.RC1.1 or later; nullptr and "" select the algorithm based on HCCL environment variables (not recommended); "fullmesh" transfers tokens directly over RDMA; "hierarchy" optimizes communication through two sends, cross-server and intra-server.
        - When `commAlg` is "hierarchy", or HCCL_INTRA_PCIE_ENABLE=1 and HCCL_INTRA_ROCE_ENABLE=0, scalesOptional must be set to nullptr. When the data type of `x` is INT32, only "fullmesh" is supported for `commAlg`, and valid data must be passed to scalesOptional.
        - The settings for `xActiveMaskOptional` depend on the value of commAlg: "fullmesh" requires a 1D Tensor with shape (BS, ), and true values must precede false values (for example, {true, false, true} is invalid); "hierarchy" is not supported in the current version, and a null pointer can be passed.
        - `expertScalesOptional` must be a 2D Tensor with shape (BS, K).
        - The value of `epWorldSize` depends on the value of commAlg: "fullmesh" supports 2, 3, 4, 5, 6, 7, 8, 16, 32, 64, 128, 192, 256, and 384; "hierarchy" supports 16, 32, and 64.
        - The value of `moeExpertNum` depends on the value of commAlg: "fullmesh" supports (0, 1024], and "hierarchy" supports (0, 512].
        - `groupTp` is not supported in the current version. Pass an empty string.
        - `tpWorldSize`, `tpRankId`, `expertShardType`, `sharedExpertNum`, and `sharedExpertRankNum` are not supported in the current version. Pass 0 for each.
        - The shape of `epRecvCountsOut` is (moeExpertNum + 2 \* globalBS \* K \* serverNum,) (the first moeExpertNum values are the numbers of received tokens, and the remaining values are pre-communication reduce-related information).
        - TP domain communication is not supported currently.
        - `expandScalesOut` must be a 1D tensor with shape (A,).
        - `quantMode` supports 0 (non-quantized) and 2 (dynamic quantization).

    <!-- end id9 -->
    <!-- npu="A3" id10 -->
    - <term>Atlas A3 products</term>:
        - `dynamicScalesOut` has an output only when the value of `quantMode` is 2.
        - `commAlg` is not supported in the current version. Pass a null pointer.
        - `xActiveMaskOptional` must be a 1D or 2D tensor (the 1D shape is (BS, ) and the 2D shape is (BS, K)). For 1D tensors, true must be placed before false. For 2D tensors, if all K values corresponding to a token are false, the token does not participate in communication.
        - `expertScalesOptional` is not supported in the current version. Pass a null pointer.
        - The value range of `epWorldSize` is [2, 768].
        - The value range of `moeExpertNum` is (0, 1024].
        - `groupTp` (reserved parameter, not supported in the current version).
        - `tpWorldSize` (reserved parameter, not supported in the current version; only 0 or 1 is supported).
        - `tpRankId` (reserved parameter, not supported in the current version; only 0 is supported).
        - `expertShardType`: Currently, only 0 is supported, indicating that shared expert cards are placed before MoE expert cards.
        - The current value range of `sharedExpertNum` is [0, 4].
        - The value range of `sharedExpertRankNum` is [0, epWorldSize). If it is set to 0, sharedExpertNum must be 0 or 1. If it is not 0, sharedExpertRankNum % sharedExpertNum = 0 must be satisfied.
        - The shape of `epRecvCountsOut` is (epWorldSize \* localExpertNum,).
        - `tpRecvCountsOut` (reserved output, not supported in the current version; simply pass an empty pointer).
        - The `expandScalesOut` output is not supported in the current version.
        - `quantMode` supports 0 (non-quantized) and 2 (dynamic quantization).

    <!-- end id10 -->
    <!-- npu="950" id11 -->
    - <term>Ascend 950PR&950DT products</term>:
        - `dynamicScalesOut` has an output when quantMode is 2, 3, 4, or 5. It also has an output when quantMode is 0 and the data type of `x` is `HIFLOAT8`, `FLOAT8_E5M2`, `FLOAT8_E4M3FN`, `FLOAT4_E2M1`, or `FLOAT4_E1M2`.
        - commAlg supports "", "fullmesh_v1", "fullmesh_v2", and "hierarchy". "hierarchy" uses intra-server UB Memory and inter-server RoCE communication.
        - `xActiveMaskOptional` must be a 1D or 2D tensor (1D shape: (BS,); 2D shape: (BS, K)). In the 1D case, true must be placed before false (for example, {true, false, true} is invalid). In the 2D case, if all K values corresponding to a token are false, the token does not participate in communication.
        - `expertScalesOptional` accepts a valid tensor or a null pointer. When a valid tensor is passed, the expertScales expert weight function is enabled, and the valid tensor is a 2D tensor with shape (BS, K). When a null pointer is passed, the expertScales expert weight function is disabled. Empty tensors are not supported.
        - `epWorldSize` value range: [2, 1024].
        - `moeExpertNum` has a value range of (0, 2048]. This limit applies to MoE experts only; shared experts are not added to this count.
        - For RoCE (`commAlg="hierarchy"`), shared experts are not supported: `sharedExpertRankNum > 0` is rejected. With `sharedExpertRankNum=0`, `sharedExpertNum=0` or the default value 1 both mean no shared experts.
        - RoCE is not supported by the `low_latency_dispatch` or `low_latency_combine` APIs. Use the ACLNN or torch_npu V2 interface.
        - For RoCE (`commAlg="hierarchy"`), `quantMode` must be 0 (`NON_QUANT`), 4 (`MX_QUANT`), or 5 (`MX_QUANT_CLIP`); all other values are rejected. For modes 4 and 5, `scalesOptional` must be a null pointer.
        - For RoCE (`commAlg="hierarchy"`), `expertScalesOptional` must be a valid FP32 ND tensor of shape `(BS, K)`. Pass null pointers for `elasticInfoOptional` and `performanceInfoOptional`.
        - `groupTp` is not supported in the current version. Pass an empty string.
        - `tpWorldSize` is not supported in the current version. Pass 0.
        - `tpRankId` is not supported in the current version. Pass 0.
        - `expertShardType`: Currently, only 0 is supported, indicating that shared expert cards are placed before MoE expert cards.
        - `sharedExpertNum` current value range: [0, 4].
        - `sharedExpertRankNum` value range: [0, epWorldSize). If sharedExpertRankNum is 0, sharedExpertNum must be 0 or 1. If sharedExpertRankNum is not 0, sharedExpertRankNum % sharedExpertNum = 0 must be satisfied.
        - `epRecvCountsOut` has a shape of (epWorldSize \* localExpertNum,).
        - `tpRecvCountsOut`: This output is not supported in the current version.
        - `expandScalesOut` is valid when the `expertScales` expert weight feature is enabled; otherwise, it is invalid.
        - `quantMode` supports 0 (non-quantized), 1 (static quantization), 2 (pertoken dynamic quantization), 3 (pergroup dynamic quantization), 4 (MX_QUANT), and 5 (MX_QUANT_CLIP).

    <!-- end id11 -->

- **Return Value**

    aclnnStatus: Return status code. For details, see [Return Codes](../../../docs/en/context/aclnn_return_code.md).

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
    <td>A required input or output parameter tensor is a null pointer.</td>
    </tr>
    <tr>
    <td>ACLNN_ERR_PARAM_INVALID</td>
    <td>161002</td>
    <td>The data types of the input and output are not within the supported range.</td>
    </tr>
    <tr>
    <td rowspan="2">ACLNN_ERR_INNER_TILING_ERROR</td>
    <td rowspan="2">561002</td>
    <td>The shapes of the input and output are not within the supported range.</td>
    </tr>
    <tr>
        <td>The parameter value is not within the supported range.</td>
    </tr>
    </tbody>
    </table>

## aclnnMoeDistributeDispatchV2

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
    </tr>
    </thead>
    <tbody>
    <tr>
    <td>workspace</td>
    <td>Input</td>
    <td>The workspace memory address allocated on the Device side.</td>
    </tr>
    <tr>
    <td>workspaceSize</td>
    <td>Input</td>
    <td>The workspace size allocated on the Device side, obtained by the first-phase API <code>aclnnMoeDistributeDispatchV2GetWorkspaceSize</code>.</td>
    </tr>
    <tr>
    <td>executor</td>
    <td>Input</td>
    <td>Executor of the operator, which contains the operator computation flow.</td>
    </tr>
    <tr>
    <td>stream</td>
    <td>Input</td>
    <td>Stream for task execution.</td>
    </tr>
    </tbody>
    </table>

- **Return Value**

aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- **Deterministic computation**:
  - aclnnMoeDistributeDispatchV2 defaults to a deterministic implementation.

- **Driver Constraints**:
  - The driver versions of all nodes in the operator communication domain must be the same.

- **API Compatibility Constraints**:
  - The `aclnnMoeDistributeDispatchV2` API must be used together with the CombineV2 series operator APIs. For details, see [Example](#example).
  - Across different product models, communication algorithms, or versions, the element values in the Tensor outputs `assistInfoForCombineOut`, `epRecvCountsOut`, `tpRecvCountsOut`, and `expandScalesOut` of `aclnnMoeDistributeDispatchV2` may differ. During use, simply pass these tensors to the corresponding parameters of the CombineV2 series operator. Other business logic of the model should not depend on these values.

- **Parameter Consistency Constraints**:
  - During the API call, the parameters `groupEp`, `epWorldSize`, `moeExpertNum`, `groupTp`, `tpWorldSize`, `expertShardType`, `sharedExpertNum`, `sharedExpertRankNum`, `globalBS`, and `commAlg`, as well as the value of `HCCL_BUFFSIZE`, must be the same on all devices, the same across different layers of the network, and the same as the corresponding parameters of the CombineV2 series operator.

- **Product-Specific Constraints**:

  <!-- npu="A3" id12 -->
  - <term>Atlas A3 products</term>: In this use case, a single device contains two DIEs (also called "die" or "bare die"). Therefore, "the current device" in Parameters refers to a single DIE.
  <!-- end id12 -->

- **Shape Variable Constraints**:

  | Variable     | Definition and Value Range                                                     |
  | :----------- | :----------------------------------------------------------------------------- |
  | A            | Maximum number of tokens to be dispatched by the current device. The value range is as follows: <ul> <li>For shared experts, A = BS \* epWorldSize \* sharedExpertNum / sharedExpertRankNum must be satisfied.</li> <li>For MoE experts, when globalBS is 0, A >= BS \* epWorldSize \* min(localExpertNum, K) must be satisfied; when globalBS is not 0, A >= globalBS \* min(localExpertNum, K) must be satisfied.</li> </ul>|
  | H (hidden size) | Size of the hidden layer. <ul><li><term>Atlas A2 products</term>: If the data type of `x` is INT32, the value range is (0, 5120] and the value must be a multiple of 32; for other data types, the value range is (0, 10240] and the value must be a multiple of 32.</li> <li><term>Atlas A3 products</term> and Ascend 950PR&950DT products: value range [1024, 8192].</li> </ul> |
  | BS           | Final number of tokens output by the current device. <ul><li><term>Atlas A2 products</term>: Depending on the value of commAlg, the value range is (0 < BS ≤ 256) for "fullmesh" and (0 < BS ≤ 512) for "hierarchy" when the driver version is ≥25.0.RC1.1;</li><li><term>Atlas A3 products</term> and Ascend 950PR&950DT products: 0 < BS ≤ 512. </li> </ul> |
  | K | Number of selected topK experts. The value range is 0 < K ≤ 16, and <code>0 < K ≤ moeExpertNum</code>. See the product-specific constraints below. |
  | serverNum    | Number of server nodes. Only 2, 4, and 8 are supported.<br>Atlas A2 products: This variable is used only in the shape of this use case.                                                  |
  | localExpertNum |  Number of experts on the current device: <ul><li>For shared expert cards, localExpertNum = 1;</li><li>For MoE expert cards, localExpertNum = <code>moeExpertNum / (epWorldSize - sharedExpertRankNum)</code>. The current version does not support TP domain communication. </li><li><term>Atlas A3 products</term> and Ascend 950PR&950DT products: 0 < localExpertNum * epWorldSize ≤ 2048 must be satisfied.</li></ul>|

  <!-- npu="A3" id27 -->
  - K: <term>Atlas A3 products</term>: When `commAlg` is "fullmesh_v2", the value range is 0 < K ≤ 12.
  <!-- end id27 -->
  <!-- npu="950" id28 -->
  - K: <term>Ascend 950PR&950DT products</term>: The ordinary fullmesh_v2 path (`commAlg` is "fullmesh_v2", non-hierarchical and non-RoCE) supports 0 < K ≤ 32; other paths retain 0 < K ≤ 16. For K > 16, the paired Combine operator must not use AddRmsNorm fusion.
  <!-- end id28 -->

- **quantMode-related constraints**:

  <!-- npu="910b" id13 -->
  - <term>Atlas A2 products</term>:
      - If `quantMode` is 0, this indicates a non-quantized scenario:
          - If the data type of `x` is `FLOAT16` or `BFLOAT16`, pass a null pointer as the input `scalesOptional`. The data type of `expandX` is the same as that of `x`.
          - If the data type of `x` is `INT32`, only `commAlg` configured as "fullmesh" is supported. The input `scalesOptional` must be a 1D Tensor with data type `FLOAT32` and shape (`BS`, ); the data type of `expandX` is `INT32`, and the output `dynamicScalesOut` is of data type `FLOAT32` with shape (`A`, ).
      - When `quantMode` is set to 2, it indicates a per-token dynamic quantization scenario, and the data type of `expandX` supports `INT8`.
          - The input `scalesOptional` can be a null pointer.
          - If valid data is passed for input `scalesOptional`, its shape is (`moeExpertNum`, `H`).
          - The shape of output `dynamicScalesOut` is `(A, )`
  <!-- end id13 -->
  <!-- npu="A3" id14 -->
  - <term>Atlas A3 products</term>:
      - When `quantMode` is 0, it indicates a non-quantized scenario. Pass a null pointer for input `scalesOptional`. The data type of `expandX` can be `FLOAT16` or `BFLOAT16`.
      - When `quantMode` is set to 2, it indicates a per-token dynamic quantization scenario, and the data type of `expandX` supports `INT8`.
          - A null pointer can be passed for input `scalesOptional`.
          - If valid data is passed for input `scalesOptional` and shared expert devices exist, its shape is (`sharedExpertNum` + `moeExpertNum`, `H`).
          - If valid data is passed for input `scalesOptional` and no shared expert devices exist, its shape is (`moeExpertNum`, `H`).
          - The shape of output `dynamicScalesOut` is `(A, )`
  <!-- end id14 -->
  <!-- npu="950" id15 -->
  - <term>Ascend 950PR&950DT products</term>:
      - When `quantMode` is 0, it indicates a non-quantized scenario.
          - When the data type of `x` is `FLOAT16` or `BFLOAT16`, the data type of `expandX` can be the same as that of `x` or `HIFLOAT8`, and `scalesOptional` must be a null pointer.
          - When the data type of `x` is `HIFLOAT8`, `FLOAT8_E5M2`, `FLOAT8_E4M3FN`, `FLOAT4_E2M1`, or `FLOAT4_E1M2`, valid data must be passed to `scalesOptional`, and the data type of `expandX` is the same as that of `x`.
              - When the data type of `x` is `HIFLOAT8`, the data type of `scalesOptional` is `FLOAT`.
              - When the data type of `x` is `FLOAT8_E5M2` or `FLOAT8_E4M3FN`, the data type of `scalesOptional` is `FLOAT` or `FLOAT8_E8M0`.
              - When the data type of `x` is `FLOAT4_E2M1` or `FLOAT4_E1M2`, the data type of `scalesOptional` is `FLOAT8_E8M0`, and H must be an even number.
              - The shape of `scalesOptional` is (`BS`, `dim1`), where `dim1` must be less than or equal to `H`.
      - When `quantMode` is 1, it indicates the static quantization scenario, and the supported data types of `expandX` are `INT8` and `HIFLOAT8`.
          - When the data type of `expandX` is `INT8`, the following cases exist:
              - `scalesOptional` represents the quantization coefficient, with a shape of (1, );
              - When `scalesOptional` represents the smoothing weights shared by all experts, the shape is (`H`, );
              - When the input `scalesOptional` represents quantization coefficients fused with the smoothing weight of each expert, its shape is (`sharedExpertNum` + `moeExpertNum`, `H`) if a shared expert card exists, and (`moeExpertNum`, `H`) if not.
          - When the data type of `expandX` is `HIFLOAT8`, the shape of `scalesOptional` must be (1, ).
      - When `quantMode` is 2, it indicates the pertoken dynamic quantization scenario. The data types of `expandX` support `INT8`, `FLOAT8_E4M3FN`, and `FLOAT8_E5M2`.
          - The input `scalesOptional` can be a null pointer.
          - If valid data is passed to the input `scalesOptional` and a shared expert card exists, its shape is (`sharedExpertNum` + `moeExpertNum`, `H`).
          - If valid data is passed to the input `scalesOptional` and no shared expert card exists, its shape is (`moeExpertNum`, `H`).
          - The shape of the output `dynamicScalesOut` is `(A, )`
      - When `quantMode` is 3, it indicates the pergroup dynamic quantization scenario. The data types of expandX support `FLOAT8_E4M3FN` and `FLOAT8_E5M2`.
          - The input `scalesOptional` can be a null pointer.
          - If valid data is passed to the input `scalesOptional` and a shared expert card exists, its shape is (`sharedExpertNum` + `moeExpertNum`, `H`).
          - If valid data is passed in `scalesOptional` and there are no shared expert cards, the shape is (`moeExpertNum`, `H`).
          - The shape of output `dynamicScalesOut` is `(A, Ceil(H, 128))`, where `Ceil(H, 128) = (H + 128 - 1) / 128`
      - When `quantMode` is 4, it indicates the mx quantization scenario. The data types supported by `expandX` are `FLOAT8_E4M3FN`, `FLOAT8_E5M2`, `FLOAT4_E2M1`, and `FLOAT4_E1M2`, and an empty pointer must be passed in for input `scalesOptional`. The shape of output `dynamicScalesOut` is `(A, Ceil(H, 64))`, where `Ceil(H, 64) = (H + 64 - 1) / 64`. If the data type of `expandX` is `FLOAT4_E2M1` or `FLOAT4_E1M2`, H must be an even number.

      - When `quantMode` is 5 (`MX_QUANT_CLIP`), the output data type supports `FLOAT8_E4M3FN` and `FLOAT8_E5M2`. `scalesOptional` must be a null pointer. `dynamicScalesOut` has data type `FLOAT8_E8M0` and shape `(A, ((H + 31) // 32 + 1) // 2 * 2)`, where `//` denotes integer division.

  <!-- end id15 -->

- **Environment Variable Constraints**:
  - **HCCL_BUFFSIZE**:
      Before calling this API, check whether the value of the <code>HCCL_BUFFSIZE</code> environment variable is appropriate. This environment variable specifies the memory size occupied by a single communication domain, in MB. The default value is 200 MB when not configured:

          <!-- npu="910b" id16 -->
          - <term>Atlas A2 products</term>:
              - commAlg is "" or nullptr: the "fullmesh" or "hierarchy" formula is selected based on HCCL environment variables.
              - commAlg is "fullmesh": when the data type of `x` is `INT32`, the size requirement is (≥ 2 \* (BS \* epWorldSize \* min(localExpertNum, K) \* (H \* sizeof(int32) + 32B) + 2MB)); for other data types, the size requirement is (≥ 2 \* (BS \* epWorldSize \* min(localExpertNum, K) \* H \* sizeof(uint16) + 2MB)).
              - commAlg is "hierarchy": the size requirement is (≥ (`moeExpertNum` + `epWorldSize` / 4) \* Align512(`maxBS` \* (`H` \* 2 + 16 \* Align8  (`K`))) \* 1B + 8MB, where Align8(x) = ((x + 8 - 1) / 8) \* 8 and Align512(x) = ((x + 512 - 1) / 512) \* 512).
          <!-- end id16 -->
          <!-- npu="A3" id17 -->
          - <term>Atlas A3 products</term>  :
              - In the ep communication domain: the size requirement is (≥ 2) and must satisfy (≥ 2 \* (localExpertNum \* maxBS \* epWorldSize \* Align512(Align32(2 \* H) + 64) + (K +   sharedExpertNum) \* maxBS \* Align512(2 \* H))) (`localExpertNum` must use the number of local experts on the MoE expert card; `Align512(x) = ((x + 512 - 1) /   512) * 512`; `Align32(x) = ((x + 32 - 1) / 32) * 32`). Currently, the TP communication domain is not supported.
          <!-- end id17 -->
          <!-- npu="950" id18 -->
          - <term>Ascend 950PR&950DT products</term>: The size set for the ep communication domain must be ≥ 2 and satisfy (≥ 2 \* (localExpertNum \* maxBS \* epWorldSize \* Align512(Align32(2 \* H) + 64) + (K + sharedExpertNum) \* maxBS \* Align512(2 \* H))) (`localExpertNum` must be the number of local experts on each MoE expert card; `Align512(x) = ((x + 512 - 1) / 512) * 512`; `Align32(x) = ((x + 32 - 1) / 32) * 32`). TP communication domain is not supported.

          <!-- end id18 -->

  - **HCCL_INTRA_PCIE_ENABLE and HCCL_INTRA_ROCE_ENABLE**:

      <!-- npu="910b" id19 -->
      - <term>Atlas A2 products</term>: This environment variable is no longer recommended. Configure `commAlg` to "hierarchy" instead.
      <!-- end id19 -->
      <!-- npu="950,A3" id20 -->
      - <term>Atlas A3 products</term>, Ascend 950PR&950DT products: This environment variable is not supported.

      <!-- end id20 -->

- **Communication Domain Usage Constraints**:
   - The CombineV2 series operator and `aclnnMoeDistributeDispatchV2` in a model support only the same EP communication domain, and no other operators are allowed in the communication domain.
   - TP domain communication is not supported currently.

   <!-- npu="A3" id21 -->
   - <term>Atlas A3 products</term>   : Nodes within a communication domain must be in a single super node. Cross-super-node configuration is not supported.

   <!-- end id21 -->

- **Communication mode constraints**:

  <!-- npu="950" id22 -->
  - <term>Ascend 950PR&950DT products</term>: Only UB Memory communication is supported.

  <!-- end id22 -->

- **Networking Constraints**:

   <!-- npu="910b" id23 -->
   - <term>Atlas A2 products</term>: In multi-machine scenarios, only switch networking is supported. Direct-connect networking between two machines is not supported.

   <!-- end id23 -->

- **Other Constraints**:
  - In the formula, "/" indicates integer division.

## Example

- Environment variable configuration:

    ```bash
    # Set an environment variable before running.
    ## ENV_DEV_NUM description: Set this variable based on the number of devices on the current machine. For example, for 16 devices on two machines, set both machines to 16.
    export ENV_DEV_NUM=16
    ```

- Number of machines:

    In a two-machine 16-NPU scenario, set the MACHINE_NUM parameter to 2, that is,

    ```Cpp
    const uint32_t MACHINE_NUM = 2;
    ```

    No modification is required for a single-machine 16-NPU scenario.

<!-- npu="910b" id24 -->
- <term>Atlas A2 products</term>:

    No need to configure the ranktable file or the RANK_TABLE_FILE and FIRST_RANK_ID environment variables.

    This example supports running the A2 operator on a single-server environment with a device count of [2, 8]. Before running, set IS_TEST_A2 to true in the sample code to ensure that the A2 branch is executed.
    In addition, if needed, set EP_WORLD_SIZE_A2 in the sample code to the number of devices and change moeExpertNum in the launchOneThreadDispatchV2AndCombineV2_A2 function so that moeExpertNum is divisible by EP_WORLD_SIZE_A2.

    The operator compilation commands are as follows. Both the moe_distribute_dispatch_v2 and moe_distribute_combine_v2 operators must be compiled, and the two operators must be executed in pairs:

    ```bash
    bash build.sh --pkg --soc=ascend910b --ops=moe_distribute_dispatch_v2,moe_distribute_combine_v2
    ```

    The sample operator execution commands are as follows:

    ```bash
    bash build.sh --run_example --ops=moe_distribute_dispatch_v2 eager cust
    ```

<!-- end id24 -->
<!-- npu="950,A3" id25 -->
- <term>Atlas A3 products</term> and <term>Ascend 950PR&950DT products</term>:

    No need to configure the ranktable file or the environment variables RANK_TABLE_FILE and FIRST_RANK_ID.

<!-- end id25 -->

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

<!-- npu="950,A3,910b" id26 -->
- <term>Atlas A2 products</term>, <term>Atlas A3 products</term>, and <term>Ascend 950PR&950DT products</term>:

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
        } while (0)

    struct Args {
        uint32_t rankId;
        uint32_t epRankId;
        HcclComm hcclEpComm;
        aclrtStream dispatchV2Stream;
        aclrtStream combineV2Stream;
        aclrtContext context;
    };

    const uint32_t MACHINE_NUM = 1;
    const char* env_dev_num = std::getenv("ENV_DEV_NUM");

    const uint32_t EP_WORLD_SIZE = 2;
    const uint32_t DEV_NUM = EP_WORLD_SIZE;

    const bool IS_TEST_A2 = false;
    const bool IS_TEST_A3A5 = true;
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
        *tensor = aclCreateTensor(
            shape.data(), shape.size(), dataType, strides.data(), 0,
            aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), *deviceAddr
        );
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

    int launchOneThreadDispatchV2AndCombineV2_A3A5(Args &args)
    {
        int ret = aclrtSetCurrentContext(args.context);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetCurrentContext failed. ret: %d\n", ret); return ret);

        char hcomEpName[128] = {0};
        ret = HcclGetCommName(args.hcclEpComm, hcomEpName);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclGetEpCommName failed. ret: %d\n", ret); return -1);
        LOG_PRINT(
            "[INFO] rank = %d, hcomEpName = %s, dispatchV2Stream = %p, combineV2Stream = %p, context = %p\n",
            args.rankId, hcomEpName, args.dispatchV2Stream, args.combineV2Stream, args.context
        );

        // Set the use case.
        int64_t BS = 8;
        int64_t H = 7168;
        int64_t K = 1;
        int64_t expertShardType = 0;
        int64_t sharedExpertNum = 0;
        int64_t sharedExpertRankNum = 0;
        int64_t moeExpertNum = EP_WORLD_SIZE - sharedExpertRankNum;
        int64_t quantMode = 0;
        int64_t globalBS = BS * EP_WORLD_SIZE;
        int64_t expertTokenNumsType = 0;
        int64_t outDtype = 0;
        int64_t commQuantMode = 0;
        int64_t groupListType = 0;
        int64_t localExpertNum;
        int64_t A;
        if (args.epRankId < sharedExpertRankNum) {
            // Shared expert device
            localExpertNum = 1;
            A = globalBS / sharedExpertRankNum;
        } else {
            // MoE expert device
            localExpertNum = moeExpertNum / (EP_WORLD_SIZE - sharedExpertRankNum);
            A = globalBS * (localExpertNum < K ? localExpertNum : K);
        }
        std::string commAlg = "";

        /* Construct device-side input and output variables based on the current use case.*/
        // Declare device-side input and output variables.
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

        // Define the dimensions of each variable for the current use case.
        std::vector<int64_t> xShape{BS, H};
        std::vector<int64_t> expertIdsShape{BS, K};
        std::vector<int64_t> scalesShape{(sharedExpertRankNum > 0) ? 1 + moeExpertNum : moeExpertNum, H};
        std::vector<int64_t> expertScalesShape{BS, K};
        std::vector<int64_t> expandXShape{A, H};
        std::vector<int64_t> dynamicScalesShape{A};
        std::vector<int64_t> expandIdxShape{A * 128};
        std::vector<int64_t> expertTokenNumsShape{localExpertNum};
        std::vector<int64_t> epRecvCountsShape{localExpertNum * EP_WORLD_SIZE};
        std::vector<int64_t> tpRecvCountsShape{1};
        std::vector<int64_t> expandScalesShape{A};

        long long xShapeSize = GetShapeSize(xShape);
        long long expertIdsShapeSize = GetShapeSize(expertIdsShape);
        long long scalesShapeSize = GetShapeSize(scalesShape);
        long long expertScalesShapeSize = GetShapeSize(expertScalesShape);
        long long expandXShapeSize = GetShapeSize(expandXShape);
        long long dynamicScalesShapeSize = GetShapeSize(dynamicScalesShape);
        long long expandIdxShapeSize = GetShapeSize(expandIdxShape);
        long long expertTokenNumsShapeSize = GetShapeSize(expertTokenNumsShape);
        long long epRecvCountsShapeSize = GetShapeSize(epRecvCountsShape);
        long long tpRecvCountsShapeSize = GetShapeSize(tpRecvCountsShape);
        long long expandScalesShapeSize = GetShapeSize(expandScalesShape);

        // Construct host-side variables.
        std::vector<op::fp16_t> xHostData(xShapeSize, 1);
        std::vector<int32_t> expertIdsHostData;
        for (int32_t token_id = 0; token_id < expertIdsShape[0]; token_id++) {
            // Each token is dispatched to moe experts {0, 1, ... k - 1}.
            for (int32_t k_id = 0; k_id < expertIdsShape[1]; k_id++) {
                expertIdsHostData.push_back(k_id);
            }
        }
        std::vector<float> scalesHostData(scalesShapeSize, 0);
        std::vector<float> expertScalesHostData(expertScalesShapeSize, 0);
        std::vector<op::fp16_t> expandXHostData(expandXShapeSize, 0);
        std::vector<float> dynamicScalesHostData(dynamicScalesShapeSize, 0);
        std::vector<int32_t> expandIdxHostData(expandIdxShapeSize, 0);
        std::vector<int64_t> expertTokenNumsHostData(expertTokenNumsShapeSize, 0);
        std::vector<int32_t> epRecvCountsHostData(epRecvCountsShapeSize, 0);
        std::vector<int32_t> tpRecvCountsHostData(tpRecvCountsShapeSize, 0);
        std::vector<float> expandScalesHostData(expandScalesShapeSize, 0);

        // Construct device-side variables.
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

        /* Declare the variables required for operator execution. */
        uint64_t dispatchV2WorkspaceSize = 0;
        aclOpExecutor *dispatchV2Executor = nullptr;
        void *dispatchV2WorkspaceAddr = nullptr;

        uint64_t combineV2WorkspaceSize = 0;
        aclOpExecutor *combineV2Executor = nullptr;
        void *combineV2WorkspaceAddr = nullptr;

        /* Execute the dispatchV2 and combineV2 operators in sequence. */
        // Call the first-phase API of the dispatchV2 operator.
        ret = aclnnMoeDistributeDispatchV2GetWorkspaceSize(
            x, expertIds,
            (quantMode > 0 ? scales : nullptr), nullptr,
            expertScales,
            hcomEpName, EP_WORLD_SIZE, args.epRankId,
            moeExpertNum, "", 0,
            0, expertShardType, sharedExpertNum,
            sharedExpertRankNum, quantMode, globalBS,
            expertTokenNumsType, commAlg.c_str(),
            expandX, dynamicScales,
            expandIdx, expertTokenNums,
            epRecvCounts, tpRecvCounts,
            expandScales, &dispatchV2WorkspaceSize,
            &dispatchV2Executor
        );
        CHECK_RET(
            ret == ACL_SUCCESS,
            LOG_PRINT("[ERROR] aclnnMoeDistributeDispatchV2GetWorkspaceSize failed. ret = %d\n", ret); return ret
        );
        // Allocate device memory based on the workspaceSize calculated by the first-phase API of the dispatchV2 operator.
        if (dispatchV2WorkspaceSize > 0) {
            ret = aclrtMalloc(&dispatchV2WorkspaceAddr, dispatchV2WorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc failed. ret = %d\n", ret); return ret);
        }
        // Call the second-phase API of the dispatchV2 operator.
        ret = aclnnMoeDistributeDispatchV2(dispatchV2WorkspaceAddr, dispatchV2WorkspaceSize, dispatchV2Executor, args.dispatchV2Stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnMoeDistributeDispatchV2 failed. ret = %d\n", ret); return ret);
        // (boilerplate) Synchronize and wait until the task execution is complete.
        ret = aclrtSynchronizeStreamWithTimeout(args.dispatchV2Stream, 10000);
        CHECK_RET(
            ret == ACL_SUCCESS,
            LOG_PRINT("[ERROR] aclrtSynchronizeStreamWithTimeout failed. ret = %d\n", ret); return ret
        );

        // Call the first-phase API of the combineV2 operator.
        ret = aclnnMoeDistributeCombineV2GetWorkspaceSize(
            expandX, expertIds, expandIdx, epRecvCounts, expertScales, tpRecvCounts,
            nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
            hcomEpName, EP_WORLD_SIZE, args.epRankId, moeExpertNum, "", 0, 0,
            expertShardType, sharedExpertNum, sharedExpertRankNum, globalBS, outDtype, commQuantMode, groupListType,
            commAlg.c_str(), x, &combineV2WorkspaceSize, &combineV2Executor);
        CHECK_RET(
            ret == ACL_SUCCESS,
            LOG_PRINT("[ERROR] aclnnMoeDistributeCombineV2GetWorkspaceSize failed. ret = %d\n", ret); return ret
        );
        // Allocate device memory based on the workspaceSize calculated by the first-phase API of the combineV2 operator.
        if (combineV2WorkspaceSize > 0) {
            ret = aclrtMalloc(&combineV2WorkspaceAddr, combineV2WorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc failed. ret = %d\n", ret); return ret);
        }
        // Call the second-phase API of the combineV2 operator.
        ret = aclnnMoeDistributeCombineV2(combineV2WorkspaceAddr, combineV2WorkspaceSize, combineV2Executor, args.combineV2Stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnMoeDistributeCombineV2 failed. ret = %d\n", ret); return ret);
        // (boilerplate) Synchronize and wait until the task execution is complete.
        ret = aclrtSynchronizeStreamWithTimeout(args.combineV2Stream, 10000);
        CHECK_RET(
            ret == ACL_SUCCESS,
            LOG_PRINT("[ERROR] aclrtSynchronizeStreamWithTimeout failed. ret = %d\n", ret); return ret
        );

        LOG_PRINT("[INFO] device_%d aclnnMoeDistributeDispatchV2 and aclnnMoeDistributeCombineV2 execute successfully.\n", args.rankId);

        // Release device resources.
        if (dispatchV2WorkspaceSize > 0) {
            aclrtFree(dispatchV2WorkspaceAddr);
        }
        if (combineV2WorkspaceSize > 0) {
            aclrtFree(combineV2WorkspaceAddr);
        }
        DestroyTensor(x);
        DestroyTensor(expertIds);
        DestroyTensor(scales);
        DestroyTensor(expertScales);
        DestroyTensor(expandX);
        DestroyTensor(dynamicScales);
        DestroyTensor(expandIdx);
        DestroyTensor(expertTokenNums);
        DestroyTensor(epRecvCounts);
        DestroyTensor(tpRecvCounts);
        DestroyTensor(expandScales);

        FreeDeviceAddr(xDeviceAddr);
        FreeDeviceAddr(expertIdsDeviceAddr);
        FreeDeviceAddr(scalesDeviceAddr);
        FreeDeviceAddr(expertScalesDeviceAddr);
        FreeDeviceAddr(expandXDeviceAddr);
        FreeDeviceAddr(dynamicScalesDeviceAddr);
        FreeDeviceAddr(expandIdxDeviceAddr);
        FreeDeviceAddr(expertTokenNumsDeviceAddr);
        FreeDeviceAddr(epRecvCountsDeviceAddr);
        FreeDeviceAddr(tpRecvCountsDeviceAddr);
        FreeDeviceAddr(expandScalesDeviceAddr);

        HcclCommDestroy(args.hcclEpComm);
        aclrtDestroyStream(args.dispatchV2Stream);
        aclrtDestroyStream(args.combineV2Stream);
        aclrtDestroyContext(args.context);
        aclrtResetDevice(args.rankId);

        return 0;
    }

    int launchOneThreadDispatchV2AndCombineV2_A2(Args &args)
    {
        int ret = aclrtSetCurrentContext(args.context);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetCurrentContext failed, ret %d\n", ret); return ret);
        char hcomEpName[128] = {0};
        ret = HcclGetCommName(args.hcclEpComm, hcomEpName);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclGetEpCommName failed, ret %d\n", ret); return -1);
        LOG_PRINT("[INFO] rank = %d, hcomEpName = %s, dispatchV2Stream = %p, combineV2Stream = %p, \
                    context = %p\n", args.rankId, hcomEpName, args.dispatchV2Stream, args.combineV2Stream,                 \
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
        std::vector<int64_t> epRecvCountsShape{localExpertNum * EP_WORLD_SIZE_A2}; // Not layered
        std::vector<int64_t> tpRecvCountsShape{1};
        std::vector<int64_t> expandScalesShape{A};

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
        ret = aclnnMoeDistributeDispatchV2GetWorkspaceSize(x, expertIds, (quantMode > 0 ? scales : nullptr), xActiveMask,
                expertScales, hcomEpName, EP_WORLD_SIZE_A2, args.epRankId, moeExpertNum, "", 0,
                0, expertShardType, sharedExpertNum, sharedExpertRankNum, quantMode, globalBS,
                expertTokenNumsType, commAlg.c_str(), expandX, dynamicScales, assistInfoForCombine, expertTokenNums, epRecvCounts,
                tpRecvCounts, expandScales, &dispatchWorkspaceSize, &dispatchExecutor);

        CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("[ERROR] aclnnMoeDistributeDispatchV2GetWorkspaceSize failed. ret = %d \n", ret); return ret);

        if (dispatchWorkspaceSize > 0) {
            ret = aclrtMalloc(&dispatchWorkspaceAddr, dispatchWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc workspace failed. ret = %d \n", ret); return ret);
        }
        // Call the second-phase API.
        ret = aclnnMoeDistributeDispatchV2(dispatchWorkspaceAddr, dispatchWorkspaceSize,
                                            dispatchExecutor, args.dispatchV2Stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnMoeDistributeDispatchV2 failed. ret = %d \n", ret);  \
                return ret);
        ret = aclrtSynchronizeStreamWithTimeout(args.dispatchV2Stream, 10000);
                    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] dispatch aclrtSynchronizeStreamWithTimeout failed. ret = %d \n", ret);  \
                return ret);
        LOG_PRINT("[INFO] device_%d aclnnMoeDistributeDispatchV2 execute successfully.\n", args.rankId);
        /**************************************** Call combine ********************************************/
        // Call the first-phase API.
        ret = aclnnMoeDistributeCombineV2GetWorkspaceSize(expandX, expertIds,
                                                            assistInfoForCombine, epRecvCounts,
                                                            expertScales, tpRecvCounts,
                                                            xActiveMask, activationScale, weightScale,
                                                            groupList, expandScales, sharedExpertX,
                                                            hcomEpName, EP_WORLD_SIZE_A2, args.epRankId, moeExpertNum,
                                                            "", 0, 0, expertShardType,
                                                            sharedExpertNum, sharedExpertRankNum, globalBS, outDtype,
                                                            commQuantMode, groupList_type, commAlg.c_str(), xOut,
                                                            &combineWorkspaceSize, &combineExecutor);
        CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("[ERROR] aclnnMoeDistributeCombineV2GetWorkspaceSize failed. ret = %d \n", ret); return ret);
        // Allocate device memory based on the workspaceSize calculated by the first-phase API.
        if (combineWorkspaceSize > 0) {
            ret = aclrtMalloc(&combineWorkspaceAddr, combineWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc workspace failed. ret = %d \n", ret); return ret);
        }

        // Call the second-phase API.
        ret = aclnnMoeDistributeCombineV2(combineWorkspaceAddr, combineWorkspaceSize, combineExecutor, args.combineV2Stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnMoeDistributeCombineV2 failed. ret = %d \n", ret);
            return ret);
        // (Boilerplate) Synchronize and wait for the task execution to complete.
        ret = aclrtSynchronizeStreamWithTimeout(args.combineV2Stream, 10000);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSynchronizeStreamWithTimeout failed. ret = %d \n", ret);
            return ret);
        LOG_PRINT("[INFO] device_%d aclnnMoeDistributeDispatchV2 and aclnnMoeDistributeCombineV2                      \
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
        FreeDeviceAddr(xOutDeviceAddr);

        HcclCommDestroy(args.hcclEpComm);
        aclrtDestroyStream(args.dispatchV2Stream);
        aclrtDestroyStream(args.combineV2Stream);
        aclrtDestroyContext(args.context);
        LOG_PRINT("[INFO] device_%d Destroy.\n", args.rankId);
        aclrtResetDevice(args.rankId);
        LOG_PRINT("[INFO] device_%d Reset.\n", args.rankId);
        return 0;
    }

    int run_example_on_A2()
    {
        int ret = aclInit(nullptr);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclInit failed, ret = %d\n", ret); return ret);

        aclrtStream dispatchV2Stream[DEV_NUM_A2];
        aclrtStream combineV2Stream[DEV_NUM_A2];
        aclrtContext context[DEV_NUM_A2];
        for (uint32_t rankId = 0; rankId < DEV_NUM_A2; rankId++) {
            ret = aclrtSetDevice(rankId);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetDevice failed, ret = %d\n", ret); return ret);
            ret = aclrtCreateContext(&context[rankId], rankId);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateContext failed, ret = %d\n", ret); return ret);
            ret = aclrtCreateStream(&dispatchV2Stream[rankId]);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateStream failed, ret = %d\n", ret); return ret);
            ret = aclrtCreateStream(&combineV2Stream[rankId]);
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
            args[rankId].dispatchV2Stream = dispatchV2Stream[rankId];
            args[rankId].combineV2Stream = combineV2Stream[rankId];
            args[rankId].context = context[rankId];
            threads[rankId].reset(new (std::nothrow) std::thread(&launchOneThreadDispatchV2AndCombineV2_A2, std::ref(args[rankId])));
        }

        for (uint32_t rankId = 0; rankId < DEV_NUM_A2; rankId++) {
            threads[rankId]->join();
        }

        aclFinalize();
        LOG_PRINT("[INFO] aclFinalize success\n");
        return 0;
    }

    int run_example_on_A3A5()
    {
        int ret = aclInit(nullptr);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclInit failed. ret = %d\n", ret); return ret);

        aclrtStream dispatchV2Stream[DEV_NUM];
        aclrtStream combineV2Stream[DEV_NUM];
        aclrtContext context[DEV_NUM];
        for (uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
            ret = aclrtSetDevice(rankId);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetDevice failed. ret = %d\n", ret); return ret);
            ret = aclrtCreateContext(&context[rankId], rankId);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateContext failed. ret = %d\n", ret); return ret);
            ret = aclrtCreateStream(&dispatchV2Stream[rankId]);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateStream failed. ret = %d\n", ret); return ret);
            ret = aclrtCreateStream(&combineV2Stream[rankId]);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateStream failed. ret = %d\n", ret); return ret);
        }

        int32_t devicesEp[EP_WORLD_SIZE];
        for (int32_t epId = 0; epId < EP_WORLD_SIZE; epId++) {
            devicesEp[epId] = epId;
        }
        // Initialize the ep communication domain.
        HcclComm commsEp[EP_WORLD_SIZE];
        ret = HcclCommInitAll(EP_WORLD_SIZE, devicesEp, commsEp);
        CHECK_RET(
            ret == ACL_SUCCESS,
            LOG_PRINT("[ERROR] HcclCommInitAll ep failed. ret = %d\n", ret); return ret
        );

        Args args[DEV_NUM];
        // Each thread calls each device to execute the operator.
        std::vector<std::unique_ptr<std::thread>> threads(DEV_NUM);
        for (uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
            uint32_t epRankId = rankId;

            args[rankId].rankId = rankId;
            args[rankId].epRankId = epRankId;
            args[rankId].hcclEpComm = commsEp[epRankId];
            args[rankId].dispatchV2Stream = dispatchV2Stream[rankId];
            args[rankId].combineV2Stream = combineV2Stream[rankId];
            args[rankId].context = context[rankId];
            threads[rankId].reset(new (std::nothrow) std::thread(&launchOneThreadDispatchV2AndCombineV2_A3A5, std::ref(args[rankId])));
        }
        for (uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
            threads[rankId]->join();
        }
        aclFinalize();
        LOG_PRINT("[INFO] aclFinalize success\n");
        return 0;
    }


    int main(int argc, char *argv[])
    {
        if (!env_dev_num) {
            LOG_PRINT("[WARNING] Please check whether environment variable ENV_DEV_NUM is set correctly.\n");
            return 0;
        }
        int actual_env_dev_num = std::stoi(std::string(env_dev_num));
        if (actual_env_dev_num < DEV_NUM) {
            LOG_PRINT("[INFO] ENV_DEV_NUM = %d is less than %d, currently not supported\n", actual_env_dev_num, DEV_NUM);
            return 0;
        }
        if (IS_TEST_A2) {
            LOG_PRINT("Example on <Atlas A2> will be executed!\n");
            int ret = run_example_on_A2();
        }
        else if (IS_TEST_A3A5) {
            LOG_PRINT("Example on <Atlas A3> or <Atlas A5> will be executed!\n");
            int ret = run_example_on_A3A5();
        }

        return 0;
    }
    ```

<!-- end id26 -->
