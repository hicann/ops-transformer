# aclnnMoeDistributeDispatchV2

[📄 View Source Code](https://gitcode.com/cann/ops-transformer/tree/master/mc2/moe_distribute_dispatch_v2)

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| Ascend 950PR/Ascend 950DT                            |    √     |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √     |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference products</term>                            |    ×     |
| <term>Atlas training products</term>                             |    ×     |

## Function

- Description: Quantizes token data (optional). When there is TP domain communication, AllToAllV communication in the EP domain is performed first, and then AllGatherV TP domain communication is performed. When there is no such communication, AllToAllV communication in the EP domain is performed.

    Compared with the `aclnnMoeDistributeDispatch` API, this API has the following changes:
    1. Outputs more detailed token information to assist CombineV2 series operators in performing efficient all-rank synchronization. Therefore, the `expandIdx` output (shape (BS × K,)) in the original interface is replaced by the `assistInfoForCombineOut` parameter (shape (A × 128,)).
    2. Adds the `commAlg` parameter to replace the `HCCL_INTRA_PCIE_ENABLE` and `HCCL_INTRA_ROCE_ENABLE` environment variables.

    For details, see the following parameter description.
- Formula:

    - Case 1: If quantMode = 0 (non-quantization scenario):

    $$
    allToAllXOut = AllToAllV(X)\\
    expandXOut =
    \begin{cases}
    AllToAllV(X), & No TP communication domain\\
    AllGatherV(allToAllXOut), & With TP communication domain\\
    \end{cases}
    $$

    - Case 2: If quantMode = 1 (static quantization scenario):

    $$
    xFp32 = CastToFp32(X) \times scales \\
    quantOut = Cast(xFp32, dstType) \\
    allToAllXOut = AllToAllV(quantOut)\\
    expandXOut =
    \begin{cases}
    AllToAllV(quantOut), & No TP communication domain\\
    AllGatherV(allToAllXOut), & With TP communication domain\\
    \end{cases}
    $$

    - Case 3: If quantMode = 2 (per-token dynamic quantization scenario):

    $$
    xFp32 = CastToFp32(X) \times scales \\
    dynamicScales = dstTypeMax/Max(Abs(xFp32)) \\
    quantOut = CastToInt8(xFp32 \times dynamicScales) \\
    allToAllXOut = AllToAllV(quantOut) \\
    allToAllDynamicScalesOut = AllToAllV(1.0/dynamicScales) \\
    expandXOut =
    \begin{cases}
    AllToAllV(quantOut), & No TP communication domain\\
    AllGatherV(allToAllXOut), & With TP communication domain\\
    \end{cases} \\
    dynamicScalesOut =
    \begin{cases}
    allToAllDynamicScalesOut, & No TP communication domain\\
    AllGatherV(allToAllDynamicScalesOut), & With TP communication domain\\
    \end{cases}
    $$

    - Case 4: If quantMode = 3 (per-tile dynamic quantization scenario):

    $$
    xFp32 = CastToFp32(X) \times scales \\
    dynamicScales = dstTypeMax/Max(Abs(xFp32)) \\
    quantOut = CastToInt8(xFp32 \times dynamicScales) \\
    allToAllXOut = AllToAllV(quantOut) \\
    allToAllDynamicScalesOut = AllToAllV(1.0/dynamicScales) \\
    expandXOut =
    \begin{cases}
    AllToAllV(quantOut), & No TP communication domain\\
    AllGatherV(allToAllXOut), & With TP communication domain\\
    \end{cases} \\
    dynamicScalesOut =
    \begin{cases}
    allToAllDynamicScalesOut, & No TP communication domain\\
    AllGatherV(allToAllDynamicScalesOut), & With TP communication domain\\
    \end{cases}
    $$

    - Case 5: If quantMode = 4 (mxfp8 quantization scenario):

    $$
    sharedExp = Floor(log_2(max(x))) - emax \\
    dynamicScales = 2^{sharedExp} \\
    quantOut = CastToFp8(X / dynamicScales) \\
    allToAllXOut = AllToAllV(quantOut) \\
    allToAllDynamicScalesOut = AllToAllV(1.0 / dynamicScales) \\
    expandXOut =
    \begin{cases}
    AllToAllV(quantOut), & No TP communication domain\\
    AllGatherV(allToAllXOut), & With TP communication domain\\
    \end{cases} \\
    dynamicScalesOut =
    \begin{cases}
    allToAllDynamicScalesOut, & No TP communication domain\\
    AllGatherV(allToAllDynamicScalesOut), & With TP communication domain\\
    \end{cases}
    $$

    $emax$ indicates the value of the exponent part corresponding to the maximum normal number of this type.

- <term>Atlas A2 training products/Atlas A2 inference products</term>: This API must be used together with `aclnnMoeDistributeCombineV2`.
- <term>Atlas A3 training series products/Atlas A3 inference series products</term> /Ascend 950PR/Ascend 950DT: This API must be used together with aclnnMoeDistributeCombineV2 or aclnnMoeDistributeCombineAddRmsNorm.

> Notes:
> The `aclnnMoeDistributeCombineV2` and `aclnnMoeDistributeCombineAddRmsNorm` operators are collectively referred to as **CombineV2 series operators** in subsequent documents.

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnMoeDistributeDispatchV2GetWorkspaceSize` is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, `aclnnMoeDistributeDispatchV2` is called to perform computation.

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
    <td>The value must be a 2D tensor.</td>
    <td>FLOAT16, BFLOAT16</td>
    <td>ND</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>expertIds</td>
    <td>Input</td>
    <td>Top K expert indexes of each token.</td>
    <td>The value must be a 2D tensor.</td>
    <td>INT32</td>
    <td>ND</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>scalesOptional</td>
    <td>Input</td>
    <td>Quantization smoothing parameter for each expert.</td>
    <td>-</td>
    <td>FLOAT32</td>
    <td>ND</td>
    <td>-</td>
    <td>√</td>
    </tr>
    <tr>
    <td>xActiveMaskOptional</td>
    <td>Input</td>
    <td>Indicates whether the tokens participate in communication.</td>
    <td><ul><li>Pass valid data or a null pointer. By default, all tokens participate in communication. </li><li>All tokens must be valid when the Bs of each rank is different.</li></ul></td>
    <td>BOOL</td>
    <td>ND</td>
    <td>-</td>
    <td>√</td>
    </tr>
    <tr>
    <td>expertScalesOptional</td>
    <td>Input</td>
    <td>Top K expert weights of each token.</td>
    <td>-</td>
    <td>FLOAT32</td>
    <td>ND</td>
    <td>-</td>
    <td>√</td>
    </tr>
    <tr>
    <td>groupEp</td>
    <td>Input</td>
    <td>Name of the EP communication domain (expert parallel communication domain).</td>
    <td><ul><li>The string length is in the range of [1, 128).</li><li>The value must be different from that of groupTp.</li></ul></td>
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
    <td><ul><li>The value must be [0, epWorldSize).</li><li>The epRankId of each card in the same EP communicator must be unique.</li></ul></td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>moeExpertNum</td>
    <td>Input</td>
    <td>Number of MoE experts.</td>
    <td>The following condition must be met: moeExpertNum % (epWorldSize - sharedExpertRankNum) = 0.</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>groupTp</td>
    <td>Input</td>
    <td>Name of the TP communication domain (data parallel communication domain).</td>
    <td>The value must be different from that of `groupEp`.</td>
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
    <td>ID of the current rank in the TP communication domain.</td>
    <td>The value of `tpRankId` must be unique for each rank in the same EP communication domain.</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>expertShardType</td>
    <td>Input</td>
    <td>Distribution type of shared expert ranks.</td>
    <td>Currently, only 0 is supported, indicating that shared expert ranks are placed before MoE expert ranks.</td>
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
    <td><ul><li> If the BSs of all ranks are the same, the value of globalBS = BS * epWorldSize is 0. </li><li> globalBS = maxBS * epWorldSize when the BS of each rank is different (maxBS indicates the maximum BS of a single card).</li></ul></td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>expertTokenNumsType</td>
    <td>Input</td>
    <td>Outputs the semantic type of the value in `expertTokenNums`.</td>
    <td>0: prefix sum of the number of tokens processed by each expert; 1: number of tokens processed by each expert.</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>√</td>
    </tr>
    <tr>
    <td>commAlg</td>
    <td>Input</td>
    <td>Indicates the communication affinity memory layout algorithm.</td>
    <td>-</td>
    <td>STRING</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>expandXOut</td>
    <td>Output</td>
    <td>Token features expanded based on `expertIds`.</td>
    <td>The value must be a 2D tensor.</td>
    <td>FLOAT16, BFLOAT16, INT8, FLOAT8_E4M3FN, FLOAT8_E5M2, HIFLOAT8</td>
    <td>ND</td>
    <td>(max(tpWorldSize, 1) × A, H)</td>
    <td>√</td>
    </tr>
    <tr>
    <td>dynamicScalesOut</td>
    <td>Output</td>
    <td>`aclTensor` on the device.</td>
    <td>The value must be a 1D tensor.</td>
    <td>FLOAT32, FLOAT8_E8M0</td>
    <td>ND</td>
    <td>(A, )</td>
    <td>√</td>
    </tr>
    <tr>
    <td>assistInfoForCombineOut</td>
    <td>Output</td>
    <td>Number of tokens sent to the same expert (corresponding to `assistInfoForCombine` in `aclnnMoeDistributeCombineV2`).</td>
    <td>The value must be a 1D tensor.</td>
    <td>INT32</td>
    <td>ND</td>
    <td>(A × 128, )</td>
    <td>√</td>
    </tr>
    <tr>
    <td>expertTokenNumsOut</td>
    <td>Output</td>
    <td>Number of tokens received by each expert.</td>
    <td>The value must be a 1D tensor.</td>
    <td>INT64</td>
    <td>ND</td>
    <td>(localExpertNum, )</td>
    <td>√</td>
    </tr>
    <tr>
    <td>epRecvCountsOut</td>
    <td>Output</td>
    <td>Number of tokens received from each rank in the EP communication domain (corresponding to `epSendCounts` in `aclnnMoeDistributeCombineV2`).</td>
    <td>The value must be a 1D tensor.</td>
    <td>INT32</td>
    <td>ND</td>
    <td>-</td>
    <td>√</td>
    </tr>
    <tr>
    <td>tpRecvCountsOut</td>
    <td>Output</td>
    <td>Number of tokens received from each rank in the TP communication domain (corresponding to `tpSendCounts` in `aclnnMoeDistributeCombineV2`).</td>
    <td>This output is available only when there is TP domain communication.</td>
    <td>INT32</td>
    <td>ND</td>
    <td>-</td>
    <td>√</td>
    </tr>
    <tr>
    <td>expandScalesOut</td>
    <td>Output</td>
    <td>Weight of the token output by the rank (corresponding to `expertScalesOptional` in `aclnnMoeDistributeCombineV2`).</td>
    <td>-</td>
    <td>FLOAT32</td>
    <td>ND</td>
    <td>-</td>
    <td>√</td>
    </tr>
    <tr>
    <td>workspaceSize</td>
    <td>Output</td>
    <td>Size of the workspace required to be allocated on the device.</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>executor</td>
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

    - <term>Atlas A2 training products/Atlas A2 inference products</term>:
        - The value of `commAlg` can be `nullptr`, `""`, `"fullmesh"`, or `"hierarchy"`. It is recommended to use `"hierarchy"` with driver version 25.0.RC1.1 or later. When set to `nullptr` or `""`, the communication algorithm is selected based on HCCL environment variables (not recommended). `"fullmesh"` indicates that tokens are directly transmitted through RDMA. `"hierarchy"` indicates a two-stage communication process: intra-server communication followed by inter-server communication, which reduces cross-server data transmission.
        - `scalesOptional` must be passed as a null pointer when `commAlg` is `hierarchy`, or when HCCL_INTRA_PCIE_ENABLE=1 and HCCL_INTRA_ROCE_ENABLE=0
        - `xActiveMaskOptional` depends on the `commAlg` value: For `"fullmesh"`, it requires a 1D tensor with shape (BS, ), where `true` must precede `false` (for example, {true, false, true} is invalid); for `"hierarchy"`, it is currently not supported and a null pointer should be passed.
        - The value of expertScalesOptional must be a 2D tensor with the shape of (BS, K).
        - The value of epWorldSize depends on the value of commAlg. For "fullmesh", the value can be 2, 3, 4, 5, 6, 7, 8, 16, 32, 64, 128, 192, 256, or 384. For "hierarchy", the value can be 16, 32, or 64.
        - The value of moeExpertNum ranges from (0, 512].
        - `groupTp` is not supported in the current version. Pass an empty string.
        - The current version does not support `tpWorldSize`, `tpRankId`, `expertShardType`, `sharedExpertNum`, and `sharedExpertRankNum`. Pass 0 for these parameters.
        - The shape of epRecvCountsOut is (moeExpertNum + 2 *globalBS* K * serverNum). The first moeExpertNum elements are the number of received tokens, and the remaining elements are the reduce information before communication.
        - Currently, TP domain communication is not supported.
        - expandScalesOut must be a 1D tensor with shape (A,).
        - `quantMode` supports 0 (non-quantization) and 2 (dynamic quantization).

    - <term>Atlas A3 training products/Atlas A3 inference products</term>:
        - `commAlg` is not supported in the current version. Pass a null pointer.
        - `xActiveMaskOptional` must be a 1D tensor with shape (BS, ) or a 2D tensor with shape (BS, K). If it is a 1D tensor, `true` must be placed before `false`. If it is a 2D tensor and the K values corresponding to tokens are all `false`, the tokens do not participate in communication.
        - `expertScalesOptional` is not supported in the current version. Pass a null pointer.
        - The value of `epWorldSize` must be in the range [2, 768].
        - The value of `moeExpertNum` must be in the range (0, 1024].
        - `groupTp` must be a string of length [0, 128) and cannot be the same as `groupEp`. This parameter can be left empty only when there is no TP domain communication.
        - The value of `tpWorldSize` must be in the range [0, 2]. 0 and 1 indicate no TP domain communication. 2 is required when TP domain communication is used.
        - The value of `tpRankId` must be in the range [0, 1]. `tpRankId` of each rank in the same TP domain must be unique. If TP domain communication is not used, pass 0.
        - The value of `expertShardType` must be 0, indicating that shared expert ranks are placed in front of MoE expert ranks.
        - The value of `sharedExpertNum` must be in the range [0, 4].
        - The value of `sharedExpertRankNum` must be in the range [0, epWorldSize). If the value is 0, `sharedExpertNum` is 0 or 1. If the value is not 0, `sharedExpertRankNum % sharedExpertNum` is 0.
        - The shape of epRecvCountsOut is (epWorldSize *max(tpWorldSize, 1)* localExpertNum,).
        - If there is communication in the TP domain, tpRecvCountsOut is a 1D shape tensor, and the shape is (tpWorldSize,).
        - `expandScalesOut` is not supported in the current version.
        - `quantMode` supports 0 (non-quantization) and 2 (dynamic quantization).

    - Ascend 950PR/Ascend 950DT:
        - `commAlg` is not supported in the current version. Pass a null pointer.
        - xActiveMaskOptional must be a 1D or 2D tensor. (When it is a 1D tensor, the shape is (BS, ). When it is a 2D tensor, the shape is (BS, K).) In 1D mode, true must be placed before false. For example, {true, false, true} is invalid. In 2D mode, if all the K values corresponding to a token are false, the token is not involved in communication.
        - `expertScalesOptional` is not supported in the current version. Pass a null pointer.
        - The value of `epWorldSize` must be in the range [2, 768].
        - The value of `moeExpertNum` must be in the range (0, 1024].
        - `groupTp` is not supported in the current version. Pass an empty string.
        - `tpWorldSize` is not supported in the current version. Pass 0.
        - `tpRankId` is not supported in the current version. Pass 0.
        - The value of `expertShardType` must be 0, indicating that shared expert ranks are placed in front of MoE expert ranks.
        - The value of `sharedExpertNum` must be in the range [0, 4].
        - The value of `sharedExpertRankNum` must be in the range [0, epWorldSize). If the value is 0, `sharedExpertNum` is 0 or 1. If the value is not 0, `sharedExpertRankNum % sharedExpertNum` is 0.
        - The shape of epRecvCountsOut is (epWorldSize *max(tpWorldSize, 1)* localExpertNum,).
        - The output tpRecvCountsOut is not supported in the current version.
        - `expandScalesOut` is not supported in the current version.
        - The value of quantMode can be 0 (non-quantization), 1 (static quantization), 2 (per-token dynamic quantization), 3 (per-group dynamic quantization), or 4 (mxfp8 dynamic quantization).

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
    <td rowspan="2">ACLNN_ERR_INNER_TILING_ERROR</td>
    <td rowspan="2">561002</td>
    <td>The input and output shapes are not supported.</td>
    </tr>
    <tr>
        <td>The parameter value is not supported.</td>
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
    <td>Size of the workspace to be allocated on the device, which is obtained by calling the first-phase API `aclnnMoeDistributeDispatchV2GetWorkspaceSize`.</td>
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

1. Deterministic computing:
     - `aclnnMoeDistributeDispatchV2` defaults to a deterministic implementation.

2. Driver restrictions:
     - The driver versions of all nodes in the operator communicator must be the same.

3. `aclnnMoeDistributeDispatchV2` and `aclnnMoeDistributeCombineV2` must be used together. For details, see [Example](#Example).

4. The element values in the `assistInfoForCombineOut`, `epRecvCountsOut`, `tpRecvCountsOut`, and `expandScalesOut` tensor output of `aclnnMoeDistributeDispatchV2` may vary depending on the product model, communication algorithm, or version. Pass the tensors directly to the corresponding parameters of `aclnnMoeDistributeCombineV2`. Other service logics of the model should not depend on the element values.

5. The values of `groupEp`, `epWorldSize`, `moeExpertNum`, `groupTp`, `tpWorldSize`, `expertShardType`, `sharedExpertNum`, `sharedExpertRankNum`, `globalBs`, `commAlg`, and `HCCL_BUFFSIZE` used during API calling must be the same for all ranks, at all network layers, and the same as those of `aclnnMoeDistributeCombineV2`.

6. <term>Atlas A3 training products/Atlas A3 inference products</term>: In this scenario, a single rank contains dual dies. Therefore, "this rank" in the parameter description indicates a single die.

7. The shape format is described as follows:

    - **A**: Maximum number of tokens that can be received by the current rank. The value range is as follows:
      - For shared experts, the following condition must be met: (A = BS *epWorldSize* sharedExpertNum / sharedExpertRankNum)
      - For MoE experts, when globalBS is 0, the following condition must be met: (A >= BS *epWorldSize* min(localExpertNum, K)). When globalBS is not 0, the following condition must be met: (A >= globalBS * min(localExpertNum, K))
    - **H**: Hidden layer size.
      - <term>Atlas A2 training products/Atlas A2 inference products</term>: The value depends on commAlg. "fullmesh" must be in the range (0, 7168] and be an integer multiple of 32. If "hierarchy" is used and the driver version is 25.0.RC1.1 or later, the value must be in the range (0, 10*1024] and be an integer multiple of 32.
      - <term>Atlas A3 training products/Atlas A3 inference products</term>: The value range is [1024, 8192].
      - Ascend 950PR/Ascend 950DT: The value is within the range of [1024, 8192].
    - **Bs**: Batch sequence size, that is, the number of tokens output by the current rank.
      - <term>Atlas A2 training products/Atlas A2 inference products</term>: The value is determined by the value of commAlg. If commAlg is set to "fullmesh", the value is within the range of (0 < BS ≤ 256). If commAlg is set to "hierarchy" and the driver version is 25.0.RC1.1 or later, the value is within the range of (0 < BS ≤ 512).
      - <term>Atlas A3 training products/Atlas A3 inference products</term>: The value range is (0 < Bs ≤ 512).
      - Ascend 950PR/Ascend 950DT: The value is within the range of (0 < BS ≤ 512).
    - **K**: Number of top K experts, which must be in the ranges 0 < K ≤ 16 and 0 < K ≤ moeExpertNum.
    - **serverNum**: Number of server nodes. The value can only be 2, 4, or 8.
    - **localExpertNum**: Number of experts on the current rank.
      - For shared expert ranks, localExpertNum = 1.
      - For MoE expert ranks, localExpertNum = moeExpertNum/(epWorldSize - sharedExpertRankNum). If localExpertNum > 1, TP domain communication is not supported.

8. **quantMode constraints**:
    - <term>Atlas A2 training products/Atlas A2 inference products</term>:
        - When quantMode is set to 0, it indicates the non-quantization scenario, and the input scales pointer is null.
        - If `quantMode` is set to 2, it indicates the pertoken dynamic quantization scenario. The data type of `expandX` can be INT8.
            - For `scales`, you can pass a null pointer.
            - If valid data is passed to `scales`, the shape is (moeExpertNum, H).
    - <term>Atlas A3 training products/Atlas A3 inference products</term>:
        - When quantMode is set to 0, it indicates the non-quantization scenario, and the input scales pointer is null.
        - If `quantMode` is set to 2, it indicates the pertoken dynamic quantization scenario. The data type of `expandX` can be INT8.
            - For `scales`, you can pass a null pointer.
            - If valid data is passed to `scales` and shared expert ranks exist, the shape is (sharedExpertNum + moeExpertNum, H).
            - If valid data is passed to `scales` but no shared expert ranks exist, the shape is (moeExpertNum, H).
    - Ascend 950PR/Ascend 950DT:
        - When quantMode is set to 0, it indicates the non-quantization scenario. The data type of expandX can be FLOAT16 or BFLOAT16. The input scales must be a null pointer.
        - When quantMode is set to 1, it indicates the static quantization scenario. The data type of expandX can be INT8 or HIFLOAT8.
            - When the data type of expandX is INT8, the following scenarios are supported:
                - The input scales represent the quantization coefficient, and the shape is (1,).
                - When the input scales represent the smooth weight shared by each expert, the shape is (H, ).
                - When the input scales represent the quantization coefficient that integrates the smooth weight of each expert, if there is a shared expert card, the shape is (sharedExpertNum + moeExpertNum, H); if there is no shared expert card, the shape is (moeExpertNum, H).
            - When the data type of expandX is HIFLOAT8, the shape of scales must be (1, ).
        - When quantMode is set to 2, it indicates the per-token dynamic quantization scenario. The data type of expandX can be INT8, FLOAT8_E4M3FN or FLOAT8_E5M2.
            - For `scales`, you can pass a null pointer.
            - If valid data is passed to `scales` and shared expert ranks exist, the shape is (sharedExpertNum + moeExpertNum, H).
            - If valid data is passed to `scales` but no shared expert ranks exist, the shape is (moeExpertNum, H).
        - When quantMode is set to 3, it indicates the per-group dynamic quantization scenario. The data type of expandX can be FLOAT8_E4M3FN or FLOAT8_E5M2.
            - For `scales`, you can pass a null pointer.
            - If valid data is passed to `scales` and shared expert ranks exist, the shape is (sharedExpertNum + moeExpertNum, H).
            - If valid data is passed to `scales` but no shared expert ranks exist, the shape is (moeExpertNum, H).
        - When quantMode is set to 4, it indicates the mxfp8 quantization scenario. The data type of expandX can be FLOAT8_E4M3FN or FLOAT8_E5M2. The input scales must be a null pointer.

9. **HCCL_BUFFSIZE**:

   Before calling this API, check whether the value of the `HCCL_BUFFSIZE` environment variable is proper. This environment variable indicates the buffer size occupied by a single communication domain, in MB. If this environment variable is not set, the default value 200 MB is used.
    - <term>Atlas A2 training products/Atlas A2 inference products</term>:
        - If `commAlg` is set to "" or a null pointer, select the "fullmesh" or "hierarchy" formula based on the HCCL environment variable.
        - If commAlg is "fullmesh", the size must be greater than or equal to 2 x (BS x epWorldSize x min(localExpertNum, K) x H x sizeof(uint16) + 2MB).
        - If commAlg is "hierarchy", the size must be (≥ (moeExpertNum + epWorldSize / 4) \* Align512(maxBS \* (H \* 2 + 16 \* Align8(K))) \* 1B + 8MB, where Align8(x) = ((x + 8 - 1) / 8) x 8, and Align512(x) = ((x + 512 - 1) / 512) x 512).
    - <term>Atlas A3 training products/Atlas A3 inference products</term>:
        - In the EP communicator, the size must be greater than or equal to 2 and meet the following condition: ≥ 2 \* (localExpertNum \* maxBS \* epWorldSize \* Align512(Align32(2 \* H) + 64) + (K + sharedExpertNum) \* maxBS \* Align512(2 \* H)). (localExpertNum indicates the number of experts on the MoE expert card. Align512(x) = ((x + 512 – 1) / 512) x 512. Align32(x) = ((x + 32 – 1) / 32) x 32.)
        - Within the TP communication domain: The value must be grater than or equal to `A * (H * 2 + 128) * 2`.
    - Ascend 950PR/Ascend 950DT: The size must be greater than or equal to 2 and meet the following condition: >= aivNum * 512 + 2 \* epWorldSize \* (maxBS \* Align512(alignedH \* 2) \* localExpertNum + 512). (aivNum indicates the number of cores. localExpertNum indicates the number of experts on the MoE expert card. Align512(x) = ((x + 512 – 1) / 512) x 512. The requirements for alignedH vary in different quantization scenarios.
        - In the pergroup dynamic quantization scenario, alignedH = Align128(H) = ((H + 128 – 1) / 128) x 128.
        - In the mx quantization scenario, alignedH = Align256(H) = ((H + 256 – 1) / 256) x 256.
        - In other quantization modes, alignedH = Align32(H) = ((H + 32 – 1) / 32) x 32.

10. **HCCL_INTRA_PCIE_ENABLE and HCCL_INTRA_ROCE_ENABLE:**
   <term>Atlas A2 training products/Atlas A2 inference products</term>: This environment variable is not recommended. You are advised to set `commAlg` to `"hierarchy"`.

11. In the formulas in this document, / denotes integer division.

12. Constraints on the use of the communication domain:
    - `aclnnMoeDistributeCombineV2` and `aclnnMoeDistributeDispatchV2` in a model support only the same EP communication domain, and no other operators are allowed in the communication domain.
    - `aclnnMoeDistributeCombineV2` and `aclnnMoeDistributeDispatchV2` in a model support only the same TP communication domain or both do not support a TP communication domain. If a TP communication domain is supported, no other operators are allowed in the communication domain.
    - <term>Atlas A3 training products/Atlas A3 inference products</term>: Nodes in a communication domain must be in the same SuperPoD. Cross-SuperPoD nodes are not supported.

13. Networking constraints:
    - <term>Atlas A2 training products/Atlas A2 inference products</term>: In multi-server scenarios, only switch-based networking is supported, and direct point-to-point networking between two servers is not supported.

## Example

- Preparing files:

    1. Create the `rank_table_m2.json` file and modify it according to the following instructions.
    
    2. Copy the project to the two servers and configure the `rank_table_m2.json` file based on the device IP addresses of the servers. Ensure that the `rank_table_m2.json` files on the two servers are the same.
    
    3. Install the CANN package and compile and run it based on [Operator Invocation](https://gitcode.com/cann/ops-nn/blob/9.0.0/docs/en/invocation/quick_op_invocation.md).

- About rankTable:

    1. You can configure the NPU resource information involved in collective communication through the ranktable file. For details, see "Communication Function Development > Cluster Information Configuration > Configuring Resource Information Through the ranktable File" in [HCCL User Guide](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/900/devaids/hccltool/HCCLpertest_16_0001.html).

    2. Run the `cat /etc/hccn.conf` or `for i in seq 0 7; do echo "===================> dev$i, NPU$((i+1))"; hccn_tool -i $i -ip -g; done` to query the device IP address. Then, set the JSON file following instructions in the collective communication guide.

    > Note: In 2-server 16-rank scenarios, the `device_ids` of both servers range from 0 to 7. The `rank_id` of one server ranges from 0 to 7, and that of the other server ranges from 8 to 15. In single-server 16-rank scenarios, both the `device_ids` and `rank_ids` range from 0 to 15.

- Environment variable settings:

    ```bash
    # Before running the code, set the three environment variables.
    ## `FIRST_RANK_ID`: For example, if there are 2 servers with 16 ranks, set `FIRST_RANK_ID` to 0 for one server and 8 for the other server.
    ## For example, export FIRST_RANK_ID=0.
    export RANK_TABLE_FILE=/home/path/to/rank_table_m2.json
    export FIRST_RANK_ID=<Start rank ID of the server>
    ## `ENV_DEV_NUM`: Set this variable based on the number of ranks on the current server. For example, if there are 2 servers with 16 ranks, set this variable to 16 for both servers.
    export ENV_DEV_NUM=16

- Set the number of servers:

    In 2-server 16-rank scenarios, set `MACHINE_NUM` to 2.

    ```Cpp
    const uint32_t MACHINE_NUM = 2;
    ```

    You do not need to set this variable in single-server 16-rank scenarios.

- <term>Atlas A2 training products/Atlas A2 inference products</term>:

    You do not need to configure the `ranktable` file or the environment variables `RANK_TABLE_FILE` and `FIRST_RANK_ID`.

    In this example, the A2 operator can run in a single-node environment with 2 to 8 ranks. Before running the example, set IS_TEST_A2 in the sample code to true to ensure that the A2 branch is executed.
    In addition, you can set EP_WORLD_SIZE_A2 to the number of ranks in the sample code and change moeExpertNum in the launchOneThreadDispatchV2AndCombineV2_A2 function so that moeExpertNum can be exactly divided by EP_WORLD_SIZE_A2.

    The operator compilation command is as follows. Both the moe_distribute_dispatch_v2 and moe_distribute_combine_v2 operators need to be compiled. These two operators must be executed in pairs.

    ```bash
    bash build.sh --pkg --soc=ascend910b --ops=moe_distribute_dispatch_v2,moe_distribute_combine_v2
    ```

    The command for executing the sample operator is as follows:

    ```bash
    bash build.sh --run_example --ops=moe_distribute_dispatch_v2 eager cust
    ```

- <term>Atlas A3 training products/Atlas A3 inference products</term>:

    You do not need to configure the `ranktable` file or the environment variables `RANK_TABLE_FILE` and `FIRST_RANK_ID`.

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

- <term>Atlas A2 training products/Atlas A2 inference products</term>, <term>Atlas A3 training products/Atlas A3 inference products</term>, and Ascend 950PR/Ascend 950DT:

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
        } while(0)

    struct Args {
        uint32_t rankId;
        uint32_t epRankId;
        uint32_t tpRankId;
        HcclComm hcclEpComm;
        HcclComm hcclTpComm;
        aclrtStream dispatchV2Stream;
        aclrtStream combineV2Stream;
        aclrtContext context;
    };

    const uint32_t MACHINE_NUM = 1;
    const char* rank_table_file = std::getenv("RANK_TABLE_FILE");
    const char* first_rank_id = std::getenv("FIRST_RANK_ID");
    const char* env_dev_num = std::getenv("ENV_DEV_NUM");

    const uint32_t EP_WORLD_SIZE = (!first_rank_id) ? 2 : 16;
    const uint32_t TP_WORLD_SIZE = (!first_rank_id) ? 1 : 0;
    const uint32_t DEV_NUM = (!first_rank_id) ? EP_WORLD_SIZE * TP_WORLD_SIZE : EP_WORLD_SIZE;

    const bool IS_TEST_A2 = false;
    const uint32_t EP_WORLD_SIZE_A2 = 8;
    const uint32_t TP_WORLD_SIZE_A2 = 1;
    const uint32_t DEV_NUM_A2 = EP_WORLD_SIZE_A2 * TP_WORLD_SIZE_A2;

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
        char hcomTpName[128] = {0};
        if (!rank_table_file && !first_rank_id) {
            ret = HcclGetCommName(args.hcclTpComm, hcomTpName);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclGetTpCommName failed. ret: %d\n", ret); return -1);
        }
        LOG_PRINT(
            "[INFO] rank = %d, hcomEpName = %s, hcomTpName = %s, dispatchV2Stream = %p, combineV2Stream = %p, context = %p\n",
            args.rankId, hcomEpName, hcomTpName, args.dispatchV2Stream, args.combineV2Stream, args.context
        );

        // Set the scenario.
        int64_t BS = 8;
        int64_t H = 7168;
        int64_t K = 1;
        int64_t expertShardType = 0;
        int64_t sharedExpertNum = 0;
        int64_t sharedExpertRankNum = 0;
        if (!rank_table_file && !first_rank_id) {
            sharedExpertNum = 1;
            sharedExpertRankNum = 1;
        } 
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
            // Shared expert ranks
            localExpertNum = 1;
            A = globalBS / sharedExpertRankNum;
        } else { 
            // MoE expert ranks
            localExpertNum = moeExpertNum / (EP_WORLD_SIZE - sharedExpertRankNum);
            A = globalBS * (localExpertNum < K ? localExpertNum : K);
        }
        std::string commAlg = "";

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
        
        // Define the dimensions of variables in the current scenario.
        std::vector<int64_t> xShape{BS, H};
        std::vector<int64_t> expertIdsShape{BS, K};
        std::vector<int64_t> scalesShape{(sharedExpertRankNum > 0) ? 1 + moeExpertNum : moeExpertNum, H};
        std::vector<int64_t> expertScalesShape{BS, K};
        std::vector<int64_t> expandXShape{(TP_WORLD_SIZE > 0 ? TP_WORLD_SIZE : 1) * A, H};
        std::vector<int64_t> dynamicScalesShape{(TP_WORLD_SIZE > 0 ? TP_WORLD_SIZE : 1) * A};
        std::vector<int64_t> expandIdxShape{A * 128};
        std::vector<int64_t> expertTokenNumsShape{localExpertNum};
        std::vector<int64_t> epRecvCountsShape{(TP_WORLD_SIZE > 0 ? TP_WORLD_SIZE : 1) * localExpertNum * EP_WORLD_SIZE};
        std::vector<int64_t> tpRecvCountsShape{TP_WORLD_SIZE > 0 ? TP_WORLD_SIZE : 1};
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

        // Construct variables on the host.
        std::vector<op::fp16_t> xHostData(xShapeSize, 1);
        std::vector<int32_t> expertIdsHostData;
        for (int32_t token_id = 0; token_id < expertIdsShape[0]; token_id++) {
            // Each token is sent to the MoE experts {0, 1, ... k - 1}.
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
        
        /* Declare the variables required for operator execution. */
        uint64_t dispatchV2WorkspaceSize = 0;
        aclOpExecutor *dispatchV2Executor = nullptr;
        void *dispatchV2WorkspaceAddr = nullptr;

        uint64_t combineV2WorkspaceSize = 0;
        aclOpExecutor *combineV2Executor = nullptr;
        void *combineV2WorkspaceAddr = nullptr;   

        /* Execute the dispatchV2 and combineV2 operators in sequence. */
        // Call the first-phase API of dispatchV2.
        ret = aclnnMoeDistributeDispatchV2GetWorkspaceSize(
            x, expertIds, 
            (quantMode > 0 ? scales : nullptr), nullptr, 
            expertScales, 
            hcomEpName, EP_WORLD_SIZE, args.epRankId,
            moeExpertNum, hcomTpName, TP_WORLD_SIZE,
            args.tpRankId, expertShardType, sharedExpertNum,
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
        // Allocate device memory based on the workspaceSize computed by the first-phase API of dispatchV2.
        if (dispatchV2WorkspaceSize > 0) {
            ret = aclrtMalloc(&dispatchV2WorkspaceAddr, dispatchV2WorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc failed. ret = %d\n", ret); return ret);
        }
        // Call the second-phase API of dispatchV2.
        ret = aclnnMoeDistributeDispatchV2(dispatchV2WorkspaceAddr, dispatchV2WorkspaceSize, dispatchV2Executor, args.dispatchV2Stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnMoeDistributedispatchV2 failed. ret = %d\n", ret); return ret);
        // (Boilerplate) Wait until the task execution is complete.
        ret = aclrtSynchronizeStreamWithTimeout(args.dispatchV2Stream, 10000);
        CHECK_RET(
            ret == ACL_SUCCESS,
            LOG_PRINT("[ERROR] aclrtSynchronizeStreamWithTimeout failed. ret = %d\n", ret); return ret
        );

        // Call the first-phase API of combineV2.
        ret = aclnnMoeDistributeCombineV2GetWorkspaceSize(
            expandX, expertIds, expandIdx, epRecvCounts, expertScales, tpRecvCounts,
            nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
            hcomEpName, EP_WORLD_SIZE, args.epRankId, moeExpertNum, hcomTpName, TP_WORLD_SIZE, args.tpRankId,
            expertShardType, sharedExpertNum, sharedExpertRankNum, globalBS, outDtype, commQuantMode, groupListType,
            commAlg.c_str(), x, &combineV2WorkspaceSize, &combineV2Executor);
        CHECK_RET(
            ret == ACL_SUCCESS,
            LOG_PRINT("[ERROR] aclnnMoeDistributeCombineV2GetWorkspaceSize failed. ret = %d\n", ret); return ret
        );
        // Allocate device memory based on the workspaceSize computed by the first-phase API of combineV2.
        if (combineV2WorkspaceSize > 0) {
            ret = aclrtMalloc(&combineV2WorkspaceAddr, combineV2WorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc failed. ret = %d\n", ret); return ret);
        }
        // Call the second-phase API of combineV2.
        ret = aclnnMoeDistributeCombineV2(combineV2WorkspaceAddr, combineV2WorkspaceSize, combineV2Executor, args.combineV2Stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnMoeDistributeCombineV2 failed. ret = %d\n", ret); return ret);
        // (Boilerplate) Wait until the task execution is complete.
        ret = aclrtSynchronizeStreamWithTimeout(args.combineV2Stream, 10000);
        CHECK_RET(
            ret == ACL_SUCCESS, 
            LOG_PRINT("[ERROR] aclrtSynchronizeStreamWithTimeout failed. ret = %d\n", ret); return ret
        );

        LOG_PRINT("[INFO] device_%d aclnnMoeDistributedispatchV2 and aclnnMoeDistributeCombineV2 execute successfully.\n", args.rankId);

        // Free device resources.
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
        FreeDeviceAddr(expandScalesDeviceAddr);
        FreeDeviceAddr(tpRecvCountsDeviceAddr);

        HcclCommDestroy(args.hcclEpComm);
        HcclCommDestroy(args.hcclTpComm);
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

        // Define the dimensions of variables in the current scenario.
        std::vector<int64_t> xShape{BS, H};
        std::vector<int64_t> expertIdsShape{BS, K};
        std::vector<int64_t> scalesShape{moeExpertNum + 1, H};
        std::vector<int64_t> expertScalesShape{BS, K};

        std::vector<int64_t> expandXShape{TP_WORLD_SIZE_A2 * A, H};
        std::vector<int64_t> dynamicScalesShape{TP_WORLD_SIZE_A2 * A};
        std::vector<int64_t> assistInfoForCombineShape{A * 128};
        std::vector<int64_t> expertTokenNumsShape{localExpertNum};
        std::vector<int64_t> epRecvCountsShape{TP_WORLD_SIZE_A2 * localExpertNum * EP_WORLD_SIZE_A2}; // non-layering
        std::vector<int64_t> tpRecvCountsShape{TP_WORLD_SIZE_A2};
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

        /**************************************** Call dispatch. ********************************************/
        // Call the first-phase API.
        ret = aclnnMoeDistributeDispatchV2GetWorkspaceSize(x, expertIds, (quantMode > 0 ? scales : nullptr), xActiveMask,
                expertScales, hcomEpName, EP_WORLD_SIZE_A2, args.epRankId, moeExpertNum, "", TP_WORLD_SIZE_A2,
                args.tpRankId, expertShardType, sharedExpertNum,sharedExpertRankNum, quantMode, globalBS,
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
        /**************************************** Call combine. ********************************************/
        // Call the first-phase API.
        ret = aclnnMoeDistributeCombineV2GetWorkspaceSize(expandX, expertIds,
                                                            assistInfoForCombine, epRecvCounts,
                                                            expertScales, tpRecvCounts,
                                                            xActiveMask, activationScale, weightScale,
                                                            groupList, expandScales, sharedExpertX,
                                                            hcomEpName, EP_WORLD_SIZE_A2, args.epRankId, moeExpertNum,
                                                            "", TP_WORLD_SIZE_A2, args.tpRankId, expertShardType,
                                                            sharedExpertNum, sharedExpertRankNum, globalBS, outDtype,
                                                            commQuantMode, groupList_type, commAlg.c_str(), xOut,
                                                            &combineWorkspaceSize, &combineExecutor);
        CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("[ERROR] aclnnMoeDistributeCombineV2GetWorkspaceSize failed. ret = %d \n", ret); return ret);
        // Allocate device memory based on the computed workspaceSize.
        if (combineWorkspaceSize > 0) {
            ret = aclrtMalloc(&combineWorkspaceAddr, combineWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc workspace failed. ret = %d \n", ret); return ret);
        }

        // Call the second-phase API.
        ret = aclnnMoeDistributeCombineV2(combineWorkspaceAddr, combineWorkspaceSize, combineExecutor, args.combineV2Stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnMoeDistributeCombineV2 failed. ret = %d \n", ret);
            return ret);
        // (Boilerplate) Wait until the task execution is complete.
        ret = aclrtSynchronizeStreamWithTimeout(args.combineV2Stream, 10000);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSynchronizeStreamWithTimeout failed. ret = %d \n", ret);
            return ret);
        LOG_PRINT("[INFO] device_%d aclnnMoeDistributeDispatchV2 and aclnnMoeDistributeCombineV2                      \
                    execute successfully.\n", args.rankId);
        // Free device resources.
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
        LOG_PRINT("[INFO] device_%d DeStroy.\n", args.rankId);
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
            uint32_t epRankId = rankId / TP_WORLD_SIZE_A2;
            uint32_t tpRankId = rankId % TP_WORLD_SIZE_A2;

            args[rankId].rankId = rankId;
            args[rankId].epRankId = epRankId;
            args[rankId].tpRankId = tpRankId;
            args[rankId].hcclEpComm = commsEp[epRankId];
            args[rankId].dispatchV2Stream = dispatchV2Stream[rankId];
            args[rankId].combineV2Stream = combineV2Stream[rankId];
            args[rankId].context = context[rankId];
            threads[rankId].reset(new(std::nothrow) std::thread(&launchOneThreadDispatchV2AndCombineV2_A2, std::ref(args[rankId])));
        }

        for(uint32_t rankId = 0; rankId < DEV_NUM_A2; rankId++) {
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

        int32_t devicesEp[TP_WORLD_SIZE][EP_WORLD_SIZE];
        for (int32_t tpId = 0; tpId < TP_WORLD_SIZE; tpId++) {
            for (int32_t epId = 0; epId < EP_WORLD_SIZE; epId++) {
                devicesEp[tpId][epId] = epId * TP_WORLD_SIZE + tpId;
            }
        }
        // Initialize the EP communication domain: ep = 8 {0,2,4,6,8,10,12,14} {1,3,5,7,9,11,13,15}.
        HcclComm commsEp[TP_WORLD_SIZE][EP_WORLD_SIZE];
        for (int32_t tpId = 0; tpId < TP_WORLD_SIZE; tpId++) {
            ret = HcclCommInitAll(EP_WORLD_SIZE, devicesEp[tpId], commsEp[tpId]);
            CHECK_RET(
                ret == ACL_SUCCESS,
                LOG_PRINT("[ERROR] HcclCommInitAll ep world %d failed. ret = %d\n", tpId, ret); return ret
            );
        }

        int32_t devicesTp[EP_WORLD_SIZE][TP_WORLD_SIZE];
        for (int32_t epId = 0; epId < EP_WORLD_SIZE; epId++) {
            for (int32_t tpId = 0; tpId < TP_WORLD_SIZE; tpId++) {
                devicesTp[epId][tpId] = epId * TP_WORLD_SIZE + tpId;
            }
        }
        // Initialize the TP communication domain: tp = 2 {0,1} {2,3} {4,5} {6,7} {8,9} {10,11} {12,13} {14,15}.
        HcclComm commsTp[EP_WORLD_SIZE][TP_WORLD_SIZE];
        for (int32_t epId = 0; epId < EP_WORLD_SIZE; epId++) {
            ret = HcclCommInitAll(TP_WORLD_SIZE, devicesTp[epId], commsTp[epId]);
            CHECK_RET(
                ret == ACL_SUCCESS,
                LOG_PRINT("[ERROR] HcclCommInitAll tp world %d failed. ret = %d\n", epId, ret); return ret
            );
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
            args[rankId].dispatchV2Stream = dispatchV2Stream[rankId];
            args[rankId].combineV2Stream = combineV2Stream[rankId];
            args[rankId].context = context[rankId];
            threads[rankId].reset(new(std::nothrow) std::thread(&launchOneThreadDispatchV2AndCombineV2_A3A5, std::ref(args[rankId])));
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
        const char* env_var_name = "RANK_TABLE_FILE and FIRST_RANK_ID";
        if (IS_TEST_A2) {
            LOG_PRINT("[INFO] %s are not identified and example on <Atlas A2> will be executed!\n", env_var_name);
            int ret = run_example_on_A2();
            return 0;
        }
        if (!env_dev_num) {
            LOG_PRINT("[WARNING] Please check whether environment variable ENV_DEV_NUM is set correctly.\n");
            return 0;
        }
        int actual_env_dev_num = std::stoi(std::string(env_dev_num));
        if (actual_env_dev_num < DEV_NUM) {
            LOG_PRINT("[INFO] ENV_DEV_NUM = %d is less than %d, currently not supported\n", actual_env_dev_num, DEV_NUM);
            return 0;
        }
        if (!rank_table_file && !first_rank_id) {
            LOG_PRINT("[INFO] %s are not identified and example on <Atlas A3> will be executed!\n", env_var_name);
            int ret = run_example_on_A3A5();
        }
        else if (rank_table_file && !first_rank_id) {
            LOG_PRINT("[INFO] %s are not identified and example on <Atlas A5> will be executed!\n", env_var_name);
            int ret = run_example_on_A3A5();
        }
        else {
            LOG_PRINT("[WARNING] Please check whether %s are set correctly.\n", env_var_name);
        }

        return 0;
    }
    ```
