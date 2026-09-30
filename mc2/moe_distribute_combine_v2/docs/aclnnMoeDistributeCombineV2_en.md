# aclnnMoeDistributeCombineV2

[📄 View Source Code](https://gitcode.com/cann/ops-transformer/tree/master/mc2/moe_distribute_combine_v2)

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 training products / Atlas A3 inference products</term>    |    √     |
| <term>Atlas A2 training products / Atlas A2 inference products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference products</term>                            |    ×     |
| <term>Atlas training products</term>                             |    ×     |

## Function

- Description: When there is TP domain communication, ReduceScatterV communication is performed first, followed by AllToAllV communication, and finally the received data is combined (multiplied by the weight and then summed). When there is no TP domain communication, AllToAllV communication is performed, and then the received data is combined (multiplied by the weight and then summed).

    Compared with the `aclnnMoeDistributeCombine` API, this API has the following changes:
    - Inputs more detailed token information to assist `aclnnMoeDistributeCombineV2` in performing efficient all-rank synchronization. Therefore, the `expandIdx` input parameter (`shape (Bs * K,`)) in the original API is replaced by the `assistInfoForCombine` parameter (shape (A * 128,)).
    - Adds the `sharedExpertXOptional` input parameter. When `sharedExpertNum` is set to 0, users can provide tokens computed by shared experts.
    - Adds the `commAlg` parameter to replace the `HCCL_INTRA_PCIE_ENABLE` and `HCCL_INTRA_ROCE_ENABLE` environment variables.

    For details, see the following parameter description.

- Formula:

$$
rsOut = ReduceScatterV(expandX)\\
ataOut = AllToAllV(rsOut)\\
xOut = Sum(expertScales * ataOut + expertScales * sharedExpertX)
$$

>Note that this API must be used together with `aclnnMoeDistributeDispatchV2`, and it performs a reverse data flow along the same path used by the `MoeDistributeDispatchV2` operator for data dispatch.

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnMoeDistributeCombineV2GetWorkspaceSize` is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, `aclnnMoeDistributeCombineV2` is called to perform computation.

```cpp
aclnnStatus aclnnMoeDistributeCombineV2GetWorkspaceSize(
    const aclTensor* expandX,
    const aclTensor* expertIds,
    const aclTensor* assistInfoForCombine,
    const aclTensor* epSendCounts,
    const aclTensor* expertScales,
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
    int64_t          globalBs,
    int64_t          outDtype,
    int64_t          commQuantMode,
    int64_t          groupListType,
    const char*      commAlg,
    aclTensor*       xOut,
    uint64_t*        workspaceSize,
    aclOpExecutor**  executor)
```

```cpp
aclnnStatus aclnnMoeDistributeCombineV2(
    void            *workspace,
    uint64_t        workspaceSize,
    aclOpExecutor   *executor,
    aclrtStream     stream)
```

## aclnnMoeDistributeCombineV2GetWorkspaceSize

- **Parameters**

    <table style="undefined;table-layout: fixed; width: 1567px"> <colgroup>
    <col style="width: 120px">
    <col style="width: 120px">
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
    <th>Precaution</th>
    <th>Data Type</th>
    <th>Data Format</th>
    <th>Dimension (Shape)</th>
    <th>Non-contiguous Tensor</th>
    </tr>
    </thead>
    <tbody>
    <tr>
    <td>expandX</td>
    <td>Input</td>
    <td>Token features expanded based on expertIds.</td>
    <td>The value must be a 2D tensor.</td>
    <td>FLOAT16, BFLOAT16</td>
    <td>ND</td>
    <td>(max(tpWorldSize, 1) * A , H)</td>
    <td>√</td>
    </tr>
    <tr>
    <td>expertIds</td>
    <td>Input</td>
    <td>Top K expert indexes of each token.</td>
    <td>The value must be a 2D tensor.</td>
    <td>INT32</td>
    <td>ND</td>
    <td>(Bs, K)</td>
    <td>√</td>
    </tr>
    <tr>
    <td>assistInfoForCombine</td>
    <td>Input</td>
    <td>Corresponds to the output of <code>assistInfoForCombineOut</code> in <code>aclnnMoeDistributeDispatchV2</code>.</td>
    <td>The value must be a 1D tensor.</td>
    <td>INT32</td>
    <td>ND</td>
    <td>(A * 128, )</td>
    <td>√</td>
    </tr>
    <tr>
    <td>epSendCounts</td>
    <td>Input</td>
    <td>Corresponds to the output of <code>epRecvCounts</code> in <code>aclnnMoeDistributeDispatchV2</code>.</td>
    <td>The value must be a 1D tensor.</td>
    <td>INT32</td>
    <td>ND</td>
    <td>-</td>
    <td>√</td>
    </tr>
    <tr>
    <td>expertScales</td>
    <td>Input</td>
    <td>Top K expert weights of each token.</td>
    <td>The value must be a 2D tensor.</td>
    <td>FLOAT32</td>
    <td>ND</td>
    <td>(Bs, K)</td>
    <td>√</td>
    </tr>
    <tr>
    <td>tpSendCountsOptional</td>
    <td>Input</td>
    <td>Corresponds to the output of <code>tpRecvCounts</code> in <code>aclnnMoeDistributeDispatchV2</code>.</td>
    <td>If there is TP domain communication, pass the parameter; otherwise, pass a null pointer.</td>
    <td>INT32</td>
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
    <td>activationScaleOptional</td>
    <td>Input</td>
    <td>Reserved parameter.</td>
    <td>It is not supported in the current version. Pass a null pointer.</td>
    <td>-</td>
    <td>ND</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>weightScaleOptional</td>
    <td>Input</td>
    <td>Reserved parameter.</td>
    <td>It is not supported in the current version. Pass a null pointer.</td>
    <td>-</td>
    <td>ND</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>groupListOptional</td>
    <td>Input</td>
    <td>Reserved parameter.</td>
    <td>It is not supported in the current version. Pass a null pointer.</td>
    <td>-</td>
    <td>ND</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>expandScalesOptional</td>
    <td>Input</td>
    <td>Corresponds to the output of <code>expandScales</code> in <code>aclnnMoeDistributeDispatchV2</code>.</td>
    <td>-</td>
    <td>FLOAT32</td>
    <td>ND</td>
    <td>-</td>
    <td>√</td>
    </tr>
    <tr>
    <td>sharedExpertXOptional</td>
    <td>Input</td>
    <td>Tokens computed by shared experts.</td>
    <td>It must have the same data type as <code>expandX</code>.</td>
    <td>FLOAT16, BFLOAT16</td>
    <td>ND</td>
    <td>-</td>
    <td>√</td>
    </tr>
    <tr>
    <td>groupEp</td>
    <td>Input</td>
    <td>Name of the EP communication domain (expert parallel communication domain).</td>
    <td>Its string length range is [1, 128). It must have a different value from <code>groupTp</code>.</td>
    <td>STRING</td>
    <td>ND</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>epWorldSize</td>
    <td>Input</td>
    <td>Size of the EP communication domain.</td>
    <td>-</td>
    <td>INT64</td>
    <td>ND</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>epRankId</td>
    <td>Input</td>
    <td>ID of the current rank in the EP communication domain.</td>
    <td>The value range is [0, epWorldSize). The epRankId of each rank in the same EP communication domain must be unique.</td>
    <td>INT64</td>
    <td>ND</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>moeExpertNum</td>
    <td>Input</td>
    <td>Number of MoE experts.</td>
    <td>It must satisfy moeExpertNum % (epWorldSize - sharedExpertRankNum) = 0.</td>
    <td>INT64</td>
    <td>ND</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>groupTp</td>
    <td>Input</td>
    <td>Name of the TP communication domain (data parallel communication domain).</td>
    <td>It must have a different value from <code>groupEp</code>.</td>
    <td>STRING</td>
    <td>ND</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>tpWorldSize</td>
    <td>Input</td>
    <td>Size of the TP communication domain.</td>
    <td>-</td>
    <td>INT64</td>
    <td>ND</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>tpRankId</td>
    <td>Input</td>
    <td>ID of the current rank in the TP communication domain.</td>
    <td>The value of <code>tpRankId</code> must be unique for each rank in the same TP communication domain.</td>
    <td>INT64</td>
    <td>ND</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>expertShardType</td>
    <td>Input</td>
    <td>Distribution type of shared expert ranks.</td>
    <td>-</td>
    <td>INT64</td>
    <td>ND</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>sharedExpertNum</td>
    <td>Input</td>
    <td>Number of shared experts. (A shared expert can be replicated and deployed on multiple ranks.)</td>
    <td>-</td>
    <td>INT64</td>
    <td>ND</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>sharedExpertRankNum</td>
    <td>Input</td>
    <td>Number of shared expert ranks.</td>
    <td>-</td>
    <td>INT64</td>
    <td>ND</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>globalBs</td>
    <td>Input</td>
    <td>Global batch size in the EP domain.</td>
    <td><ul><li>If all ranks have the same Bs, the value of <code>globalBs</code> is Bs * epWorldSize is 0. </li><li>If each rank has a different Bs, the value of <code>globalBs</code> is maxBs * epWorldSize (maxBs indicates the maximum Bs of a single rank).</li></ul></td>
    <td>INT64</td>
    <td>ND</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>outDtype</td>
    <td>Input</td>
    <td>Data type of the output x. It is a reserved parameter.</td>
    <td>It is not supported in the current version. Pass 0.</td>
    <td>INT64</td>
    <td>ND</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>commQuantMode</td>
    <td>Input</td>
    <td>Communication quantization type.</td>
    <td>The value is 0 or 2. The value 0 indicates no quantization, whereas the value 2 indicates int8 quantization.</td>
    <td>INT64</td>
    <td>ND</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>groupListType</td>
    <td>Input</td>
    <td>Group list format. It is a reserved parameter.</td>
    <td>It is not supported in the current version. Pass 0.</td>
    <td>INT64</td>
    <td>ND</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>commAlg</td>
    <td>Input</td>
    <td>Communication affinity memory layout algorithm, which is of the string data type.</td>
    <td>-</td>
    <td>STRING</td>
    <td>ND</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>xOut</td>
    <td>Output</td>
    <td>Processed tokens.</td>
    <td>The value must be a 2D tensor.</td>
    <td>FLOAT16, BFLOAT16</td>
    <td>ND</td>
    <td>(Bs, H)</td>
    <td>-</td>
    </tr>
    <tr>
    <td>workspaceSize</td>
    <td>Output</td>
    <td>Size of the workspace required to be allocated on the device.</td>
    <td>-</td>
    <td>UINT64</td>
    <td>ND</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>executor</td>
    <td>Output</td>
    <td>Operator executor, containing the operator computation process.</td>
    <td>-</td>
    <td>aclOpExecutor*</td>
    <td>ND</td>
    <td>-</td>
    <td>-</td>
    </tr>
    </tbody>
    </table>

    - <term>Atlas A2 training products / Atlas A2 inference products</term>:
    
        - Shared experts are not supported.
        - The shape of `epSendCounts` is (`moeExpertNum + 2 * globalBs * K * serverNum,`), where K indicates the number of top K experts, moeExpertNum indicates the number of tokens received from each rank in the EP communication domain, and `2 * globalBs * K * serverNum` indicates the number of tokens and communication area offset that can be combined for the reduce operation before communication across storage servers or within a storage server. If `globalBs` is 0, the shape is calculated as follows: Bs * epWorldSize.
        - Currently, TP domain communication is not supported.
        - `xActiveMaskOptional` depends on the `commAlg` value: For `"fullmesh"`, it requires a 1D tensor with shape (Bs, ), where `true` must precede `false` (for example, {true, false, true} is invalid); for `"hierarchy"`, it is currently not supported and a null pointer should be passed.
        - `exapndScalesOptional` must be a 1D tensor with shape (A, ).
        - `sharedExpertXOptional` is a reserved parameter, which is not supported in the current version. Pass a null pointer.
        - The value of `epWorldSize` depends on the `commAlg` value: For `"fullmesh"`, it supports 16, 32, 64, 128, and 256; for `"hierarchy"`, it supports 16, 32, and 64.
        - The value of `moeExpertNum` must be in the range (0, 512] and satisfy moeExpertNum / (epWorldSize - sharedExpertRankNum) <= 24.
        - `groupTp` is not supported in the current version. Pass an empty string.
        - `tpWorldSize` is not supported in the current version. Pass 0.
        - `tpRankId` is not supported in the current version. Pass 0.
        - `expertShardType` is not supported in the current version. Pass 0.
        - `sharedExpertNum` is not supported in the current version. Pass 0.
        - `sharedExpertRankNum` is not supported in the current version. Pass 0.
        - The value of `commQuantMode` is 2. It is supported only when `commAlg` is `hierarchy` or HCCL_INTRA_PCIE_ENABLE=1, HCCL_INTRA_ROCE_ENABLE=0, and the driver version is 25.0.RC1.1 or later.
        - The value of `commAlg` can be `nullptr`, `""`, `"fullmesh"`, or `"hierarchy"`. It is recommended to use `"hierarchy"` with driver version 25.0.RC1.1 or later. When set to `nullptr` or `""`, the communication algorithm is selected based on HCCL environment variables (not recommended). `"fullmesh"` indicates that tokens are directly transmitted through RDMA. `"hierarchy"` indicates a two-stage communication process: intra-server communication followed by inter-server communication, which reduces cross-server data transmission.

    - <term>Atlas A3 training products / Atlas A3 inference products</term>:

        - The shape of `epSendCounts` is `(epWorldSize * max(tpWorldSize, 1) * localExpertNum,)`.
        - When there is TP domain communication, `tpSendCountsOptional` is a 1D tensor with shape (tpWorldSize, ).
        - `xActiveMaskOptional` must be a 1D tensor with shape (BS, ) or a 2D tensor with shape (BS, K). If it is a 1D tensor, `true` must be placed before `false`. If it is a 2D tensor and the K values corresponding to tokens are all `false`, the tokens do not participate in communication.
        - `exapndScalesOptional` is a reserved parameter, which is not supported in the current version. Pass a null pointer.
        - `sharedExpertXOptional` must be a 2D tensor with shape (Bs, H) or a 3D tensor (the product of the first two dimensions equals Bs and the third dimension equals H). This parameter is optional. When it is provided, `sharedExpertRankNum` must be set to 0.
        - The value of `epWorldSize` must be in the range [2, 768].
        - The value of `moeExpertNum` must be in the range (0, 1024].
        - The string length of `groupTp` must be in the range [1, 128). It cannot have the same value with `groupEp`.
        - The value of `tpWorldSize` must be in the range [0, 2]. 0 and 1 indicate no TP domain communication. 2 is required when TP domain communication is used.
        - The value of `tpRankId` must be in the range [0, 1]. `tpRankId` of each rank in the same TP domain must be unique. If TP domain communication is not used, pass 0.
        - The value of `expertShardType` must be 0, indicating that shared expert ranks are placed in front of MoE expert ranks.
        - The value of `sharedExpertNum` must be in the range [0, 4].
        - The value of `sharedExpertRankNum` must be in the range [0, epWorldSize). If the value is 0, `sharedExpertNum` is 0 or 1. If the value is not 0, `sharedExpertRankNum % sharedExpertNum` is 0.
        - The value of `commQuantMode` is 2, and can be enabled only when `tpWorldSize` is less than 2.
        - `commAlg` is not supported in the current version. Pass a null pointer.

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

## aclnnMoeDistributeCombineV2

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
    <td>Size of the workspace to be allocated on the device, which is obtained by calling <code>aclnnMoeDistributeCombineV2GetWorkspaceSize</code>.</td>
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
     - `aclnnMoeDistributeCombineV2` defaults to a deterministic implementation.

2. `aclnnMoeDistributeDispatchV2` and `aclnnMoeDistributeCombineV2` must be used together. For details, see [Example](#example).

3. The element values in the `assistInfoForCombineOut`, `epRecvCountsOut`, `tpRecvCountsOut`, and `expandScalesOut` tensor output of `aclnnMoeDistributeDispatchV2` may vary depending on the product model, communication algorithm, or version. Pass the tensors directly to the corresponding parameters of `aclnnMoeDistributeCombineV2`. Other service logics of the model should not depend on the element values.

4. The values of `groupEp`, `epWorldSize`, `moeExpertNum`, `groupTp`, `tpWorldSize`, `expertShardType`, `sharedExpertNum`, `sharedExpertRankNum`, `globalBs`, `commAlg`, and `HCCL_BUFFSIZE` used during API calling must be the same for all ranks, at all network layers, and the same as those of `aclnnMoeDistributeDispatchV2`.

5. The shape format is described as follows:
    - **A**: Maximum number of tokens to be distributed by the current rank. The value range is as follows:
      - For shared experts, A = `Bs * epWorldSize * sharedExpertNum / sharedExpertRankNum`.
      - For MoE experts, when `globalBs` is 0, A >= `Bs * epWorldSize * min(localExpertNum, K)`. When `globalBs` is not 0, A >= globalBs * min(localExpertNum, K).
    - **H**: Hidden layer size.
      - <term>Atlas A2 training products / Atlas A2 inference products</term>: The value depends on `commAlg`. `"fullmesh"` must be in the range (0, 7168] and be an integer multiple of 32. If `"hierarchy"` is used and the driver version is 25.0.RC1.1 or later, the value must be in the range (0, 10*1024] and be an integer multiple of 32.
      - <term>Atlas A3 training products / Atlas A3 inference products</term>: The value range is [1024, 8192].
    - **Bs**: Batch sequence size, that is, the number of tokens output by the current rank.
      - <term>Atlas A2 training products / Atlas A2 inference products</term>: The value range is (0 < Bs ≤ 256).
      - <term>Atlas A3 training products / Atlas A3 inference products</term>: The value range is (0 < Bs ≤ 512).
    - **K**: Number of top K experts, which must be in the ranges 0 < K ≤ 16 and 0 < K ≤ moeExpertNum.
    - **serverNum**: Number of server nodes. The value can only be 2, 4, or 8.
    - **localExpertNum**: Number of experts on the current rank.
      - For shared expert ranks, localExpertNum = 1.
      - For MoE expert ranks, localExpertNum = moeExpertNum/(epWorldSize - sharedExpertRankNum). If localExpertNum > 1, TP domain communication is not supported.

6. **HCCL_BUFFSIZE**:

   Before calling this API, check whether the value of the `HCCL_BUFFSIZE` environment variable is proper. This environment variable indicates the buffer size occupied by a single communication domain, in MB. If this environment variable is not set, the default value 200 MB is used.
   - <term>Atlas A2 training products / Atlas A2 inference products</term>:
     - If `commAlg` is set to `""` or `nullptr`, select `"fullmesh"` or `"hierarchy"` formula based on the HCCL environment variable.
     - If `commAlg` is set to `"fullmesh"`, the value requirement is (`≥ 2 * (Bs * epWorldSize * min(localExpertNum, K) * H * sizeof(uint16) + 2MB)`).
     - If `commAlg` is set to `"hierarchy"`, the value requirement is (`≥ moeExpertNum * Bs * (H * sizeof(dtypeX) + 4 * ((K + 7) / 8 * 8) * sizeof(uint32)`) + 4MB + 100MB), where (moeExpertNum / (epWorldSize - sharedExpertRankNum) ≤ 24) is not required.
   - <term>Atlas A3 training products / Atlas A3 inference products</term>:
     - Within an EP communication domain: The value must be greater than or equal to 2 and satisfy (`≥ 2 * (localExpertNum * maxBs * epWorldSize * Align512(Align32(2 * H) + 44) + (K + sharedExpertNum) * maxBs * Align512(2 * H))`) (`localExpertNum` indicates the number of experts assigned to the current rank when using MoE; `Align512(x) = ((x + 512 - 1) / 512) * 512`; `Align32(x) = ((x + 32 - 1) / 32) * 32`).
     - Within a TP communication domain: The value must satisfy >=`A * (H * 2 + 128) * 2`.

7. **HCCL_INTRA_PCIE_ENABLE and HCCL_INTRA_ROCE_ENABLE:**
   <term>Atlas A2 training products / Atlas A2 inference products</term>: This environment variable is not recommended. You are advised to set `commAlg` to `"hierarchy"`.

8. In the formulas in this document, `/` denotes integer division.

9. Constraints on the use of communication domains:
   - `aclnnMoeDistributeCombineV2` and `aclnnMoeDistributeDispatchV2` in a model support only the same EP communication domain, and no other operators are allowed in the communication domain.
   - `aclnnMoeDistributeCombineV2` and `aclnnMoeDistributeDispatchV2` in a model support only the same TP communication domain or both do not support a TP communication domain. If a TP communication domain is supported, no other operators are allowed in the communication domain.
   - <term>Atlas A3 training products / Atlas A3 inference products</term>: Nodes in a communication domain must be in the same SuperPoD. Cross-SuperPoD nodes are not supported.

10. Networking constraints:

   - <term>Atlas A2 training products / Atlas A2 inference products</term>: In multi-server scenarios, only switch-based networking is supported, and direct point-to-point networking between two servers is not supported.

## Example

- <term>Atlas A2 training products / Atlas A2 inference products</term>:
    
    - Preparing files:
        
        1. Create **rank_table_m2.json** and modify it.
        
        2. Copy the project to the two servers and configure the **rank_table_m2.json** file based on the device IP address. Ensure that the **rank_table_m2.json** files on the two servers are the same.
        
        3. Install the CANN package and compile and run it based on [Operator Invocation](../../../docs/zh/invocation/quick_op_invocation.md).

    - About rankTable:
    
        1. You can configure the NPU resource information involved in collective communication through the ranktable file. For details, see "Communication Function Development > Cluster Information Configuration > Configuring Resource Information Through the Ranktable File" in [HCCL](https://www.hiascend.com/document/detail/en/canncommercial/800/hcclug/hcclug/hcclug_000014.html).

        2. Run the `cat /etc/hccn.conf` or `for i in seq 0 7; do echo "===================> dev$i, NPU$((i+1))"; hccn_tool -i $i -ip -g; done` command to query the device IP address. Then, set the JSON file following instructions in the collective communication guide.

        > Note: In 2-server 16-rank scenarios, the device_ids of both servers range from 0 to 7. The rank_id of one server ranges from 0 to 7, and that of the other server ranges from 8 to 15. In single-server 16-rank scenarios, both the device_ids and rank_ids range from 0 to 15.

    - Environment variable settings:

        ```bash
        # Before running the code, set the three environment variables.
        ## FIRST_RANK_ID description: For example, if there are 2 servers with 16 ranks, set FIRST_RANK_ID to 0 for one server and 8 for the other server.
        ## For example, export FIRST_RANK_ID=0.
        export RANK_TABLE_FILE=/home/path/to/rank_table_m2.json
        export FIRST_RANK_ID=<Start rank ID of the server>
        ## EP_WORLD_SIZE description: Set this variable based on the number of ranks on the current server. For example, if there are 2 servers with 16 ranks, set this variable to 16 for both servers.
        export EP_WORLD_SIZE=16
        ```
    
    - Set the number of servers:
        In 2-server 16-rank scenarios, set MACHINE_NUM to 2.

        ```Cpp
        const uint32_t MACHINE_NUM = 2;

        ```
        
        You do not need to set this variable in single-server 16-rank scenarios.

- <term>Atlas A3 training products / Atlas A3 inference products</term>:
    
    You do not need to configure the **ranktable** file or the environment variables **RANK_TABLE_FILE** and **FIRST_RANK_ID**.
       
The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

- <term>Atlas A2 training products / Atlas A2 inference products</term> and <term>Atlas A3 training products / Atlas A3 inference products</term>:

    ```Cpp
    #include <thread>
    #include <iostream>
    #include <string>
    #include <vector>
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
    const char* ep_world_size_env = std::getenv("EP_WORLD_SIZE");

    const uint32_t EP_WORLD_SIZE = (!rank_table_file && !first_rank_id) ? 2 : 16;
    const uint32_t TP_WORLD_SIZE = (!rank_table_file && !first_rank_id) ? 1 : 0;
    const uint32_t DEV_NUM = (!rank_table_file && !first_rank_id) ? EP_WORLD_SIZE * TP_WORLD_SIZE : EP_WORLD_SIZE;

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

    int launchOneThreadDispatchV2AndCombineV2(Args &args)
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
            LOG_PRINT("[ERROR] aclnnMoeDistributedispatchV2GetWorkspaceSize failed. ret = %d\n", ret); return ret
        );
        // Allocate device memory based on the workspaceSize computed by the first-phase API of dispatchV2.
        if (dispatchV2WorkspaceSize > 0) {
            ret = aclrtMalloc(&dispatchV2WorkspaceAddr, dispatchV2WorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc failed. ret = %d\n", ret); return ret);
        }
        // Call the second-phase API of dispatchV2.
        ret = aclnnMoeDistributeDispatchV2(dispatchV2WorkspaceAddr, dispatchV2WorkspaceSize, dispatchV2Executor, args.dispatchV2Stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnMoeDistributedispatchV2 failed. ret = %d\n", ret); return ret);
        // (Fixed writing) Wait until the task execution is complete.
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
            LOG_PRINT("[ERROR] aclnnMoeDistributecombineV2GetWorkspaceSize failed. ret = %d\n", ret); return ret
        );
        // Allocate device memory based on the workspaceSize computed by the first-phase API of combineV2.
        if (combineV2WorkspaceSize > 0) {
            ret = aclrtMalloc(&combineV2WorkspaceAddr, combineV2WorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc failed. ret = %d\n", ret); return ret);
        }
        // Call the second-phase API of combineV2.
        ret = aclnnMoeDistributeCombineV2(combineV2WorkspaceAddr, combineV2WorkspaceSize, combineV2Executor, args.combineV2Stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnMoeDistributecombineV2 failed. ret = %d\n", ret); return ret);
        // (Fixed writing) Wait until the task execution is complete.
        ret = aclrtSynchronizeStreamWithTimeout(args.combineV2Stream, 10000);
        CHECK_RET(
            ret == ACL_SUCCESS, 
            LOG_PRINT("[ERROR] aclrtSynchronizeStreamWithTimeout failed. ret = %d\n", ret); return ret
        );

        LOG_PRINT("[INFO] device_%d aclnnMoeDistributedispatchV2 and aclnnMoeDistributecombineV2 execute successfully.\n", args.rankId);

        // Release device resources.
        if (dispatchV2WorkspaceSize > 0) {
            aclrtFree(dispatchV2WorkspaceAddr);
        }
        if (combineV2WorkspaceSize > 0) {
            aclrtFree(combineV2WorkspaceAddr);
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

        HcclCommDestroy(args.hcclEpComm);
        HcclCommDestroy(args.hcclTpComm);
        aclrtDestroyStream(args.dispatchV2Stream);
        aclrtDestroyStream(args.combineV2Stream);
        aclrtDestroyContext(args.context);
        aclrtResetDevice(args.rankId);
        
        return 0;
    }

    int run_example_on_A2(int rankId, const char* RANK_TABLE_FILE, const char* FIRST_RANK_ID)
    {
        Args args;
        aclrtStream dispatchV2Stream;
        aclrtStream combineV2Stream;
        aclrtContext context;

        int ret = aclrtSetDevice(rankId);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetDevice failed. ret = %d", ret));
        ret = aclrtCreateContext(&context, rankId);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateContext failed. ret = %d", ret));
        ret = aclrtCreateStream(&dispatchV2Stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateStream failed. ret = %d", ret));
        ret = aclrtCreateStream(&combineV2Stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateStream failed. ret = %d", ret));

        int first_rank_id = std::stoi(std::string(FIRST_RANK_ID));
        HcclComm hcclComm = nullptr;
        int rank_id = rankId + first_rank_id;
        ret = HcclCommInitClusterInfo(RANK_TABLE_FILE, rank_id, &hcclComm);
        if (ret != HCCL_SUCCESS) {
            std::cout << "[ERROR] HCCL CommInitClusterInfo failed. ret = " << ret << std::endl;
            return ret;
        }
        std::cout << "[INFO] HcclCommInitClusterInfo success, rank_id:" << rank_id << ", rankSize:" << DEV_NUM
                << ", hcclComm:" << hcclComm << std::endl;

        args.rankId = rankId;
        args.epRankId = rankId;
        args.tpRankId = 0;
        args.hcclEpComm = hcclComm;
        args.dispatchV2Stream = dispatchV2Stream;
        args.combineV2Stream = combineV2Stream;
        args.context = context;

        launchOneThreadDispatchV2AndCombineV2(args);
        return 0;
    }

    int run_example_on_A3()
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
            args[rankId].dispatchV2Stream = dispatchV2Stream[rankId];
            args[rankId].combineV2Stream = combineV2Stream[rankId];
            args[rankId].context = context[rankId];
            threads[rankId].reset(new(std::nothrow) std::thread(&launchOneThreadDispatchV2AndCombineV2, std::ref(args[rankId])));
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
        int ep_world_size_cur = std::stoi(std::string(ep_world_size_env));
        if (!rank_table_file && !first_rank_id) {
            LOG_PRINT("[INFO] %s are not identified and example on <Atlas A3> will be executed!\n", env_var_name);
            int ret = run_example_on_A3();   
        }
        else if (rank_table_file && first_rank_id) {
            LOG_PRINT("[INFO] %s are identified and example on <Atlas A2> will be executed!\n", env_var_name);
            if (ep_world_size_cur < 16) {
                LOG_PRINT("[INFO] EP_WORLD_SIZE = %d is less than 16, currently not supported <Atlas A2> \n", ep_world_size_cur);
                return 0; // moe_distribute_combine_v2 A2 supports at least 16 ranks. Therefore, it is not maintained currently.
            } else {
                uint32_t single_machine_dev_num = EP_WORLD_SIZE / MACHINE_NUM;
                std::vector<std::unique_ptr<std::thread>> threads(single_machine_dev_num);
                auto ret = aclInit(nullptr);
                CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclInit failed. ret = %d\n", ret); return ret);
                for (int rankId = 0; rankId < single_machine_dev_num; ++rankId) {
                    threads[rankId] = std::make_unique<std::thread>([rankId,&ret]()
                    {
                        ret = run_example_on_A2(rankId, rank_table_file, first_rank_id);
                        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] run example on A2 failed. ret = %d\n", ret); return ret);
                    });
                }
                for (int rankId = 0; rankId < single_machine_dev_num; ++rankId) {
                    threads[rankId]->join();
                }
                aclFinalize();
                LOG_PRINT("[INFO] aclFinalize success\n");
            }
        }
        else {
            LOG_PRINT("[WARNING] Please check whether %s are set correctly.\n", env_var_name);
        }
        return 0;
    }
    ```
