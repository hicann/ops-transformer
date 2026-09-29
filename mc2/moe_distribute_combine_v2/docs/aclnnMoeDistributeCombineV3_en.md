# aclnnMoeDistributeCombineV3

[📄 View Source Code](https://gitcode.com/cann/ops-transformer/tree/master/mc2/moe_distribute_combine_v2)

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

- Description: When there is TP domain communication, ReduceScatterV communication is performed first, followed by AllToAllV communication, and finally the received data is combined (multiplied by the weight and then summed). When there is no TP domain communication, AllToAllV communication is performed, and then the received data is combined (multiplied by the weight and then summed).
- Formula:
    - When there is no TP domain communication:

    $$
    ataOut = AllToAllV(expandX)\\
    xOut = Sum(expertScales * ataOut + expertScales * sharedExpertX)
    $$

    - When there is TP domain communication:

    $$
    rsOut = ReduceScatterV(expandX)\\
    ataOut = AllToAllV(rsOut)\\
    xOut = Sum(expertScales * ataOut + expertScales * sharedExpertX)
    $$

> Note that this API must be used together with `aclnnMoeDistributeDispatchV3`, and it performs a reverse data flow along the same path used by `aclnnMoeDistributeDispatchV3` for data dispatch.

Compared with the `aclnnMoeDistributeCombineV2` API, this API has the following changes:

- Special expert scenarios are supported:

  - **zeroExpertNum≠0**: Enabled by setting the `zeroExpertNum` parameter to a value greater than 0.

    $$Moe(oriXOptional) = 0$$

  - **copyExpertNum≠0**: Enabled by setting the `copyExpertNum` parameter to a value greater than 0 and setting a valid value for the oriXOptional parameter.

    $$Moe(oriXOptional) = oriXOptional$$

  - **constExpertNum≠0**: Enabled by setting the constExpertNum parameter to a value greater than 0 and setting valid values for the oriXOptional, constExpertAlpha1Optional, constExpertAlpha2Optional, and constExpertVOptional parameters.

    $$Moe(oriXOptional) = constExpertAlpha1Optional * oriXOptional + constExpertAlpha2Optional * constExpertVOptional$$

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnMoeDistributeCombineV3GetWorkspaceSize` is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, `aclnnMoeDistributeCombineV3` is called to perform computation.

```cpp
aclnnStatus aclnnMoeDistributeCombineV3GetWorkspaceSize(
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
    const aclTensor* elasticInfoOptional,
    const aclTensor* oriXOptional,
    const aclTensor* constExpertAlpha1Optional,
    const aclTensor* constExpertAlpha2Optional,
    const aclTensor* constExpertVOptional,
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
    int64_t          zeroExpertNum,
    int64_t          copyExpertNum,
    int64_t          constExpertNum,
    aclTensor*       xOut,
    uint64_t*        workspaceSize,
    aclOpExecutor**  executor)
```

```cpp
aclnnStatus aclnnMoeDistributeCombineV3(
    void*           workspace,
    uint64_t        workspaceSize,
    aclOpExecutor*  executor,
    aclrtStream     stream)
```

## aclnnMoeDistributeCombineV3GetWorkspaceSize

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
    <td>Token features expanded based on expertIds. </td>
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
    <td>(BS, K)</td>
    <td>√</td>
    </tr>
    <tr>
    <td>assistInfoForCombine</td>
    <td>Input</td>
    <td>Corresponds to the <code>assistInfoForCombineOut</code> output of <code>aclnnMoeDistributeDispatchV3</code>.</td>
    <td>The value must be a 1D tensor.</td>
    <td>INT32</td>
    <td>ND</td>
    <td>(A * 128, )</td>
    <td>√</td>
    </tr>
    <tr>
    <td>epSendCounts</td>
    <td>Input</td>
    <td>Corresponds to the <code>epRecvCounts</code> output of <code>aclnnMoeDistributeDispatchV3</code>.</td>
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
    <td><code>(BS, K)</code></td>
    <td>√</td>
    </tr>
    <tr>
    <td>tpSendCountsOptional</td>
    <td>Input</td>
    <td>Corresponds to the <code>tpRecvCounts</code> output of <code>aclnnMoeDistributeDispatchV3</code>.</td>
    <td>The parameter is required when there is TP domain communication. Otherwise, pass a null pointer.</td>
    <td>INT32</td>
    <td>ND</td>
    <td>When there is TP domain communication, the shape is <code>(tpWorldSize, )</code>.</td>
    <td>√</td>
    </tr>
    <tr>
    <td>xActiveMaskOptional</td>
    <td>Input</td>
    <td>Indicates whether the tokens participate in communication.</td>
    <td>The value must be a 1D or 2D tensor. You can pass valid data or a null pointer.-</td>
    <td>BOOL</td>
    <td>ND</td>
    <td>When the input is 1D, the shape is <code>(BS,)</code>. When the input is 2D, the shape is <code>(BS, K)</code></td>
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
    <td>Corresponds to the <code>expandScales</code> output of <code>aclnnMoeDistributeDispatchV3</code>.</td>
    <td>-</td>
    <td>FLOAT32</td>
    <td>ND</td>
    <td><code>(A, )</code></td>
    <td>√</td>
    </tr>
    <tr>
    <td>sharedExpertXOptional</td>
    <td>Input</td>
    <td>Tokens computed by shared experts.</td>
    <td>-</td>
    <td>FLOAT16, BFLOAT16</td>
    <td>ND</td>
    <td><code>(BS, H)</code></td>
    <td>√</td>
    </tr>
    <tr>
    <td>elasticInfoOptional</td>
    <td>Input</td>
    <td>Dynamic scale-in information of the EP communication domain.</td>
    <td>-</td>
    <td>INT32</td>
    <td>ND</td>
    <td>-</td>
    <td>√</td>
    </tr>
    <tr>
    <td>oriXOptional</td>
    <td>Input</td>
    <td>Token data that has not passed through the feedforward neural network (FNN).</td>
    <td>An input is required when copyExpert or constExpert is enabled. You can pass a valid input or an empty pointer. When <code>copyExpertNum</code> or <code>constExpertNum</code> is not 0, a valid input must be passed. When a valid input is passed, it must be a 2D tensor, and the data type must be the same as that of expandX.</td>
    <td>FLOAT16, BFLOAT16</td>
    <td>ND</td>
    <td><code>(BS, H)</code></td>
    <td>√</td>
    </tr>
    <tr>
    <td>constExpertAlpha1Optional</td>
    <td>Input</td>
    <td>Calculation coefficient required for enabling constExpert.</td>
    <td>-</td>
    <td>FLOAT16, BFLOAT16</td>
    <td>ND</td>
    <td>-</td>
    <td>√</td>
    </tr>
    <tr>
    <td>constExpertAlpha2Optional</td>
    <td>Input</td>
    <td>Calculation coefficient required for enabling constExpert.</td>
    <td>-</td>
    <td>FLOAT16, BFLOAT16</td>
    <td>ND</td>
    <td>-</td>
    <td>√</td>
    </tr>
    <tr>
    <td>constExpertVOptional</td>
    <td>Input</td>
    <td>aclTensor on the device. The computation coefficient needs to be input when constExpert is enabled.</td>
    <td>-</td>
    <td>FLOAT16, BFLOAT16</td>
    <td>ND</td>
    <td>-</td>
    <td>√</td>
    </tr>
    <tr>
    <td>groupEp</td>
    <td>Input</td>
    <td>Name of the EP communication domain.</td>
    <td>Its string length range is [1, 128). It must have a different value from <code>groupTp</code>.</td>
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
    <td>-</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>groupTp</td>
    <td>Input</td>
    <td>Name of the TP communication domain (data parallelism).</td>
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
    <td>Number of shared experts.</td>
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
    <td>globalBS</td>
    <td>Input</td>
    <td>Global batch size in the EP domain.</td>
    <td><br>If the number of BSs of each rank is the same, the value is <code>globalBS = BS * epWorldSize</code> or 0.<br>If the number of BSs of each rank is different, the value is <code>globalBS = maxBS * epWorldSize</code>, where maxBS indicates the maximum BS value of a single card.</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>outDtype</td>
    <td>Input</td>
    <td>Reserved parameter, which specifies the output data type.</td>
    <td>It is not supported in the current version. Pass 0.</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>commQuantMode</td>
    <td>Input</td>
    <td>Communication quantization type.</td>
    <td>The value is 0 or 2. The value 0 indicates no quantization during communication, whereas the value 2 indicates int8 quantization during communication.</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>groupListType</td>
    <td>Input</td>
    <td>Reserved parameter, in group list format.</td>
    <td>It is not supported in the current version. Pass 0.</td>
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
    <td>xOut</td>
    <td>Output</td>
    <td>Processed tokens.</td>
    <td>2D tensor. It has the same data type and format as expandX.</td>
    <td>FLOAT16, BFLOAT16</td>
    <td>ND</td>
    <td><code>(BS, H)</code></td>
    <td>-</td>
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

    - The value of `commAlg` can be `nullptr`, `""`, `"fullmesh"`, or `"hierarchy"`. It is recommended to use `"hierarchy"` with driver version 25.0.RC1.1 or later. When set to `nullptr` or `""`, the communication algorithm is selected based on HCCL environment variables (not recommended). `"fullmesh"` indicates that tokens are directly transmitted through RDMA. `"hierarchy"` indicates a two-stage communication process: intra-server communication followed by inter-server communication, which reduces cross-server traffic.
    - Shared experts are not supported.
    - The shape of epSendCounts is (moeExpertNum + 2 *globalBS* K *serverNum,). K indicates the number of top K experts. The first moeExpertNum elements indicate the number of tokens received from each card in the EP communicator. The last 2* globalBS *K* serverNum elements are used to store the number of tokens that can be reduced by the combine operation in advance and the offset of the communication area before inter-server or intra-server communication. If globalBS is 0, the value is calculated based on BS * epWorldSize.
    - Currently, TP domain communication is not supported.
    - The value of xActiveMaskOptional depends on the value of commAlg. For "fullmesh", xActiveMaskOptional must be a 1D tensor with the shape of (BS,). true must be placed before false. For example, {true, false, true} is invalid. For "hierarchy", the current version does not support this parameter. You can pass a null pointer.
    - expandScalesOptional must be a 1D tensor with shape (A,).
    - `sharedExpertXOptional` is a reserved parameter, which is not supported in the current version. Pass a null pointer.
    - The value of epWorldSize depends on the value of commAlg. For "fullmesh", the value can be 2, 3, 4, 5, 6, 7, 8, 16, 32, 64, 128, 192, 256, or 384. For "hierarchy", the value can be 16, 32, or 64.
    - The value of moeExpertNum ranges from (0, 512].
    - `groupTp` is not supported in the current version. Pass an empty string.
    - `tpWorldSize` is not supported in the current version. Pass 0.
    - `tpRankId` is not supported in the current version. Pass 0.
    - `expertShardType` is not supported in the current version. Pass 0.
    - `sharedExpertNum` is not supported in the current version. Pass 0.
    - `sharedExpertRankNum` is not supported in the current version. Pass 0.
    - The value of `commQuantMode` is 0 or 2 (0 indicates no quantization and 2 indicates int8 quantization). The value 2 is supported only when commAlg is `"hierarchy"`, or when HCCL_INTRA_PCIE_ENABLE=1, HCCL_INTRA_ROCE_ENABLE=0, and the driver version is 25.0.RC1.1 or later.
    - `expandScalesOptional` must be a 1D tensor.
    - `elasticInfoOptional` is not supported in the current version. Pass a null pointer.
    - `oriXOptional` is not supported in the current version when commAlg is `"hierarchy"`. Pass a null pointer.
    - `constExpertAlpha1Optional` is a reserved parameter, which is not supported in the current version. Pass a null pointer.
    - `constExpertAlpha2Optional` is a reserved parameter, which is not supported in the current version. Pass a null pointer.
    - `constExpertVOptional` is a reserved parameter, which is not supported in the current version. Pass a null pointer.
    - When commAlg is `"fullmesh"`, the value of `zeroExpertNum` must be in the range [0, MAX_INT32), where MAX_INT32 = 2^31 - 1. Valid zero expert IDs must be in the range [<code>moeExpertNum</code>, <code>moeExpertNum + zeroExpertNum</code>).
    - When commAlg is `"fullmesh"`, the value of `copyExpertNum` must be in the range [0, MAX_INT32), where MAX_INT32 = 2^31 - 1. Valid copy expert IDs must be in the range [<code>moeExpertNum + zeroExpertNum</code>, <code>moeExpertNum + zeroExpertNum + copyExpertNum</code>).
    - `constExpertNum` is not supported in the current version. Pass 0.
    </details>

    <details>
    <summary><term>Atlas A3 training products/Atlas A3 inference products</term></summary>

    - `commAlg` is not supported in the current version. Pass a null pointer.
    - The shape of epSendCounts is (epWorldSize *max(tpWorldSize, 1)* localExpertNum, ).
    - When there is communication in the TP domain, tpSendCountsOptional is a 1D shape tensor, and the shape is (tpWorldSize, ).
    - `xActiveMaskOptional` must be a 1D tensor with shape (BS, ) or a 2D tensor with shape (BS, K). If it is a 1D tensor, `true` must be placed before `false`. If it is a 2D tensor and the K values corresponding to tokens are all `false`, the tokens do not participate in communication.
    - `expandScalesOptional` is a reserved parameter, which is not supported in the current version. Pass a null pointer.
    - sharedExpertXOptional must be a 2D or 3D tensor. (When it is a 2D tensor, the shape is (BS, H). When it is a 3D tensor, the product of the first two dimensions is equal to BS, and the third dimension is equal to H.) This parameter can be passed or not. If this parameter is passed, sharedExpertRankNum must be set to 0.
    - The value of `epWorldSize` must be in the range [2, 768].
    - The value of `moeExpertNum` must be in the range (0, 1024].
    - groupTp must be a string of length [0, 128) and cannot be the same as groupEp. This parameter can be left empty only when there is no TP domain communication.
    - The value of `tpWorldSize` must be in the range [0, 2]. 0 and 1 indicate no TP domain communication. 2 is required when TP domain communication is used.
    - The value of `tpRankId` must be in the range [0, 1]. `tpRankId` of each rank in the same TP domain must be unique. If TP domain communication is not used, pass 0.
    - The value of `expertShardType` must be 0, indicating that shared expert ranks are placed in front of MoE expert ranks.
    - The value of `sharedExpertNum` must be in the range [0, 4].
    - The value of `sharedExpertRankNum` must be in the range [0, epWorldSize). If the value is 0, `sharedExpertNum` is 0 or 1. If the value is not 0, `sharedExpertRankNum % sharedExpertNum` is 0.
    - The value of `commQuantMode` is 0 or 2 (0 indicates no quantization and 2 indicates int8 quantization). The value 2 is supported only when tpWorldSize < 2.
    - `expandScalesOptional` is a reserved parameter, which is not supported in the current version. Pass a null pointer.
    - `elasticInfoOptional` is not supported in the current version. Pass a null pointer.
    - constExpertAlpha1Optional can be set to valid data or a null pointer. When constExpertNum is not 0, valid input must be passed. When valid data is passed, it must be a 2D tensor whose shape is <code>(constExpertNum, H)</code>. The data type must be the same as that of expandX.
    - constExpertAlpha2Optional can be set to valid data or a null pointer. When constExpertNum is not 0, valid input must be passed. When valid data is passed, it must be a 2D tensor whose shape is <code>(constExpertNum, H)</code>. The data type must be the same as that of expandX.
    - constExpertVOptional can be set to valid data or a null pointer. When constExpertNum is not 0, valid input must be passed. When valid data is passed, it must be a 2D tensor whose shape is <code>(constExpertNum, H)</code>. The data type must be the same as that of expandX.
    - The value of `zeroExpertNum` must be in the range [0, MAX_INT32), where MAX_INT32 = 2^31 - 1. Valid zero expert IDs must be in the range [<code>moeExpertNum</code>, <code>moeExpertNum + zeroExpertNum</code>).
    - The value of `copyExpertNum` must be in the range [0, MAX_INT32), where MAX_INT32 = 2^31 - 1. Valid copy expert IDs must be in the range [<code>moeExpertNum + zeroExpertNum</code>, <code>moeExpertNum + zeroExpertNum + copyExpertNum</code>).
    - The value of `constExpertNum` must be in the range [0, MAX_INT32), where MAX_INT32 = 2^31 - 1. Valid constant expert IDs must be in the range [<code>moeExpertNum + zeroExpertNum + copyExpertNum</code>, <code>moeExpertNum + zeroExpertNum + copyExpertNum + constExpertNum</code>).
    </details>
    
    <details>
    <summary>Ascend 950PR/Ascend 950DT</summary>
    - `commAlg` is not supported in the current version. Pass a null pointer.
    - The shape of `epSendCounts` is (epWorldSize * max(tpWorldSize, 1) * localExpertNum, ).
    - The TP communication domain is not supported.
    - `xActiveMaskOptional` must be a 1D tensor with shape (BS, ) or a 2D tensor with shape (BS, K). If it is a 1D tensor, `true` must be placed before `false`. If it is a 2D tensor and the K values corresponding to tokens are all `false`, the tokens do not participate in communication.
    - `expandScalesOptional` is a reserved parameter, which is not supported in the current version. Pass a null pointer.
    - The sharedExpertXOptional parameter is a 2D or 3D tensor. (When it is a 2D tensor, the shape is (BS, H). When it is a 3D tensor, the product of the first two dimensions is equal to BS, and the third dimension is equal to H.) This parameter can be passed or not. If it is passed, sharedExpertRankNum must be set to 0.
    - The value of `epWorldSize` must be in the range [2, 768].
    - The value of `moeExpertNum` must be in the range (0, 1024].
    - `groupTp` is not supported in the current version. Pass an empty string.
    - `tpWorldSize` is not supported in the current version. Pass 0.
    - `tpRankId` is not supported in the current version. Pass 0.
    - The value of `expertShardType` must be 0, indicating that shared expert ranks are placed in front of MoE expert ranks.
    - The value of `sharedExpertNum` must be in the range [0, 4].
    - The value of `sharedExpertRankNum` must be in the range [0, epWorldSize). If the value is 0, `sharedExpertNum` is 0 or 1. If the value is not 0, `sharedExpertRankNum % sharedExpertNum` is 0.
    - The value of `commQuantMode` is 0 or 2 (0 indicates no quantization and 2 indicates int8 quantization). The value 2 is supported only when tpWorldSize < 2.
    - `expandScalesOptional` is a reserved parameter, which is not supported in the current version. Pass a null pointer.
    - The elasticInfoOptional parameter is reserved and is not supported in the current version. You can pass a null pointer.
    - The constExpertAlpha1Optional parameter can be set to a valid data or null pointer. If constExpertNum is not 0, valid input must be passed. If valid data is passed, the data must be a 2D tensor with the shape of <code>(constExpertNum, H)</code>. The data type must be the same as that of expandX.
    - The constExpertAlpha2Optional parameter can be set to a valid data or null pointer. If constExpertNum is not 0, valid input must be passed. If valid data is passed, the data must be a 2D tensor with the shape of <code>(constExpertNum, H)</code>. The data type must be the same as that of expandX.
    - constExpertVOptional can be set to a valid data or null pointer. When constExpertNum is not 0, valid input must be passed. When valid data is passed, it must be a 2D tensor with the shape of <code>(constExpertNum, H)</code>. The data type must be the same as that of expandX.
    - The value of `zeroExpertNum` must be in the range [0, MAX_INT32), where MAX_INT32 = 2^31 - 1. Valid zero expert IDs must be in the range [<code>moeExpertNum</code>, <code>moeExpertNum + zeroExpertNum</code>).
    - The value of `copyExpertNum` must be in the range [0, MAX_INT32), where MAX_INT32 = 2^31 - 1. Valid copy expert IDs must be in the range [<code>moeExpertNum + zeroExpertNum</code>, <code>moeExpertNum + zeroExpertNum + copyExpertNum</code>).
    - The value of `constExpertNum` must be in the range [0, MAX_INT32), where MAX_INT32 = 2^31 - 1. Valid constant expert IDs must be in the range [<code>moeExpertNum + zeroExpertNum + copyExpertNum</code>, <code>moeExpertNum + zeroExpertNum + copyExpertNum + constExpertNum</code>).
    </details>

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

## aclnnMoeDistributeCombineV3

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
        <td>Address of the workspace to be allocated on the device.</td>
    </tr>
    <tr>
        <td>workspaceSize</td>
        <td>Input</td>
        <td>Size of the workspace to be allocated on the device, which is obtained by calling the first-phase API **aclnnMoeDistributeCombineV3GetWorkspaceSize**.</td>
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

- **Deterministic Computation**
  - **aclnnMoeDistributeCombineV3** defaults to a deterministic implementation.

- Driver restrictions:
  - The driver versions of all nodes in the operator communicator must be the same.

- **API constraints:**:
  - `aclnnMoeDistributeDispatchV3` and `aclnnMoeDistributeCombineV3` must be used together. The `assistInfoForCombineOut`, `epRecvCountsOut`, `tpRecvCountsOut`, and `expandScalesOut` outputs of `aclnnMoeDistributeDispatchV3` must be directly passed to the corresponding parameters of `aclnnMoeDistributeCombineV3`. The service logic cannot depend on the specific values of these tensors.

- **Parameter consistency constraints:**
  - The values of the groupEp, epWorldSize, moeExpertNum, groupTp, tpWorldSize, expertShardType, sharedExpertNum, sharedExpertRankNum, globalBS, commAlg, and HCCL_BUFFSIZE parameters of all devices must be the same and consistent with those of the corresponding parameters of aclnnMoeDistributeDispatchV3.

- **Product constraints**:
  - <term>Atlas A3 training products/Atlas A3 inference products</term>: In this scenario, a single rank contains dual dies. Therefore, the "rank" in the parameter description indicates a single die.

- **Shape variable constraints**:

    <table style="undefined;table-layout: fixed; width: 1189px"><colgroup>
    <col style="width: 186px">
    <col style="width: 1003px">
    </colgroup>
    <thead>
    <tr>
        <th>Variable</th>
        <th>Definition and Value Range</th>
    </tr></thead>
    <tbody>
    <tr>
        <td>A</td>
        <td>Maximum number of tokens to be distributed by the card. The value range is as follows:<ul><li>For shared experts, the following condition must be met: A = BS x epWorldSize x sharedExpertNum / sharedExpertRankNum. </li><li>For MoE experts, when globalBS is 0, the following condition must be met: A >= BS * epWorldSize * min(localExpertNum, K). When globalBS is not 0, the following condition must be met: A >= globalBS * min(localExpertNum, K).</li></ul></td>
    </tr>
    <tr>
        <td>H</td>
        <td>Hidden size of the hidden layer. The value range is as follows:<ul><li>Atlas A2 training products/Atlas A2 inference products: The value is determined by the value of commAlg. For "fullmesh", the value is an integer multiple of 32 and is within the range (0, 7168]. For "hierarchy" and driver version ≥ 25.0.RC1.1, the value is an integer multiple of 32 and is within the range (0, 10 x 1024].</li><li>Atlas A3 training products/Atlas A3 inference products: [1024, 8192].</li></ul></td>
    </tr>
    <tr>
        <td>BS</td>
        <td>Number of tokens output by the card. The value range is as follows:<ul><li>Atlas A2 training products/Atlas A2 inference products: The value is determined by the value of commAlg. For "fullmesh", the value is within the range (0, 256]. For "hierarchy" and driver version ≥ 25.0.RC1.1, the value is within the range (0, 512].</li><li>Atlas A3 training products/Atlas A3 inference products: 0 < BS ≤ 512.</li></ul></td>
    </tr>
    <tr>
        <td>K</td>
        <td>Number of top K experts selected. The value range is as follows:<br>0 < K ≤16, and 0 < K ≤ moeExpertNum+zeroExpertNum+copyExpertNum+constExpertNum.</td>
    </tr>
    <tr>
        <td>serverNum</td>
        <td>Number of server nodes:<br>Atlas A2 training products/Atlas A2 inference products: This variable is used only in shapes in this scenario. The value is 2, 4, or 8.</td>
    </tr>
    <tr>
        <td>localExpertNum</td>
        <td>Number of experts on the card: <ul><li>For a shared expert card, localExpertNum = 1. </li><li>For an MoE expert card, localExpertNum = moeExpertNum/(epWorldSize-sharedExpertRankNum). TP communication is not supported when localExpertNum > 1. </li><li>Atlas A3 training products/Atlas A3 inference products: The value must satisfy 0 < localExpertNum * epWorldSize ≤ 2048.</li></ul></td>
    </tr>
    </tbody></table>

- **Environment variables constraints**:
  - **HCCL_BUFFSIZE**: Before calling this API, check whether the value of the HCCL_BUFFSIZE environment variable is proper. This environment variable indicates the buffer size occupied by a single communication domain, in MB. If this environment variable is not set, the default value 200 MB is used.
      - <term>Atlas A2 training products/Atlas A2 inference products</term>:
        - If `commAlg` is set to `""` or `nullptr`, select the `"fullmesh"` or `"hierarchy"` formula based on the **HCCL_INTRA_PCIE_ENABLE** and **HCCL_INTRA_ROCE_ENABLE** environment variables.
        - If `commAlg` is set to `"fullmesh"`, the value must satisfy <code>>= 2 \* (BS \* epWorldSize \* min(localExpertNum, K) \* H \* sizeof(uint16) + 2MB)</code>.
        - If commAlg is set to "hierarchy", the size must be greater than or equal to (≥ (moeExpertNum + epWorldSize / 4) \* Align512(maxBS \* (H \* 2 + 16 \* Align8(K))) \* 1B + 8MB, where Align8(x) = ((x + 8 - 1) / 8) *8 and Align512(x) = ((x + 512 - 1) / 512)* 512.
      - <term>Atlas A3 training products/Atlas A3 inference products</term>:
        - Within an EP communication domain: The value must satisfy <code>>= 2</code> and <code>>= 2 \* (localExpertNum \* maxBS \* epWorldSize \* Align512(Align32(2 \* H) + 44) + (K + sharedExpertNum) \* maxBS \* Align512(2 \* H))</code>. Set <code>localExpertNum</code> to the number of experts assigned to the current rank when using MoE, where <code>Align512(x) = ((x + 512 - 1) / 512) \* 512</code> and <code>Align32(x) = ((x + 32 - 1) / 32) \* 32</code>.
        - Within the TP communication domain: The value must be grater than or equal to `A * (H * 2 + 128) * 2`.

  - **HCCL_INTRA_PCIE_ENABLE/HCCL_INTRA_ROCE_ENABLE**:
    - <term>Atlas A2 training products/Atlas A2 inference products</term>: This environment variable is not recommended. You are advised to set commAlg to `"hierarchy"`.

- **Constraints on the use of communication domains**:
  - `aclnnMoeDistributeCombineV3` and `aclnnMoeDistributeDispatchV3` in a model support only the same EP communication domain, and no other operators are allowed in the communication domain.
  - `aclnnMoeDistributeCombineV3` and `aclnnMoeDistributeDispatchV3` in a model support only the same TP communication domain or both do not support a TP communication domain. If a TP communication domain is supported, no other operators are allowed in the communication domain.
  - <term>Atlas A3 training products/Atlas A3 inference products</term>: Nodes in a communication domain must be in the same SuperPoD. Cross-SuperPoD nodes are not supported.

- **Networking constraints**:
  - <term>Atlas A2 training products/Atlas A2 inference products</term>: In multi-server scenarios, only switch-based networking is supported, and direct point-to-point networking between two servers is not supported.

- **Other constraints**:
  - In the formulas, `/` denotes integer division.
  - <code>moeExpertNum + zeroExpertNum + copyExpertNum + constExpertNum < MAX_INT32</code>

## Example

- <term>Atlas A2 training products/Atlas A2 inference products</term>

    In this example, the A2 operator can run in a single-server environment with 2 to 8 ranks. You can set EP_WORLD_SIZE_A2 to the number of devices and change the value of moeExpertNum in the sample code to ensure that moeExpertNum can be exactly divided by EP_WORLD_SIZE_A2.

    - Operator build: The following is the operator build command. Both the moe_distribute_dispatch_v2 and moe_distribute_combine_v2 operators need to be built and executed in pairs.

        ```bash
        bash build.sh --pkg --soc=ascend910b --ops=moe_distribute_dispatch_v2,moe_distribute_combine_v2
        ```

    - Create the sample code for the <term>Atlas A2 training products/Atlas A2 inference products</term>. After the compilation is complete, create a test file test_aclnn_moe_distribute_combine_v3.cpp using the Atlas A2 sample code in the operator [examples](https://gitcode.com/cann/ops-transformer/tree/9.0.0/mc2/moe_distribute_combine_v2/examples/) directory by referring to the existing [test_aclnn_moe_distribute_combine_v2.cpp](https://gitcode.com/cann/ops-transformer/tree/9.0.0/mc2/moe_distribute_combine_v2/examples/test_aclnn_moe_distribute_combine_v2.cpp) file.

    - Run the operator sample. The following command will execute all sample code files in the operator [examples](https://gitcode.com/cann/ops-transformer/tree/9.0.0/mc2/moe_distribute_combine_v2/examples/) directory.

        ```bash
        bash build.sh --run_example --ops=moe_distribute_combine_v2 eager cust
        ```

    - <term>Atlas A2 training products/Atlas A2 inference products</term> sample code:

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
            } while(0)

        struct Args {
            uint32_t rankId;
            uint32_t epRankId;
            uint32_t tpRankId;
            HcclComm hcclEpComm;
            HcclComm hcclTpComm;
            aclrtStream dispatchV3Stream;
            aclrtStream combineV3Stream;
            aclrtContext context;
        };

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
                strides[i] = shape[i +1] * strides[i + 1];
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
            int64_t constExpertNum = 0; // Only A3
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

            //Input in the zero-expert scenario.
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

            // Define the dimensions of variables in the current scenario.
            std::vector<int64_t> xShape{BS, H};
            std::vector<int64_t> expertIdsShape{BS, K};
            std::vector<int64_t> scalesShape{moeExpertNum + 1, H};
            std::vector<int64_t> expertScalesShape{BS, K};

            std::vector<int64_t> expandXShape{TP_WORLD_SIZE_A2 * A, H};
            std::vector<int64_t> dynamicScalesShape{TP_WORLD_SIZE_A2 * A};
            std::vector<int64_t> assistInfoForCombineShape{A * 128};
            std::vector<int64_t> expertTokenNumsShape{localExpertNum};
            std::vector<int64_t> epRecvCountsShape{TP_WORLD_SIZE_A2 * localExpertNum * EP_WORLD_SIZE_A2}; // (not layered)
            std::vector<int64_t> tpRecvCountsShape{TP_WORLD_SIZE_A2};
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

            /**************************************** Call dispatch. ********************************************/
            // Call the first-phase API.
            ret = aclnnMoeDistributeDispatchV3GetWorkspaceSize(x, expertIds, (quantMode > 0 ? scales : nullptr), xActiveMask,
                    expertScales, elasticInfo, hcomEpName, EP_WORLD_SIZE_A2, args.epRankId, moeExpertNum, "", TP_WORLD_SIZE_A2,
                    args.tpRankId, expertShardType, sharedExpertNum,sharedExpertRankNum, quantMode, globalBS,
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
            /**************************************** Call combine. ********************************************/
            // Call the first-phase API.
            ret = aclnnMoeDistributeCombineV3GetWorkspaceSize(expandX, expertIds,
                                                                assistInfoForCombine, epRecvCounts,
                                                                expertScales, tpRecvCounts,
                                                                xActiveMask, activationScale, weightScale,
                                                                groupList, expandScales, sharedExpertX,
                                                                elasticInfo, oriX, constExpertAlpha1, constExpertAlpha2, constExpertV,
                                                                hcomEpName, EP_WORLD_SIZE_A2, args.epRankId, moeExpertNum,
                                                                "", TP_WORLD_SIZE_A2, args.tpRankId, expertShardType,
                                                                sharedExpertNum, sharedExpertRankNum, globalBS, outDtype,
                                                                commQuantMode, groupList_type, commAlg.c_str(), zeroExpertNum, copyExpertNum, constExpertNum, xOut,
                                                                &combineWorkspaceSize, &combineExecutor);
            CHECK_RET(ret == ACL_SUCCESS,
                LOG_PRINT("[ERROR] aclnnMoeDistributeCombineV3GetWorkspaceSize failed. ret = %d \n", ret); return ret);
            // Allocate device memory based on the workspaceSize computed by the first-phase API.
            if (combineWorkspaceSize > 0) {
                ret = aclrtMalloc(&combineWorkspaceAddr, combineWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
                CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc workspace failed. ret = %d \n", ret); return ret);
            }

            // Call the second-phase API.
            ret = aclnnMoeDistributeCombineV3(combineWorkspaceAddr, combineWorkspaceSize, combineExecutor, args.combineV3Stream);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnMoeDistributeCombineV3 failed. ret = %d \n", ret);
                return ret);
            // (Fixed writing) Wait until the task execution is complete.
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
                uint32_t epRankId = rankId / TP_WORLD_SIZE_A2;
                uint32_t tpRankId = rankId % TP_WORLD_SIZE_A2;

                args[rankId].rankId = rankId;
                args[rankId].epRankId = epRankId;
                args[rankId].tpRankId = tpRankId;
                args[rankId].hcclEpComm = commsEp[epRankId];
                args[rankId].dispatchV3Stream = dispatchV3Stream[rankId];
                args[rankId].combineV3Stream = combineV3Stream[rankId];
                args[rankId].context = context[rankId];
                threads[rankId].reset(new(std::nothrow) std::thread(&launchOneThreadDispatchV3AndCombineV3_A2, std::ref(args[rankId])));
            }

            for(uint32_t rankId = 0; rankId < DEV_NUM_A2; rankId++) {
                threads[rankId]->join();
            }

            aclFinalize();
            LOG_PRINT("[INFO] aclFinalize success\n");
            return 0;
        }
        ```

- Ascend 950PR/Ascend 950DT: Refer to the preparation section and sample code in the [aclnnMoeDistributeCombineV2](./aclnnMoeDistributeCombineV2.md) API. Set the involved variables again according to the preceding restrictions. For the scenario parameters added in the V4 API compared with the V3 API, pass values according to the preceding parameter description.

- <term>Atlas A3 training products/Atlas A3 inference products</term>:
       
    For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

- The sample code is as follows:

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

        // Define the dimensions of variables in the current scenario.
        std::vector<int64_t> xShape{BS, H};
        std::vector<int64_t> expertIdsShape{BS, K};
        std::vector<int64_t> scalesShape{moeExpertNum + 1, H};
        std::vector<int64_t> expertScalesShape{BS, K};

        std::vector<int64_t> expandXShape{TP_WORLD_SIZE * A, H};
        std::vector<int64_t> dynamicScalesShape{TP_WORLD_SIZE * A};
        std::vector<int64_t> expandIdxShape{A * 128};
        std::vector<int64_t> expertTokenNumsShape{localExpertNum};
        std::vector<int64_t> epRecvCountsShape{TP_WORLD_SIZE * localExpertNum * EP_WORLD_SIZE};
        std::vector<int64_t> tpRecvCountsShape{TP_WORLD_SIZE};
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
        ret = aclnnMoeDistributeDispatchV3GetWorkspaceSize(x, expertIds, (quantMode > 0 ? scales : nullptr), nullptr,
                expertScales, nullptr, hcomEpName, EP_WORLD_SIZE, args.epRankId, moeExpertNum, hcomTpName, TP_WORLD_SIZE,
                args.tpRankId, expertShardType, sharedExpertNum,sharedExpertRankNum, quantMode, globalBS,
                expertTokenNumsType, nullptr, zeroExpertNum, copyExpertNum, constExpertNum, expandX, dynamicScales, expandIdx, expertTokenNums, epRecvCounts,
                tpRecvCounts, expandScales, &dispatchWorkspaceSize, &dispatchExecutor);

        CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("[ERROR] warm up aclnnMoeDistributeDispatchV3GetWorkspaceSize failed. ret = %d \n", ret); return ret);

        if (dispatchWorkspaceSize > 0) {
            ret = aclrtMalloc(&dispatchWorkspaceAddr, dispatchWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] warm up aclrtMalloc workspace failed. ret = %d \n", ret); return ret);
        }
        // Call the second-phase API.
        ret = aclnnMoeDistributeDispatchV3(dispatchWorkspaceAddr, dispatchWorkspaceSize,
                                            dispatchExecutor, args.dispatchStream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] warm up aclnnMoeDistributeDispatchV3 failed. ret = %d \n", ret);  \
                return ret);
        ret = aclrtSynchronizeStreamWithTimeout(args.dispatchStream, 10000);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] warm up aclrtSynchronizeStreamWithTimeout failed. ret = %d \n", ret);  \
            return ret);

        /**************************************** Call dispatch. ********************************************/
        if (availableRank.find(args.rankId) != availableRank.end()) {
            // Call the first-phase API.
        ret = aclnnMoeDistributeDispatchV3GetWorkspaceSize(x, expertIds, (quantMode > 0 ? scales : nullptr), nullptr,
                expertScales, elasticInfo, hcomEpName, EP_WORLD_SIZE, args.epRankId, moeExpertNum, hcomTpName, TP_WORLD_SIZE,
                args.tpRankId, expertShardType, sharedExpertNum,sharedExpertRankNum, quantMode, globalBS,
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
        }
        /**************************************** Call combine. ********************************************/
        // Call the first-phase API.
        if (availableRank.find(args.rankId) != availableRank.end()) {
        ret = aclnnMoeDistributeCombineV3GetWorkspaceSize(expandX, expertIds,
                                                            expandIdx, epRecvCounts,
                                                            expertScales, tpRecvCounts,
                                                            nullptr, nullptr, nullptr,
                                                            nullptr, nullptr, nullptr,
                                                            elasticInfo, oriX, constExpertAlpha1, constExpertAlpha2, constExpertV,
                                                            hcomEpName, EP_WORLD_SIZE, args.epRankId, moeExpertNum,
                                                            hcomTpName, TP_WORLD_SIZE, args.tpRankId, expertShardType,
                                                            sharedExpertNum, sharedExpertRankNum, globalBS, outDtype,
                                                            commQuantMode, groupList_type, nullptr, zeroExpertNum, copyExpertNum, constExpertNum, xOut,
                                                            &combineWorkspaceSize, &combineExecutor);
        CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("[ERROR] aclnnMoeDistributeCombineV3GetWorkspaceSize failed. ret = %d \n", ret); return ret);
        // Allocate device memory based on the workspaceSize computed by the first-phase API.
        if (combineWorkspaceSize > 0) {
            ret = aclrtMalloc(&combineWorkspaceAddr, combineWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc workspace failed. ret = %d \n", ret); return ret);
        }

        // Call the second-phase API.
        ret = aclnnMoeDistributeCombineV3(combineWorkspaceAddr, combineWorkspaceSize, combineExecutor, args.combineStream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnMoeDistributeCombineV3 failed. ret = %d \n", ret);
            return ret);
        // (Fixed writing) Wait until the task execution is complete.
        ret = aclrtSynchronizeStreamWithTimeout(args.combineStream, 10000);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSynchronizeStreamWithTimeout failed. ret = %d \n", ret);
            return ret);
        LOG_PRINT("[INFO] device_%d aclnnMoeDistributeDispatchV3 and aclnnMoeDistributeCombineV3                      \
                    execute successfully.\n", args.rankId);
        }
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
