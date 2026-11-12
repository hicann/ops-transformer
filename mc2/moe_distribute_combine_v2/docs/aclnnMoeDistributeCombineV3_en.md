# aclnnMoeDistributeCombineV3

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T14:11:01.359Z -->

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

- Description: Performs AllToAllV communication and then combines the received data. If expertScales is non-empty, weights are multiplied before summing; if empty, data is summed directly. TP domain communication is not supported.
- Formula:

    $$
    dataOut = AllToAllV(expandX)\\
    xOut = Sum(expertScales * dataOut) + sharedExpertX
    $$

    When `expertScales` is an empty tensor, weight multiplication is skipped and the expert outputs are summed directly.

> Note: This API must be used together with `aclnnMoeDistributeDispatchV3`. It is equivalent to returning data along the same path as the data collection in the `aclnnMoeDistributeDispatchV3` API.

Compared with the `aclnnMoeDistributeCombineV2` API, this API has the following changes:

- Added support for special expert scenarios:

  - **zeroExpertNum≠0**: Enabled by passing a `zeroExpertNum` parameter greater than 0.

    $$Moe(oriXOptional) = 0$$

  - **copyExpertNum≠0**: Enabled by passing a `copyExpertNum` parameter greater than 0. A valid `oriXOptional` parameter must be passed in.

    $$Moe(oriXOptional) = oriXOptional$$

  - **constExpertNum≠0**: Enabled by passing a `constExpertNum` parameter greater than 0. Valid `oriXOptional`, `constExpertAlpha1Optional`, `constExpertAlpha2Optional`, and `constExpertVOptional` parameters must be passed in.

    $$Moe(oriXOptional) = constExpertAlpha1Optional * oriXOptional + constExpertAlpha2Optional * constExpertVOptional$$

## Function Prototype

Each operator is a two-phase API. First call the "aclnnMoeDistributeCombineV3GetWorkspaceSize" API to obtain the workspace size required for computation and the executor containing the operator computation process, and then call the "aclnnMoeDistributeCombineV3" API to perform the computation.

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
    <td>expandX</td>
    <td>Input</td>
    <td>Token features expanded based on <code>expertIds</code>.</td>
    <td>Requires a 2D tensor.</td>
    <td>FLOAT16, BFLOAT16</td>
    <td>ND</td>
    <td><code>(A, H)</code></td>
    <td>√</td>
    </tr>
    <tr>
    <td>expertIds</td>
    <td>Input</td>
    <td>topK expert indexes for each token.</td>
    <td>Requires a 2D tensor.</td>
    <td>INT32</td>
    <td>ND</td>
    <td><code>(BS, K)</code></td>
    <td>√</td>
    </tr>
    <tr>
    <td>assistInfoForCombine</td>
    <td>Input</td>
    <td>Corresponds to the <code>assistInfoForCombineOut</code> output of <code>aclnnMoeDistributeDispatchV3</code>.</td>
    <td>Requires a 1D tensor.</td>
    <td>INT32</td>
    <td>ND</td>
    <td><code>(A * 128, )</code></td>
    <td>√</td>
    </tr>
    <tr>
    <td>epSendCounts</td>
    <td>Input</td>
    <td>Corresponds to the <code>epRecvCounts</code> output of <code>aclnnMoeDistributeDispatchV3</code>.</td>
    <td>Requires a 1D Tensor.</td>
    <td>INT32</td>
    <td>ND</td>
    <td>-</td>
    <td>√</td>
    </tr>
    <tr>
    <td>expertScales</td>
    <td>Input</td>
    <td>Weights of the topK experts for each token.</td>
    <td>2D tensor required when non-empty. For an empty tensor, at least one dimension is 0 and the number of dimensions is not fixed.</td>
    <td>FLOAT32</td>
    <td>ND</td>
    <td>Non-empty: <code>(BS, K)</code>; empty tensor: at least one dimension is 0</td>
    <td>√</td>
    </tr>
    <tr>
    <td>tpSendCountsOptional</td>
    <td>Input</td>
    <td>Corresponds to the <code>tpRecvCounts</code> output of <code>aclnnMoeDistributeDispatchV3</code>.</td>
    <td>Reserved parameter. No longer supported for TP domain communication. Pass a null pointer.</td>
    <td>INT32</td>
    <td>ND</td>
    <td>-</td>
    <td>√</td>
    </tr>
    <tr>
    <td>xActiveMaskOptional</td>
    <td>Input</td>
    <td>Indicates whether a token participates in communication.</td>
    <td><ul><li>Must be a 1D or 2D tensor. Valid data or a null pointer can be passed. By default, all tokens participate in the communication.</li><li>All tokens must be valid if BS differs across devices.</li></ul></td>
    <td>BOOL</td>
    <td>ND</td>
    <td>If the input is 1D, the shape is <code>(BS, )</code>; if the input is 2D, the shape is <code>(BS, K)</code></td>
    <td>√</td>
    </tr>
    <tr>
    <td>activationScaleOptional</td>
    <td>Input</td>
    <td>Reserved parameter.</td>
    <td>Not supported in the current version. Pass a null pointer.</td>
    <td>-</td>
    <td>ND</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>weightScaleOptional</td>
    <td>Input</td>
    <td>Reserved parameter.</td>
    <td>Not supported in the current version. Pass a null pointer.</td>
    <td>-</td>
    <td>ND</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>groupListOptional</td>
    <td>Input</td>
    <td>Reserved parameter.</td>
    <td>Not supported in the current version. Pass a null pointer.</td>
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
    <td>Indicates tokens computed by the shared expert.</td>
    <td>-</td>
    <td>FLOAT16, BFLOAT16</td>
    <td>ND</td>
    <td><code>(BS, H)</code></td>
    <td>√</td>
    </tr>
    <tr>
    <td>elasticInfoOptional</td>
    <td>Input</td>
    <td>Dynamic scale-down information of the EP communicator.</td>
    <td>-</td>
    <td>INT32</td>
    <td>ND</td>
    <td>-</td>
    <td>√</td>
    </tr>
    <tr>
    <td>oriXOptional</td>
    <td>Input</td>
    <td>Represents token data that has not passed through the FFN (Feed-Forward Neural network).</td>
    <td>Required in use cases where copyExpert or constExpert is enabled. You can pass either valid data or a null pointer. If <code>copyExpertNum</code> is not 0 or <code>constExpertNum</code> is not 0, valid input must be passed. If valid data is passed, it must be a 2D tensor, and its data type must keep consistent with <code>expandX</code>.</td>
    <td>FLOAT16, BFLOAT16</td>
    <td>ND</td>
    <td><code>(BS, H)</code></td>
    <td>√</td>
    </tr>
    <tr>
    <td>constExpertAlpha1Optional</td>
    <td>Input</td>
    <td>Calculation coefficient to be input when the constExpert use case is enabled.</td>
    <td>-</td>
    <td>FLOAT16, BFLOAT16</td>
    <td>ND</td>
    <td>-</td>
    <td>√</td>
    </tr>
    <tr>
    <td>constExpertAlpha2Optional</td>
    <td>Input</td>
    <td>Computation coefficient to be input in the use case where constExpert is enabled.</td>
    <td>-</td>
    <td>FLOAT16, BFLOAT16</td>
    <td>ND</td>
    <td>-</td>
    <td>√</td>
    </tr>
    <tr>
    <td>constExpertVOptional</td>
    <td>Input</td>
    <td>aclTensor on the device. Computation coefficient to be input in the use case where constExpert is enabled.</td>
    <td>-</td>
    <td>FLOAT16, BFLOAT16</td>
    <td>ND</td>
    <td>-</td>
    <td>√</td>
    </tr>
    <tr>
    <td>groupEp</td>
    <td>Input</td>
    <td>Name of the EP communicator (expert parallelism communicator).</td>
    <td>String length: <code>[1, 128)</code>.</td>
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
    <td>ID of the current device in the EP domain.</td>
    <td>Value range <code>[0, epWorldSize)</code>. The <code>epRankId</code> values of devices in the same EP communicator must be unique.</td>
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
    <td>Cannot be the same as <code>groupEp</code>.</td>
    <td>STRING</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>tpWorldSize</td>
    <td>Input</td>
    <td>TP communication domain size (reserved parameter, not supported in the current version, only 0 or 1 allowed).</td>
    <td>-</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>tpRankId</td>
    <td>Input</td>
    <td>TP domain device ID.</td>
    <td>Reserved parameter. Not supported in the current version. Set it to 0.</td>
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
    <td>Number of devices for shared experts (a shared expert can be replicated and deployed on multiple devices).</td>
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
    <td><br>If the BS is the same on each rank, <code>globalBS = BS * epWorldSize </code>or 0 <br>If the BS is different on each rank, <code>globalBS = maxBS * epWorldSize</code>, where maxBS indicates the maximum BS on a single device.</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>outDtype</td>
    <td>Input</td>
    <td>Specifies the data type of output x. Reserved parameter.</td>
    <td>Not supported in the current version. Pass 0.</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>commQuantMode</td>
    <td>Input</td>
    <td>Communication quantization type.</td>
    <td>-</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>groupListType</td>
    <td>Input</td>
    <td>group List format, reserved parameter.</td>
    <td>Not supported in the current version. Set to 0.</td>
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
    <td>Must be a 2D tensor. The data type/format must be consistent with <code>expandX</code>.</td>
    <td>FLOAT16, BFLOAT16</td>
    <td>ND</td>
    <td><code>(BS, H)</code></td>
    <td>-</td>
    </tr>
    <tr>
    <td>workspaceSize</td>
    <td>Output</td>
    <td>Returns the size of workspace required on the device.</td>
    <td>-</td>
    <td>UINT64</td>
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

    <!-- npu="910b" id7 -->
    <details>
    <summary><term>Atlas A2 products</term>:</summary>

    - commAlg supports nullptr, "", "fullmesh", and "hierarchy". It is recommended to configure "hierarchy" with a driver of version ≥25.0.RC1.1. nullptr and "" select an algorithm based on HCCL environment variables (not recommended). "fullmesh" transfers tokens directly over RDMA. "hierarchy" sends tokens twice, first within a server and then across servers, reducing the amount of data transmitted across servers.
    - The shared expert scenario is not supported.
    - The shape of epSendCounts is (moeExpertNum + 2 \* globalBS \* K \* serverNum, ), where K indicates the number of topK experts. The first moeExpertNum values indicate the number of tokens received from each device in the EP communicator. The last 2 \* globalBS \* K \* serverNum values store the number of tokens on which combine can perform reduce in advance and the offsets in the communication area before inter-server/intra-server communication. If globalBS is 0, BS \* epWorldSize is used for calculation.
    - TP domain communication is not supported currently.
    - xActiveMaskOptional depends on the value of commAlg. For "fullmesh", it must be a 1D Tensor with shape (BS, ). true values must be placed before false values (for example, {true, false, true} is invalid). "hierarchy" is not supported in the current version; pass a null pointer.
    - expandScalesOptional must be a 1D Tensor with shape (A, ).
    - sharedExpertXOptional is a reserved parameter and is not supported in the current version. Pass a null pointer.
    - epWorldSize depends on the value of commAlg. For "fullmesh", supported values are 2, 3, 4, 5, 6, 7, 8, 16, 32, 64, 128, 192, 256, 384. For "hierarchy", supported values are 16, 32, 64.
    - moeExpertNum depends on the value of commAlg. For "fullmesh", the supported range is (0, 1024]. For "hierarchy", the supported range is (0, 512].
    - groupTp is not supported in the current version. Pass an empty string.
    - tpWorldSize is not supported in the current version. Pass 0.
    - tpRankId is not supported in the current version. Pass 0.
    - expertShardType is not supported in the current version. Pass 0.
    - sharedExpertNum is not supported in the current version. Pass 0.
    - sharedExpertRankNum is not supported in the current version. Pass 0.
    - commQuantMode: value range 0 or 2 (0 indicates no quantization, and 2 indicates int8 quantization). The value 2 is supported only when commAlg is "hierarchy", or when HCCL_INTRA_PCIE_ENABLE=1, HCCL_INTRA_ROCE_ENABLE=0, and the driver version is 25.0.RC1.1 or later.
    - expandScalesOptional must be a 1D Tensor.
    - elasticInfoOptional is not supported in the current version. Pass an empty pointer.
    - oriXOptional is not supported in the current version when commAlg is "hierarchy". Pass an empty pointer.
    - constExpertAlpha1Optional: reserved parameter. Not supported in the current version. Pass a null pointer.
    - constExpertAlpha2Optional: reserved parameter. Not supported in the current version. Pass a null pointer.
    - constExpertVOptional: reserved parameter. Not supported in the current version. Pass a null pointer.
    - zeroExpertNum: when commAlg="fullmesh", value range: [0, MAX_INT32), where MAX_INT32 = 2^31 - 1. Valid zero expert IDs are [<code>moeExpertNum</code>, <code>moeExpertNum + zeroExpertNum</code>).
    - copyExpertNum: when commAlg="fullmesh", value range: [0, MAX_INT32), where MAX_INT32 = 2^31 - 1. Valid copy expert IDs are [<code>moeExpertNum + zeroExpertNum</code>, <code>moeExpertNum + zeroExpertNum + copyExpertNum</code>).
    - constExpertNum: Not supported in the current version. Pass 0.
    </details>
    <!-- end id7 -->

    <!-- npu="A3" id8 -->
    <details>
    <summary><term>Atlas A3 products</term>  :</summary>

    - expertScales: only valid tensors of shape (BS, K) are supported for weighted aggregation. Empty tensors are not supported.
    - commAlg: Not supported in the current version. Pass a null pointer.
    - The shape of epSendCounts is (epWorldSize \* localExpertNum, ).
    - tpSendCountsOptional is a reserved parameter. TP domain communication is no longer supported. Pass a null pointer.
    - xActiveMaskOptional must be a 1D or 2D tensor (the shape is (BS, ) for 1D and (BS, K) for 2D). For 1D, true must precede false. For 2D, if the K values of a token are all false, the token does not participate in communication.
    - expandScalesOptional is a reserved parameter and is not supported in the current version. Pass a null pointer.
    - sharedExpertXOptional must be a 2D or 3D tensor (the shape is (BS, H) for 2D; for 3D, the product of the first two dimensions equals BS and the third dimension equals H). This parameter is optional. If it is passed, sharedExpertRankNum must be 0.
    - The value range of epWorldSize is [2, 768].
    - The value range of moeExpertNum is (0, 1024].
    - groupTp is a reserved parameter. TP domain communication is no longer supported. Passing an empty value is supported only when TP domain communication is not used.
    - tpWorldSize is a reserved parameter. TP domain communication is no longer supported. Only 0 or 1 is allowed.
    - tpRankId is a reserved parameter. TP domain communication is no longer supported. Pass 0.
    - expertShardType: only 0 is supported currently, indicating that shared expert devices are placed before MoE expert devices.
    - The current value range of sharedExpertNum is [0, 4].
    - The value range of sharedExpertRankNum is [0, epWorldSize). If sharedExpertRankNum is 0, sharedExpertNum must be 0 or 1. If it is not 0, sharedExpertRankNum % sharedExpertNum = 0 must be satisfied.
    - The value range of commQuantMode is 0 or 2 (0 indicates no quantization and 2 indicates int8 quantization). The value 2 can be used only when tpWorldSize < 2.
    - expandScalesOptional is a reserved parameter and is not supported in the current version. Pass a null pointer.
    - elasticInfoOptional is not supported in the current version. Pass a null pointer.
    - constExpertAlpha1Optional can be set to valid data or a null pointer. Valid input is mandatory when constExpertNum is not 0. If valid data is passed, it must be a 2D tensor with shape <code>(constExpertNum, H)</code>, and its data type must be kept consistent with that of expandX.
    - constExpertAlpha2Optional can be set to valid data or a null pointer. Valid input is mandatory when constExpertNum is not 0. If valid data is passed, it must be a 2D tensor with shape <code>(constExpertNum, H)</code>, and its data type must be kept consistent with that of expandX.
    - constExpertVOptional can be set to valid data or a null pointer. Valid input is mandatory when constExpertNum is not 0. If valid data is passed, it must be a 2D tensor with shape <code>(constExpertNum, H)</code>, and its data type must be kept consistent with that of expandX.
    - The value range of zeroExpertNum is [0, MAX_INT32). MAX_INT32 = 2^31 - 1. Valid zero expert IDs are in the range [<code>moeExpertNum</code>, <code>moeExpertNum + zeroExpertNum</code>).
    - copyExpertNum value range: [0, MAX_INT32), where MAX_INT32 = 2^31 - 1. Valid copy expert IDs are [<code>moeExpertNum</code>, <code>moeExpertNum + zeroExpertNum</code>).
    - constExpertNum value range: [0, MAX_INT32), where MAX_INT32 = 2^31 - 1. Valid constant expert IDs are [<code>moeExpertNum + zeroExpertNum + copyExpertNum</code>, <code>moeExpertNum + zeroExpertNum + copyExpertNum + constExpertNum</code>).
    </details>
    <!-- end id8 -->

    <!-- npu="950" id9 -->
    <details>
    <summary><term>Ascend 950PR&950DT products</term>:</summary>

    - expertScales determines whether to enable the expert weight function. If expertScales is a valid tensor of shape (BS, K), weighting is enabled. If expertScales is an empty tensor, weighting is disabled and expert outputs are directly summed.
    - commAlg is not supported in the current version. Pass a null pointer.
    - The shape of epSendCounts is (epWorldSize \* localExpertNum, ).
    - TP domain communication is not supported currently.
    - xActiveMaskOptional must be a 1D or 2D tensor (1D shape: (BS, ); 2D shape: (BS, K)). For a 1D tensor, true values must be placed before false values. For a 2D tensor, if all K values corresponding to a token are false, the token does not participate in communication.
    - expandScalesOptional is a reserved parameter and is not supported in the current version. Pass a null pointer.
    - sharedExpertXOptional must be a 2D or 3D tensor (2D shape: (BS, H); for 3D, the product of the first two dimensions is BS and the third dimension is H). This parameter is optional. If passed in, sharedExpertRankNum must be 0.
    - epWorldSize supports values in [2, 1024].
    - The value range of moeExpertNum is (0, 1024].
    - groupTp is not supported in the current version. Pass an empty string.
    - tpWorldSize is not supported in the current version. Pass 0.
    - tpRankId is not supported in the current version. Pass 0.
    - In the current version, only 0 is supported for expertShardType, indicating that shared expert devices are placed before MoE expert devices.
    - The current value range of sharedExpertNum is [0, 4].
    - The value range of sharedExpertRankNum is [0, epWorldSize). If it is 0, sharedExpertNum must be 0 or 1. If it is not 0, sharedExpertRankNum % sharedExpertNum = 0 must be satisfied.
    - The value range of commQuantMode is 0, 2, 3, or 4 (0 indicates no quantization, 2 indicates int8 quantization, 3 indicates mxfp8 e5m2 quantization, and 4 indicates mxfp8 e4m3 quantization).
    - expandScalesOptional is a reserved parameter and is not supported in the current version. Pass a null pointer.
    - elasticInfoOptional is a reserved parameter, not supported in the current version. Pass a null pointer.
    - constExpertAlpha1Optional can be passed with valid data or a null pointer. Valid input is required when constExpertNum is not 0. If valid data is passed, it must be a 2D tensor with shape <code>(constExpertNum, H)</code>. Its data type must keep consistent with expandX.
    - constExpertAlpha2Optional can be passed with valid data or a null pointer. Valid input is required when constExpertNum is not 0. If valid data is passed, it must be a 2D tensor with shape <code>(constExpertNum, H)</code>. Its data type must keep consistent with expandX.
    - constExpertVOptional can be passed with valid data or a null pointer. Valid input is required when constExpertNum is not 0. If valid data is passed, it must be a 2D tensor with shape <code>(constExpertNum, H)</code>. Its data type must keep consistent with expandX.
    - zeroExpertNum value range: [0, MAX_INT32), where MAX_INT32 = 2^31 - 1. Valid zero expert IDs are [<code>moeExpertNum</code>, <code>moeExpertNum + zeroExpertNum</code>).
    - copyExpertNum value range: [0, MAX_INT32), where MAX_INT32 = 2^31 - 1. Valid copy expert IDs are [<code>moeExpertNum + zeroExpertNum</code>, <code>moeExpertNum + zeroExpertNum + copyExpertNum</code>).
    - constExpertNum value range: [0, MAX_INT32), where MAX_INT32 = 2^31 - 1. Valid constant expert IDs are [<code>moeExpertNum + zeroExpertNum + copyExpertNum</code>, <code>moeExpertNum + zeroExpertNum + copyExpertNum + constExpertNum</code>).
    </details>
    <!-- end id9 -->

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
    <th>Return value</th>
    <th>Error Code</th>
    <th>Description</th>
    </tr>
    </thead>
    <tbody>
    <tr>
    <td>ACLNN_ERR_PARAM_NULLPTR</td>
    <td>161001</td>
    <td>The required input or output Tensor parameter is a null pointer.</td>
    </tr>
    <tr>
    <td>ACLNN_ERR_PARAM_INVALID</td>
    <td>161002</td>
    <td>The data type of input or output is not supported.</td>
    </tr>
    <tr>
    <td rowspan="2">ACLNN_ERR_INNER_TILING_ERROR</td>
    <td rowspan="2">561002</td>
    <td>The input and output shapes are out of the supported range.</td>
    </tr>
    <tr>
        <td>The parameter value is out of the supported range.</td>
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
        <td>Workspace size allocated on the Device. Obtained by the first-phase API aclnnMoeDistributeCombineV3GetWorkspaceSize.</td>
    </tr>
    <tr>
        <td>executor</td>
        <td>Input</td>
        <td>Operator executor, which contains the operator computation process.</td>
    </tr>
    <tr>
        <td>stream</td>
        <td>Input</td>
        <td>Specifies the Stream for task execution.</td>
    </tr>
    </tbody>
    </table>

- **Return Value**

    aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- **Deterministic computation**:
  - aclnnMoeDistributeCombineV3 defaults to a deterministic implementation.

- **Driver Constraints**:
  - The driver versions of all nodes in the operator communication domain must be the same.

- **API Compatibility Constraints**:
  - `aclnnMoeDistributeDispatchV3` and `aclnnMoeDistributeCombineV3` must be used together. The `assistInfoForCombineOut`, `epRecvCountsOut`, `tpRecvCountsOut`, and `expandScalesOut` output by the former must be directly passed to the corresponding parameters of the latter. The service logic must not rely on the specific values of these tensors.

- **Parameter Consistency Constraints**:
  - The `groupEp`, `epWorldSize`, `moeExpertNum`, `groupTp`, `tpWorldSize`, `expertShardType`, `sharedExpertNum`, `sharedExpertRankNum`, `globalBS`, `commAlg` parameters and the `HCCL_BUFFSIZE` value must be consistent across all cards, and must be consistent with the corresponding parameters of `aclnnMoeDistributeDispatchV4`.

- **Product-Specific Constraints**:

  <!-- npu="A3" id10 -->
  - <term>Atlas A3 products</term>: Each device contains two DIEs (dies). In Parameters, "current device" refers to a single DIE.

  <!-- end id10 -->

- **Shape Variable Constraints**:

  | Variable     | Definition and value range                                                     |
  | :----------- | :----------------------------------------------------------------------------- |
  | A            | Maximum number of tokens to be distributed on the current device. Value range: <ul><li>For shared experts, <code>A = BS \* epWorldSize \* sharedExpertNum / sharedExpertRankNum</code> must be met.</li><li>For MoE experts, if globalBS is 0, <code>A >= BS \* epWorldSize \* min(localExpertNum, K)</code> must be met; if globalBS is not 0, <code>A >= globalBS \* min(localExpertNum, K)</code> must be met.</li></ul>|
  | H            | Indicates the hidden size:<ul><li> <term>Atlas A2 products</term>: (0, 10240] and a multiple of 32.</li><li><term>Atlas A3 products</term> and Ascend 950PR&950DT products: [1024, 8192]. </li></ul>|
  | BS           | Final number of output tokens on the current device:<ul><li> <term>Atlas A2 products</term>: Depends on commAlg. "fullmesh" supports (0, 256]; "hierarchy" supports (0, 512] when the driver version is ≥ 25.0.RC1.1;</li><li><term>Atlas A3 products</term> and Ascend 950PR&950DT products: 0 < BS ≤ 512.</li></ul> |
  | K            | Indicates that topK experts are selected:<br> 0 < K ≤ 16 and 0 < K ≤ <code>moeExpertNum+zeroExpertNum+copyExpertNum+constExpertNum</code>. |
  | serverNum    | Number of server nodes:<br><term>Atlas A2 products</term>: This variable is used only by the shape in this use case, and only 2, 4, and 8 are supported.
  | localExpertNum | Number of experts on the current device:<ul><li>For a shared expert device, localExpertNum = 1;</li><li>For a MoE expert device, localExpertNum = <code>moeExpertNum/(epWorldSize-sharedExpertRankNum)</code>. If localExpertNum > 1, TP communication is not supported. </li><li><term>Atlas A3 products</term> and <term>Ascend 950PR&950DT products</term>: 0 < localExpertNum * epWorldSize ≤ 2048 must be met.</li></ul>|

- **Environment Variable Constraints**:
  - **HCCL_BUFFSIZE**: Before calling this API, check whether the value of the HCCL_BUFFSIZE environment variable is reasonable. This environment variable indicates the memory size occupied by a single communication domain, in MB. The default value is 200 MB when not configured.

      <!-- npu="910b" id11 -->
      - <term>Atlas A2 products</term>:
        - commAlg is configured as "" or nullptr: Based on the HCCL_INTRA_PCIE_ENABLE and HCCL_INTRA_ROCE_ENABLE environment variable configuration, selects the "fullmesh" or "hierarchy" formula.
        - commAlg set to "fullmesh": Set the size requirement to <code>= 2 \* (BS \* epWorldSize \* min(localExpertNum, K) \* H \* sizeof(uint16) + 2MB)</code>.
        - commAlg set to "hierarchy": Set the size requirement to (≥ (`moeExpertNum` + `epWorldSize` / 4) \* Align512(`maxBS` \* (`H` \* 2 + 16 \* Align8(`K`))) \* 1B + 8MB, where Align8(x) = ((x + 8 - 1) / 8) \* 8, Align512(x) = ((x + 512 - 1) / 512) \* 512).
      <!-- end id11 -->
      <!-- npu="950,A3" id12 -->
      - <term>Atlas A3 products</term>, Ascend 950PR&950DT products:
        - Within the EP communication domain: Set the size requirement to <code>>= 2 and >= 2 \* (localExpertNum \* maxBS \* epWorldSize \* Align512(Align32(2 \* H) + 44) + (K + sharedExpertNum) \* maxBS \* Align512(2 \* H))</code>, where <code>localExpertNum</code> must use the number of experts on the local MoE expert card, and <code>Align512(x) = ((x + 512 - 1) / 512) \* 512, Align32(x) = ((x + 32 - 1) / 32) \* 32</code>.
        - Within the TP communication domain: the value must be >= A \* (H \* 2 + 128) \* 2.

      <!-- end id12 -->

  - **HCCL_INTRA_PCIE_ENABLE/HCCL_INTRA_ROCE_ENABLE**:

    <!-- npu="910b" id13 -->
    - <term>Atlas A2 products</term>: These environment variables are no longer recommended. It is recommended to configure commAlg as "hierarchy".
    <!-- end id13 -->
    <!-- npu="950,A3" id14 -->
    - <term>Atlas A3 products</term>, Ascend 950PR&950DT products: This environment variable is not supported.

    <!-- end id14 -->

- **Communication Domain Usage Constraints**:
  - The `aclnnMoeDistributeCombineV3` series operators and `aclnnMoeDistributeDispatchV3` in a model support only the same EP communication domain, and no other operators are allowed in this communication domain.
  - TP domain communication is not supported currently.

  <!-- npu="A3" id15 -->
  - <term>Atlas A3 products</term>  : Nodes in a communicator must reside in the same SuperPoD. Cross-SuperPoD communication is not supported.

  <!-- end id15 -->

- **Communication mode constraints**:

  <!-- npu="950" id16 -->
  - <term>Ascend 950PR&950DT products</term>: Only UB Memory communication is supported.

  <!-- end id16 -->

- **Networking Constraints**:

  <!-- npu="910b" id17 -->
  - <term>Atlas A2 products</term>: In multi-machine scenarios, only switch networking is supported. Direct-connect networking between two machines is not supported.

  <!-- end id17 -->

- **Other Constraints**:
  - In the formula, "/" indicates integer division.
  - <code>moeExpertNum + zeroExpertNum + copyExpertNum + constExpertNum < MAX_INT32</code>.

## Example

<!-- npu="910b" id18 -->
- <term>Atlas A2 products</term> :

    This sample supports running the A2 operator in a single-node environment with a card count in the range [2, 8]. You can set EP_WORLD_SIZE_A2 to the card count in the sample code as needed, and modify moeExpertNum so that moeExpertNum is divisible by EP_WORLD_SIZE_A2.

    - Compile the operator: The operator compilation command is as follows. Both the moe_distribute_dispatch_v2 and moe_distribute_combine_v2 operators need to be compiled, and these two operators must be executed in pairs.

        ```bash
        bash build.sh --pkg --soc=ascend910b --ops=moe_distribute_dispatch_v2,moe_distribute_combine_v2
        ```

    - Create A2 example code: After compilation, refer to the existing [test_aclnn_moe_distribute_combine_v2.cpp](../examples/test_aclnn_moe_distribute_combine_v2.cpp) file in the operator [examples](../examples/) directory and create a new test file test_aclnn_moe_distribute_combine_v2.cpp using the A2 example code.

    - Execute operator samples: The sample operator execution command is as follows. This command executes all sample code files in the operator [examples](https://gitcode.com/cann/ops-transformer/tree/9.1.0/mc2/moe_distribute_dispatch_v2/examples/) directory.

        ```bash
        bash build.sh --run_example --ops=moe_distribute_combine_v2 eager cust
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
            void *expandScalesDeviceAddr = nullptr;

            // Zero-expert use case input
            void *oriXDeviceAddr = nullptr;
            void *tpRecvCountsDeviceAddr = nullptr;

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
            aclTensor *expandScales = nullptr;

            aclTensor *activationScale = nullptr; // reserved parameter
            aclTensor *weightScale = nullptr; // reserved parameter
            aclTensor *groupList = nullptr; // reserved parameter

            aclTensor *sharedExpertX = nullptr; // A3

            aclTensor *oriX = nullptr;
            aclTensor *constExpertAlpha1 = nullptr; // A3
            aclTensor *constExpertAlpha2 = nullptr; // A3
            aclTensor *constExpertV = nullptr; // A3

            aclTensor *tpRecvCounts = nullptr;

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
            std::vector<int64_t> epRecvCountsShape{localExpertNum * EP_WORLD_SIZE_A2}; // No layering
            std::vector<int64_t> expandScalesShape{A};

            std::vector<int64_t> oriXShape{BS, H};
            std::vector<int64_t> tpRecvCountsShape{1};
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
            int64_t expandScalesShapeSize = GetShapeSize(expandScalesShape);

            int64_t oriXSize = GetShapeSize(oriXShape);
            int64_t tpRecvCountsShapeSize = GetShapeSize(tpRecvCountsShape);

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
            std::vector<float> expandScalesHostData(expandScalesShapeSize, 0);

            std::vector<int16_t> oriXHostData(oriXSize, 1);
            std::vector<int32_t> tpRecvCountsHostData(tpRecvCountsShapeSize, 0);
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
            ret = CreateAclTensor(expandScalesHostData, expandScalesShape, &expandScalesDeviceAddr, aclDataType::ACL_FLOAT, &expandScales);
            CHECK_RET(ret == ACL_SUCCESS, return ret);

            ret = CreateAclTensor(oriXHostData, oriXShape, &oriXDeviceAddr, aclDataType::ACL_BF16, &oriX);
            CHECK_RET(ret == ACL_SUCCESS, return ret);
            ret = CreateAclTensor(tpRecvCountsHostData, tpRecvCountsShape, &tpRecvCountsDeviceAddr, aclDataType::ACL_INT32, &tpRecvCounts);
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
            // Allocate device memory based on the workspaceSize calculated by the first-phase API.
            if (combineWorkspaceSize > 0) {
                ret = aclrtMalloc(&combineWorkspaceAddr, combineWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
                CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc workspace failed. ret = %d \n", ret); return ret);
            }

            // Call the second-phase API.
            ret = aclnnMoeDistributeCombineV3(combineWorkspaceAddr, combineWorkspaceSize, combineExecutor, args.combineV3Stream);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnMoeDistributeCombineV3 failed. ret = %d \n", ret);
                return ret);
            // (Boilerplate) Synchronously wait for the task execution to complete.
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
            DestroyTensor(expandScales);
            DestroyTensor(activationScale);
            DestroyTensor(weightScale);
            DestroyTensor(groupList);
            DestroyTensor(sharedExpertX);
            DestroyTensor(oriX);
            DestroyTensor(constExpertAlpha1);
            DestroyTensor(constExpertAlpha2);
            DestroyTensor(constExpertV);
            DestroyTensor(tpRecvCounts);
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
            FreeDeviceAddr(expandScalesDeviceAddr);
            FreeDeviceAddr(oriXDeviceAddr);
            FreeDeviceAddr(tpRecvCountsDeviceAddr);
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

<!-- end id18 -->
<!-- npu="950" id19 -->
- <term>Ascend 950PR&950DT products</term>: For details, see the preparation part and example code in the Example section of [aclnnMoeDistributeCombineV2](./aclnnMoeDistributeCombineV2_en.md). Reset the involved variables according to the preceding Constraints. For the use case parameters newly added in the V4 API compared with the V3 API, pass values as described in the preceding Parameters.

<!-- end id19 -->
<!-- npu="A3" id20 -->
- <term>Atlas A3 products</term>:

    For details about the compilation and execution process, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

<!-- end id20 -->
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
        void *expandScalesDeviceAddr = nullptr;
        void *residualXDeviceAddr = nullptr;
        void *sharedExpertXDeviceAddr = nullptr;

        void *elasticInfoDeviceAddr = nullptr;
        void *oriXDeviceAddr = nullptr;
        void *constExpertAlpha1DeviceAddr = nullptr;
        void *constExpertAlpha2DeviceAddr = nullptr;
        void *constExpertVDeviceAddr = nullptr;
        void *tpRecvCountsDeviceAddr = nullptr;

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
        aclTensor *expandScales = nullptr;
        aclTensor *residualX = nullptr;
        aclTensor *sharedExpertX = nullptr;


        aclTensor *elasticInfo = nullptr;
        aclTensor *oriX = nullptr;
        aclTensor *constExpertAlpha1 = nullptr;
        aclTensor *constExpertAlpha2 = nullptr;
        aclTensor *constExpertV = nullptr;
        aclTensor *tpRecvCounts = nullptr;

        aclTensor *xOut = nullptr;

        // Define the dimensions of each variable in the current use case.
        std::vector<int64_t> xShape{BS, H};
        std::vector<int64_t> expertIdsShape{BS, K};
        std::vector<int64_t> scalesShape{moeExpertNum + 1, H};
        std::vector<int64_t> expertScalesShape{BS, K};

        std::vector<int64_t> expandXShape{A, H};
        std::vector<int64_t> dynamicScalesShape{A};
        std::vector<int64_t> expandIdxShape{A * 128};
        std::vector<int64_t> expertTokenNumsShape{localExpertNum};
        std::vector<int64_t> epRecvCountsShape{localExpertNum * EP_WORLD_SIZE};
        std::vector<int64_t> expandScalesShape{A};
        std::vector<int64_t> sharedExpertXShape{BS, 1, H};


        std::vector<int64_t> elasticInfoShape{4 + EP_WORLD_SIZE * 2};
        std::vector<int64_t> oriXShape{BS, H};
        std::vector<int64_t> constExpertAlpha1Shape{constExpertNum, H};
        std::vector<int64_t> constExpertAlpha2Shape{constExpertNum, H};
        std::vector<int64_t> constExpertVShape{constExpertNum, H};
        std::vector<int64_t> tpRecvCountsShape{1};

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
        int64_t expandScalesShapeSize = GetShapeSize(expandScalesShape);
        int64_t sharedExpertXShapeSize = GetShapeSize(sharedExpertXShape);

        int64_t elasticInfoSize = GetShapeSize(elasticInfoShape);
        int64_t oriXSize = GetShapeSize(oriXShape);
        int64_t constExpertAlpha1Size = GetShapeSize(constExpertAlpha1Shape);
        int64_t constExpertAlpha2Size = GetShapeSize(constExpertAlpha2Shape);
        int64_t constExpertVSize = GetShapeSize(constExpertVShape);
        int64_t tpRecvCountsShapeSize = GetShapeSize(tpRecvCountsShape);

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
        std::vector<float> expandScalesHostData(expandScalesShapeSize, 0);
        std::vector<int16_t> sharedExpertXHostData(sharedExpertXShapeSize, 1);

        std::vector<int16_t> oriXHostData(oriXSize, 1);
        std::vector<int16_t> constExpertAlpha1HostData(constExpertAlpha1Size, 0);
        std::vector<int16_t> constExpertAlpha2HostData(constExpertAlpha2Size, 0);
        std::vector<int16_t> constExpertVHostData(constExpertVSize, 0);
        std::vector<int32_t> tpRecvCountsHostData(tpRecvCountsShapeSize, 0);

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
        ret = CreateAclTensor(tpRecvCountsHostData, tpRecvCountsShape, &tpRecvCountsDeviceAddr, aclDataType::ACL_INT32, &tpRecvCounts);
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
        // (Boilerplate) Synchronize and wait for the task to complete.
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
        if (tpRecvCounts != nullptr) {
            aclDestroyTensor(tpRecvCounts);
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
        if (tpRecvCountsDeviceAddr != nullptr) {
            aclrtFree(tpRecvCountsDeviceAddr);
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
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclCommInitAll ep failed, ret = %d\n", ret); return ret);

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
