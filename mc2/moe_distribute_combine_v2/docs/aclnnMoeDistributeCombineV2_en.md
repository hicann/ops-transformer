# aclnnMoeDistributeCombineV2

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T14:15:55.075Z -->

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

- Description: Performs AllToAllV communication and integrates the received data. If expertScales is not empty, the outputs are multiplied by weights and then summed. If expertScales is empty, the outputs are directly summed. TP domain communication is not supported.
- Formula:

    $$
    ataOut = AllToAllV(expandX)\\
    xOut = Sum(expertScales * ataOut + expertScales * sharedExpertX)
    $$

    If `expertScales` is an empty tensor, no weight multiplication is performed, and the expert outputs are directly summed.

>Note that this API must be used together with `aclnnMoeDistributeDispatchV2`. It is equivalent to returning data along the path used by the `MoeDistributeDispatchV2` operator to collect data.

Compared with the `aclnnMoeDistributeCombine` API, the changes of this API are as follows:

- More detailed token information is input to help `aclnnMoeDistributeCombineV2` perform efficient synchronization across all devices. Therefore, the `expandIdx` input parameter with the shape `(BS * K,)` in the original API is replaced by the `assistInfoForCombine` parameter with the shape `(A * 128,)`;
- The `sharedExpertXOptional` input parameter is added. It supports user input of tokens computed by the shared expert when `sharedExpertNum` is 0;
- The `commAlg` input parameter is added, replacing the `HCCL_INTRA_PCIE_ENABLE` and `HCCL_INTRA_ROCE_ENABLE` environment variables.

For details, refer to the following Parameters.

## Function Prototype

Each operator is implemented as a two-phase API. Call the "aclnnMoeDistributeCombineV2GetWorkspaceSize" API first to obtain the workspace size required for the calculation and the executor that contains the calculation process of the operator, and then call the "aclnnMoeDistributeCombineV2" API to perform the calculation.

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
    int64_t          globalBS,
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
    <td>TopK expert indexes of each token.</td>
    <td>Must be a 2D tensor.</td>
    <td>INT32</td>
    <td>ND</td>
    <td><code>(BS, K)</code></td>
    <td>√</td>
    </tr>
    <tr>
    <td>assistInfoForCombine</td>
    <td>Input</td>
    <td>Corresponding to the <code>assistInfoForCombineOut</code> output in <code>aclnnMoeDistributeDispatchV2</code>.</td>
    <td>Must be a 1D tensor.</td>
    <td>INT32</td>
    <td>ND</td>
    <td><code>(A * 128, )</code></td>
    <td>√</td>
    </tr>
    <tr>
    <td>epSendCounts</td>
    <td>Input</td>
    <td>Corresponds to the <code>epRecvCounts</code> output in <code>aclnnMoeDistributeDispatchV2</code>.</td>
    <td>Requirement is a 1D Tensor.</td>
    <td>INT32</td>
    <td>ND</td>
    <td>-</td>
    <td>√</td>
    </tr>
    <tr>
    <td>expertScales</td>
    <td>Input</td>
    <td>Weights of the topK experts for each token. Empty tensors are supported. If an empty tensor is passed in, no weighting is performed, and the expert outputs are directly summed.</td>
    <td>A 2D tensor is required when non-empty. For an empty tensor, at least one dimension must be 0, and the number of dimensions is not fixed.</td>
    <td>FLOAT32</td>
    <td>ND</td>
    <td>Non-empty: <code>(BS, K)</code>; empty tensor: at least one dimension is 0</td>
    <td>√</td>
    </tr>
    <tr>
    <td>tpSendCountsOptional</td>
    <td>Input</td>
    <td>Reserved parameter. TP domain communication is no longer supported. Simply pass a null pointer</td>
    <td>Reserved parameter. TP domain communication is no longer supported. Simply pass a null pointer.</td>
    <td>INT32</td>
    <td>ND</td>
    <td>-</td>
    <td>√</td>
    </tr>
    <tr>
    <td>xActiveMaskOptional</td>
    <td>Input</td>
    <td>Indicates whether tokens participate in communication.</td>
    <td><ul><li>Requirement is a 1D or 2D tensor. Pass valid data or a null pointer. By default, all tokens participate in communication.</li><li>When BS differs across devices, all tokens must be valid.</li></ul></td>
    <td>BOOL</td>
    <td>ND</td>
    <td>When the input is 1D, the shape is <code>(BS, )</code>; when the input is 2D, the shape is <code>(BS, K)</code></td>
    <td>√</td>
    </tr>
    <tr>
    <td>activationScaleOptional</td>
    <td>Input</td>
    <td>Reserved parameter.</td>
    <td>Not supported in the current version. Simply pass a null pointer.</td>
    <td>-</td>
    <td>ND</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>weightScaleOptional</td>
    <td>Input</td>
    <td>Reserved parameter.</td>
    <td>Not supported in the current version. Simply pass a null pointer.</td>
    <td>-</td>
    <td>ND</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>groupListOptional</td>
    <td>Input</td>
    <td>Reserved parameter.</td>
    <td>Not supported in the current version. Simply pass a null pointer.</td>
    <td>-</td>
    <td>ND</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>expandScalesOptional</td>
    <td>Input</td>
    <td>Corresponding to the <code>expandScales</code> output in <code>aclnnMoeDistributeDispatchV2</code>.</td>
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
    <td>The data type must be the same as that of expandX.</td>
    <td>FLOAT16, BFLOAT16</td>
    <td>ND</td>
    <td><code>(BS, H)</code></td>
    <td>√</td>
    </tr>
    <tr>
    <td>groupEp</td>
    <td>Input</td>
    <td>Name of the EP communicator (expert parallel communicator).</td>
    <td>The string length is in the range <code>[1, 128)</code>.</td>
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
    <td>ID of the local device in the EP domain.</td>
    <td>Value range <code>[0, epWorldSize)</code>. The <code>epRankId</code> of each device in the same EP communicator must be unique.</td>
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
    <td>TP communicator name (Reserved parameter, not supported in the current version).</td>
    <td>Cannot be the same as <code>groupEp</code>.</td>
    <td>STRING</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>tpWorldSize</td>
    <td>Input</td>
    <td>TP communicator size (Reserved parameter. Not supported in the current version. Only 0 or 1 is allowed).</td>
    <td>-</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>tpRankId</td>
    <td>Input</td>
    <td>TP domain local device ID.</td>
    <td>Reserved parameter. Not supported in the current version. Simply pass 0.</td>
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
    <td>Number of shared experts (one shared expert can be copied and deployed to multiple devices).</td>
    <td>-</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>sharedExpertRankNum</td>
    <td>Input</td>
    <td>Number of devices for shared experts. (One shared expert can be copied and deployed to multiple devices)</td>
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
    <td><ul><li>When all ranks have the same BS, <code>globalBS = BS * epWorldSize</code> or 0.</li><li>When ranks have different BS values, <code>globalBS = maxBS * epWorldSize</code> (maxBS indicates the maximum BS of a single device).</li></ul></td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>outDtype</td>
    <td>Input</td>
    <td>Specifies the data type of output x. Reserved parameter.</td>
    <td>Not supported in the current version. Simply pass 0.</td>
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
    <td>group List format. Reserved parameter.</td>
    <td>Not supported in the current version. Simply pass 0.</td>
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
    <td>xOut</td>
    <td>Output</td>
    <td>Processed token.</td>
    <td>Must satisfy: 2D Tensor, data type/format consistent with <code>expandX</code>.</td>
    <td>FLOAT16, BFLOAT16</td>
    <td>ND</td>
    <td><code>(BS, H)</code></td>
    <td>-</td>
    </tr>
    <tr>
    <td>workspaceSize</td>
    <td>Output</td>
    <td>Returns the workspace size to be allocated on the Device side.</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>executor</td>
    <td>Output</td>
    <td>Returns the op executor containing the operator computation flow.</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    </tbody>
    </table>

    <!-- npu="910b" id7 -->
    - <term>Atlas A2 products</term>:
        - `commAlg` supports nullptr, "", "fullmesh", and "hierarchy"; "hierarchy" is recommended, together with a driver of version ≥25.0.RC1.1; nullptr and "" select the algorithm based on HCCL environment variables (not recommended); "fullmesh" transfers tokens directly over RDMA; "hierarchy" sends tokens twice (intra-node and cross-node) to reduce the cross-node data volume.
        - The shared expert scenario is not supported.
        - The shape of `epSendCounts` is (moeExpertNum + 2 \* globalBS \* K \* serverNum, ), where K indicates the number of topK experts. The first moeExpertNum values indicate the number of tokens received from each card in the EP communication domain, and the last 2 \* globalBS \* K \* serverNum values are used to store the number of tokens and communication domain offsets that can be reduced in advance by combine before inter-server/intra-server communication. When globalBS = 0, the calculation is based on BS \* epWorldSize.
        - TP domain communication is not supported currently.
        - The value of `xActiveMaskOptional` depends on `commAlg`. For "fullmesh", the requirement is a 1D Tensor with shape (BS, ); true must be placed before false (for example, {true, false, true} is invalid); "hierarchy" is not supported in the current version. Simply pass a null pointer.
        - The requirement of `expandScalesOptional` is a 1D Tensor with shape (A, ).
        - `sharedExpertXOptional` is a reserved parameter and is not supported in the current version. Simply pass a null pointer.
        - The value of `epWorldSize` depends on `commAlg`. For "fullmesh", 2, 3, 4, 5, 6, 7, 8, 16, 32, 64, 128, 192, 256, and 384 are supported. For "hierarchy", 16, 32, and 64 are supported.
        - The value of `moeExpertNum` depends on `commAlg`. For "fullmesh", the value range is (0, 1024]. For "hierarchy", the value range is (0, 512].
        - `groupTp` is not supported in the current version. Pass an empty string.
        - `tpWorldSize` is not supported in the current version. Simply pass 0.
        - `tpRankId` is not supported in the current version. Pass 0.
        - `expertShardType` is not supported in the current version. Simply pass 0.
        - `sharedExpertNum` is not supported in the current version. Simply pass 0.
        - `sharedExpertRankNum` is not supported in the current version. Simply pass 0.
        - `commQuantMode` value range is 0 or 2 (0 indicates non-quantized, 2 indicates int8 quantization). The value 2 is supported only when commAlg is "hierarchy" or when HCCL_INTRA_PCIE_ENABLE=1, HCCL_INTRA_ROCE_ENABLE=0, and the driver version is ≥25.0.RC1.1.

    <!-- end id7 -->
    <!-- npu="A3" id8 -->
    - <term>Atlas A3 products</term>:
        - `expertScales` supports only a valid Tensor of shape (BS, K) for weighted aggregation. Empty tensors are not supported.
        - `commAlg` is not supported in the current version. Simply pass a null pointer.
        - The shape of `epSendCounts` is (epWorldSize \* max(tpWorldSize, 1) \* localExpertNum, ).
        - When TP domain communication is involved, `tpSendCountsOptional` is a 1D Tensor with shape (tpWorldSize, ).
        - `xActiveMaskOptional` requirement is a 1D or 2D Tensor (for 1D, shape is (BS, ); for 2D, shape is (BS, K)); for 1D, true must be placed before false; for 2D, if all K values corresponding to a token are false, the token does not participate in communication.
        - `expandScalesOptional` is a reserved parameter and is not supported in the current version. Simply pass a null pointer.
        - `sharedExpertXOptional` requirement is a 2D or 3D Tensor (for 2D, shape is (BS, H); for 3D, the product of the first two dimensions equals BS and the third dimension equals H); it can be passed or omitted. If passed, sharedExpertRankNum must be 0.
        - `epWorldSize` supports the value range [2, 768].
        - `moeExpertNum` value range is (0, 1024].
        - `groupTp` is a reserved parameter. TP domain communication is no longer supported. Passing null is supported only when no TP domain communication is used.
        - `tpWorldSize` is a reserved parameter. TP domain communication is no longer supported. Only 0 or 1 is allowed.
        - `tpRankId` is a reserved parameter. TP domain communication is no longer supported. Simply pass 0.
        - `expertShardType`: Currently, only 0 is supported, indicating that shared expert cards are placed before MoE expert cards.
        - The current value range of `sharedExpertNum` is [0, 4].
        - The value range of `sharedExpertRankNum` is [0, epWorldSize). When it is 0, sharedExpertNum must be 0 or 1. When it is not 0, sharedExpertRankNum % sharedExpertNum = 0 must be satisfied.
        - The value range of `commQuantMode` is 0 or 2 (0 indicates non-quantized, and 2 indicates int8 quantization). The value 2 can be enabled only when tpWorldSize < 2.

    <!-- end id8 -->
    <!-- npu="950" id9 -->
    - <term>Ascend 950PR&950DT products</term>:
        - `expertScales` optionally enables the expert weight function. Pass a valid Tensor of shape (BS, K) to enable it and apply weighting. Pass an empty Tensor to disable it and directly sum the expert outputs.
        - `commAlg` is not supported in the current version. Simply pass a null pointer.
        - The shape of `epSendCounts` is (epWorldSize \* max(tpWorldSize, 1) \* localExpertNum, ).
        - `tpSendCountsOptional` is not supported in the current version. Simply pass a null pointer.
        - `xActiveMaskOptional`: Requirement is a 1D or 2D tensor (shape (BS, ) for 1D tensors, shape (BS, K) for 2D tensors); for 1D tensors, true values must be placed before false values (e.g., {true, false, true} is invalid); for 2D tensors, a token whose K values are all false does not participate in communication.
        - `expandScalesOptional`: Reserved parameter. Not supported in the current version. Simply pass a null pointer.
        - `sharedExpertXOptional`: Requirement is a 2D or 3D tensor (shape (BS, H) for 2D tensors; for 3D tensors, the product of the first two dimensions equals BS and the third dimension equals H). The parameter is optional. If passed, sharedExpertRankNum must be 0.
        - `epWorldSize`: Supported values: [2, 1024].
        - `moeExpertNum`: Value range: (0, 1024].
        - `groupTp` is not supported in the current version. Pass an empty string.
        - `tpWorldSize`: Not supported in the current version. Simply pass 0.
        - `tpRankId` is not supported in the current version. Pass 0.
        - `expertShardType`: Currently, only 0 is supported, indicating that shared expert cards are placed before MoE expert cards.
        - `sharedExpertNum`: Current value range: [0, 4].
        - `sharedExpertRankNum`: Value range: [0, epWorldSize). If the value is 0, sharedExpertNum must be 0 or 1; if the value is not 0, sharedExpertRankNum % sharedExpertNum = 0 must be satisfied.
        - `commQuantMode`: Value range: 0, 2, 3, or 4 (0 indicates non-quantized, 2 indicates int8 quantization, 3 indicates mxfp8 e5m2 quantization, 4 indicates mxfp8 e4m3 quantization).

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
    <td>The required input or output parameter Tensor is a null pointer.</td>
    </tr>
    <tr>
    <td>ACLNN_ERR_PARAM_INVALID</td>
    <td>161002</td>
    <td>The input and output data types are not in the supported range.</td>
    </tr>
    <tr>
    <td rowspan="2">ACLNN_ERR_INNER_TILING_ERROR</td>
    <td rowspan="2">561002</td>
    <td>The input and output shapes are not within the supported range.</td>
    </tr>
    <tr>
        <td>The parameter value is not within the supported range.</td>
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
        <th>Parameter</th>
        <th>Input/Output</th>
        <th>Description</th>
    </tr></thead>
    <tbody>
    <tr>
        <td>workspace</td>
        <td>Input</td>
        <td>Workspace memory address allocated on the device.</td>
    </tr>
    <tr>
        <td>workspaceSize</td>
        <td>Input</td>
        <td>Workspace size allocated on the device. Obtained by the first-phase API aclnnMoeDistributeCombineV2GetWorkspaceSize.</td>
    </tr>
    <tr>
        <td>executor</td>
        <td>Input</td>
        <td>Op executor, which contains the operator computation flow.</td>
    </tr>
    <tr>
        <td>stream</td>
        <td>Input</td>
        <td>Stream for task execution.</td>
    </tr>
    </tbody>
    </table>

- **Return Value**

    Returns an aclnnStatus status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- **Deterministic computation**:
  - aclnnMoeDistributeCombineV2 defaults to a deterministic implementation.

- **Driver Constraints**:
  - The driver versions of all nodes in the operator communication domain must be the same.

- **API Compatibility Constraints**:
  - The `aclnnMoeDistributeDispatchV2` API must be used together with the `aclnnMoeDistributeCombineV2` API. For details, see [Example](#example). The element values in the Tensor outputs `assistInfoForCombineOut`, `epRecvCountsOut`, and `expandScalesOut` of `aclnnMoeDistributeDispatchV2` may differ across different product models, communication algorithms, or versions. Simply pass the preceding Tensors to the corresponding parameters of `aclnnMoeDistributeCombineV2`. Other business logic of the model must not depend on them.

- **Parameter Consistency Constraints**:
  - The `groupEp`, `epWorldSize`, `moeExpertNum`, `expertShardType`, `sharedExpertNum`, `sharedExpertRankNum`, `globalBS`, and `commAlg` parameters and the `HCCL_BUFFSIZE` value used in the API call must be consistent on all devices and across different layers in the network, and must be consistent with the corresponding parameters of `aclnnMoeDistributeDispatchV2`.

- **Product-Specific Constraints**:

  <!-- npu="A3" id10 -->
  - <term>Atlas A3 products</term>: In this use case, a single device contains dual dies (referred to as "die"), so "local device" in the parameter description refers to a single die.

  <!-- end id10 -->

- **Shape Variable Constraints**:

  | Variable     | Definition and value range                                                     |
  | :----------- | :----------------------------------------------------------------------------- |
  | A            | Maximum number of tokens to be dispatched on the local device. Value ranges are as follows: <ul><li>For shared experts, must satisfy <code>A = BS \* epWorldSize \* sharedExpertNum / sharedExpertRankNum</code>.</li><li>For MoE experts, when globalBS is 0, must satisfy <code>A >= BS \* epWorldSize \* min(localExpertNum, K)</code>; when globalBS is not 0, must satisfy <code>A >= globalBS \* min(localExpertNum, K)</code>.</li></ul>|
  | H            |Indicates the hidden size:<ul><li> <term>Atlas A2 products</term>: (0, 10240] and a multiple of 32.</li><li><term>Atlas A3 products</term>, Ascend 950PR&950DT products: [1024, 8192]. </li></ul>|
  | BS           | Indicates batch sequence size (number of tokens finally output by the local device):<ul><li> <term>Atlas A2 products</term>: Depending on the value of commAlg, "fullmesh" supports (0, 256]; "hierarchy" supports (0, 512] when the driver version is ≥ 25.0.RC1.1;</li><li><term>Atlas A3 products</term>, Ascend 950PR&950DT products: 0 < BS ≤ 512.</li></ul> |
  | K            |Indicates selecting topK experts. Value range is (0 < K ≤ 16) and must satisfy (0 < K ≤ moeExpertNum). |
  | serverNum    | Number of server nodes:<br><term>Atlas A2 products</term>: This variable is used only by the shapes in this use case. Only 2, 4, and 8 are supported.
  | localExpertNum | Number of local card experts:<ul><li>For shared expert devices, localExpertNum = 1;</li><li>For MoE expert devices, localExpertNum = <code>moeExpertNum/(epWorldSize-sharedExpertRankNum)</code>. TP communication is not supported when localExpertNum > 1. </li><li><term>Atlas A3 products</term>, <term>Ascend 950PR&950DT products</term>: Must satisfy 0 < localExpertNum * epWorldSize ≤ 2048.</li></ul>|

- **Environment Variable Constraints**:
  - **HCCL_BUFFSIZE**: Before calling this API, check whether the value of the `HCCL_BUFFSIZE` environment variable is reasonable. This environment variable indicates the memory size occupied by a single communicator, in MB. It defaults to 200MB if not configured.

    <!-- npu="910b" id11 -->
    - <term>Atlas A2 products</term>:
      - If commAlg is "" or nullptr: Select the "fullmesh" or "hierarchy" formula based on HCCL environment variables.
      - If commAlg is "fullmesh": Set size requirement is (≥ 2 \* (BS \* epWorldSize \* min(localExpertNum, K) \* H \* sizeof(uint16) + 2MB)).
      - If commAlg is "hierarchy": Set size requirement is (≥ (`moeExpertNum` + `epWorldSize` / 4) \* Align512(`maxBS` \* (`H` \* 2 + 16 \* Align8(`K`))) \* 1B + 8MB, where Align8(x) = ((x + 8 - 1) / 8) \* 8 and Align512(x) = ((x + 512 - 1) / 512) \* 512.
    <!-- end id11 -->
    <!-- npu="A3" id12 -->
    - <term>Atlas A3 products</term>:
      - In the EP communicator: Set size requirement is (≥ 2) and must satisfy (≥ 2 \* (localExpertNum \* maxBS \* epWorldSize \* Align512(Align32(2 \* H) + 44) + (K + sharedExpertNum) \* maxBS \* Align512(2 \* H))) (`localExpertNum` must be the number of local card experts on the MoE expert device; `Align512(x) = ((x + 512 - 1) / 512) * 512`; `Align32(x) = ((x + 32 - 1) / 32) * 32`).
      - Within the TP communication domain: the value must be >= A \* (H \* 2 + 128) \* 2.
    <!-- end id12 -->
    <!-- npu="950" id13 -->
    - <term>Ascend 950PR&950DT products</term>: Set size requirement (≥ 2) and must satisfy (≥ 2 \* (localExpertNum \* maxBS \* epWorldSize \* Align512(Align32(2 \* H) + 44) + (K + sharedExpertNum) \* maxBS \* Align512(2 \* H))) (`localExpertNum` must be the number of local card experts on the MoE expert device; `Align512(x) = ((x + 512 - 1) / 512) * 512`; `Align32(x) = ((x + 32 - 1) / 32) * 32`).

    <!-- end id13 -->

  - **HCCL_INTRA_PCIE_ENABLE and HCCL_INTRA_ROCE_ENABLE**:

    <!-- npu="910b" id14 -->
    - <term>Atlas A2 products</term>: This environment variable is no longer recommended. Configure `commAlg` as "hierarchy" instead.
    <!-- end id14 -->
    <!-- npu="950,A3" id15 -->
    - <term>Atlas A3 products</term> and Ascend 950PR&950DT products: This environment variable is not supported.

    <!-- end id15 -->

- **Communication Domain Usage Constraints**:
  - In one model, `aclnnMoeDistributeCombineV2` and `aclnnMoeDistributeDispatchV2` support only the same EP communicator, and no other operators are allowed in this communicator.
  - TP domain communication is not supported currently.

  <!-- npu="A3" id16 -->
  - <term>Atlas A3 products</term>  : Nodes in a communicator must be in the same SuperPoD. Cross-SuperPoD communication is not supported.

  <!-- end id16 -->

- **Communication method constraints**:

  <!-- npu="950" id17 -->
  - <term>Ascend 950PR&950DT products</term>: Only UB Memory communication is supported.

  <!-- end id17 -->

- **Networking Constraints**:

  <!-- npu="910b" id18 -->
  - <term>Atlas A2 products</term>: In multi-machine scenarios, only switch networking is supported. Direct-connect networking between two machines is not supported.

  <!-- end id18 -->

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

<!-- npu="910b" id19 -->
- <term>Atlas A2 products</term>:

    No need to configure the ranktable file or the environment variables RANK_TABLE_FILE and FIRST_RANK_ID.

    This example supports running the A2 operator in a single-node environment with [2, 8] devices. Before running, set IS_TEST_A2 to true in the sample code to ensure that the A2 branch is executed.
    In addition, set EP_WORLD_SIZE_A2 to the number of devices in the sample code as needed, and change moeExpertNum in the launchOneThreadDispatchV2AndCombineV2_A2 function so that moeExpertNum is divisible by EP_WORLD_SIZE_A2.

    The operator build commands are as follows. Both the moe_distribute_dispatch_v2 and moe_distribute_combine_v2 operators must be built, and the two operators must be executed in pairs:

    ```bash
    bash build.sh --pkg --soc=ascend910b --ops=moe_distribute_dispatch_v2,moe_distribute_combine_v2
    ```

    The command to run the example operators is as follows:

    ```bash
    bash build.sh --run_example --ops=moe_distribute_combine_v2 eager cust
    ```

<!-- end id19 -->
<!-- npu="950,A3" id20 -->
- <term>Atlas A3 products</term>, <term>Ascend 950PR&950DT products</term>:

    No need to configure the ranktable file or the environment variables RANK_TABLE_FILE and FIRST_RANK_ID.

<!-- end id20 -->

The following sample code is for reference only. For the build and execution process, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

<!-- npu="950,A3,910b" id21 -->
- <term>Atlas A2 products</term>, <term>Atlas A3 products</term>, <term>Ascend 950PR&950DT products</term>:

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
            // Moe expert device
            localExpertNum = moeExpertNum / (EP_WORLD_SIZE - sharedExpertRankNum);
            A = globalBS * (localExpertNum < K ? localExpertNum : K);
        }
        std::string commAlg = "";

        /* Construct device-side input and output variables based on the current use case.*/
        // Declare device-side input and output variables
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

        // Define the dimensions of each variable in the current use case
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

        // Construct host-side variables
        std::vector<op::fp16_t> xHostData(xShapeSize, 1);
        std::vector<int32_t> expertIdsHostData;
        for (int32_t token_id = 0; token_id < expertIdsShape[0]; token_id++) {
            // Each token is sent to Moe experts {0, 1, ... k - 1}
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

        // Construct device-side variables
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

        /* Declare variables required for operator execution. */
        uint64_t dispatchV2WorkspaceSize = 0;
        aclOpExecutor *dispatchV2Executor = nullptr;
        void *dispatchV2WorkspaceAddr = nullptr;

        uint64_t combineV2WorkspaceSize = 0;
        aclOpExecutor *combineV2Executor = nullptr;
        void *combineV2WorkspaceAddr = nullptr;

        /* Execute the dispatchV2 and combineV2 operators in sequence. */
        // Call the first-phase API of the dispatchV2 operator
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
        // Allocate device memory based on the workspaceSize calculated by the first-phase API of dispatchV2.
        if (dispatchV2WorkspaceSize > 0) {
            ret = aclrtMalloc(&dispatchV2WorkspaceAddr, dispatchV2WorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc failed. ret = %d\n", ret); return ret);
        }
        // Call the second-phase API of dispatchV2.
        ret = aclnnMoeDistributeDispatchV2(dispatchV2WorkspaceAddr, dispatchV2WorkspaceSize, dispatchV2Executor, args.dispatchV2Stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnMoeDistributeDispatchV2 failed. ret = %d\n", ret); return ret);
        // (Boilerplate) Wait for task execution to complete synchronously.
        ret = aclrtSynchronizeStreamWithTimeout(args.dispatchV2Stream, 10000);
        CHECK_RET(
            ret == ACL_SUCCESS,
            LOG_PRINT("[ERROR] aclrtSynchronizeStreamWithTimeout failed. ret = %d\n", ret); return ret
        );

        // Call the first-phase API of combineV2.
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
        // Allocate device memory based on the workspaceSize calculated by the first-phase API of combineV2.
        if (combineV2WorkspaceSize > 0) {
            ret = aclrtMalloc(&combineV2WorkspaceAddr, combineV2WorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc failed. ret = %d\n", ret); return ret);
        }
        // Call the second-phase API of combineV2.
        ret = aclnnMoeDistributeCombineV2(combineV2WorkspaceAddr, combineV2WorkspaceSize, combineV2Executor, args.combineV2Stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnMoeDistributeCombineV2 failed. ret = %d\n", ret); return ret);
        // (Boilerplate) Wait for task execution to complete synchronously.
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
        int64_t zeroExpertNum = 0;
        int64_t copyExpertNum = 0;
        int64_t constExpertNum = 0; // A3
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
        aclTensor *groupList = nullptr; // Reserved parameter.

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
        std::vector<int64_t> epRecvCountsShape{localExpertNum * EP_WORLD_SIZE_A2}; // Not layered.
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
        // (Boilerplate) Synchronize and wait for task execution to complete.
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
        // Initialize the ep communicator.
        HcclComm commsEp[EP_WORLD_SIZE];
        ret = HcclCommInitAll(EP_WORLD_SIZE, devicesEp, commsEp);
        CHECK_RET(
            ret == ACL_SUCCESS,
            LOG_PRINT("[ERROR] HcclCommInitAll ep failed. ret = %d\n", ret); return ret
        );

        Args args[DEV_NUM];
        // Each thread calls the operator on each device.
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
            LOG_PRINT("[ERROR] Please check whether environment variable ENV_DEV_NUM is set correctly.\n");
            LOG_PRINT("[WARNING] For details related to ENV_DEV_NUM, see aclnnMoeDistributeCombineV2.md.\n");
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

<!-- end id21 -->
