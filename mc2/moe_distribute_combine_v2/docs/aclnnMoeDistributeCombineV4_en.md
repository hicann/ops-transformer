# aclnnMoeDistributeCombineV4

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T14:10:22.852Z -->

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

- Description: Performs AllToAllV communication and then combines the received data. If expertScales is not empty, the weights are multiplied before summing; if it is empty, the outputs are summed directly. TP domain communication is not supported.

    Compared with the `aclnnMoeDistributeCombineV3` API, this API has the following changes:

    Adds the communication overhead collection function to record the communication time of each device. This feature is enabled by passing in the `performanceInfoOptional` parameter. It is recommended to use this function with the [DeepXTrace](https://github.com/antgroup/DeepXTrace) tool. In a single operator call, the communication overhead of each device is accumulated to this tensor. Clear the tensor as needed before use.

- Formula:

    $$
    ataOut = AllToAllV(expandX)\\
    xOut = Sum(expertScales * ataOut + expertScales * sharedExpertX)
    $$

    If `expertScales` is an empty tensor, weight multiplication is skipped and the expert outputs are directly summed.

> NOTE The API must be used together with `aclnnMoeDistributeDispatchV4`, which means data is returned along the same path used by `aclnnMoeDistributeDispatchV4` to collect data.

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call the "aclnnMoeDistributeCombineV4GetWorkspaceSize" API first to obtain the workspace size required for computation and the executor that contains the operator computation process, and then call the "aclnnMoeDistributeCombineV4" API to execute the computation.

```cpp
aclnnStatus aclnnMoeDistributeCombineV4GetWorkspaceSize(
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
aclnnStatus aclnnMoeDistributeCombineV4(
    void*           workspace,
    uint64_t        workspaceSize,
    aclOpExecutor*  executor,
    aclrtStream     stream)
```

## aclnnMoeDistributeCombineV4GetWorkspaceSize

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
    <th>Shape</th>
    <th>Non-contiguous Tensor</th>
    </tr>
    </thead>
    <tbody>
    <tr>
    <td>expandX</td>
    <td>Input</td>
    <td>Token features expanded based on <code>expertIds</code>.</td>
    <td>Required to be a 2D tensor.</td>
    <td>FLOAT16, BFLOAT16</td>
    <td>ND</td>
    <td><code>(A, H)</code></td>
    <td>√</td>
    </tr>
    <tr>
    <td>expertIds</td>
    <td>Input</td>
    <td>TopK expert indices of each token.</td>
    <td>Required to be a 2D tensor.</td>
    <td>INT32</td>
    <td>ND</td>
    <td><code>(BS, K)</code></td>
    <td>√</td>
    </tr>
    <tr>
    <td>assistInfoForCombine</td>
    <td>Input</td>
    <td>Corresponds to the <code>assistInfoForCombineOut</code> output of <code>aclnnMoeDistributeDispatchV4</code>.</td>
    <td>Required to be a 1D tensor.</td>
    <td>INT32</td>
    <td>ND</td>
    <td><code>(A * 128, )</code></td>
    <td>√</td>
    </tr>
    <tr>
    <td>epSendCounts</td>
    <td>Input</td>
    <td>Corresponds to the <code>epRecvCounts</code> output of <code>aclnnMoeDistributeDispatchV4</code>.</td>
    <td>Required to be a 1D Tensor.</td>
    <td>INT32</td>
    <td>ND</td>
    <td>-</td>
    <td>√</td>
    </tr>
    <tr>
    <td>expertScales</td>
    <td>Input</td>
    <td>Weights of the topK experts of each token. Empty tensors are supported. When an empty tensor is passed, no weighting is performed and the expert outputs are directly summed.</td>
    <td>Required to be a 2D tensor when non-empty. An empty tensor must have at least one dimension of 0, and the number of dimensions is not fixed.</td>
    <td>FLOAT32</td>
    <td>ND</td>
    <td>Not empty: <code>(BS, K)</code>; empty tensor: at least one dimension is 0</td>
    <td>√</td>
    </tr>
    <tr>
    <td>tpSendCountsOptional</td>
    <td>Input</td>
    <td>Corresponds to the <code>tpRecvCounts</code> output of <code>aclnnMoeDistributeDispatchV4</code>.</td>
    <td>Reserved parameter. TP domain communication is no longer supported. Pass a null pointer.</td>
    <td>INT32</td>
    <td>ND</td>
    <td>-</td>
    <td>√</td>
    </tr>
    <tr>
    <td>xActiveMaskOptional</td>
    <td>Input</td>
    <td>Indicates whether the token participates in communication.</td>
    <td><ul><li>Required to be a 1D or 2D tensor. Valid data or a null pointer can be passed. By default, all tokens participate in communication.</li><li>When BS differs across devices, all tokens must be valid.</li></ul></td>
    <td>BOOL</td>
    <td>ND</td>
    <td>If the input is 1D, the shape is <code>(BS, )</code>; if the input is 2D, the shape is <code>(BS, K)</code>.</td>
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
    <td>Corresponds to the <code>expandScales</code> output of <code>aclnnMoeDistributeDispatchV4</code>.</td>
    <td>-</td>
    <td>FLOAT32</td>
    <td>ND</td>
    <td><code>(A, )</code></td>
    <td>√</td>
    </tr>
    <tr>
    <td>sharedExpertXOptional</td>
    <td>Input</td>
    <td>Indicates the tokens calculated by the shared expert.</td>
    <td>-</td>
    <td>FLOAT16, BFLOAT16</td>
    <td>ND</td>
    <td><code>(BS, H)</code></td>
    <td>√</td>
    </tr>
    <tr>
    <td>elasticInfoOptional</td>
    <td>Input</td>
    <td>Dynamic scale-in information of the EP communicator.</td>
    <td>-</td>
    <td>INT32</td>
    <td>ND</td>
    <td>-</td>
    <td>√</td>
    </tr>
    <tr>
    <td>oriXOptional</td>
    <td>Input</td>
    <td>Indicates the token data that has not gone through the FFN (Feed-Forward Neural Network).</td>
    <td>This input is required when copyExpert or constExpert is enabled. Valid data or a null pointer can be passed. When <code>copyExpertNum</code> is not 0 or <code>constExpertNum</code> is not 0, valid input must be passed. When valid data is passed, it must be a 2D tensor whose data type is the same as that of <code>expandX</code>.</td>
    <td>FLOAT16, BFLOAT16</td>
    <td>ND</td>
    <td><code>(BS, H)</code></td>
    <td>√</td>
    </tr>
    <tr>
    <td>constExpertAlpha1Optional</td>
    <td>Input</td>
    <td>Calculation coefficient required as input when constExpert is enabled.</td>
    <td>-</td>
    <td>FLOAT16, BFLOAT16</td>
    <td>ND</td>
    <td>-</td>
    <td>√</td>
    </tr>
    <tr>
    <td>constExpertAlpha2Optional</td>
    <td>Input</td>
    <td>Computation coefficient required as input when constExpert is enabled.</td>
    <td>-</td>
    <td>FLOAT16, BFLOAT16</td>
    <td>ND</td>
    <td>-</td>
    <td>√</td>
    </tr>
    <tr>
    <td>constExpertVOptional</td>
    <td>Input</td>
    <td>aclTensor on the device. Computation coefficient required as input when constExpert is enabled.</td>
    <td>-</td>
    <td>FLOAT16, BFLOAT16</td>
    <td>ND</td>
    <td>-</td>
    <td>√</td>
    </tr>
    <tr>
    <td>performanceInfoOptional</td>
    <td>Input</td>
    <td>Indicates the communication time for the current device to wait for data from all devices, in us (microseconds).</td>
    <td>In a single operator call, the communication overhead of each device is accumulated on this Tensor. The operator does not automatically clear it to zero. Therefore, clear the Tensor to zero each time before enabling it to start recording the overhead.</td>
    <td>INT64</td>
    <td>ND</td>
    <td>-</td>
    <td>√</td>
    </tr>
    <tr>
    <td>groupEp</td>
    <td>Input</td>
    <td>Name of the EP communicator (expert parallel communicator).</td>
    <td>String length range: <code>[1, 128)</code>.</td>
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
    <td>Local device ID in the EP communicator.</td>
    <td>Value range: <code>[0, epWorldSize)</code>. The <code>epRankId</code> of each device in the same EP communicator must be unique.</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>moeExpertNum</td>
    <td>Input</td>
    <td>Number of MoE experts.</td>
    <td>Satisfies <code>moeExpertNum % (epWorldSize - sharedExpertRankNum) = 0</code>.</td>
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
    <td>TP communicator size (reserved parameter, not supported in the current version, only 0 or 1 allowed).</td>
    <td>-</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>tpRankId</td>
    <td>Input</td>
    <td>Local device ID in the TP domain.</td>
    <td>Reserved parameter, not supported in the current version. Pass 0.</td>
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
    <td>Number of devices for shared experts (a shared expert can be replicated across multiple devices).</td>
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
    <td><ul><li>If all ranks have the same BS, <code>globalBS = BS * epWorldSize</code> or 0.</li><li>If the BS values differ across ranks, <code>globalBS = maxBS * epWorldSize</code> (maxBS is the upper limit of the single-rank BS capacity configured by the service, and must be no less than the actual BS of any rank). All ranks must pass in the same non-zero globalBS.</li></ul></td>
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
    <td>Not supported in the current version. Pass 0.</td>
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
    <td>Required to be a 2D tensor, with the same data type/format as <code>expandX</code>.</td>
    <td>FLOAT16, BFLOAT16</td>
    <td>ND</td>
    <td><code>(BS, H)</code></td>
    <td>-</td>
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
    <td>Returns the op executor that contains the operator computation flow.</td>
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

    - commAlg supports nullptr, "", "fullmesh", and "hierarchy". "hierarchy" is recommended together with a driver of version 25.0.RC1.1 or later. For nullptr and "", the algorithm is selected based on HCCL environment variables (not recommended). For "fullmesh", tokens are transferred directly over RDMA. For "hierarchy", data is sent twice (intra-node and cross-node) to reduce the amount of data transferred across nodes.
    - The shared expert scenario is not supported.
    - The shape of epSendCounts is (moeExpertNum + 2 \* globalBS \* K \* serverNum, ), where K indicates the number of topK experts. The first moeExpertNum numbers indicate the number of tokens received from each device in the EP communicator. The last 2 \* globalBS \* K \* serverNum numbers are used to store the number of tokens on which combine can perform reduce in advance before inter-node/intra-node communication, and the offsets in the communication area. If globalBS is 0, the value is calculated as BS \* epWorldSize.
    - TP domain communication is not supported currently.
    - xActiveMaskOptional depends on the value of commAlg. For "fullmesh", the parameter is required to be a 1D tensor with shape (BS, ), and true must be placed before false (for example, {true, false, true} is invalid). For "hierarchy", the parameter is not supported in the current version; pass a null pointer.
    - expandScalesOptional is required to be a 1D tensor with shape (A, ).
    - sharedExpertXOptional is a reserved parameter and is not supported in the current version; pass a null pointer.
    - epWorldSize depends on the value of commAlg. For "fullmesh", 2, 3, 4, 5, 6, 7, 8, 16, 32, 64, 128, 192, 256, and 384 are supported. For "hierarchy", 16, 32, and 64 are supported.
    - moeExpertNum depends on the value of commAlg. For "fullmesh", the value range is (0, 1024]. For "hierarchy", the value range is (0, 512].
    - groupTp is not supported in the current version; pass an empty string.
    - tpWorldSize is not supported in the current version; pass 0.
    - tpRankId is not supported in the current version. Pass 0.
    - expertShardType is not supported in the current version. Pass 0.
    - sharedExpertNum is not supported in the current version. Pass 0.
    - sharedExpertRankNum is not supported in the current version. Pass 0.
    - commQuantMode: value range is 0 or 2 (0 indicates no quantization, and 2 indicates int8 quantization). The value 2 is only supported when commAlg is "hierarchy", or when HCCL_INTRA_PCIE_ENABLE=1, HCCL_INTRA_ROCE_ENABLE=0, and the driver version is ≥ 25.0.RC1.1.
    - expandScalesOptional is required to be a 1D tensor.
    - elasticInfoOptional is not supported in the current version. Pass a null pointer.
    - oriXOptional is not supported in the current version when commAlg="hierarchy". Pass a null pointer.
    - constExpertAlpha1Optional is a reserved parameter. It is not supported in the current version. Pass a null pointer.
    - constExpertAlpha2Optional is a reserved parameter. It is not supported in the current version. Pass a null pointer.
    - constExpertVOptional is a reserved parameter and is not supported in the current version. Pass a null pointer.
    - zeroExpertNum: when commAlg="fullmesh", the value range is [0, MAX_INT32), where MAX_INT32 = 2^31 - 1. The value of a valid zero expert ID is in the range [<code>moeExpertNum</code>, <code>moeExpertNum + zeroExpertNum</code>).
    - copyExpertNum: when commAlg="fullmesh", the value range is [0, MAX_INT32), where MAX_INT32 = 2^31 - 1. The value of a valid copy expert ID is in the range [<code>moeExpertNum + zeroExpertNum</code>, <code>moeExpertNum + zeroExpertNum + copyExpertNum</code>).
    - constExpertNum is not supported in the current version. Pass 0.
    - performanceInfoOptional: valid data or a null pointer can be passed. Passing a null pointer indicates that recording of communication overhead is not enabled. When valid data is passed, the tensor is required to be a 1D tensor whose shape is (epWorldSize,). The supported data type is int64, and the required data format is ND.
    </details>
    <!-- end id7 -->

    <!-- npu="A3" id8 -->
    <details>
    <summary><term>Atlas A3 products</term>  :</summary>

    - expertScales: only valid tensors with shape (BS, K) are supported for weighted aggregation. Empty tensors are not supported.
    - commAlg supports two input values: "" and "hierarchy". "": the default value, indicating that the hierarchy cross-SuperPoD template is not enabled. "hierarchy": enables the cross-SuperPoD template, which supports different BS values across ranks. This template only supports use cases where tpWorldSize is 1 and the number of shared experts is 0, and does not support 2D mask, special experts, or the performanceInfo use case. For variable BS use cases, the same non-zero globalBS capacity must be configured for all ranks, and xActiveMaskOptional must not be passed.
    - epSendCounts: the shape is (epWorldSize \* localExpertNum, ).
    - tpSendCountsOptional is a reserved parameter. TP domain communication is no longer supported; pass a null pointer.
    - xActiveMaskOptional is required to be a 1D or 2D Tensor (1D shape: (BS, ); 2D shape: (BS, K)). For 1D tensors, true must be placed before false. For 2D tensors, a token does not participate in communication if all of its K values are false.
    - expandScalesOptional is a reserved parameter and is not supported in the current version; pass a null pointer.
    - sharedExpertXOptional is required to be a 2D or 3D Tensor (2D shape: (BS, H); for 3D, the product of the first two dimensions equals BS and the third dimension equals H). It is optional. If passed, sharedExpertRankNum must be 0.
    - epWorldSize has a value range of [2, 768]. In the commAlg="hierarchy" use case, the value range is [16, 256] and the value must be an integer multiple of 16.
    - moeExpertNum has a value range of (0, 1024]. In the commAlg="hierarchy" use case, the value range is (0, 512].
    - groupTp is a reserved parameter. TP domain communication is no longer supported; passing null is only supported when TP domain communication is not used.
    - tpWorldSize is a reserved parameter. TP domain communication is no longer supported; only 0 or 1 is allowed.
    - tpRankId is a reserved parameter. TP domain communication is no longer supported; pass 0.
    - expertShardType only supports 0 in the current version, indicating that shared expert devices are placed before MoE expert devices.
    - sharedExpertNum: current value range [0, 4].
    - sharedExpertRankNum: value range [0, epWorldSize). If the value is 0, sharedExpertNum must be 0 or 1. If the value is not 0, sharedExpertRankNum % sharedExpertNum = 0 must be satisfied.
    - commQuantMode: value range 0 or 2 (0 indicates no quantization and 2 indicates int8 quantization). The value 2 is only supported when tpWorldSize < 2.
    - expandScalesOptional: reserved parameter, not supported in the current version. Pass a null pointer.
    - elasticInfoOptional: reserved parameter, not supported in the current version. Pass a null pointer.
    - constExpertAlpha1Optional: can be set to valid data or a null pointer. Valid input is required when constExpertNum is not 0. If valid data is passed, it must be a 2D Tensor with shape <code>(constExpertNum, H)</code>, and its data type must be the same as that of expandX.
    - constExpertAlpha2Optional: can be set to valid data or a null pointer. Valid input is required when constExpertNum is not 0. If valid data is passed, it must be a 2D Tensor with shape <code>(constExpertNum, H)</code>, and its data type must be the same as that of expandX.
    - constExpertVOptional: can be set to valid data or a null pointer. Valid input is required when constExpertNum is not 0. If valid data is passed, it must be a 2D Tensor with shape <code>(constExpertNum, H)</code>, and its data type must be the same as that of expandX.
    - zeroExpertNum: value range [0, MAX_INT32), where MAX_INT32 = 2^31 - 1. The value of the ID of a valid zero expert is [<code>moeExpertNum</code>, <code>moeExpertNum + zeroExpertNum</code>).
    - copyExpertNum: value range [0, MAX_INT32), where MAX_INT32 = 2^31 - 1. The value of the ID of a valid copy expert is [<code>moeExpertNum + zeroExpertNum</code>, <code>moeExpertNum + zeroExpertNum + copyExpertNum</code>).
    - constExpertNum value range: [0, MAX_INT32), where MAX_INT32 = 2^31 - 1. The value is the ID of valid constant experts: [<code>moeExpertNum + zeroExpertNum + copyExpertNum</code>, <code>moeExpertNum + zeroExpertNum + copyExpertNum + constExpertNum</code>).
    - performanceInfoOptional can be set to valid data or a null pointer. Passing a null pointer indicates that the function of recording communication overhead is disabled. When valid data is passed, it is required to be a 1D tensor with shape (epWorldSize,); the supported data type is int64, and the data format is required to be ND.
    </details>
    <!-- end id8 -->

    <!-- npu="950" id9 -->
    <details>
    <summary><term>Ascend 950PR&950DT products</term>:</summary>

    - expertScales can be used to optionally enable the expert weight function. Passing a valid tensor with shape (BS, K) for expertScales enables the function and performs weighted summation; passing an empty tensor disables the function and directly sums the expert outputs.
    - commAlg is not supported in the current version; simply pass a null pointer.
    - The shape of epSendCounts is (epWorldSize \* localExpertNum, ).
    - TP domain communication is not supported currently.
    - xActiveMaskOptional is required to be a 1D or 2D tensor (shape (BS, ) for 1D and shape (BS, K) for 2D). For 1D, true must be placed before false; for 2D, a token does not participate in communication if all of the K values corresponding to the token are false.
    - expandScalesOptional is a reserved parameter, which is not supported in the current version; simply pass a null pointer.
    - sharedExpertXOptional is required to be a 2D or 3D tensor (shape (BS, H) for 2D; for 3D, the product of the first two dimensions is BS and the third dimension is H). This parameter can be passed or omitted; if it is passed, sharedExpertRankNum must be 0.
    - The supported value range of epWorldSize is [2, 1024].
    - The value range of moeExpertNum is (0, 1024].
    - groupTp is not supported in the current version; pass an empty string.
    - tpWorldSize is not supported in the current version; pass 0.
    - tpRankId is not supported in the current version; pass 0.
    - expertShardType only supports passing 0 in the current version, indicating that shared expert devices are placed before MoE expert devices.
    - The current value range of sharedExpertNum is [0, 4].
    - The value range of sharedExpertRankNum is [0, epWorldSize); when the value is 0, sharedExpertNum must be 0 or 1; when the value is not 0, sharedExpertRankNum % sharedExpertNum = 0 must be satisfied.
    - The value range of commQuantMode is 0, 2, 3, or 4 (0 indicates no quantization, 2 indicates int8 quantization, 3 indicates mxfp8 quantization (e5m2), and 4 indicates mxfp8 quantization (e4m3)).
    - expandScalesOptional is a reserved parameter and is not supported in the current version; pass a null pointer.
    - elasticInfoOptional is a reserved parameter, which is not supported in the current version. Pass a null pointer.
    - constExpertAlpha1Optional can be set to valid data or a null pointer. Valid input must be passed when constExpertNum is not 0. If valid data is passed, it must be a 2D tensor with shape <code>(constExpertNum, H)</code>, and its data type must be the same as that of expandX.
    - constExpertAlpha2Optional can be set to valid data or a null pointer. Valid input must be passed when constExpertNum is not 0. If valid data is passed, it must be a 2D tensor with shape <code>(constExpertNum, H)</code>, and its data type must be the same as that of expandX.
    - constExpertVOptional can be set to valid data or a null pointer. Valid input must be passed when constExpertNum is not 0. If valid data is passed, it must be a 2D tensor with shape <code>(constExpertNum, H)</code>, and its data type must be the same as that of expandX.
    - zeroExpertNum value range: [0, MAX_INT32), where MAX_INT32 = 2^31 - 1. Valid zero expert IDs are [<code>moeExpertNum</code>, <code>moeExpertNum + zeroExpertNum</code>).
    - copyExpertNum value range: [0, MAX_INT32), where MAX_INT32 = 2^31 - 1. Valid copy expert IDs are [<code>moeExpertNum + zeroExpertNum</code>, <code>moeExpertNum + zeroExpertNum + copyExpertNum</code>).
    - constExpertNum value range: [0, MAX_INT32), where MAX_INT32 = 2^31 - 1. Valid const expert IDs are [<code>moeExpertNum + zeroExpertNum + copyExpertNum</code>, <code>moeExpertNum + zeroExpertNum + copyExpertNum + constExpertNum</code>).
    - performanceInfoOptional can be set to valid data or a null pointer. Passing a null pointer indicates that the communication overhead recording function is disabled. If valid data is passed, it must be a 1D tensor with shape (epWorldSize,). The supported data type is int64, and the data format is required to be ND.
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
    <th>Return Value</th>
    <th>Error Code</th>
    <th>Description</th>
    </tr>
    </thead>
    <tbody>
    <tr>
    <td>ACLNN_ERR_PARAM_NULLPTR</td>
    <td>161001</td>
    <td>A required input or output tensor is a null pointer.</td>
    </tr>
    <tr>
    <td>ACLNN_ERR_PARAM_INVALID</td>
    <td>161002</td>
    <td>The data type of an input or output is out of the supported range.</td>
    </tr>
    <tr>
    <td rowspan="2">ACLNN_ERR_INNER_TILING_ERROR</td>
    <td rowspan="2">561002</td>
    <td>The shapes of the input and output are not within the supported range.</td>
    </tr>
    <tr>
        <td>The value of the parameter is not within the supported range.</td>
    </tr>
    </tbody>
    </table>

## aclnnMoeDistributeCombineV4

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
        <td>Size of the workspace allocated on the device, obtained by calling the first-phase API aclnnMoeDistributeCombineV4GetWorkspaceSize.</td>
    </tr>
    <tr>
        <td>executor</td>
        <td>Input</td>
        <td>Op executor, which contains the operator computation flow.</td>
    </tr>
    <tr>
        <td>stream</td>
        <td>Input</td>
        <td>Specifies the Stream for executing tasks.</td>
    </tr>
    </tbody>
    </table>

- **Return Value**

    Returns an aclnnStatus status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- **Deterministic computation**:
  - aclnnMoeDistributeCombineV4 defaults to a deterministic implementation.

- **Driver Constraints**:
  - The driver versions of all nodes in the operator communication domain must be the same.

- **API Compatibility Constraints**:

  - `aclnnMoeDistributeDispatchV4` and `aclnnMoeDistributeCombineV4` must be used together. The `assistInfoForCombineOut`, `epRecvCountsOut`, `tpRecvCountsOut`, and `expandScalesOut` output by the former must be directly passed to the corresponding parameters of the latter. The business logic must not rely on the specific values of these tensors.

- **Parameter Consistency Constraints**:

  - The `groupEp`, `epWorldSize`, `moeExpertNum`, `groupTp`, `tpWorldSize`, `expertShardType`, `sharedExpertNum`, `sharedExpertRankNum`, `globalBS`, `commAlg` parameters and the `HCCL_BUFFSIZE` value must be consistent across all cards, and must be consistent with the corresponding parameters of `aclnnMoeDistributeDispatchV4`.

- **Product-Specific Constraints**:

  <!-- npu="A3" id10 -->
  - <term>Atlas A3 products</term>: Each device contains two DIEs (dies/bare dies), and "this device" in Parameters refers to a single DIE.

  <!-- end id10 -->

- **Shape Variable Constraints**:

  | Variable     | Definition and Value Range                                                     |
  | :----------- | :----------------------------------------------------------------------------- |
  | A            | Maximum number of tokens to be dispatched by this device. The value range is as follows: <ul><li>For shared experts, <code>A = BS \* epWorldSize \* sharedExpertNum / sharedExpertRankNum</code> must be satisfied.</li><li>For MoE experts, when globalBS is 0, <code>A >= BS \* epWorldSize \* min(localExpertNum, K)</code> must be satisfied; when globalBS is non-0, <code>A >= globalBS \* min(localExpertNum, K)</code> must be satisfied.</li></ul>|
  | H            | Indicates the hidden size:<ul><li><term>Atlas A2 products</term>: (0, 10240] and a multiple of 32.</li><li><term>Atlas A3 products</term>: Depends on the value of commAlg. When commAlg is "hierarchy", the value range of H is [1024, 7168] and H must be a multiple of 32. In other use cases, the value range is [1024, 8192].</li><li><term>Ascend 950PR&950DT products</term>: The value range is [1024, 8192].</li></ul> |
  | BS           | Final number of output tokens of this device:<ul><li><term>Atlas A2 products</term>: Depends on the value of commAlg. "fullmesh" supports (0, 256]; "hierarchy" supports (0, 512] when the driver version is ≥ 25.0.RC1.1.</li><li><term>Atlas A3 products</term>: Depends on the value of commAlg. The value range is (0 < BS ≤ 256) for "fullmesh_v2" and "hierarchy", and (0 < BS ≤ 512) for "fullmesh_v1" and "".</li><li><term>Ascend 950PR&950DT products</term>: 0 < BS ≤ 512.</li></ul>|
  | K | Number of selected topK experts. The value range is 0 < K ≤ 16, and <code>0 < K ≤ moeExpertNum + zeroExpertNum + copyExpertNum + constExpertNum</code>. See the product-specific constraints below. |
  | serverNum    | Number of server nodes:<br>Atlas A2 products: This variable is used only in the shape of this use case. Only 2, 4, and 8 are supported.
  | localExpertNum | Number of experts on this device:<ul><li>For a shared expert device, localExpertNum = 1.</li><li>For an MoE expert device, localExpertNum = <code>moeExpertNum/(epWorldSize-sharedExpertRankNum)</code>. TP communication is not supported when localExpertNum > 1.</li><li><term>Atlas A3 products</term>: 0 < localExpertNum \* epWorldSize ≤ 2048 must be satisfied. When commAlg is "hierarchy", localExpertNum ≤ 24 and localExpertNum \* epWorldSize ≤ 512 must be satisfied.</li><li><term>Ascend 950PR&950DT products</term>: 0 < localExpertNum \* epWorldSize ≤ 2048 must be satisfied.</li></ul>|

  <!-- npu="A3" id23 -->
  - K: <term>Atlas A3 products</term>: When `commAlg` is "fullmesh_v2", the value range is 0 < K ≤ 16.
  <!-- end id23 -->
  <!-- npu="950" id24 -->
  - K: <term>Ascend 950PR&950DT products</term>: The ordinary MTE path (`commAlg` is nullptr or "", non-hierarchical and without AddRmsNorm fusion) supports 0 < K ≤ 32; other paths retain 0 < K ≤ 16.
  <!-- end id24 -->

- **Environment Variable Constraints**:
  - **HCCL_BUFFSIZE**: Before calling this API, check whether the value of the HCCL_BUFFSIZE environment variable is reasonable. This environment variable indicates the memory size occupied by a single communication domain, in MB. The default value is 200 MB when not configured.

      <!-- npu="910b" id11 -->
      - <term>Atlas A2 products</term>:
        - commAlg is configured as "" or nullptr: Based on the HCCL_INTRA_PCIE_ENABLE and HCCL_INTRA_ROCE_ENABLE environment variable configuration, selects the "fullmesh" or "hierarchy" formula.
        - commAlg configured as "fullmesh": required to be <code>= 2 \* (BS \* epWorldSize \* min(localExpertNum, K) \* H \* sizeof(uint16) + 2MB)</code>.
        - commAlg is configured as "hierarchy": Must be >= (`moeExpertNum` + `epWorldSize` / 4) \* Align512(`maxBS` \* (`H` \* 2 + 16 \* Align8(`K`))) \* 1B + 8MB, where Align8(x) = ((x + 8 - 1) / 8) \* 8, Align512(x) = ((x + 512 - 1) / 512) \* 512.
      <!-- end id11 -->
      <!-- npu="A3" id12 -->
      - <term>Atlas A3 products</term>:
        - When commAlg is configured as "" or not configured: required to be <code>>= 2 and satisfy >= 2 \* (localExpertNum \* maxBS \* epWorldSize \* Align512(Align32(2 \* H) + 44) + (K + sharedExpertNum) \* maxBS \* Align512(2 \* H))</code>, where <code>localExpertNum</code> must use the number of experts on the local MoE expert card, and <code>Align512(x) = ((x + 512 - 1) / 512) \* 512, Align32(x) = ((x + 32 - 1) / 32) \* 32</code>.
        - When commAlg is configured as "hierarchy": the value must satisfy <code>(moeExpertNum \* maxBS \* (H \* 2 + (3 \* (K + 7) / 8 \* 8)) \* 4 + 64) + 404 \* 1024 \* 1024</code>.
      <!-- end id12 -->
      <!-- npu="950" id13 -->
      - <term>Ascend 950PR&950DT products</term>:
        - When commAlg is configured as "" or not configured: required to be <code>>= 2 and satisfy >= 2 \* (localExpertNum \* maxBS \* epWorldSize \* Align512(Align32(2 \* H) + 44) + (K + sharedExpertNum) \* maxBS \* Align512(2 \* H))</code>, where <code>localExpertNum</code> must use the number of experts on the local MoE expert card, and <code>Align512(x) = ((x + 512 - 1) / 512) \* 512, Align32(x) = ((x + 32 - 1) / 32) \* 32</code>.

      <!-- end id13 -->

  - **HCCL_INTRA_PCIE_ENABLE/HCCL_INTRA_ROCE_ENABLE**:

    <!-- npu="910b" id14 -->
    - <term>Atlas A2 products</term>: These environment variables are no longer recommended. It is recommended to configure commAlg as "hierarchy".
    <!-- end id14 -->
    <!-- npu="950,A3" id15 -->
    - <term>Atlas A3 products</term> and Ascend 950PR&950DT products: This environment variable is not supported.
    <!-- end id15 -->

  - **HCCL_LOGIC_SUPERPOD_ID**

    <!-- npu="A3" id16 -->
    - <term>Atlas A3 products</term>: When commAlg is set to "hierarchy", this environment variable must be configured based on different SuperPoDs. For example, the two servers are set to `export HCCL_LOGIC_SUPERPOD_ID=0` and `export HCCL_LOGIC_SUPERPOD_ID=1` respectively.

    <!-- end id16 -->

- **Communication Domain Usage Constraints**:
  - The `aclnnMoeDistributeCombineV4` series operators and `aclnnMoeDistributeDispatchV4` in a model only support the same EP communication domain, and no other operators are allowed in this communication domain.
  - TP domain communication is not supported currently.

  <!-- npu="A3" id17 -->
  - <term>Atlas A3 products</term>: The communicator supports cross-SuperPoD communication only when commAlg is set to "hierarchy". In other use cases, nodes in a communicator must be within one SuperPoD, and cross-SuperPoD is not supported.

  <!-- end id17 -->

- **Communication mode constraint**:

  <!-- npu="950" id18 -->
  - <term>Ascend 950PR&950DT products</term>: Only UB Memory communication is supported.

  <!-- end id18 -->

- **Networking Constraints**:

  <!-- npu="910b" id19 -->
  - <term>Atlas A2 products</term>: In multi-machine scenarios, only switch networking is supported. Direct-connect networking between two machines is not supported.

  <!-- end id19 -->

- **Other Constraints**:
  - In the formula, "/" indicates integer division.
  - <code>moeExpertNum + zeroExpertNum + copyExpertNum + constExpertNum < MAX_INT32</code>.

## Example

<!-- npu="910b" id20 -->
- <term>Atlas A2 products</term>:

    This sample supports running the A2 operator in a single-node environment with a card count in the range [2, 8]. You can set EP_WORLD_SIZE_A2 to the card count in the sample code as needed, and modify moeExpertNum so that moeExpertNum is divisible by EP_WORLD_SIZE_A2.

    - Compile the operator: The operator compile command is as follows. The underlying compile target of the V4 API is moe_distribute_combine_v2, which reuses the same build artifact as the V2/V3 APIs, so you do not need to compile a separate V4 version. Both the moe_distribute_dispatch_v2 and moe_distribute_combine_v2 operators must be compiled, and these two operators must be executed in pairs.

        ```bash
        bash build.sh --pkg --soc=ascend910b --ops=moe_distribute_dispatch_v2,moe_distribute_combine_v2
        ```

    - Create the A2 example code: After the compilation is complete, refer to the existing [test_aclnn_moe_distribute_combine_v2.cpp](../examples/test_aclnn_moe_distribute_combine_v2.cpp) file in the operator [examples](../examples/) directory and create a new test file test_aclnn_moe_distribute_combine_v4.cpp based on the A2 example code.

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
            int64_t constExpertNum = 0; // Only for A3
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

            // Zero-expert use case input
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
            std::vector<int64_t> epRecvCountsShape{localExpertNum * EP_WORLD_SIZE_A2}; // No layering
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
            std::vector<int64_t> performanceInfoHostData(performanceInfoShapeSize, 0);
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
            ret = CreateAclTensor(performanceInfoHostData, performanceInfoShape, &performanceInfoDeviceAddr, aclDataType::ACL_INT64, &performanceInfo);
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
            // (boilerplate) Synchronize and wait for the task execution to complete.
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

<!-- end id20 -->
<!-- npu="950" id21 -->
- <term>Ascend 950PR&950DT products</term>: For details, see the preparation part and example code in the Example section of [aclnnMoeDistributeCombineV2](./aclnnMoeDistributeCombineV2_en.md). Reset the involved variables according to the preceding Constraints. For the use case parameters newly added in the V4 API compared with the V3 API, pass values as described in the preceding Parameters.

<!-- end id21 -->
<!-- npu="A3" id22 -->
- <term>Atlas A3 products</term>:

    For details about the compilation and execution procedure, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

<!-- end id22 -->
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
                expertScales, elasticInfo, nullptr, // performanceInfoOptional
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
        /**************************************** Call combine ********************************************/
        // Call the first-phase API.
        ret = aclnnMoeDistributeCombineV4GetWorkspaceSize(expandX, expertIds,
                                                            expandIdx, epRecvCounts,
                                                            expertScales, tpRecvCounts,
                                                            nullptr, nullptr, nullptr,
                                                            nullptr, nullptr, nullptr,
                                                            elasticInfo, oriX, constExpertAlpha1, constExpertAlpha2, constExpertV,
                                                            nullptr, // performanceInfoOptional
                                                            hcomEpName, EP_WORLD_SIZE, args.epRankId, moeExpertNum,
                                                            "", 0, 0, expertShardType,
                                                            sharedExpertNum, sharedExpertRankNum, globalBS, outDtype,
                                                            commQuantMode, groupList_type, nullptr, zeroExpertNum, copyExpertNum, constExpertNum, xOut,
                                                            &combineWorkspaceSize, &combineExecutor);
        CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("[ERROR] aclnnMoeDistributeCombineV4GetWorkspaceSize failed. ret = %d \n", ret); return ret);
        // Allocate device memory based on the workspaceSize computed by the first-phase API.
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
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclCommInitAll ep failed, ret %d\n", ret); return ret);

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
