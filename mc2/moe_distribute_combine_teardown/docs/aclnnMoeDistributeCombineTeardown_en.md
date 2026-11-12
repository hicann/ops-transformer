# aclnnMoeDistributeCombineTeardown

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T14:00:04.633Z -->

## Applicable Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 products</term>: Not supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 products</term>: Not supported
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

- **Description**: Receives data from aclnnMoeDistributeCombineSetup and integrates the data (multiplies by weights and then adds).
- **Formula**:

    $$
    dataOut = AllToAllV(expandX)\\
    xOut = Sum(expertScales * dataOut + expertScales * sharedExpertX)
    $$

    Data collected by the MoeDistributeDispatchSetup and MoeDistributeDispatchTeardown operators is returned along the original path. This operator only performs communication status confirmation and data sorting. It confirms the communication status of the data sent by the aclnnMoeDistributeCombineSetup API and performs a weighted sum based on the expert scale weights.

- **NOTE**: This API must be used together with aclnnMoeDistributeDispatchSetup, aclnnMoeDistributeDispatchTeardown, and aclnnMoeDistributeCombineSetup.

## Function Prototype

Each operator uses a two-phase API. First, call the "`aclnnMoeDistributeCombineTeardownGetWorkspaceSize`" API to obtain the input parameters, calculate the workspace size required for computation based on the computation process, and obtain the executor that contains the computation process. Then, call the "`aclnnMoeDistributeCombineTeardown`" API to execute the computation.

```cpp
aclnnStatus aclnnMoeDistributeCombineTeardownGetWorkspaceSize(
    const aclTensor* expandX,
    const aclTensor* quantExpandX,
    const aclTensor* expertIds,
    const aclTensor* expandIdx,
    const aclTensor* expertScales,
    const aclTensor* commCmdInfo,
    const aclTensor* xActiveMaskOptional,
    const aclTensor* sharedExpertXOptional,
    const char* groupEp,
    int64_t epWorldSize,
    int64_t epRankId,
    int64_t moeExpertNum,
    int64_t expertShardType,
    int64_t sharedExpertNum,
    int64_t sharedExpertRankNum,
    int64_t globalBs,
    int64_t commQuantMode,
    int64_t commType,
    const char* commAlg,
    aclTensor* xOut,
    uint64_t* workspaceSize,
    aclOpExecutor** executor)
```

```cpp
aclnnStatus aclnnMoeDistributeCombineTeardown(
    void* workspace,
    uint64_t workspaceSize,
    aclOpExecutor* executor,
    aclrtStream stream)
```

## aclnnMoeDistributeCombineTeardownGetWorkspaceSize

- **Parameters**

    <table style="undefined;table-layout: fixed; width: 1548px"><colgroup>
    <col style="width: 162px">
    <col style="width: 94px">
    <col style="width: 485px">
    <col style="width: 458px">
    <col style="width: 84px">
    <col style="width: 78px">
    <col style="width: 76px">
    <col style="width: 111px">
    </colgroup>
    <thead>
    <tr>
        <th>Parameter</th>
        <th>Input/Output</th>
        <th>Description</th>
        <th>Instruction</th>
        <th>Data Type</th>
        <th>Data Format</th>
        <th>Dimension</th>
        <th>Non-contiguous Tensor</th>
    </tr></thead>
    <tbody>
    <tr>
        <td>expandX (aclTensor*)</td>
        <td>Input</td>
        <td>Token features expanded based on expertIds.</td>
        <td>Empty tensors are not supported.</td>
        <td>FLOAT16, BFLOAT16</td>
        <td>ND</td>
        <td>(A, H)</td>
        <td>√</td>
    </tr>
    <tr>
        <td>quantExpandX (aclTensor*)</td>
        <td>Input</td>
        <td>quantExpandX output in aclnnMoeDistributeCombineSetup.</td>
        <td>Does not support empty tensors.</td>
        <td>INT8</td>
        <td>ND</td>
        <td>(A, tokenMsgSize)</td>
        <td>√</td>
    </tr>
    <tr>
        <td>expertIds (aclTensor*)</td>
        <td>Input</td>
        <td>topK expert indexes of each token.</td>
        <td>Does not support empty tensors.</td>
        <td>INT32</td>
        <td>ND</td>
        <td>(Bs, K)</td>
        <td>√</td>
    </tr>
    <tr>
        <td>expandIdx (aclTensor*)</td>
        <td>Input</td>
        <td>Corresponds to the expandIdx output in aclnnMoeDistributeDispatchSetup.</td>
        <td>Does not support empty tensors.</td>
        <td>INT32</td>
        <td>ND</td>
        <td>(BS * K, )</td>
        <td>√</td>
    </tr>
    <tr>
        <td>expertScales (aclTensor*)</td>
        <td>Input</td>
        <td>Weight of the topK experts for each token.</td>
        <td>Empty tensors are not supported.</td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>(BS, K)</td>
        <td>√</td>
    </tr>
    <tr>
        <td>commCmdInfo (aclTensor*)</td>
        <td>Input</td>
        <td>Output of aclnnMoeDistributeCombineSetup, cmd information for communication.</td>
        <td>Does not support empty tensor.</td>
        <td>INT32</td>
        <td>ND</td>
        <td>((A + epWorldSize) * 16, )</td>
        <td>√</td>
    </tr>
    <tr>
        <td>xActiveMaskOptional (aclTensor*)</td>
        <td>Optional input</td>
        <td>Indicates whether tokens participate in communication.</td>
        <td>Does not support empty tensors. You can pass in valid data or a null pointer. A null pointer indicates that all tokens participate in communication.</td>
        <td>BOOL</td>
        <td>ND</td>
        <td>(BS, )</td>
        <td>√</td>
    </tr>
    <tr>
        <td>sharedExpertXOptional (aclTensor*)</td>
        <td>Optional input</td>
        <td>Indicates tokens computed by the shared expert.</td>
        <td>Does not support empty tensors. Must be a 2D or 3D tensor. For a 2D tensor, the shape is (BS,H). For a 3D tensor, the product of the first two dimensions must equal BS, and the third dimension must equal H. The data type must remain consistent with expandX. You can pass in valid data or a null pointer. If valid data is passed in, sharedExpertNum must be 0.</td>
        <td>FLOAT16, BFLOAT16</td>
        <td>ND</td>
        <td>(BS, H) or (a, b, H)</td>
        <td>√</td>
    </tr>
    <tr>
        <td>groupEp</td>
        <td>Input</td>
        <td>Name of the EP communicator.</td>
        <td>String length: [1, 128)</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>epWorldSize</td>
        <td>Input</td>
        <td>EP communicator size</td>
        <td>Supported values: [2, 384]</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>epRankId</td>
        <td>Input</td>
        <td>ID of the local device in the EP domain.</td>
        <td>Value range: [0, epWorldSize).<br>The epRankId of each device in the same EP communicator is unique.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>moeExpertNum</td>
        <td>Input</td>
        <td>MoE expert count.</td>
        <td>Value range: (0, 512].<br>moeExpertNum % (epWorldSize - sharedExpertRankNum) = 0 must be satisfied.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>expertShardType</td>
        <td>Input</td>
        <td>Distribution type of shared expert cards.</td>
        <td>Currently only supports passing 0, indicating that shared expert cards are placed before MoE expert cards.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>sharedExpertNum</td>
        <td>Input</td>
        <td>Shared expert count.</td>
        <td>Current value range: [0, 4]. 0 indicates no shared expert. Currently, only the case without shared experts is supported.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>sharedExpertRankNum</td>
        <td>Input</td>
        <td>Quantity of shared expert cards.</td>
        <td>Current value range [0, epWorldSize / 2].</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>globalBs</td>
        <td>Input</td>
        <td>Global batch size in the EP domain.</td>
        <td>If the Bs count is the same on each rank, globalBs = Bs * epWorldSize or globalBs = 0; if the Bs count differs on each rank, globalBs = maxBs * epWorldSize, where maxBs indicates the maximum Bs value on a single device.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>commQuantMode</td>
        <td>Input</td>
        <td>Communication quantization type.</td>
        <td>Value range [0, 2]. 0 indicates no quantization during communication. Currently only supports 0.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>commType</td>
        <td>Input</td>
        <td>Selected communication solution.</td>
        <td>Value range [0, 2]. 0 indicates the AICPU-SDMA solution, 1 indicates the CCU solution, and 2 indicates the URMA solution. Currently only supports 2.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>commAlg</td>
        <td>Input</td>
        <td>Communication algorithm selection.</td>
        <td>Only supports passing in a null pointer or empty string.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>xOut (aclTensor*)</td>
        <td>Output</td>
        <td>Indicates the processed token.</td>
        <td>Does not support empty tensors. The data type must remain consistent with expandX.</td>
        <td>FLOAT16, BFLOAT16</td>
        <td>ND</td>
        <td>(BS, H)</td>
        <td>√</td>
    </tr>
    <tr>
        <td>workspaceSize</td>
        <td>Output</td>
        <td>Returns the workspace size that requires side application on the Device.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>executor</td>
        <td>Output</td>
        <td>Returns the op executor that contains the operator computation process.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    </tbody></table>

  <!-- npu="950" id7 -->
  - Ascend 950PR&950DT products:
    - The shared expert scenario is not supported.
    - epWorldSize currently only supports 2 or 8.
    - moeExpertNum indicates the MoE expert count. Currently, you can only pass in 32.
    - expertShardType currently only supports 0, indicating that shared expert cards are placed before MoE expert cards.
    - sharedExpertNum indicates the shared expert count. Shared experts are currently not supported. You can only pass in 0.
    - sharedExpertRankNum indicates the number of shared expert cards. Shared experts are currently not supported. You can only pass in 0.
    - commQuantMode currently only supports passing in 0, indicating that quantization is not performed.
    - commType has a value range of [0, 2]. Currently, only 2 is supported, indicating the URMA path.
    - commAlg is not supported in the current version. Simply pass null pointer.

  <!-- end id7 -->
  - Atlas A3 products:
    - The shared expert scenario is not supported.
    - epWorldSize: the current value only supports 2 and 8.
    - moeExpertNum indicates the MoE expert count. Currently, you can only pass in 32.
    - expertShardType currently only supports passing in 0, indicating that shared expert cards are placed before MoE expert cards.
    - sharedExpertNum indicates the shared expert count. Shared experts are currently not supported. You can only pass in 0.
    - sharedExpertRankNum indicates the number of shared expert cards. Shared experts are currently not supported. You can only pass in 0.
    - commQuantMode currently only supports 0, indicating that no quantization is performed.
    - The value range of commType is [0, 2]. Currently only 2 is supported, indicating the URMA path.
    - commAlg is not supported in the current version. Pass null pointer.

- **Return Value**

    aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

    The first-phase API performs input parameter validation and returns an error in the following scenarios:

    <table style="undefined;table-layout: fixed; width: 1000px"><colgroup>
    <col style="width: 300px">
    <col style="width: 150px">
    <col style="width: 550px">
    </colgroup>
    <thead>
    <tr>
        <th>Return value</th>
        <th>Error code</th>
        <th>Description</th>
    </tr></thead>
    <tbody>
    <tr>
        <td>ACLNN_ERR_PARAM_NULLPTR </td>
        <td>161001</td>
        <td>A required input or output tensor is a null pointer.</td>
    </tr>
    <tr>
        <td>ACLNN_ERR_PARAM_INVALID </td>
        <td>161002</td>
        <td>The data types of input and output are not within the supported range.</td>
    </tr>
    <tr>
        <td rowspan="2">ACLNN_ERR_INNER_TILING_ERROR</td>
        <td rowspan="2">561002</td>
        <td>The shapes of input and output are not within the supported range.</td>
    </tr>
    <tr>
        <td>The parameter value is not within the supported range.</td>
    </tr>
    </tbody>
    </table>

## aclnnMoeDistributeCombineTeardown

- **Parameters:**

    <table style="undefined;table-layout: fixed; width: 1000px"><colgroup>
    <col style="width: 200px">
    <col style="width: 130px">
    <col style="width: 670px">
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
        <td>Workspace memory address applied on the Device side.</td>
    </tr>
    <tr>
        <td>workspaceSize</td>
        <td>Input</td>
        <td>Workspace size applied on the Device side, obtained by calling the first-phase API aclnnMoeDistributeCombineTeardownGetWorkspaceSize.</td>
    </tr>
    <tr>
        <td rowspan="2">executor</td>
        <td rowspan="2">Input</td>
        <td>op executor, containing the operator computation flow.</td>
    </tr>
    <tr>
        <td>Specifies the stream for task execution.</td>
    </tr>
    </tbody>
    </table>

- **Return value**

    aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnMoeDistributeCombineTeardown defaults to a deterministic implementation.
- The aclnnMoeDistributeDispatchSetup API, aclnnMoeDistributeDispatchTeardown API, aclnnMoeDistributeCombineSetup API, and aclnnMoeDistributeCombineTeardown API must be used together.
- The values of the `groupEp`, `epWorldSize`, `moeExpertNum`, `expertShardType`, `sharedExpertNum`, `sharedExpertRankNum`, `globalBs`, `commQuantMode`, `commType`, and `commAlg` parameters used in the API call must remain consistent across all devices; the values of the `groupEp`, `epWorldSize`, `expertShardType`, `sharedExpertNum`, `sharedExpertRankNum`, `globalBs`, `commQuantMode`, `commType`, and `commAlg` parameters must also remain consistent across different layers in the network and with the corresponding parameters of the aclnnMoeDistributeDispatchSetup, aclnnMoeDistributeDispatchTeardown, and aclnnMoeDistributeCombineSetup APIs.
- Description of the shape format in Parameters:

  - A: the maximum number of tokens that the current card needs to distribute. The value range is as follows:
    - For shared experts, A must satisfy A = BS \* epWorldSize \* sharedExpertNum / sharedExpertRankNum; when globalBs is not 0, A must satisfy A = globalBs\* sharedExpertNum / sharedExpertRankNum.
    - For MoE experts, when globalBs is 0, A >= BS \* epWorldSize \* min(localExpertNum, K) must be satisfied; when globalBs is non-0, A >= globalBs \* min(localExpertNum, K) must be satisfied.
  - H: indicates the hidden size. The value range is [1024, 8196]. Currently only 4096 and 7168 are supported.
  - HS: indicates the data size in communication, which is Align512( Align32(H) + Align8(H) / 8 *sizeof(float) ), where Align512(x) = ((x + 512 - 1) / 512) \* 512, Align32(x) = ((x + 32 - 1) / 32) \* 32, Align8(x) = ((x + 8 - 1) / 8) \* 8.
  - BS: indicates the batch sequence size, that is, the final number of tokens output by this device. The value range is 0 < BS ≤ 512. Currently only 8, 16, and 256 are supported.
  - K: indicates selecting topK experts. The value range is 0 < K ≤ 16 and must satisfy 0 < K ≤ moeExpertNum. Currently only 6 and 8 are supported.
  - localExpertNum: indicates the number of experts on the current card.
    - For a shared expert card, localExpertNum = 1.
    - For MoE expert cards, localExpertNum = moeExpertNum / (epWorldSize - sharedExpertRankNum). moeExpertNum currently only supports 32.
  - tokenMsgSize: indicates the dimension information of each token during data communication. The calculation formula is Align512(Align32(H)+Align8(H)/8\*sizeof(float)), where AlignN(x)=((x+N-1)/N*N).
  - Shared experts are currently not supported. sharedExpertNum and sharedExpertRankNum currently only support 0.
- HCCL_BUFFSIZE:
  Before calling this API, check whether the value of the HCCL_BUFFSIZE environment variable is reasonable. This environment variable indicates the memory size occupied by a single communication domain, in MB. The default value is 200 MB when not configured.

  <!-- npu="950" id8 -->
  - Ascend 950PR&950DT products:
    - Requires >= 2 and satisfies >= 4 \* (localExpertNum \* maxBs \* epWorldSize \* Align512(Align32(2 \* H) + 44) + (K + sharedExpertNum) \* maxBs \* Align512(2 \* H)), localExpertNum must use the local expert count of the MoE expert card, where Align512(x) = ((x + 512 - 1) / 512) \* 512, Align32(x) = ((x + 32 - 1) / 32) \* 32
  <!-- end id8 -->
  - Atlas A3 products:
    - Requires >= 2 and satisfies >= 2 \* (localExpertNum \* maxBs \* epWorldSize \* Align512(Align32(2 \* H) + 44) + (K + sharedExpertNum) \* maxBs \* Align512(2 \* H)), localExpertNum must use the local expert count of the MoE expert card, where Align512(x) = ((x + 512 - 1) / 512) \* 512, Align32(x) = ((x + 32 - 1) / 32) \* 32
- Communication domain usage constraints:
  - The aclnnMoeDistributeDispatchSetup, aclnnMoeDistributeDispatchTeardown, aclnnMoeDistributeCombineSetup, and aclnnMoeDistributeCombineTeardown APIs in a model only support the same EP communicator, and no other operator is allowed in the communicator.
- Communication mode constraints:

    <!-- npu="950" id9 -->
    - <term>Ascend 950PR&950DT products</term>: only supports URMA communication.

    <!-- end id9 -->

## Example

<!-- npu="950" id10 -->
- Ascend 950PR&950DT products: aclnnMoeDistributeCombineTeardown must be called after aclnnMoeDistributeCombineSetup is called. See the example in [aclnnMoeDistributeCombineSetup](../../moe_distribute_combine_setup/docs/aclnnMoeDistributeCombineSetup_en.md).

<!-- end id10 -->
