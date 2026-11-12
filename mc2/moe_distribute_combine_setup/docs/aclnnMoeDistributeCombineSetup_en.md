# aclnnMoeDistributeCombineSetup

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T14:33:47.460Z -->

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

- **Description**: Performs AlltoAllV communication and writes data to the peer GM.
- **Formula**:

    $$
    dataOut = AllToAllV(expandX)\\
    $$

    Data is returned along the same path used by the MoeDistributeDispatchSetup and MoeDistributeDispatchTeardown operators to collect data. This operator only sends communication status and communication data, and exits immediately after the data is sent without waiting for communication completion. Communication status confirmation and data post-processing are completed by aclnnMoeDistributeCombineTeardown.

- **NOTE**: This API must be used together with aclnnMoeDistributeDispatchSetup, aclnnMoeDistributeDispatchTeardown, and aclnnMoeDistributeCombineTeardown.

## Function Prototype

This operator is a two-phase API. Call the `aclnnMoeDistributeCombineSetupGetWorkspaceSize` API first to obtain the input parameters, calculate the required workspace size based on the computation process, and obtain the executor that contains the operator computation process. Then call the `aclnnMoeDistributeCombineSetup` API to perform the computation. The aclnnMoeDistributeCombineSetupTeardownCalcOutputSize API is provided to calculate the output size of the aclnnMoeDistributeCombineSetup phase.

```cpp
aclnnStatus aclnnMoeDistributeCombineSetupGetWorkspaceSize(
    const aclTensor* expandX,
    const aclTensor* expertIds,
    const aclTensor* assistInfoForCombine,
    const char*      groupEp,
    int64_t          epWorldSize,
    int64_t          epRankId,
    int64_t          moeExpertNum,
    int64_t          expertShardType,
    int64_t          sharedExpertNum,
    int64_t          sharedExpertRankNum,
    int64_t          globalBs,
    int64_t          commQuantMode,
    int64_t          commType,
    const char*      commAlg,
    aclTensor*       quantExpandXOut,
    aclTensor*       commCmdInfoOut,
    uint64_t*        workspaceSize,
    aclOpExecutor**  executor)
```

```cpp
aclnnStatus aclnnMoeDistributeCombineSetup(
    void             *workspace,
    uint64_t         workspaceSize,
    aclOpExecutor    *executor,
    aclrtStream      stream)
```

```cpp
aclnnStatus aclnnMoeDistributeCombineSetupTeardownCalcOutputSize(
    const aclTensor *expandX,
    const aclTensor *expertIds,
    const aclTensor *assistInfoForCombine,
    const char      *groupEp,
    int64_t         epWorldSize,
    int64_t         epRankId,
    int64_t         moeExpertNum,
    int64_t         expertShardType,
    int64_t         sharedExpertNum,
    int64_t         sharedExpertRankNum,
    int64_t         globalBs,
    int64_t         commQuantMode,
    int64_t         commType,
    const char      *commAlg,
    uint64_t        &tokenMsgSize,
    uint64_t        &commCmdInfoOutSize)
```

## aclnnMoeDistributeCombineSetupGetWorkspaceSize

- **Parameters**

    <table style="undefined;table-layout: fixed; width: 1434px"><colgroup>
    <col style="width: 156px">
    <col style="width: 71px">
    <col style="width: 338px">
    <col style="width: 450px">
    <col style="width: 87px">
    <col style="width: 67px">
    <col style="width: 170px">
    <col style="width: 95px">
    </colgroup>
    <thead>
    <tr>
        <th>Parameter</th>
        <th>Input/Output</th>
        <th>Description</th>
        <th>Instruction</th>
        <th>Data Type</th>
        <th>Data Format</th>
        <th>Dimensions</th>
        <th>Non-contiguous Tensor</th>
    </tr></thead>
    <tbody>
    <tr>
        <td>expandX</td>
        <td>Input</td>
        <td>Self-refresh parameter, token features expanded based on expertIds.</td>
        <td>Does not support null tensors.</td>
        <td>FLOAT16, BFLOAT16</td>
        <td>ND</td>
        <td>(A, H)</td>
        <td>√</td>
    </tr>
    <tr>
        <td>expertIds</td>
        <td>Input</td>
        <td>topK expert indices of each token.</td>
        <td>Does not support null tensors.</td>
        <td>INT32</td>
        <td>ND</td>
        <td>(Bs, K)</td>
        <td>√</td>
    </tr>
    <tr>
        <td>assistInfoForCombine</td>
        <td>Input</td>
        <td>Corresponds to the assistInfoForCombine output in aclnnMoeDistributeDispatchTeardown.</td>
        <td>Does not support null tensors.</td>
        <td>INT32</td>
        <td>ND</td>
        <td>(A * 128, )</td>
        <td>√</td>
    </tr>
    <tr>
        <td>groupEp</td>
        <td>Input</td>
        <td>EP communication domain name.</td>
        <td>String length range [1, 128)</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>epWorldSize</td>
        <td>Input</td>
        <td>EP communication domain size.</td>
        <td>Value range [2, 384]</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>epRankId</td>
        <td>Input</td>
        <td>Local device ID in the EP domain.</td>
        <td>Value range [0, epWorldSize).<br>The epRankId of each device in the same EP communication domain must be unique.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>moeExpertNum</td>
        <td>Input</td>
        <td>MoE expert count.</td>
        <td>Value range (0, 512].<br>Must satisfy moeExpertNum % (epWorldSize - sharedExpertRankNum) = 0.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>expertShardType</td>
        <td>Input</td>
        <td>Distribution type of shared expert cards.</td>
        <td>Currently only supports passing in 0, indicating that shared expert cards are placed before MoE expert cards.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>sharedExpertNum </td>
        <td>Input</td>
        <td>shared expert count</td>
        <td>Current value range [0, 4]. 0 indicates no shared expert.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>sharedExpertRankNum</td>
        <td>Input</td>
        <td>Number of shared expert cards.</td>
        <td>The current value range is [0, epWorldSize / 2].</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>globalBs</td>
        <td>Input</td>
        <td>Global batch size in the EP domain.</td>
        <td>When each rank has the same Bs, globalBs = Bs * epWorldSize or globalBs = 0. When each rank has different Bs values, globalBs = maxBs * epWorldSize, among which maxBs indicates the maximum Bs on a single device.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>commQuantMode</td>
        <td>Input</td>
        <td>Communication quantization type.</td>
        <td>Currently only supports passing in 0, indicating that no quantization is performed.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>commType</td>
        <td>Input</td>
        <td>Communication scheme selection.</td>
        <td>Currently only supports 2, indicating the URMA path.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>commAlg</td>
        <td>Input</td>
        <td>Communication algorithm selection.</td>
        <td>Reserved field, not supported in the current version. Passing a null pointer or an empty string is sufficient.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>quantExpandXOut</td>
        <td>Output</td>
        <td>Quantized token.</td>
        <td>Does not support null tensors.</td>
        <td>INT8</td>
        <td>ND</td>
        <td>(A, tokenMsgSize)</td>
        <td>√</td>
    </tr>
    <tr>
        <td>commCmdInfoOut</td>
        <td>Output</td>
        <td>cmd information of the communication.</td>
        <td>Does not support null tensors.</td>
        <td>INT32</td>
        <td>ND</td>
        <td>((A + epWorldSize) * 16, )</td>
        <td>√</td>
    </tr>
    <tr>
        <td>workspaceSize</td>
        <td>Output</td>
        <td>Returns the workspace size that needs to be applied by the Device side.</td>
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
    </tbody></table>

  <!-- npu="950" id7 -->
  - Ascend 950PR&950DT products:
    - The shared expert scenario is not supported.
    - The current value of epWorldSize only supports 2 and 8.
    - moeExpertNum indicates the MoE expert count. Currently, only 32 can be passed in.
    - expertShardType currently only supports passing in 0, indicating that shared expert cards are arranged before MoE expert cards.
    - sharedExpertNum indicates the shared expert count. Currently, shared experts are not supported; only 0 can be passed in.
    - sharedExpertRankNum indicates the number of shared expert cards. Currently, shared experts are not supported; only 0 can be passed in.
    - commQuantMode currently only supports passing in 0, indicating that no quantization is performed.
    - commType has a value range of [0, 2]. Currently only 2 is supported, indicating the URMA path.
    - commAlg is not supported in the current version. Passing a null pointer is sufficient.

  <!-- end id7 -->

- **Return Value**

    aclnnStatus: Return status code. See specifically [aclnn return code](../../../docs/en/context/aclnn_return_code.md).

    The first-phase API performs input parameter validation and returns an error in the following scenarios:

    <table style="undefined;table-layout: fixed; width: 1000px"><colgroup>
    <col style="width: 300px">
    <col style="width: 150px">
    <col style="width: 550px">
    </colgroup>
    <thead>
    <tr>
        <th>Return Value</th>
        <th>Error Code</th>
        <th>Description</th>
    </tr></thead>
    <tbody>
    <tr>
        <td>ACLNN_ERR_PARAM_NULLPTR </td>
        <td>161001</td>
        <td>Required input and output parameter tensors are null pointers.</td>
    </tr>
    <tr>
        <td>ACLNN_ERR_PARAM_INVALID </td>
        <td>161002</td>
        <td>The data type of input and output is not within the supported range.</td>
    </tr>
    <tr>
        <td rowspan="2">ACLNN_ERR_INNER_TILING_ERROR</td>
        <td rowspan="2">561002</td>
        <td>The shape of input and output is not within the supported range.</td>
    </tr>
    <tr>
        <td>The parameter value is not within the supported range.</td>
    </tr>
    </tbody>
    </table>

## aclnnMoeDistributeCombineSetup

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
        <td>Memory address of the workspace applied on the Device side.</td>
    </tr>
    <tr>
        <td>workspaceSize</td>
        <td>Input</td>
        <td>Size of the workspace applied on the Device side, obtained by calling the first-phase API aclnnMoeDistributeCombineSetupGetWorkspaceSize.</td>
    </tr>
    <tr>
        <td>executor</td>
        <td>Input</td>
        <td>Op executor, which contains the operator computation flow.</td>
    </tr>
    <tr>
        <td>stream</td>
        <td>Input</td>
        <td>Stream for executing tasks.</td>
    </tr>
    </tbody>
    </table>

- **Return value**

    Returns aclnnStatus status code, see specifically [aclnn return code](../../../docs/en/context/aclnn_return_code.md).

## aclnnMoeDistributeCombineSetupTeardownCalcOutputSize

- **Parameters**

    <table style="undefined;table-layout: fixed; width: 1434px"><colgroup>
    <col style="width: 156px">
    <col style="width: 71px">
    <col style="width: 338px">
    <col style="width: 450px">
    <col style="width: 87px">
    <col style="width: 67px">
    <col style="width: 170px">
    <col style="width: 95px">
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
        <td>expandX</td>
        <td>Input</td>
        <td>Self-refresh parameter, token features expanded based on expertIds.</td>
        <td>Null tensor is not supported.</td>
        <td>FLOAT16, BFLOAT16</td>
        <td>ND</td>
        <td>(A, H)</td>
        <td>√</td>
    </tr>
    <tr>
        <td>expertIds</td>
        <td>Input</td>
        <td>TopK expert indexes of each token.</td>
        <td>Does not support null tensors.</td>
        <td>INT32</td>
        <td>ND</td>
        <td>(Bs, K)</td>
        <td>√</td>
    </tr>
    <tr>
        <td>assistInfoForCombine</td>
        <td>Input</td>
        <td>Corresponds to the assistInfoForCombine output in aclnnMoeDistributeDispatchTeardown.</td>
        <td>Does not support null tensors.</td>
        <td>INT32</td>
        <td>ND</td>
        <td>(A * 128, )</td>
        <td>√</td>
    </tr>
    <tr>
        <td>groupEp</td>
        <td>Input</td>
        <td>EP communication domain name.</td>
        <td>String length range [1, 128).</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>epWorldSize</td>
        <td>Input</td>
        <td>EP communication domain size</td>
        <td>Supported value range [2, 384]</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>epRankId</td>
        <td>Input</td>
        <td>Local device ID in the EP domain.</td>
        <td>Value range [0, epWorldSize).<br>The epRankId of each device in the same EP communication domain is unique.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>moeExpertNum</td>
        <td>Input</td>
        <td>MoE expert count.</td>
        <td>Value range (0, 512].<br>Must satisfy moeExpertNum % (epWorldSize - sharedExpertRankNum) = 0.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>expertShardType</td>
        <td>Input</td>
        <td>Shared expert card distribution type.</td>
        <td>Currently only supports 0, indicating that shared expert cards are arranged before MoE expert cards.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>sharedExpertNum </td>
        <td>Input</td>
        <td>Shared expert count.</td>
        <td>Current value range [0, 4]. 0 indicates no shared expert.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>sharedExpertRankNum</td>
        <td>Input</td>
        <td>Quantity of shared expert devices.</td>
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
        <td>When the number of Bs values is the same on each rank, globalBs = Bs * epWorldSize or globalBs = 0; when the number of Bs values differs on each rank, globalBs = maxBs * epWorldSize, among which maxBs indicates the maximum Bs value on a single device.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>commQuantMode</td>
        <td>Input</td>
        <td>Communication quantization type.</td>
        <td>Currently only supports passing in 0, indicating that quantization is not performed.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>commType</td>
        <td>Input</td>
        <td>Communication scheme selection.</td>
        <td>Currently only supports 2, indicating the URMA path.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>commAlg</td>
        <td>Input</td>
        <td>communication algorithm selection</td>
        <td>Reserved field. Not supported in the current version. Passing a null pointer or an empty string is sufficient.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>tokenMsgSize</td>
        <td>Output</td>
        <td>Size of tokenMsgSize, the second dimension of quantExpandX in the aclnnMoeDistributeCombineSetup API. It indicates the dimension information of each token during data communication.</td>
        <td>-</td>
        <td>INT64</td>
        <td>-</td>
        <td>Align512(Align32(H) + Align8(H) / 8 * sizeof(float)), among which AlignN(x) = ((x + N - 1) / N) * N.</td>
        <td>-</td>
    </tr>
    <tr>
        <td>commCmdInfoOutSize</td>
        <td>Output</td>
        <td>Size of commCmdInfoOut of the aclnnMoeDistributeCombineSetup API.</td>
        <td>-</td>
        <td>INT64</td>
        <td>-</td>
        <td>(A + epWorldSize) * 16</td>
        <td>-</td>
    </tr>
    <tr>
        <td>workspaceSize</td>
        <td>Output</td>
        <td>Returns the size of the workspace to be allocated on the Device side.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>executor</td>
        <td>Output</td>
        <td>Returns the op executor, which contains the operator computation flow.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    </tbody></table>

- **Return Value**

    aclnnStatus: return code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnMoeDistributeCombineSetup defaults to a deterministic implementation.
- The aclnnMoeDistributeDispatchSetup, aclnnMoeDistributeDispatchTeardown, aclnnMoeDistributeCombineSetup, and aclnnMoeDistributeCombineTeardown APIs must be used together.
- During the API call, the values of the `groupEp`, `epWorldSize`, `moeExpertNum`, `expertShardType`, `sharedExpertNum`, `sharedExpertRankNum`, `globalBs`, `commQuantMode`, `commType`, and `commAlg` parameters must be the same on all devices; the values of the `groupEp`, `epWorldSize`, `expertShardType`, `sharedExpertNum`, `sharedExpertRankNum`, `globalBs`, `commQuantMode`, `commType`, and `commAlg` parameters must be the same across different layers in the network and must be the same as the corresponding parameters of the aclnnMoeDistributeDispatchSetup, aclnnMoeDistributeDispatchTeardown, and aclnnMoeDistributeCombineTeardown APIs.
- Description of the shape format in Parameters:
  - A: the maximum number of tokens that the current card needs to distribute. The value range is as follows:
    - For shared experts, when globalBs is 0, it must satisfy A = BS \* epWorldSize \* sharedExpertNum / sharedExpertRankNum; when globalBs is not 0, it must satisfy A = globalBs\* sharedExpertNum / sharedExpertRankNum.
    - For MoE experts, when globalBs is 0, it must satisfy A >= BS \* epWorldSize \* min(localExpertNum, K); when globalBs is not 0, it must satisfy A >= globalBs \* min(localExpertNum, K).
  - H: Indicates the hidden size, that is, the size of the hidden layer. The value range is [1024, 8192]. Currently only supports 4096 and 7168.
  - BS: Indicates the batch sequence size, that is, the quantity of tokens finally output by the current device. The value range is 0 < BS ≤ 512. Currently only supports 8, 16, and 256.
  - K: Indicates that topK experts are selected. The value range is 0 < K ≤ 16, and 0 < K ≤ moeExpertNum must also be satisfied. Currently only supports 6 and 8.
  - localExpertNum: indicates the number of experts on the current card.
    - For a shared expert card, localExpertNum = 1.
    - For MoE expert cards, localExpertNum = moeExpertNum / (epWorldSize - sharedExpertRankNum). moeExpertNum currently only supports 32.
  - tokenMsgSize: Indicates the dimension information of each token during data communication. The formula is Align512(Align32(H)+Align8(H)/8\*sizeof(float)), where AlignN(x)=((x+N-1)/N*N).
  - Shared experts are currently not supported. sharedExpertNum and sharedExpertRankNum currently only support 0.
- HCCL_BUFFSIZE:
  Before calling this API, check whether the value of the HCCL_BUFFSIZE environment variable is reasonable. This environment variable indicates the memory size occupied by a single communication domain, in MB. The default value is 200 MB when not configured.

  <!-- npu="950" id8 -->
  - Ascend 950PR&950DT products:
    - The value must be >= 2 and >= 4 \* (localExpertNum \* maxBs \* epWorldSize \* Align512(Align32(2 \* H) + 44) + (K + sharedExpertNum) \* maxBs \* Align512(2 \* H)), where localExpertNum must use the expert count of the current MoE expert card, Align512(x) = ((x + 512 - 1) / 512) \* 512, and Align32(x) = ((x + 32 - 1) / 32) \* 32.
  <!-- end id8 -->

- Communication domain usage constraints:
  - In a model, aclnnMoeDistributeDispatchSetup, aclnnMoeDistributeDispatchTeardown, aclnnMoeDistributeCombineSetup, and aclnnMoeDistributeCombineTeardown only support the same EP communication domain, and no other operators are allowed in this communication domain.
- Communication mode constraints:

    <!-- npu="950" id9 -->
    - <term>Ascend 950PR&950DT products</term>: only supports URMA communication.

    <!-- end id9 -->

## Example

- File preparation:

    1. Create and modify the rank_table_m2.json file as instructed below.

    2. Copy the project to both servers, and configure the rank_table_m2.json file content based on the device IP of each machine. Ensure that the rank_table_m2.json files on both servers remain consistent.

    3. Install the cann package, and compile and run it by referring to [Operator Invocation](../../../docs/zh/invocation/quick_op_invocation.md).

- About rankTable:

    1. Developers can use the ranktable file to configure NPU resource information for collective communication. For details, see "Communication Function Development > Cluster Information Configuration > ranktable File Configuration of Resource Information" in [*HCCL User Guide*](https://hiascend.com/en/document/redirect/CannCommunityHcclUg).

    2. Run `cat /etc/hccn.conf` or `for i in seq 0 7; do echo "===================> dev$i, NPU$((i+1))"; hccn_tool -i $i -ip -g; done` to query the device IP of the machine. Then fill in the JSON file by referring to the collective communication documentation.

    > Note: In a two-machine 16-card scenario, the device_id of both machines is 0 to 7, the rank_id of one machine is 0 to 7, and the rank_id of the other machine is 8 to 15. In a single-machine 16-card scenario, both device_id and rank_id are 0 to 15.

- Environment variable configuration:

    ```bash
    # Three environment variables must be set before running.
    ## FIRST_RANK_ID description: For example, with two machines and 16 devices, set the value to 0 on one machine and 8 on the other.
    ## For example, export FIRST_RANK_ID=0.
    export RANK_TABLE_FILE=/home/path/to/rank_table_m2.json
    export FIRST_RANK_ID=<starting rank_id of the device>
    ## ENV_DEV_NUM description: Set this variable to the number of devices on the current machine. For example, with two machines and 16 devices, set the variable to 16 on both machines.
    export ENV_DEV_NUM=16

- Machine quantity setting:

    In the two-machine 16-card use case, set the parameter MACHINE_NUM to 2, that is,

    ```Cpp
    const uint32_t MACHINE_NUM = 2;
    ```

    For the single-machine 16-card use case, no modification is required.

<!-- npu="950" id10 -->
- Ascend 950PR&950DT products:

    The sample code below is for reference only. For the specific compilation and execution process, see [Compilation and Running Example](../../../docs/en/context/compile_and_run_sample.md).

    ```Cpp
    #include <thread>
    #include <iostream>
    #include <string>
    #include <vector>
    #include <cstring>
    #include "acl/acl.h"
    #include "hccl/hccl.h"
    #include "aclnnop/aclnn_moe_distribute_combine_setup.h"
    #include "aclnnop/aclnn_moe_distribute_combine_teardown.h"

    #define CHECK_RET(cond, return_expr)                                                                                   \
        do {                                                                                                               \
            if (!(cond)) {                                                                                                 \
                return_expr;                                                                                               \
            }                                                                                                              \
        } while (0)

    #define LOG_PRINT(message, ...)                                                                                        \
        do {                                                                                                               \
            printf(message, ##__VA_ARGS__);                                                                                \
        } while (0)

    constexpr int DEV_NUM = 2;

    template <typename Func>
    class Guard {
    public:
        explicit Guard(Func &func) : func_(func)
        {
        }
        ~Guard()
        {
            func_();
        }

    private:
        Func &func_;
    };

    int64_t AlignN(int64_t x, int64_t n)
    {
        return (x + n - 1) / n * n;
    }

    int64_t GetShapeSize(const std::vector<int64_t> &shape)
    {
        int64_t shape_size = 1;
        for (auto i : shape) {
            shape_size *= i;
        }
        return shape_size;
    }

    struct Args {
        uint32_t rankId;
        uint32_t epRankId;
        HcclComm hcclEpComm;
        aclrtStream combinesetupstream;
        aclrtStream combineteardownstream;
        aclrtContext context;
    };

    template <typename T>
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

    void DestroyTensor(aclTensor *tensor)
    {
        if (tensor != nullptr) {
            aclDestroyTensor(tensor);
        }
    }

    void FreeDeviceAddr(void *deviceAddr)
    {
        if (deviceAddr != nullptr) {
            aclrtFree(deviceAddr);
        }
    }

    int LaunchOneProcess(Args &args)
    {
        int ret = aclrtSetCurrentContext(args.context);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetCurrentContext failed. ret: %d\n", ret); return ret);

        char hcomEpName[128] = {0};
        ret = HcclGetCommName(args.hcclEpComm, hcomEpName);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclGetCommName failed. ret: %d\n", ret); return ret);

        auto destroyFunc = [&args]() {
            std::cout << "== begin to destroy " << std::endl;
            HcclCommDestroy(args.hcclEpComm);
            aclrtDestroyStream(args.combinesetupstream);
            aclrtDestroyStream(args.combineteardownstream);
            aclrtDestroyContext(args.context);
            aclrtResetDevice(args.rankId);
        };
        auto guard = Guard<decltype(destroyFunc)>(destroyFunc);

        // Use case parameters.
        int64_t bs = 16;
        int64_t h = 4096;
        int64_t k = 6;
        int64_t expertShardType = 0;
        int64_t sharedExpertNum = 0;
        int64_t sharedExpertRankNum = 0;
        int64_t moeExpertNum = 32;
        int64_t commQuantMode = 0;
        int64_t tpWorldSize = 1;
        int64_t epWorldSize = DEV_NUM;
        int64_t commType = 2;
        int64_t timeOut = 100000000;

        int64_t globalBS = bs * epWorldSize;
        int64_t localExpertNum;
        int64_t localToken;

        if (args.epRankId < sharedExpertRankNum) {
            localExpertNum = 1;
            localToken = globalBS / sharedExpertRankNum;
        } else {
            localExpertNum = moeExpertNum / (epWorldSize - sharedExpertRankNum);
            localToken = globalBS * (localExpertNum < k ? localExpertNum : k);
        }

        uint64_t assistInfoForCombineOutSize = localToken * 128;

        // Define shape.
        std::vector<int64_t> expandXShape{tpWorldSize * localToken, h};
        std::vector<int64_t> expertIdsShape{bs, k};
        std::vector<int64_t> expertScalesShape{bs, k};
        std::vector<int64_t> expandIdxShape{bs * k};
        std::vector<int64_t> assistInfoForCombineOutShape{assistInfoForCombineOutSize};
        std::vector<int64_t> quantExpandXOutShape{tpWorldSize * localToken,
                                                AlignN(AlignN(h, 32) + AlignN(h, 8) / 8 * sizeof(float), 512)};
        std::vector<int64_t> commCmdInfoOutShapeforcombine{(localToken + epWorldSize) * 16};
        std::vector<int64_t> xOutShape{bs, h};

        int64_t expandXShapeSize = GetShapeSize(expandXShape);
        int64_t expertIdsShapeSize = GetShapeSize(expertIdsShape);
        int64_t expertScalesShapeSize = GetShapeSize(expertScalesShape);
        int64_t expandIdxShapeSize = GetShapeSize(expandIdxShape);
        int64_t assistInfoForCombineOutShapeSize = GetShapeSize(assistInfoForCombineOutShape);
        int64_t quantExpandXOutShapeSize = GetShapeSize(quantExpandXOutShape);
        int64_t commCmdInfoOutShapeSizeforcombine = GetShapeSize(commCmdInfoOutShapeforcombine);
        int64_t xOutShapeSize = GetShapeSize(xOutShape);

        // Construct host data.
        std::vector<int16_t> expandXHostData(expandXShapeSize, 0);
        std::vector<int32_t> expertIdsHostData;
        for (int32_t token_id = 0; token_id < expertIdsShape[0]; token_id++) {
            for (int32_t k_id = 0; k_id < expertIdsShape[1]; k_id++) {
                expertIdsHostData.push_back(k_id);
            }
        }
        std::vector<float> expertScalesHostData(expertScalesShapeSize, 0);
        std::vector<int32_t> expandIdxHostData(expandIdxShapeSize, 0);
        std::vector<int32_t> assistInfoForCombineOutHostData(assistInfoForCombineOutShapeSize, 0);
        std::vector<int8_t> quantExpandXOutHostData(quantExpandXOutShapeSize, 0);
        std::vector<int32_t> commCmdInfoOutforCombineHostData(commCmdInfoOutShapeSizeforcombine, 0);
        std::vector<float> xOutHostData(xOutShapeSize, 0);

        // Declare device addresses and tensors.
        void *expandXDeviceAddr = nullptr;
        void *expertIdsDeviceAddr = nullptr;
        void *expertScalesDeviceAddr = nullptr;
        void *expandIdxDeviceAddr = nullptr;
        void *assistInfoForCombineOutDeviceAddr = nullptr;
        void *quantExpandXOutDeviceAddr = nullptr;
        void *commCmdInfoOutforCombineDeviceAddr = nullptr;
        void *xOutDeviceAddr = nullptr;

        aclTensor *expandX = nullptr;
        aclTensor *expertIds = nullptr;
        aclTensor *expertScales = nullptr;
        aclTensor *expandIdx = nullptr;
        aclTensor *assistInfoForCombineOut = nullptr;
        aclTensor *quantExpandXOut = nullptr;
        aclTensor *commCmdInfoOutforCombine = nullptr;
        aclTensor *xOut = nullptr;

        // Create tensors.
        ret = CreateAclTensor(expandXHostData, expandXShape, &expandXDeviceAddr, aclDataType::ACL_FLOAT16, &expandX);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(expertIdsHostData, expertIdsShape, &expertIdsDeviceAddr, aclDataType::ACL_INT32, &expertIds);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(expertScalesHostData, expertScalesShape, &expertScalesDeviceAddr, aclDataType::ACL_FLOAT,
                            &expertScales);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(expandIdxHostData, expandIdxShape, &expandIdxDeviceAddr, aclDataType::ACL_INT32, &expandIdx);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(assistInfoForCombineOutHostData, assistInfoForCombineOutShape,
                            &assistInfoForCombineOutDeviceAddr, aclDataType::ACL_INT32, &assistInfoForCombineOut);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(quantExpandXOutHostData, quantExpandXOutShape, &quantExpandXOutDeviceAddr,
                            aclDataType::ACL_INT8, &quantExpandXOut);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(commCmdInfoOutforCombineHostData, commCmdInfoOutShapeforcombine,
                            &commCmdInfoOutforCombineDeviceAddr, aclDataType::ACL_INT32, &commCmdInfoOutforCombine);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(xOutHostData, xOutShape, &xOutDeviceAddr, aclDataType::ACL_FLOAT16, &xOut);
        CHECK_RET(ret == ACL_SUCCESS, return ret);

        // Declare variables required for operator execution.
        uint64_t workspaceSize = 0;
        aclOpExecutor *executor = nullptr;
        void *workspaceAddr = nullptr;

        /******************************Call combine_setup********************************************/
        ret = aclnnMoeDistributeCombineSetupGetWorkspaceSize(
            expandX, expertIds, assistInfoForCombineOut, hcomEpName, epWorldSize, args.epRankId, moeExpertNum,
            expertShardType, sharedExpertNum, sharedExpertRankNum, globalBS, commQuantMode, commType, nullptr,
            quantExpandXOut, commCmdInfoOutforCombine, &workspaceSize, &executor);
        CHECK_RET(ret == ACL_SUCCESS,
                LOG_PRINT("[ERROR] aclnnMoeDistributeCombineSetupGetWorkspaceSize failed. ret = %d \n", ret);
                return ret);

        if (workspaceSize > 0) {
            ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] CombineSetup aclrtMalloc failed. ret = %d\n", ret);
                    return ret);
        }

        ret = aclnnMoeDistributeCombineSetup(workspaceAddr, workspaceSize, executor, args.combinesetupstream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnMoeDistributeCombineSetup failed. ret = %d \n", ret);
                return ret);

        ret = aclrtSynchronizeStreamWithTimeout(args.combinesetupstream, timeOut);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSynchronizeStreamWithTimeout failed. ret = %d \n", ret);
                return ret);
        LOG_PRINT("[INFO] device_%d aclnnMoeDistributeCombineSetup execute successfully.\n", args.rankId);

        /******************************Call combine_teardown********************************************/
        ret = aclnnMoeDistributeCombineTeardownGetWorkspaceSize(
            expandX, quantExpandXOut, expertIds, expandIdx, expertScales, commCmdInfoOutforCombine, nullptr, nullptr,
            hcomEpName, epWorldSize, args.epRankId, moeExpertNum, expertShardType, sharedExpertNum, sharedExpertRankNum,
            globalBS, commQuantMode, commType, nullptr, xOut, &workspaceSize, &executor);
        CHECK_RET(ret == ACL_SUCCESS,
                LOG_PRINT("[ERROR] aclnnMoeDistributeCombineTeardownGetWorkspaceSize failed. ret = %d \n", ret);
                return ret);

        if (workspaceSize > 0) {
            ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] CombineTeardown aclrtMalloc failed. ret = %d\n", ret);
                    return ret);
        }

        ret = aclnnMoeDistributeCombineTeardown(workspaceAddr, workspaceSize, executor, args.combineteardownstream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnMoeDistributeCombineTeardown failed. ret = %d \n", ret);
                return ret);

        ret = aclrtSynchronizeStreamWithTimeout(args.combineteardownstream, timeOut);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSynchronizeStreamWithTimeout failed. ret = %d \n", ret);
                return ret);
        LOG_PRINT("[INFO] device_%d aclnnMoeDistributeCombineTeardown execute successfully.\n", args.rankId);

        // Release resources.
        if (workspaceSize > 0) {
            aclrtFree(workspaceAddr);
        }
        DestroyTensor(expandX);
        DestroyTensor(expertIds);
        DestroyTensor(expertScales);
        DestroyTensor(expandIdx);
        DestroyTensor(assistInfoForCombineOut);
        DestroyTensor(quantExpandXOut);
        DestroyTensor(commCmdInfoOutforCombine);
        DestroyTensor(xOut);

        FreeDeviceAddr(expandXDeviceAddr);
        FreeDeviceAddr(expertIdsDeviceAddr);
        FreeDeviceAddr(expertScalesDeviceAddr);
        FreeDeviceAddr(expandIdxDeviceAddr);
        FreeDeviceAddr(assistInfoForCombineOutDeviceAddr);
        FreeDeviceAddr(quantExpandXOutDeviceAddr);
        FreeDeviceAddr(commCmdInfoOutforCombineDeviceAddr);
        FreeDeviceAddr(xOutDeviceAddr);

        return 0;
    }

    int main(int argc, char *argv[])
    {
        int ret = aclInit(nullptr);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclInit failed. ret = %d \n", ret); return ret);

        aclrtStream combineSetupStream[DEV_NUM];
        aclrtStream combineTeardownStream[DEV_NUM];
        aclrtContext context[DEV_NUM];
        for (uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
            ret = aclrtSetDevice(rankId);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetDevice failed. ret = %d \n", ret); return ret);
            ret = aclrtCreateContext(&context[rankId], rankId);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateContext failed. ret = %d \n", ret); return ret);
            ret = aclrtCreateStream(&combineSetupStream[rankId]);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateStream failed. ret = %d \n", ret); return ret);
            ret = aclrtCreateStream(&combineTeardownStream[rankId]);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateStream failed. ret = %d \n", ret); return ret);
        }

        int32_t devices[DEV_NUM];
        for (int i = 0; i < DEV_NUM; i++) {
            devices[i] = i;
        }

        HcclComm comms[DEV_NUM];
        ret = HcclCommInitAll(DEV_NUM, devices, comms);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclCommInitAll failed. ret = %d \n", ret); return ret);

        Args args[DEV_NUM];
        std::vector<std::unique_ptr<std::thread>> threads(DEV_NUM);
        for (uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
            args[rankId].rankId = rankId;
            args[rankId].epRankId = rankId;
            args[rankId].hcclEpComm = comms[rankId];
            args[rankId].combinesetupstream = combineSetupStream[rankId];
            args[rankId].combineteardownstream = combineTeardownStream[rankId];
            args[rankId].context = context[rankId];
            threads[rankId].reset(new (std::nothrow) std::thread(&LaunchOneProcess, std::ref(args[rankId])));
        }
        for (uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
            threads[rankId]->join();
        }
        aclFinalize();
        return 0;
    }
    \`\`\`

<!-- end id10 -->
