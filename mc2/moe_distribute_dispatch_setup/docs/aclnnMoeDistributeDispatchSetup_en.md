# aclnnMoeDistributeDispatchSetup

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T14:25:27.003Z -->

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

- Description:
    - Performs quantization on Token data (optional), and performs AllToAllV communication in the EP (Expert Parallelism) domain based on the topK experts selected by each token. Only data sending and communication status sending are performed. The operator exits immediately after issuing the communication instruction, without waiting for communication completion. Data reception and postprocessing are completed by the aclnnMoeDistributeDispatchTeardown API.

    - Note that this API must be used together with aclnnMoeDistributeDispatchTeardown, aclnnMoeDistributeCombineSetup, and aclnnMoeDistributeCombineTeardown.

## Function Prototype

Each operator is a two-phase API. Call the "aclnnMoeDistributeDispatchSetupGetWorkspaceSize" API first to obtain the workspace size required for computation and the executor that includes the computation process of the operator. Then, call the "aclnnMoeDistributeDispatchSetup" API to perform the computation. The "aclnnMoeDistributeDispatchSetupTeardownCalcOutputSize" API is provided to calculate the size of partial outputs of "aclnnMoeDistributeDispatchSetup".

```cpp
aclnnStatus aclnnMoeDistributeDispatchSetupGetWorkspaceSize(
    const aclTensor* x,
    const aclTensor* expertIds,
    const aclTensor* scalesOptional,
    const aclTensor* xActiveMaskOptional,
    const char*      groupEp,
    int64_t          epWorldSize,
    int64_t          epRankId,
    int64_t          moeExpertNum,
    int64_t          expertShardType,
    int64_t          sharedExpertNum,
    int64_t          sharedExpertRankNum,
    int64_t          quantMode,
    int64_t          globalBs,
    int64_t          commType,
    const char*      commAlg,
    aclTensor*       yOut,
    aclTensor*       expandIdxOut,
    aclTensor*       commCmdInfoOut,
    uint64_t*        workspaceSize,
    aclOpExecutor**  executor)
```

```cpp
aclnnStatus aclnnMoeDistributeDispatchSetup(
    void             *workspace,
    uint64_t         workspaceSize,
    aclOpExecutor    *executor,
    aclrtStream      stream)
```

```cpp
aclnnStatus aclnnMoeDistributeDispatchSetupTeardownCalcOutputSize(
    const aclTensor* x,
    const aclTensor* expertIds,
    const aclTensor* scalesOptional,
    const aclTensor* xActiveMaskOptional,
    const char*      groupEp,
    int64_t          epWorldSize,
    int64_t          epRankId,
    int64_t          moeExpertNum,
    int64_t          expertShardType,
    int64_t          sharedExpertNum,
    int64_t          sharedExpertRankNum,
    int64_t          quantMode,
    int64_t          globalBs,
    int64_t          expertTokenNumsType,
    int64_t          commType,
    const char*      commAlg,
    uint64_t&        tokenMsgSize,
    uint64_t&        expandIdxOutSize,
    uint64_t&        assistInfoForCombineOutSize,
    uint64_t&        commCmdInfoOutSize)
```

## aclnnMoeDistributeDispatchSetupGetWorkspaceSize

- **Parameters**

    <table style="undefined;table-layout: fixed; width: 1550px"> <colgroup>
    <col style="width: 110px">
    <col style="width: 120px">
    <col style="width: 305px">
    <col style="width: 330px">
    <col style="width: 210px">
    <col style="width: 100px">
    <col style="width: 180px">
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
    <td>Indicates the token data sent by this card.</td>
    <td>Required to be a 2D Tensor.</td>
    <td>FLOAT16, BFLOAT16</td>
    <td>ND</td>
    <td>(Bs, H)</td>
    <td>√</td>
    </tr>
    <tr>
    <td>expertIds</td>
    <td>Input</td>
    <td>topK expert indexes of each token.</td>
    <td>Required to be a 2D Tensor.</td>
    <td>INT32</td>
    <td>ND</td>
    <td>(Bs, K)</td>
    <td>√</td>
    </tr>
    <tr>
    <td>scalesOptional</td>
    <td>Input</td>
    <td>Quantization smoothing parameter of each expert.</td>
    <td>Required to be a 2D Tensor. Pass a null pointer in non-quantized use cases. In dynamic quantization, valid data or a null pointer can be optionally passed.</td>
    <td>FLOAT32</td>
    <td>ND</td>
    <td>(sharedExpertNum + moeExpertNum, H)</td>
    <td>√</td>
    </tr>
    <tr>
    <td>xActiveMaskOptional</td>
    <td>Input</td>
    <td>Indicates whether tokens participate in communication.</td>
    <td>Required to be a 1D Tensor. You can optionally pass input valid data or pass a null pointer. A null pointer indicates that all tokens will participate in communication.</td>
    <td>BOOL</td>
    <td>ND</td>
    <td>(Bs, )</td>
    <td>√</td>
    </tr>
    <tr>
    <td>groupEp</td>
    <td>Input</td>
    <td>EP communication domain name (expert parallelism communication domain).</td>
    <td>The string length range is [1, 128).</td>
    <td>STRING</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>epWorldSize</td>
    <td>Input</td>
    <td>EP communication domain size.</td>
    <td>-</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>epRankId</td>
    <td>Input</td>
    <td>ID of this card in the EP domain.</td>
    <td>Value range: [0, epWorldSize). The epRankId of each card in the same EP communication domain must be unique.</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>moeExpertNum</td>
    <td>Input</td>
    <td>MoE expert count.</td>
    <td>Value range: (0, 512]. moeExpertNum % (epWorldSize - sharedExpertRankNum) = 0 must be satisfied.</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>expertShardType</td>
    <td>Input</td>
    <td>Indicates the shared expert card distribution type.</td>
    <td>Currently only supports passing 0, indicating that shared expert cards are arranged before MoE expert cards.</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>sharedExpertNum</td>
    <td>Input</td>
    <td>Indicates the shared expert count.</td>
    <td>Value range: [0, 4].</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>sharedExpertRankNum</td>
    <td>Input</td>
    <td>Indicates the number of shared expert cards.</td>
    <td>Value range [0, epWorldSize / 2].</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>quantMode</td>
    <td>Input</td>
    <td>Indicates the quantization mode.</td>
    <td>Value range [0, 4]. 0 indicates non-quantization, 1 indicates static quantization, 2 indicates Pertoken dynamic quantization, 3 indicates Pergroup dynamic quantization, and 4 indicates MX quantization. Currently only supports 0 and 4.</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>globalBs</td>
    <td>Input</td>
    <td>Global batch size in the EP domain.</td>
    <td><ul><li>If all ranks have the same Bs, globalBs = Bs * epWorldSize or 0.</li><li>If ranks have different Bs values, globalBs = maxBs * epWorldSize (maxBs is the maximum Bs of a single device).</li></ul></td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>commType</td>
    <td>Input</td>
    <td>Indicates the communication solution selection.</td>
    <td>Value range [0, 2]. 0 indicates the AICPU-SDMA solution, 1 indicates the CCU solution, 2 indicates the URMA solution. The current version only supports 2.</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>commAlg</td>
    <td>Input</td>
    <td>Indicates the communication affinity memory layout algorithm.</td>
    <td>Reserved field. Not supported in the current version. Pass a null pointer or an empty string.</td>
    <td>STRING</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>yOut</td>
    <td>Output</td>
    <td>Indicates the communication data to be sent by this card. The communication data is an algorithmic rearrangement of the input token data. If quantization is required, the input tokens are quantized first and then the data is rearranged.</td>
    <td>Required to be a 2D Tensor.</td>
    <td>FLOAT16, BFLOAT16, INT8, HiFP8, FP8E5M2, FP8E4M3</td>
    <td>ND</td>
    <td>(BS * (K + sharedExpertNum), tokenMsgSize)</td>
    <td>√</td>
    </tr>
    <tr>
    <td>expandIdxOut</td>
    <td>Output</td>
    <td>Indicates the number of tokens sent to the same expert, corresponding to expandIdx in the Combine series operators.</td>
    <td>Required to be a 1D Tensor.</td>
    <td>INT32</td>
    <td>ND</td>
    <td>(BS * K, )</td>
    <td>√</td>
    </tr>
    <tr>
    <td>commCmdInfoOut</td>
    <td>Output</td>
    <td>Communication cmd information.</td>
    <td>Required to be a 1D Tensor.</td>
    <td>INT32</td>
    <td>ND</td>
    <td>(BS * (K + sharedExpertNum) + epWorldSize * localExpertNum) * 16</td>
    <td>√</td>
    </tr>
    <tr>
    <td>workspaceSize</td>
    <td>Output</td>
    <td>Returns the workspace size that needs to be requested on the Device side.</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>executor</td>
    <td>Output</td>
    <td>Returns the op executor, which includes the computation flow of the operator.</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    </tbody>
    </table>

    <!-- npu="950" id7 -->
    - <term>Ascend 950PR&950DT products</term>:
        - scalesOptional: pass a null pointer in non-quantized use cases. For dynamic quantization, you can optionally pass valid data or a null pointer.
        - xActiveMaskOptional: you can optionally pass valid data or a null pointer. Passing a null pointer indicates that all tokens will participate in communication.
        - The string length range of groupEp is [1, 128).
        - The value range of epWorldSize is [2, 384]. Currently, only 2 and 8 are supported.
        - The value range of epRankId is [0, epWorldSize). The epRankId of each card in the same EP communication domain must be unique.
        - The value range of moeExpertNum is (0, 512].
        - expertShardType: currently only 0 is supported, indicating that shared expert cards are arranged before MoE expert cards.
        - The current value range of sharedExpertNum is [0, 4].
        - The value range of sharedExpertRankNum is [0, epWorldSize / 2].
        - globalBs: if the number of Bs is the same on each rank, globalBs = Bs *epWorldSize or globalBs = 0; if the number of Bs is different on each rank, globalBs = maxBs* epWorldSize, where maxBs indicates the maximum value of Bs on a single card.
        - commType currently only supports 2.
        - commAlg is not supported in the current version. Pass a null pointer.

    <!-- end id7 -->

- **Return value**

  Returns an aclnnStatus status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

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
  <td>Required input and output tensors are null pointers.</td>
  </tr>
  <tr>
  <td>ACLNN_ERR_PARAM_INVALID</td>
  <td>161002</td>
  <td>The data types of the input and output are out of the supported range.</td>
  </tr>
  <tr>
  <td rowspan="2">ACLNN_ERR_INNER_TILING_ERROR</td>
  <td rowspan="2">561002</td>
  <td>The shapes of the input and output are out of the supported range.</td>
  </tr>
  <tr>
      <td>The parameter value is out of the supported range.</td>
  </tr>
  </tbody>
  </table>

## aclnnMoeDistributeDispatchSetup

- **Parameters**

    <table style="undefined;table-layout: fixed; width: 1100px"><colgroup>
    <col style="width: 200px">
    <col style="width: 130px">
    <col style="width: 770px">
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
        <td>Workspace memory address requested on the Device.</td>
    </tr>
    <tr>
        <td>workspaceSize</td>
        <td>Input</td>
        <td>Workspace size requested on the Device, obtained by calling the first-phase API aclnnMoeDistributeDispatchSetupGetWorkspaceSize.</td>
    </tr>
    <tr>
        <td>executor</td>
        <td>Input</td>
        <td>op executor, includes the operator computation flow.</td>
    </tr>
    <tr>
        <td>stream</td>
        <td>Input</td>
        <td>Specifies the Stream for task execution.</td>
    </tr>
    </tbody>
    </table>

- **Return Value**

    Returns an aclnnStatus status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## aclnnMoeDistributeDispatchSetupTeardownCalcOutputSize

- **Parameters**

    <table style="undefined;table-layout: fixed; width: 1550px"> <colgroup>
    <col style="width: 110px">
    <col style="width: 120px">
    <col style="width: 305px">
    <col style="width: 330px">
    <col style="width: 210px">
    <col style="width: 100px">
    <col style="width: 180px">
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
    <td>Indicates the token data sent by this card.</td>
    <td>Required to be a 2D tensor.</td>
    <td>FLOAT16, BFLOAT16</td>
    <td>ND</td>
    <td>(Bs, H)</td>
    <td>√</td>
    </tr>
    <tr>
    <td>expertIds</td>
    <td>Input</td>
    <td>The topK expert indexes of each token.</td>
    <td>Required to be a 2D Tensor.</td>
    <td>INT32</td>
    <td>ND</td>
    <td>(Bs, K)</td>
    <td>√</td>
    </tr>
    <tr>
    <td>scalesOptional</td>
    <td>Input</td>
    <td>Quantization smoothing parameters of each expert.</td>
    <td>Required to be a 2D Tensor. For non-quantized fields, pass a null pointer. For dynamic quantization, you can optionally pass valid data or a null pointer.</td>
    <td>FLOAT32</td>
    <td>ND</td>
    <td>(sharedExpertNum + moeExpertNum, H)</td>
    <td>√</td>
    </tr>
    <tr>
    <td>xActiveMaskOptional</td>
    <td>Input</td>
    <td>Indicates whether a token participates in communication.</td>
    <td>Required to be a 1D tensor. You can optionally pass input valid data or pass a null pointer. Passing a null pointer indicates that all tokens will participate in communication.</td>
    <td>BOOL</td>
    <td>ND</td>
    <td>(Bs, )</td>
    <td>√</td>
    </tr>
    <tr>
    <td>groupEp</td>
    <td>Input</td>
    <td>Name of the EP communication domain (expert parallel communication domain).</td>
    <td>String length range is [1, 128).</td>
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
    <td>ID of this card in the EP domain.</td>
    <td>Value range [0, epWorldSize). epRankId of each card in the same EP communication domain must be unique.</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>moeExpertNum</td>
    <td>Input</td>
    <td>MoE expert count.</td>
    <td>Value range (0, 512]. moeExpertNum % (epWorldSize - sharedExpertRankNum) = 0 must be satisfied.</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>expertShardType</td>
    <td>Input</td>
    <td>Indicates the distribution type of shared expert cards.</td>
    <td>Currently only supports 0, indicating that shared expert cards are arranged before MoE expert cards.</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>sharedExpertNum</td>
    <td>Input</td>
    <td>Indicates the shared expert count.</td>
    <td>Value range: [0, 4].</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>sharedExpertRankNum</td>
    <td>Input</td>
    <td>Indicates the number of shared expert cards.</td>
    <td>Value range: [0, epWorldSize / 2].</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>quantMode</td>
    <td>Input</td>
    <td>Indicates the quantization mode.</td>
    <td>Value range: [0, 4]. 0 indicates no quantization, 1 indicates static quantization, 2 indicates Pertoken dynamic quantization, 3 indicates Pergroup dynamic quantization, 4 indicates MX quantization. Currently, only 0 and 4 are supported.</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>globalBs</td>
    <td>Input</td>
    <td>Global batch size in the EP domain.</td>
    <td><ul><li>When the Bs values of all ranks are the same, globalBs = Bs * epWorldSize or 0.</li><li>When the Bs values of ranks are different, globalBs = maxBs * epWorldSize (maxBs is the maximum Bs of a single device).</li></ul></td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>expertTokenNumsType</td>
    <td>Input</td>
    <td>Semantic type of the values in output expertTokenNums.</td>
    <td><ul><li>Value 0: the output in expertTokenNums is the prefix sum of the number of tokens processed by each expert.</li><li>Value 1: the output in expertTokenNums is the number of tokens processed by each expert.</li></ul></td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>commType</td>
    <td>Input</td>
    <td>Indicates the communication solution selection.</td>
    <td>Value range [0, 2]. 0 indicates the AICPU-SDMA solution, 1 indicates the CCU solution, 2 indicates the URMA solution. Only 2 is supported in the current version.</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>commAlg</td>
    <td>Input</td>
    <td>Indicates the communication-affinity memory layout algorithm.</td>
    <td>Reserved field. Not supported in the current version. Pass a null pointer or an empty string.</td>
    <td>STRING</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>tokenMsgSize</td>
    <td>Output</td>
    <td>Second-dimension output of yOut of the aclnnMoeDistributeDispatchSetup API. Indicates the dimension information of each token during data communication.</td>
    <td>-</td>
    <td>INT64</td>
    <td>-</td>
    <td>In non-quantization use cases, tokenMsgSize = Align256(H). In quantization use cases, when quantmode = 1, tokenMsgSize = Align512(H); when quantmode = 2, tokenMsgSize = Align512(Align32(H) + 4); when quantmode = 3, tokenMsgSize = Align512(Align128(H) + ceilDiv(H,128)); when quantmode = 4, tokenMsgSize = Align512(Align128(H) + Align2(ceilDiv(H,32))), where AlignN(x) = ((x + N - 1) / N) * N, ceilDiv(n,m) = ((n + m - 1) / m).</td>
    <td>-</td>
    </tr>
    <tr>
    <td>expandIdxOutSize</td>
    <td>Output</td>
    <td>Size of the expandIdxOut space of the aclnnMoeDistributeDispatchSetup interface.</td>
    <td>-</td>
    <td>INT64</td>
    <td>-</td>
    <td>(BS * K)</td>
    <td>-</td>
    </tr>
    <tr>
    <td>assistInfoForCombineOutSize</td>
    <td>Output</td>
    <td>Memory size of assistInfoForCombineOut of the aclnnMoeDistributeDispatchSetup API.</td>
    <td>-</td>
    <td>INT64</td>
    <td>-</td>
    <td>(A * 128)</td>
    <td>-</td>
    </tr>
    <tr>
    <td>commCmdInfoOutSize</td>
    <td>Output</td>
    <td>Size of commCmdInfoOut of the aclnnMoeDistributeDispatchSetup interface.</td>
    <td>-</td>
    <td>INT64</td>
    <td>-</td>
    <td>(BS * (K + sharedExpertNum) + epWorldSize * localExpertNum) * 16</td>
    <td>-</td>
    </tr>
    </tbody>
    </table>

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

1. Deterministic computation:
     - aclnnMoeDistributeDispatchSetup defaults to a deterministic implementation.

2. The aclnnMoeDistributeDispatchSetup API must be used together with the aclnnMoeDistributeDispatchTeardown, aclnnMoeDistributeCombineSetup, and aclnnMoeDistributeCombineTeardown APIs.

3. The values of the `groupEp`, `epWorldSize`, `moeExpertNum`, `expertShardType`, `sharedExpertNum`, `sharedExpertRankNum`, `globalBs`, `commQuantMode`, `commType`, and `commAlg` parameters used in the API call must be the same across all cards. The values of the `groupEp`, `epWorldSize`, `expertShardType`, `sharedExpertNum`, `sharedExpertRankNum`, `globalBs`, `commQuantMode`, `commType`, and `commAlg` parameters must also be the same in different layers of the network, and consistent with the corresponding parameters of `aclnnMoeDistributeDispatchTeardown`, `aclnnMoeDistributeCombineSetup`, and `aclnnMoeDistributeCombineTeardown`.

4. Description of the shape format in Parameters:
    - A: indicates the maximum number of tokens that this card can receive. The value range is as follows:

      - For MoE experts, when `globalBs` is 0, A >= `BS` *`epWorldSize`* min(`localExpertNum`, `K`) must be satisfied; when `globalBs` is non-zero, A >= `globalBs` * min(`localExpertNum`, `K`) must be satisfied.
      - For shared experts, when `globalBs` is 0, A = `BS` *`epWorldSize`* `sharedExpertNum` / `sharedExpertRankNum` must be satisfied; when `globalBs` is non-zero, A = `globalBs` * `sharedExpertNum` / `sharedExpertRankNum` must be satisfied.
    - H: indicates the hidden size. The value range is [1024, 8192]. Currently, only 4096 and 7168 are supported.
    - BS: indicates batch sequence size, that is, the number of tokens finally output by this card. The value range is 0 < BS ≤ 512. Currently, only 8, 16, and 256 are supported.
    - K: indicates that topK experts are selected. The value range is 0 < `K` ≤ 16, and 0 < `K` ≤ `moeExpertNum` must also be satisfied. Currently, only 6 and 8 are supported.
    - localExpertNum: indicates the number of experts on the current card.

      - For a shared expert card, localExpertNum = 1.
      - For MoE expert cards, localExpertNum = `moeExpertNum` / (`epWorldSize` - `sharedExpertRankNum`). moeExpertNum currently only supports 32.
    - tokenMsgSize: indicates the dimension information of each token during data communication.
      - In non-quantization use cases, tokenMsgSize = Align256(H).
      - In quantization use cases, tokenMsgSize = Align512(Align32(H) + 4 ), where AlignN(x) = ((x + N - 1) / N) * N.

    - The current version does not support shared experts. sharedExpertNum and sharedExpertRankNum currently only support 0.

5. HCCL_BUFFSIZE:

    Before calling this API, check whether the value of the `HCCL_BUFFSIZE` environment variable is proper. This environment variable indicates the memory size occupied by a single communication domain, in MB. The default value is 200MB if the variable is not configured. The value must be >= 2 and satisfy >= 4 *(`localExpertNum`* `maxBs` *`epWorldSize`* Align512(Align32(2 *H) + 44) + (`K` + `sharedExpertNum`)* `maxBs` *Align512(2* `H`)). `localExpertNum` indicates the number of local experts on this card when MoE expert cards are used, where Align512(x) = ((x + 512 - 1) / 512) *512 and Align32(x) = ((x + 32 - 1) / 32)* 32.

6. Communication domain usage constraints:
    - The aclnnMoeDistributeDispatchSetup, aclnnMoeDistributeDispatchTeardown, aclnnMoeDistributeCombineSetup, and aclnnMoeDistributeCombineTeardown APIs in a model only support the same EP communication domain, and no other operators are allowed in the communication domain.

7. Communication mode constraints:

  <!-- npu="950" id8 -->
  - <term>Ascend 950PR&950DT products</term>: Only URMA communication is supported.

  <!-- end id8 -->

## Example

- File preparation:

    1. Create and modify the rank_table_m2.json file as instructed below.

    2. Copy the project to both servers, and configure the rank_table_m2.json file content based on the device IP of each machine. Ensure that the rank_table_m2.json files on both servers remain consistent.

    3. Install the CANN package, then compile and run the code by referring to [Operator Invocation](../../../docs/zh/invocation/quick_op_invocation.md).

- About rankTable:

    1. Developers can use the ranktable file to configure the NPU resource information for collective communication. For details, see the section "Communication Function Development > Cluster Information Configuration > Configuring Resource Information Using a ranktable File" in [*Collective Communication User Guide*](https://hiascend.com/en/document/redirect/CannCommunityHcclUg).

    2. Use `cat /etc/hccn.conf` or `for i in seq 0 7; do echo "===================> dev$i, NPU$((i+1))"; hccn_tool -i $i -ip -g; done` to query the device IP of the server. Then fill in the JSON file by referring to the collective communication documentation.

    > Note: In a two-machine 16-card scenario, the device_id of both machines is 0 to 7, the rank_id of one machine is 0 to 7, and the rank_id of the other machine is 8 to 15. In a single-machine 16-card scenario, both device_id and rank_id are 0 to 15.

<!-- npu="950" id9 -->
- <term>Ascend 950PR&950DT products</term>:

    The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

    ```cpp
    #include <cstdint>
    #include <cstdio>
    #include <iostream>
    #include <thread>
    #include <vector>

    #include "acl/acl.h"
    #include "hccl/hccl.h"
    #include "aclnnop/aclnn_moe_distribute_dispatch_setup.h"
    #include "aclnnop/aclnn_moe_distribute_dispatch_teardown.h"

    #define CHECK_RET(cond, return_expr) \
        do { \
            if (!(cond)) { \
                return_expr; \
            } \
        } while (0)

    #define LOG_PRINT(message, ...) \
        do { \
            printf(message, ##__VA_ARGS__); \
        } while (0)

    constexpr int32_t DEV_NUM = 2;
    constexpr int64_t BS = 8;
    constexpr int64_t H = 1024;
    constexpr int64_t K = 1;
    constexpr int64_t MOE_EXPERT_NUM = 2;
    constexpr int64_t EP_WORLD_SIZE = DEV_NUM;
    constexpr int64_t GLOBAL_BS = BS * EP_WORLD_SIZE;
    constexpr int64_t LOCAL_EXPERT_NUM = MOE_EXPERT_NUM / EP_WORLD_SIZE;
    constexpr int64_t LOCAL_TOKEN_NUM = GLOBAL_BS * ((LOCAL_EXPERT_NUM < K) ? LOCAL_EXPERT_NUM : K);
    constexpr int64_t EXPERT_SHARD_TYPE = 0;
    constexpr int64_t SHARED_EXPERT_NUM = 0;
    constexpr int64_t SHARED_EXPERT_RANK_NUM = 0;
    constexpr int64_t QUANT_MODE = 0;
    constexpr int64_t EXPERT_TOKEN_NUMS_TYPE = 1;
    constexpr int64_t COMM_TYPE = 2;
    constexpr int64_t TIMEOUT = 100000000;

    template <typename Func>
    class Guard {
    public:
        explicit Guard(Func &func)
            : func_(func)
        {}

        ~Guard()
        {
            func_();
        }

    private:
        Func &func_;
    };

    struct Args {
        uint32_t rankId;
        HcclComm hcclComm;
        aclrtContext context;
        aclrtStream setupStream;
        aclrtStream teardownStream;
    };

    int64_t GetShapeSize(const std::vector<int64_t> &shape)
    {
        int64_t size = 1;
        for (int64_t dim : shape) {
            size *= dim;
        }
        return size;
    }

    template <typename T>
    int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, aclDataType dataType,
                        void **deviceAddr, aclTensor **tensor)
    {
        const size_t size = static_cast<size_t>(GetShapeSize(shape)) * sizeof(T);
        int ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc failed. ret = %d\n", ret); return ret);

        ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMemcpy failed. ret = %d\n", ret); return ret);

        std::vector<int64_t> strides(shape.size(), 1);
        for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
            strides[i] = shape[i + 1] * strides[i + 1];
        }
        *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                                shape.data(), shape.size(), *deviceAddr);
        CHECK_RET(*tensor != nullptr, LOG_PRINT("[ERROR] aclCreateTensor failed.\n"); return -1);
        return ACL_SUCCESS;
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
        std::cout << "[INFO] device_" << args.rankId << " worker start." << std::endl;
        int ret = aclrtSetCurrentContext(args.context);
        CHECK_RET(ret == ACL_SUCCESS,
                LOG_PRINT("[ERROR] device_%u aclrtSetCurrentContext failed. ret = %d\n", args.rankId, ret);
                return ret);

        char groupEp[128] = {0};
        ret = HcclGetCommName(args.hcclComm, groupEp);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] device_%u HcclGetCommName failed. ret = %d\n", args.rankId, ret);
                return ret);

        auto runtimeCleanup = [&args]() {
            HcclCommDestroy(args.hcclComm);
            aclrtDestroyStream(args.setupStream);
            aclrtDestroyStream(args.teardownStream);
            aclrtDestroyContext(args.context);
            aclrtResetDevice(args.rankId);
        };
        auto runtimeGuard = Guard<decltype(runtimeCleanup)>(runtimeCleanup);

        // setup input/output shapes
        const std::vector<int64_t> xShape{BS, H};
        const std::vector<int64_t> expertIdsShape{BS, K};
        const std::vector<int64_t> yShape{BS * K, H};
        const std::vector<int64_t> expandIdxShape{BS * K};
        const std::vector<int64_t> commCmdInfoShape{(BS * K + EP_WORLD_SIZE * LOCAL_EXPERT_NUM) * 16};

        // teardown output shapes
        const std::vector<int64_t> expandXShape{LOCAL_TOKEN_NUM, H};
        // The current teardown ACLNN wrapper requires a non-null tensor here even when quantMode is 0.
        const std::vector<int64_t> dynamicScalesShape{1};
        const std::vector<int64_t> assistInfoShape{LOCAL_TOKEN_NUM * 128};
        const std::vector<int64_t> expertTokenNumsShape{LOCAL_EXPERT_NUM};

        std::vector<int16_t> xHostData(GetShapeSize(xShape), 1);
        std::vector<int32_t> expertIdsHostData(GetShapeSize(expertIdsShape));
        for (int64_t i = 0; i < BS; ++i) {
            expertIdsHostData[i] = static_cast<int32_t>(i % MOE_EXPERT_NUM);
        }
        std::vector<int16_t> yHostData(GetShapeSize(yShape), 0);
        std::vector<int32_t> expandIdxHostData(GetShapeSize(expandIdxShape), 0);
        std::vector<int32_t> commCmdInfoHostData(GetShapeSize(commCmdInfoShape), 0);
        std::vector<int16_t> expandXHostData(GetShapeSize(expandXShape), 0);
        std::vector<float> dynamicScalesHostData(GetShapeSize(dynamicScalesShape), 0);
        std::vector<int32_t> assistInfoHostData(GetShapeSize(assistInfoShape), 0);
        std::vector<int64_t> expertTokenNumsHostData(GetShapeSize(expertTokenNumsShape), 0);

        void *xAddr = nullptr;
        void *expertIdsAddr = nullptr;
        void *yAddr = nullptr;
        void *expandIdxAddr = nullptr;
        void *commCmdInfoAddr = nullptr;
        void *expandXAddr = nullptr;
        void *dynamicScalesAddr = nullptr;
        void *assistInfoAddr = nullptr;
        void *expertTokenNumsAddr = nullptr;
        void *setupWorkspace = nullptr;
        void *teardownWorkspace = nullptr;

        aclTensor *x = nullptr;
        aclTensor *expertIds = nullptr;
        aclTensor *y = nullptr;
        aclTensor *expandIdx = nullptr;
        aclTensor *commCmdInfo = nullptr;
        aclTensor *expandX = nullptr;
        aclTensor *dynamicScales = nullptr;
        aclTensor *assistInfo = nullptr;
        aclTensor *expertTokenNums = nullptr;

        auto tensorCleanup = [&]() {
            DestroyTensor(x);
            DestroyTensor(expertIds);
            DestroyTensor(y);
            DestroyTensor(expandIdx);
            DestroyTensor(commCmdInfo);
            DestroyTensor(expandX);
            DestroyTensor(dynamicScales);
            DestroyTensor(assistInfo);
            DestroyTensor(expertTokenNums);
            FreeDeviceAddr(xAddr);
            FreeDeviceAddr(expertIdsAddr);
            FreeDeviceAddr(yAddr);
            FreeDeviceAddr(expandIdxAddr);
            FreeDeviceAddr(commCmdInfoAddr);
            FreeDeviceAddr(expandXAddr);
            FreeDeviceAddr(dynamicScalesAddr);
            FreeDeviceAddr(assistInfoAddr);
            FreeDeviceAddr(expertTokenNumsAddr);
            FreeDeviceAddr(setupWorkspace);
            FreeDeviceAddr(teardownWorkspace);
        };
        auto tensorGuard = Guard<decltype(tensorCleanup)>(tensorCleanup);

        ret = CreateAclTensor(xHostData, xShape, aclDataType::ACL_FLOAT16, &xAddr, &x);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(expertIdsHostData, expertIdsShape, aclDataType::ACL_INT32, &expertIdsAddr, &expertIds);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(yHostData, yShape, aclDataType::ACL_FLOAT16, &yAddr, &y);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(expandIdxHostData, expandIdxShape, aclDataType::ACL_INT32, &expandIdxAddr, &expandIdx);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret =
            CreateAclTensor(commCmdInfoHostData, commCmdInfoShape, aclDataType::ACL_INT32, &commCmdInfoAddr, &commCmdInfo);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(expandXHostData, expandXShape, aclDataType::ACL_FLOAT16, &expandXAddr, &expandX);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(dynamicScalesHostData, dynamicScalesShape, aclDataType::ACL_FLOAT, &dynamicScalesAddr,
                            &dynamicScales);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(assistInfoHostData, assistInfoShape, aclDataType::ACL_INT32, &assistInfoAddr, &assistInfo);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(expertTokenNumsHostData, expertTokenNumsShape, aclDataType::ACL_INT64, &expertTokenNumsAddr,
                            &expertTokenNums);
        CHECK_RET(ret == ACL_SUCCESS, return ret);

        uint64_t setupWorkspaceSize = 0;
        aclOpExecutor *setupExecutor = nullptr;
        ret = aclnnMoeDistributeDispatchSetupGetWorkspaceSize(
            x, expertIds, nullptr, nullptr, groupEp, EP_WORLD_SIZE, args.rankId, MOE_EXPERT_NUM, EXPERT_SHARD_TYPE,
            SHARED_EXPERT_NUM, SHARED_EXPERT_RANK_NUM, QUANT_MODE, GLOBAL_BS, COMM_TYPE, nullptr, y, expandIdx, commCmdInfo,
            &setupWorkspaceSize, &setupExecutor);
        CHECK_RET(ret == ACL_SUCCESS,
                LOG_PRINT("[ERROR] device_%u DispatchSetupGetWorkspaceSize failed. ret = %d\n", args.rankId, ret);
                return ret);

        if (setupWorkspaceSize > 0) {
            ret = aclrtMalloc(&setupWorkspace, setupWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CHECK_RET(ret == ACL_SUCCESS,
                    LOG_PRINT("[ERROR] device_%u setup workspace malloc failed. ret = %d\n", args.rankId, ret);
                    return ret);
        }
        ret = aclnnMoeDistributeDispatchSetup(setupWorkspace, setupWorkspaceSize, setupExecutor, args.setupStream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] device_%u DispatchSetup failed. ret = %d\n", args.rankId, ret);
                return ret);
        ret = aclrtSynchronizeStreamWithTimeout(args.setupStream, TIMEOUT);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] device_%u setup synchronize failed. ret = %d\n", args.rankId, ret);
                return ret);
        LOG_PRINT("[INFO] device_%u dispatch setup success.\n", args.rankId);

        uint64_t teardownWorkspaceSize = 0;
        aclOpExecutor *teardownExecutor = nullptr;
        ret = aclnnMoeDistributeDispatchTeardownGetWorkspaceSize(
            x, y, expertIds, commCmdInfo, groupEp, EP_WORLD_SIZE, args.rankId, MOE_EXPERT_NUM, EXPERT_SHARD_TYPE,
            SHARED_EXPERT_NUM, SHARED_EXPERT_RANK_NUM, QUANT_MODE, GLOBAL_BS, EXPERT_TOKEN_NUMS_TYPE, COMM_TYPE, nullptr,
            expandX, dynamicScales, assistInfo, expertTokenNums, &teardownWorkspaceSize, &teardownExecutor);
        CHECK_RET(ret == ACL_SUCCESS,
                LOG_PRINT("[ERROR] device_%u DispatchTeardownGetWorkspaceSize failed. ret = %d\n", args.rankId, ret);
                return ret);

        if (teardownWorkspaceSize > 0) {
            ret = aclrtMalloc(&teardownWorkspace, teardownWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CHECK_RET(ret == ACL_SUCCESS,
                    LOG_PRINT("[ERROR] device_%u teardown workspace malloc failed. ret = %d\n", args.rankId, ret);
                    return ret);
        }
        ret = aclnnMoeDistributeDispatchTeardown(teardownWorkspace, teardownWorkspaceSize, teardownExecutor,
                                                args.teardownStream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] device_%u DispatchTeardown failed. ret = %d\n", args.rankId, ret);
                return ret);
        ret = aclrtSynchronizeStreamWithTimeout(args.teardownStream, TIMEOUT);
        CHECK_RET(ret == ACL_SUCCESS,
                LOG_PRINT("[ERROR] device_%u teardown synchronize failed. ret = %d\n", args.rankId, ret);
                return ret);
        LOG_PRINT("[INFO] device_%u dispatch teardown success.\n", args.rankId);
        return ACL_SUCCESS;
    }

    int main(int argc, char *argv[])
    {
        int ret = aclInit(nullptr);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclInit failed. ret = %d\n", ret); return ret);

        aclrtContext contexts[DEV_NUM];
        aclrtStream setupStreams[DEV_NUM];
        aclrtStream teardownStreams[DEV_NUM];
        for (uint32_t rankId = 0; rankId < DEV_NUM; ++rankId) {
            ret = aclrtSetDevice(rankId);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetDevice failed. ret = %d\n", ret); return ret);
            ret = aclrtCreateContext(&contexts[rankId], rankId);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateContext failed. ret = %d\n", ret); return ret);
            ret = aclrtCreateStream(&setupStreams[rankId]);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateStream failed. ret = %d\n", ret); return ret);
            ret = aclrtCreateStream(&teardownStreams[rankId]);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateStream failed. ret = %d\n", ret); return ret);
        }

        int32_t devices[DEV_NUM] = {0, 1};
        HcclComm comms[DEV_NUM];
        ret = HcclCommInitAll(DEV_NUM, devices, comms);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclCommInitAll failed. ret = %d\n", ret); return ret);

        Args args[DEV_NUM];
        int results[DEV_NUM] = {ACL_SUCCESS, ACL_SUCCESS};
        std::thread threads[DEV_NUM];
        for (uint32_t rankId = 0; rankId < DEV_NUM; ++rankId) {
            args[rankId] = {rankId, comms[rankId], contexts[rankId], setupStreams[rankId], teardownStreams[rankId]};
            threads[rankId] =
                std::thread([&args, &results, rankId]() { results[rankId] = LaunchOneProcess(args[rankId]); });
        }
        for (auto &thread : threads) {
            thread.join();
        }

        ret = aclFinalize();
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclFinalize failed. ret = %d\n", ret); return ret);
        for (uint32_t rankId = 0; rankId < DEV_NUM; ++rankId) {
            CHECK_RET(results[rankId] == ACL_SUCCESS,
                    LOG_PRINT("[ERROR] device_%u failed. ret = %d\n", rankId, results[rankId]);
                    return results[rankId]);
        }
        return ACL_SUCCESS;
    }
    ```

<!-- end id9 -->
