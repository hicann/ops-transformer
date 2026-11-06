# MoeDistributeCombineV2

English | [简体中文](./README.md)

## Product Support

| Product | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950DT products</term>                             |    √     |
| <term>Atlas A3 products</term>       |    √     |
| <term>Atlas A2 products</term> |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                               |    ×     |
| <term>Atlas training products</term>                              |    ×     |

## Function Description

- Interface function: Performs AllToAllV communication and then integrates the received data.

    Compared with the MoeDistributeCombine operator, this operator has the following changes:
    - More detailed token information is input to help `MoeDistributeCombineV2` perform synchronization across all devices efficiently. Therefore, the `expandIdx` input parameter with the shape of (`BS` * `K`,) in the original operator is replaced by the `assistInfoForCombine` parameter with the shape of (`A` * 128,);
    - The `sharedExpertXOptional` input parameter is added. When `sharedExpertNum` is 0, you can input the tokens computed by the shared expert;
    - The `commAlg` input parameter is added to replace the `HCCL_INTRA_PCIE_ENABLE` and `HCCL_INTRA_ROCE_ENABLE` environment variables.
    For details, see the following parameter description.
- Computation formula:

    $$
    ataOut = AllToAllV(expandX)\\
    xOut = Sum(expertScales * ataOut) + sharedExpertX
    $$

    When `expertScales` is an empty tensor, the results of routing experts are summed directly. If `sharedExpertXOptional` is passed, the results are still accumulated directly without multiplying the routing weights.

    Note that this operator must be used together with MoeDistributeDispatchV2. It is equivalent to returning the data along the same path used by the MoeDistributeDispatchV2 operator to collect data.

## Parameter Description

<table style="undefined;table-layout: fixed; width: 1392px"> <colgroup>
 <col style="width: 120px">
 <col style="width: 120px">
 <col style="width: 160px">
 <col style="width: 150px">
 <col style="width: 80px">
 </colgroup>
 <thead>
  <tr>
   <th>Parameter Name</th>
   <th>Input/Output/Attribute</th>
   <th>Description</th>
   <th>Data Type</th>
   <th>Data Format</th>
  </tr>
 </thead>
 <tbody>
  <tr>
   <td>expandX</td>
   <td>Input</td>
   <td>Token features expanded based on expertIds.</td>
   <td>FLOAT16, BFLOAT16</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>expertIds</td>
   <td>Input</td>
   <td>Indexes of the topK experts for each token.</td>
   <td>INT32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>assistInfoForCombine</td>
   <td>Input</td>
   <td>Indicates the number of tokens received by the same expert. Corresponds to the assistInfoForCombineOut output in MoeDistributeDispatchV2.</td>
   <td>INT32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>epSendCounts</td>
   <td>Input</td>
   <td>Number of tokens received from each device in the EP communicator. Corresponds to the epRecvCounts output in MoeDistributeDispatchV2.</td>
   <td>INT32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>expertScales</td>
   <td>Input</td>
   <td>Weights of the topK experts for each token.</td>
   <td>FLOAT32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>tpSendCountsOptional</td>
   <td>Optional input</td>
   <td>Reserved parameter. TP domain communication is no longer supported. Pass a null pointer.</td>
   <td>INT32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>xActiveMaskOptional</td>
   <td>Optional input</td>
   <td>Indicates whether a token participates in communication. You can pass valid data or a null pointer. For a 1D tensor, true values must be placed before false values (for example, {true, false, true} is invalid). For a 2D tensor, a token does not participate in communication if all K values corresponding to the token are false. By default, all tokens participate in communication.</td>
   <td>BOOL</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>activationScaleOptional</td>
   <td>Optional input</td>
   <td>Reserved parameter. Not supported in the current version. Pass a null pointer.</td>
   <td>-</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>weightScaleOptional</td>
   <td>Optional input</td>
   <td>Reserved parameter. Not supported in the current version. Pass a null pointer.</td>
   <td>-</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>groupListOptional</td>
   <td>Optional input</td>
   <td>Reserved parameter. Not supported in the current version. Pass a null pointer.</td>
   <td>-</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>expandScalesOptional</td>
   <td>Optional input</td>
   <td>Indicates the weights of tokens on the current device. Corresponds to the expandScales output in MoeDistributeDispatchV2.</td>
   <td>FLOAT32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>sharedExpertXOptional</td>
   <td>Optional input</td>
   <td>Indicates the tokens computed by the shared expert. The data type must be the same as that of expandX.</td>
   <td>FLOAT16, BFLOAT16</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>elasticInfoOptional</td>
   <td>Optional input</td>
   <td>Dynamic scale-in information of the EP communicator.</td>
   <td>INT32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>oriXOptional</td>
   <td>Optional input</td>
   <td>Indicates the token data that does not pass through the FFN (Feed-Forward Neural network). This input data is required when copyExpert or constExpert is enabled.</td>
   <td>FLOAT16, BFLOAT16</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>constExpertAlpha1Optional</td>
   <td>Optional input</td>
   <td>Calculation coefficient that needs to be input when constExpert is enabled.</td>
   <td>FLOAT16, BFLOAT16</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>constExpertAlpha2Optional</td>
   <td>Optional input</td>
   <td>Calculation coefficient that needs to be input when constExpert is enabled.</td>
   <td>FLOAT16, BFLOAT16</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>constExpertVOptional</td>
   <td>Optional input</td>
   <td>Calculation coefficient that needs to be input when constExpert is enabled.</td>
   <td>FLOAT16, BFLOAT16</td>
   <td>ND</td>
  </tr>
  <tr>
    <td>performanceInfoOptional</td>
    <td>Optional input</td>
    <td>Indicates the communication time that the current device spends waiting for data from other devices, in us (microseconds). The communication time consumed by each device in a single operator call is accumulated on this tensor. The operator does not automatically clear the tensor. Therefore, you need to clear the tensor each time before enabling the tensor to record the time consumption.</td>
    <td>INT64</td>
    <td>ND</td>
  </tr>
  <tr>
   <td>groupEp</td>
   <td>Attribute</td>
   <td>Name of the EP communicator (expert parallel communication domain). The string length ranges from [1, 128).</td>
   <td>STRING</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>epWorldSize</td>
   <td>Attribute</td>
   <td>Size of the EP communicator.</td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>epRankId</td>
   <td>Attribute</td>
   <td>ID of the current device in the EP domain. Value range: [0, epWorldSize). The epRankId of each device in the same EP communicator must be unique.</td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>moeExpertNum</td>
   <td>Attribute</td>
   <td>Number of MoE experts. moeExpertNum % (epWorldSize - sharedExpertRankNum) = 0 must be satisfied.</td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>groupTp</td>
   <td>Optional attribute</td>
   <td><li>Name of the TP communicator (data parallel communication domain). Reserved parameter. TP domain communication is no longer supported.</li><li>Default value: "".</li></td>
   <td>STRING</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>tpWorldSize</td>
   <td>Optional attribute</td>
   <td><li>Size of the TP communicator. Reserved parameter. TP domain communication is no longer supported. Only 0 or 1 is allowed.</li><li>Default value: 0.</li></td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>tpRankId</td>
   <td>Optional attribute</td>
   <td><li>ID of the current device in the TP domain. When there is no TP domain communication, pass 0.</li><li>Default value: 0.</li></td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>expertShardType</td>
   <td>Optional attribute</td>
   <td><li>Indicates the distribution type of shared expert devices. Currently, only 0 is supported, indicating that shared expert devices are placed before MoE expert devices.</li><li>Default value: 0.</li></td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>sharedExpertNum</td>
   <td>Optional attribute</td>
   <td><li>Indicates the number of shared experts (one shared expert can be replicated and deployed on multiple devices).</li><li>Default value: 1.</li></td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>sharedExpertRankNum</td>
   <td>Optional attribute</td>
   <td><li>Indicates the number of shared expert devices. Value range: [0, epWorldSize). When the value is 0, sharedExpertNum must be 0 or 1. When the value is not 0, sharedExpertRankNum % sharedExpertNum = 0 must be satisfied.</li><li>Default value: 0.</li></td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>globalBs</td>
   <td>Optional attribute</td>
   <td><li>Global batch size capacity in the EP domain. When the BS values of all ranks are the same, globalBs = BS * epWorldSize or 0. When the BS values differ across ranks, globalBs = maxBS * epWorldSize (maxBS is the upper limit of the single-rank BS capacity configured by the service and must not be smaller than the actual BS of any rank). The globalBs values passed by all ranks must be the same.</li><li>Default value: 0. When the BS values differ across ranks, 0 cannot be passed.</li></td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>outDtype</td>
   <td>Optional attribute</td>
   <td><li>Specifies the data type of the output x. Reserved parameter. Not supported in the current version. Pass 0.</li><li>Default value: 0.</li></td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>commQuantMode</td>
   <td>Optional attribute</td>
   <td><li>Communication quantization type. Value range: 0 or 2. 0 indicates that quantization is not performed during communication, and 2 indicates int8 quantization during communication.</li><li>Default value: 0.</li></td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>groupListType</td>
   <td>Optional attribute</td>
   <td><li>Format of groupList. Reserved parameter. Not supported in the current version. Pass 0.</li><li>Default value: 0.</li></td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>commAlg</td>
   <td>Optional attribute</td>
   <td><li>Indicates the communication-affinity memory layout algorithm.</li><li>Default value: "".</li></td>
   <td>STRING</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>zeroExpertNum</td>
   <td>Optional attribute</td>
   <td><li>Number of zero experts.</li><li>Default value: 0.</li></td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>copyExpertNum</td>
   <td>Optional attribute</td>
   <td><li>Number of copy experts.</li><li>Default value: 0.</li></td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>constExpertNum</td>
   <td>Optional attribute</td>
   <td><li>Number of constant experts.</li><li>Default value: 0.</li></td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>xOut</td>
   <td>Output</td>
   <td>Indicates the processed tokens. The data type and data format are the same as those of expandX.</td>
   <td>FLOAT16, BFLOAT16</td>
   <td>ND</td>
  </tr>
 </tbody>
</table>

* <term>Atlas A2 products</term>:
    * The shared expert scenario is not supported, and `expertShardType`, `sharedExpertNum`, `sharedExpertRankNum`, and `sharedExpertXOptional` are not supported.
    * Only the EP domain is supported, and there is no TP domain. The `groupTp`, `tpWorldSize`, and `tpRankId` attributes are not supported, and the `tpRecvCounts` output is invalid.
    * The dynamic scale-in scenario is not supported, and `elasticInfoOptional` is not supported.
    * When `commAlg` = "hierarchy", `expandScalesOptional` must be passed.
    * The constant expert scenario is not supported. `constExpertNum`, `constExpertAlpha1Optional`, `constExpertAlpha2Optional`, and `constExpertVOptional` are not supported; use the default values.

- <term>Atlas A3 products</term>:
    * When `commAlg` = "hierarchy", `expandScalesOptional` must be passed.
    * `expertScales` supports only a valid tensor with the shape of (BS, K) for weighted aggregation. Empty tensors are not supported.
    * commAlg supports "", "fullmesh_v1", "fullmesh_v2", and "hierarchy". "": Default value. The fullmesh_v2 template is not enabled. "fullmesh_v1": The fullmesh_v2 template is not enabled. "fullmesh_v2": The fullmesh_v2 template is enabled. "hierarchy": Cross hyper node communication is enabled. Different `BS` values across ranks are supported. Only the scenario with zero shared experts is supported. 2D masks, special experts, and performanceInfo scenarios are not supported. In the variable BS scenario, all ranks must be configured with the same non-zero `globalBs` capacity, and `xActiveMaskOptional` must not be passed.
    * The value range of epWorldSize is [2, 768]. In the commAlg="hierarchy" scenario, the value range is [16, 256], and the value must be a multiple of 16.
    * The value range of moeExpertNum is (0, 1024]. In the commAlg="hierarchy" scenario, the value range is (0, 512].

- <term>Ascend 950DT products</term>:
    * You can choose whether to enable the expertScales expert weight function. When a valid tensor with the shape of (BS, K) is passed for `expertScales`, the function is enabled and weighting is performed. When an empty tensor is passed, the function is disabled and the expert outputs are summed directly.
    * `expandScalesOptional` is not supported.
    * `commAlg` is not supported.
    * Only the EP domain is supported, and there is no TP domain. The `groupTp`, `tpWorldSize`, and `tpRankId` attributes are not supported, and the `tpRecvCounts` output is invalid.

## Constraints

- `MoeDistributeDispatchV2` and the `CombineV2` series operators must be used together. For details, see the calling example.

- The driver versions of all nodes in the operator communicator must be the same.

- For different product models, different communication algorithms, or different versions, the element values in the `assistInfoForCombineOut`, `epRecvCountsOut`, `tpRecvCountsOut`, and `expandScalesOut` tensor outputs of `MoeDistributeDispatchV2` may differ. When using them, pass these tensors directly to the corresponding parameters of the `CombineV2` series operators. Other business logic in the model should not depend on them.

- The values of the `groupEp`, `epWorldSize`, `moeExpertNum`, `groupTp`, `tpWorldSize`, `expertShardType`, `sharedExpertNum`, `sharedExpertRankNum`, `globalBs`, and `commAlg` parameters used when calling the operator, and the `HCCL_BUFFSIZE` value, must be the same on all devices, the same across different layers in the network, and consistent with the corresponding parameters of the `MoeDistributeDispatchV2` operator.

- Shape format description in Parameter Description:
    - `A`: Indicates the maximum number of tokens that the current device may receive. The value range is as follows:
        - For shared experts, `A` = `BS` *`epWorldSize`* `sharedExpertNum` / `sharedExpertRankNum` must be satisfied.
        - For MoE experts, when `globalBs` is 0, `A` >= `BS` *`epWorldSize`* min(`localExpertNum`, `K`) must be satisfied. When `globalBs` is not 0, `A` >= `globalBs` * min(`localExpertNum`, `K`) must be satisfied.
    - `K`: Indicates that topK experts are selected. The value range is 0 < `K` ≤ 16, and 0 < `K` ≤ `moeExpertNum` + `zeroExpertNum` + `copyExpertNum` + `constExpertNum` must also be satisfied.
    - `localExpertNum`: Indicates the number of experts on the current device.
        - For shared expert devices, `localExpertNum` = 1
        - For MoE expert devices, `localExpertNum` = `moeExpertNum` / (`epWorldSize` - `sharedExpertRankNum`). TP domain communication is not supported in the current version.

- Parameter constraints:
    - `zeroExpertNum`: Value range: [0, MAX_INT32), where MAX_INT32 = 2^31 - 1. The ID of a valid zero expert is in the range [`moeExpertNum`, `moeExpertNum` + `zeroExpertNum`).
    - `copyExpertNum`: Value range: [0, MAX_INT32), where MAX_INT32 = 2^31 - 1. The ID of a valid copy expert is in the range [`moeExpertNum` + `zeroExpertNum`, `moeExpertNum` + `zeroExpertNum` + `copyExpertNum`).
    - `constExpertNum`: Value range: [0, MAX_INT32), where MAX_INT32 = 2^31 - 1. The ID of a valid constant expert is in the range [`moeExpertNum` + `zeroExpertNum` + `copyExpertNum`, `moeExpertNum` + `zeroExpertNum` + `copyExpertNum` + `constExpertNum`).
    - `oriXOptional`: You can pass valid data or a null pointer. When `copyExpertNum` is not 0 or `constExpertNum` is not 0, valid input must be passed. When valid data is passed, the shape must be (`BS`, `H`), and the data type must be the same as that of `expandX`.
    - `constExpertAlpha1Optional`: You can pass valid data or a null pointer. When `constExpertNum` is not 0, valid input must be passed. When valid data is passed, the shape must be (`constExpertNum`, `H`), and the data type must be the same as that of `expandX`.
    - `constExpertAlpha2Optional`: You can pass valid data or a null pointer. When `constExpertNum` is not 0, valid input must be passed. When valid data is passed, the shape must be (`constExpertNum`, `H`), and the data type must be the same as that of `expandX`.
    - `constExpertVOptional`: You can pass valid data or a null pointer. When `constExpertNum` is not 0, valid input must be passed. When valid data is passed, the shape must be (`constExpertNum`, `H`), and the data type must be the same as that of `expandX`.

- In the formulas in this document, "/" indicates integer division.
- Communicator usage constraints:
    - Within a model, `MoeDistributeCombineV2` and `MoeDistributeDispatchV2` support only the same EP communicator, and no other operators are allowed in the communicator.
    - TP domain communication is currently not supported.

- Communication mode constraints:
    - <term>Ascend 950DT products</term>: Only UB Memory communication is supported.

- <term>Atlas A2 products</term>:
    - `commAlg`: The current version supports four input values: nullptr, "", "fullmesh", and "hierarchy". If "hierarchy" is configured, it is recommended to use a driver of version 25.0.RC1.1 or later.
        - nullptr and "": Only in this scenario do the `HCCL_INTRA_PCIE_ENABLE` and `HCCL_INTRA_ROCE_ENABLE` settings take effect. When `HCCL_INTRA_PCIE_ENABLE`=1&&`HCCL_INTRA_ROCE_ENABLE`=0, the "hierarchy" algorithm is called. Otherwise, the "fullmesh" algorithm is called. This method is not recommended.
        - "fullmesh": Token data is directly sent to the devices where the topk target experts reside through RDMA.
        - "hierarchy": Token data is sent twice, across servers and within a server. RDMA communication is used only between devices with the same ID on different servers, and HCCS communication is used within a server.
    - `HCCL_INTRA_PCIE_ENABLE` and `HCCL_INTRA_ROCE_ENABLE`: Using these environment variables to control the communication algorithm is not recommended. The original scenario of `HCCL_INTRA_PCIE_ENABLE`=1&&`HCCL_INTRA_ROCE_ENABLE`=0 is replaced by `commAlg`="hierarchy" in the following text, and the default scenario is replaced by `commAlg`="fullmesh".
    - When `commAlg` = "hierarchy", `xActiveMaskOptional`, `oriXOptional`, `zeroExpertNum`, and `copyExpertNum` are not supported.
    - Shape format description in Parameter Description:
        - `H`: Indicates the hidden size. The value range is (0, 10240], and the value must be a multiple of 32.
        - `BS`: Indicates the batch sequence size, that is, the number of tokens finally output by the current device.
            - `commAlg` = "fullmesh": The value range is [1, 256].
            - `commAlg` = "hierarchy": The value range is [1, 512].
        - `performanceInfoOptional`: You can pass valid data or a null pointer. A null pointer indicates that the function of recording communication time consumption is disabled. When valid data is passed, the tensor must be 1D with the shape of (ep\_world\_size,), the data type must be int64, and the data format must be ND.
    - Attribute constraints:
        - `epWorldSize`: The value depends on commAlg. "fullmesh" supports 2, 3, 4, 5, 6, 7, 8, 16, 32, 64, 128, 192, 256, and 384. "hierarchy" supports 16, 32, and 64.
        - `moeExpertNum`: The value depends on commAlg. "fullmesh" supports (0, 1024], and "hierarchy" supports (0, 512].
        - `commQuantMode`: 2 enables int8 quantization during communication. This value is supported only when `commAlg` = "hierarchy" and the driver version is 25.0.RC1.1 or later.
    - `HCCL_BUFFSIZE`: Before calling this operator, check whether the value of the `HCCL_BUFFSIZE` environment variable is reasonable. This environment variable indicates the memory size occupied by a single communicator, in MB. If it is not configured, the default value is 200MB.
        - `commAlg` = "fullmesh": The value must be >= (`BS` *`epWorldSize`* min(`localExpertNum`, `K`) *`H`* 4B + 4MB).
        - `commAlg` = "hierarchy": The value must be >= (`moeExpertNum` + `epWorldSize` / 4) *Align512(`maxBS`* (`H` *2 + 16* Align8(`K`))) *1B + 8MB, where Align8(x) = ((x + 8 - 1) / 8)* 8 and Align512(x) = ((x + 512 - 1) / 512) * 512.

- <term>Atlas A3 products</term>:
    - In this scenario, a single device contains two DIEs (also called dies). Therefore, "the current device" in Parameter Description refers to a single DIE.
    - Shape format description in Parameter Description:
        - `H`: Indicates the hidden size. The value range is [1024, 8192].
        - `BS`: Indicates the batch sequence size, that is, the number of tokens finally output by the current device. The value range is [1, 512].
    - Parameter constraints:
        - `epWorldSize`: Supported values: 8, 16, 32, 64, 128, 144, 256, and 288.
        - `moeExpertNum`: Value range: (0, 1024].
        - `groupTp`: Reserved parameter. TP domain communication is no longer supported.
        - `tpWorldSize`: Reserved parameter. TP domain communication is no longer supported. Only 0 or 1 is allowed.
        - `tpRankId`: Reserved parameter. TP domain communication is no longer supported. Pass 0.
        - `sharedExpertNum`: Value range: [0, 4].
        - `commQuantMode`: int8 quantization can be enabled if and only if `tpWorldSize` < 2.
        - `performanceInfoOptional`: Reserved parameter. Not supported in the current version. Pass a null pointer.
        - `commAlg`: The current version supports two input values: "" and "hierarchy".
            - "": Default value. The communicator does not cross hyper nodes.
            - "hierarchy": Enables the ROCE hierarchical direct-drive capability. You need to set the `HCCL_LOGIC_SUPERPOD_ID` environment variable based on different logical hyper nodes. For example, set `export HCCL_LOGIC_SUPERPOD_ID=0` and `export HCCL_LOGIC_SUPERPOD_ID=1` on the two servers respectively.
    - `HCCL_BUFFSIZE`: Before calling this operator, check whether the value of the `HCCL_BUFFSIZE` environment variable is reasonable. This environment variable indicates the memory size occupied by a single communicator, in MB. If it is not configured, the default value is 200MB. The value must be >= 2 and satisfy >= 2 *(`localExpertNum`* `maxBS` *`epWorldSize`* Align512(Align32(2 *`H`) + 64) + (`K` + `sharedExpertNum`)* `maxBS` *Align512(2* `H`)). localExpertNum must use the number of experts on the MoE expert device, where Align512(x) = ((x + 512 - 1) / 512) *512 and Align32(x) = ((x + 32 - 1) / 32)* 32.

- <term>Ascend 950DT products</term>:
    - Shape format description in Parameter Description:
        - `H`: Indicates the hidden size. The value range is [1024, 8192].
        - `BS`: Indicates the batch sequence size, that is, the number of tokens finally output by the current device. The value depends on commAlg: for "fullmesh_v2" and "hierarchy", the value range is (0 < BS ≤ 256); for "fullmesh_v1" and "", the value range is (0 < BS ≤ 512).
    - Parameter constraints:
        - `epWorldSize`: Supported value range: [2, 1024].
        - `moeExpertNum`: Value range: (0, 2048].
        - `groupTp` is not supported in the current version. Pass an empty string.
        - `tpWorldSize` is not supported in the current version. Pass 0.
        - `tpRankId` is not supported in the current version. Pass 0.
        - Currently, only 0 can be passed for `expertShardType`, indicating that shared expert devices are placed before MoE expert devices.
        - The value range of `sharedExpertNum` is [0, 4].
        - The value range of `sharedExpertRankNum` is [0, epWorldSize). When the value is 0, sharedExpertNum must be 0 or 1. When the value is not 0, sharedExpertRankNum % sharedExpertNum = 0 must be satisfied.
        - The value range of `commQuantMode` is 0 or 2 (0 indicates no quantization, and 2 indicates int8 quantization).
        - `performanceInfoOptional`: Reserved parameter. Not supported in the current version. Pass a null pointer.
    - `HCCL_BUFFSIZE`: Before calling this operator, check whether the value of the `HCCL_BUFFSIZE` environment variable is reasonable. This environment variable indicates the memory size occupied by a single communicator, in MB. If it is not configured, the default value is 200MB. The value must be >= 2 and satisfy:
        - `commAlg` = "fullmesh_v1" or "": >= 2 * (`localExpertNum` * `maxBS` * `epWorldSize` * Align512(Align32(2 * `H`) + 64) + (`K` + `sharedExpertNum`) * `maxBS` * 480Align512(2 * `H`)).
        - `commAlg` = "fullmesh_v2": >= 2 * (`localExpertNum` * `maxBS` * `epWorldSize` * 480Align512(Align32(2 * `H`) + 64) + (`K` + `sharedExpertNum`) * `maxBS` * 480Align512(2 * `H`)).
        - Where `480Align512(x) = ((x + 480 - 1) / 480) * 512`, `Align512(x) = ((x + 512 - 1) / 512) * 512`, and `Align32(x) = ((x + 32 - 1) / 32) * 32`.
        - For <term>Ascend 950DT products</term>, when `quantMode` on the dispatch side is 0 (non-quantization) and `scalesOptional` is not empty, the length of each token in the formula is `max(Align32(2 * H) + 44, Align32(Align32(2 * H) + 4 * H) + 16)`, and the result is aligned to 512B for fullmesh_v1 or 480B for fullmesh_v2.

## Calling Instructions

| Calling Method | Sample Code | Description |
| :--------: | :----------------------------------------: | :-------------------------------------------------------: |
| aclnn API | [test_aclnn_moe_distribute_combine_v2.cpp](./examples/test_aclnn_moe_distribute_combine_v2.cpp) | Calls the moe_distribute_combine_v2 operator through the [aclnnMoeDistributeCombineV2](docs/aclnnMoeDistributeCombineV2_en.md) API. |
