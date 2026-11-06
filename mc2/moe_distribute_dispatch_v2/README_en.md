# MoeDistributeDispatchV2

English | [简体中文](./README.md)

## Product Support

| Product                                                     |  Supported   |
| :----------------------------------------------------------- |:-------:|
| <term>Ascend 950DT products</term>                             |    √    |
| <term>Atlas A3 products</term>       |    √    |
| <term>Atlas A2 products</term> |    √    |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×    |
| <term>Atlas inference products</term>                               |    ×    |
| <term>Atlas training products</term>                              |    ×    |

## Function Description

Operator function: Quantizes token data (optional) and performs AllToAllV communication in the EP (Expert Parallelism) domain. TP (Tensor Parallelism) domain communication is no longer supported in the current version. TP-related parameters are reserved parameters.

- Scenario 1: If quantMode=0 (non-quantization scenario):

$$
allToAllXOut = AllToAllV(X)\\
expandXOut = AllToAllV(X)
$$

- Scenario 2: If quantMode=1 (static quantization scenario):

$$
xFp32 = CastToFp32(X) \times scales \\
quantOut = Cast(xFp32, dstType) \\
allToAllXOut = AllToAllV(quantOut)\\
expandXOut = AllToAllV(quantOut)
$$

- Scenario 3: If quantMode=2 (pertoken dynamic quantization scenario):

$$
xFp32 = CastToFp32(X) \times scales \\
dynamicScales = dstTypeMax/Max(Abs(xFp32)) \\
quantOut = CastToInt8(xFp32 \times dynamicScales) \\
allToAllXOut = AllToAllV(quantOut) \\
allToAllDynamicScalesOut = AllToAllV(1.0/dynamicScales) \\
expandXOut = AllToAllV(quantOut) \\
dynamicScalesOut = allToAllDynamicScalesOut
$$

- Scenario 4: If quantMode=3 (pertile dynamic quantization scenario):

$$
xFp32 = CastToFp32(X) \times scales \\
dynamicScales = dstTypeMax/Max(Abs(xFp32)) \\
quantOut = CastToInt8(xFp32 \times dynamicScales) \\
allToAllXOut = AllToAllV(quantOut) \\
allToAllDynamicScalesOut = AllToAllV(1.0/dynamicScales) \\
expandXOut = AllToAllV(quantOut) \\
dynamicScalesOut = allToAllDynamicScalesOut
$$

- Scenario 5: If quantMode=4 (mx quantization scenario):

$$
sharedExp = Floor(log_2(max(x))) - emax \\
dynamicScales = 2^{sharedExp} \\
quantOut = CastToFp8(X / dynamicScales) \\
allToAllXOut = AllToAllV(quantOut) \\
allToAllDynamicScalesOut = AllToAllV(1.0 / dynamicScales) \\
expandXOut = AllToAllV(quantOut) \\
dynamicScalesOut = allToAllDynamicScalesOut
\end{cases}
$$

Here, $emax$ indicates the exponent value corresponding to the largest normal number of the type.

- <term>Atlas A2 products</term>: This operator must be used together with `MoeDistributeCombineV2`.
- <term>Atlas A3 products</term> and <term>Ascend 950DT products</term>: This operator must be used together with `MoeDistributeCombineV2` or `MoeDistributeCombineAddRmsNorm`.

> Note: MoeDistributeCombineV2 and MoeDistributeCombineAddRmsNorm are collectively referred to as the CombineV2 series operators in the following text.

Compared with the `MoeDistributeDispatch` operator, changes in this operator are as follows:

- It outputs more detailed token information to help the `CombineV2` series operators efficiently synchronize across all devices. Therefore, the `expandIdx` output parameter with shape (`BS` * `K`,) in the original operator is replaced by the `assistInfoForCombineOut` parameter with shape (`A` * 128,);
- It adds the `commAlg` input parameter, replacing the `HCCL_INTRA_PCIE_ENABLE` and `HCCL_INTRA_ROCE_ENABLE` environment variables.

For details, see the following parameter description.

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
<td>x</td>
    <td>Input</td>
    <td>Token data sent by the current device.</td>
    <td>FLOAT16, BFLOAT16, FLOAT8_E5M2, FLOAT8_E4M3FN, HIFLOAT8, FLOAT4_E2M1, FLOAT4_E1M2</td>
    <td>ND</td>
   </tr>
   <tr>
    <td>expertIds</td>
   <td>Input</td>
   <td>TopK expert indices of each token.</td>
   <td>INT32</td>
   <td>ND</td>
  </tr>
  <tr>
<td>scalesOptional</td>
    <td>Optional Input</td>
    <td>Quantization smoothing parameter of each expert.</td>
    <td>FLOAT32, FLOAT8_E8M0</td>
    <td>ND</td>
   </tr>
  <tr>
   <td>xActiveMaskOptional</td>
   <td>Optional Input</td>
   <td>Whether a token participates in communication. You can pass valid data or a null pointer. In the 1D case, true values must be placed before false values (for example, {true, false, true} is invalid). In the 2D case, a token does not participate in communication if all K values corresponding to the token are false. By default, all tokens participate in communication.</td>
   <td>BOOL</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>expertScalesOptional</td>
   <td>Optional Input</td>
   <td>TopK expert weights of each token.</td>
   <td>FLOAT32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>elasticInfoOptional</td>
   <td>Optional Input</td>
   <td>Dynamic scale-in information of the EP communicator. When some communication devices are removed from the communicator due to faults, the actual number of devices participating in communication can be obtained from this parameter.</td>
   <td>FLOAT32</td>
   <td>ND</td>
  </tr>
  <tr>
    <td>performanceInfoOptional</td>
    <td>Optional Input</td>
    <td>Communication time spent by the current device waiting for data from other devices, in us (microseconds). In a single operator call, the communication time spent by each device is accumulated into this tensor. The operator does not automatically clear the tensor. Therefore, before you enable this tensor to start recording time each time, you must clear the tensor.</td>
    <td>INT64</td>
    <td>ND</td>
  </tr>
  <tr>
   <td>groupEp</td>
   <td>Attribute</td>
    <td>Name of the EP communicator (expert parallelism communicator). The string length ranges from [1, 128).</td>
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
   <td>ID of the current device in the EP domain. Value range: [0, epWorldSize). The epRankId values of devices in the same EP communicator must be unique.</td>
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
   <td>Optional Attribute</td>
   <td><li>Name of the TP communicator (reserved parameter, not supported in the current version).</li><li>Default value: "".</li></td>
   <td>STRING</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>tpWorldSize</td>
   <td>Optional Attribute</td>
   <td><li>Size of the TP communicator (reserved parameter, not supported in the current version; only 0 or 1 is supported).</li><li>Default value: 0.</li></td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>tpRankId</td>
   <td>Optional Attribute</td>
   <td><li>ID of the current device in the TP domain (reserved parameter, not supported in the current version; only 0 is supported).</li><li>Default value: 0.</li></td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>expertShardType</td>
   <td>Optional Attribute</td>
   <td><li>Distribution type of shared expert devices. Currently only 0 is supported, indicating that shared expert devices are placed before MoE expert devices.</li><li>Default value: 0.</li></td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>sharedExpertNum</td>
   <td>Optional Attribute</td>
   <td><li>Number of shared experts (a shared expert can be replicated and deployed on multiple devices).</li><li>Default value: 1.</li></td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>sharedExpertRankNum</td>
   <td>Optional Attribute</td>
   <td><li>Number of shared expert devices. Value range: [0, epWorldSize). If it is 0, sharedExpertNum must be 0 or 1. If it is not 0, sharedExpertRankNum % sharedExpertNum = 0 must be satisfied.</li><li>Default value: 0.</li></td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>quantMode</td>
   <td>Optional Attribute</td>
   <td><li>Quantization mode. Default value: 0.</li></td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>globalBs</td>
   <td>Optional Attribute</td>
   <td><li>Global batch size capacity in the EP domain. If the BS values of all ranks are the same, globalBs = BS * epWorldSize or 0. If the BS values of ranks are different, globalBs = maxBS * epWorldSize (maxBS is the upper limit of the BS capacity of a single rank configured by the service, and is not smaller than the actual BS of any rank). The globalBs values passed by all ranks must be the same.</li><li>Default value: 0. It cannot be 0 when the BS values of ranks are different.</li></td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>expertTokenNumsType</td>
   <td>Optional Attribute</td>
   <td><li>Semantic type of the values in the output expertTokenNums. Supported values: 0 (the output in expertTokenNums is the prefix sum of the number of tokens processed by each expert) and 1 (the output in expertTokenNums is the number of tokens processed by each expert).</li><li>Default value: 1.</li></td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>commAlg</td>
   <td>Optional Attribute</td>
   <td><li>Communication-affinity memory layout algorithm.</li><li>Default value: "".</li></td>
   <td>STRING</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>zeroExpertNum</td>
   <td>Optional Attribute</td>
   <td><li>Number of zero experts.</li><li>Default value: 0.</li></td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>copyExpertNum</td>
   <td>Optional Attribute</td>
   <td><li>Number of copy experts.</li><li>Default value: 0.</li></td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>constExpertNum</td>
   <td>Optional Attribute</td>
   <td><li>Number of constant experts.</li><li>Default value: 0.</li></td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
    <td>expandXOut</td>
    <td>Output</td>
    <td>Token features expanded based on expertIds.</td>
    <td>FLOAT16, BFLOAT16, INT8, FLOAT8_E4M3FN, FLOAT8_E5M2, HIFLOAT8, FLOAT4_E2M1, FLOAT4_E1M2</td>
    <td>ND</td>
   </tr>
   <tr>
    <td>dynamicScalesOut</td>
    <td>Output</td>
    <td>In a quantization scenario, the quantization factor of the tokens output by the current device.</td>
    <td>FLOAT32, FLOAT8_E8M0</td>
    <td>ND</td>
   </tr>
  <tr>
   <td>assistInfoForCombineOut</td>
   <td>Output</td>
   <td>Number of tokens sent to the same expert (corresponds to assistInfoForCombine in the CombineV2 series operators).</td>
   <td>INT32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>expertTokenNumsOut</td>
   <td>Output</td>
   <td>Number of tokens received by each expert.</td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>epRecvCountsOut</td>
   <td>Output</td>
   <td>Number of tokens received from each device in the EP communicator (corresponds to epSendCounts in the CombineV2 series operators).</td>
   <td>INT32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>tpRecvCountsOut</td>
   <td>Output</td>
   <td>Number of tokens received from each device in the TP communicator (reserved output, not supported in the current version; pass a null pointer). This output is not available when there is no TP domain communication.</td>
   <td>INT32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>expandScalesOut</td>
   <td>Output</td>
   <td>Weight of the tokens output by the current device (corresponds to expandScalesOptional in the CombineV2 series operators).</td>
   <td>FLOAT32</td>
   <td>ND</td>
  </tr>
 </tbody>
</table>

* <term>Atlas A2 products</term>:
    * The shared expert scenario is not supported, and the `expertShardType`, `sharedExpertNum`, and `sharedExpertRankNum` attributes are not supported.
    * Only the EP domain is supported and there is no TP domain. The `groupTp`, `tpWorldSize`, and `tpRankId` attributes are not supported, and the `tpRecvCounts` output has no valid content.
    * `elasticInfoOptional` is not supported.
    * The content of `expandScalesOut` is valid when `commAlg` = "hierarchy".
    * The constant expert scenario is not supported and `constExpertNum` is not supported. Use the default value.
* <term>Atlas A3 products</term>  :
    * commAlg supports the following input methods: "", "fullmesh_v1", "fullmesh_v2", and "hierarchy". "": default value, which does not enable the fullmesh_v2 template. "fullmesh_v1": does not enable the fullmesh_v2 template. "fullmesh_v2": enables the fullmesh_v2 template, which cannot be enabled when `BS` values differ across devices, when xActiveMask is used as input, or in special expert scenarios. "hierarchy": enables the cross-supernode template, supports different `BS` values across ranks, supports only the scenario where the number of shared experts is 0, and does not support the 2D mask, special expert, or performanceInfo scenarios. In the variable-BS scenario, configure the same non-zero `globalBs` capacity for all ranks and do not pass `xActiveMaskOptional`.
    * When `commAlg` = "hierarchy", `expertScalesOptional` must be a 2D tensor with shape (BS, K). When `commAlg` = "", "fullmesh_v1", or "fullmesh_v2", you can pass a valid tensor or a null pointer. The valid tensor is a 2D tensor with shape (BS, K). When a null pointer is passed, `expertScalesOptional` is invalid. Empty tensors are not supported.
    * The value range of epWorldSize is [2, 768]. When commAlg = "hierarchy", the value range is [16, 256] and the value must be a multiple of 16.
    * The value range of moeExpertNum is (0, 1024]. When commAlg = "hierarchy", the value range is (0, 512].
    * When `commAlg` = "hierarchy", `expandScalesOut` must be a 1D tensor with shape (A,). When `commAlg` = "", "fullmesh_v1", or "fullmesh_v2", this output is not supported yet.
* <term>Ascend 950DT products</term>:
    * Only the EP domain is supported and there is no TP domain. The `groupTp`, `tpWorldSize`, and `tpRankId` attributes are not supported, and the `tpRecvCounts` output has no valid content.
    * You can pass a valid tensor or a null pointer to `expertScalesOptional`. When a valid tensor is passed, the expertScales expert weight function is enabled. The valid tensor is a 2D tensor with shape (BS, K). When a null pointer is passed, the expertScales expert weight function is disabled. Empty tensors are not supported.
    * When the expertScales expert weight function is enabled, the `expandScalesOut` output is valid. When the function is disabled, this output is invalid.

## Constraints

- The `MoeDistributeDispatchV2` operator must be used together with the `CombineV2` series operators. For details, see the invocation example.

- The driver versions of all nodes in the operator communicator must be the same.

- Across different product models, communication algorithms, or versions, the element values in the `assistInfoForCombineOut`, `epRecvCountsOut`, `tpRecvCountsOut`, and `expandScalesOut` tensor outputs of `MoeDistributeDispatchV2` may differ. When using them, pass these tensors directly to the corresponding parameters of the `CombineV2` series operators. Other business logic in the model should not depend on them.

- The values of the `groupEp`, `epWorldSize`, `moeExpertNum`, `groupTp`, `tpWorldSize`, `expertShardType`, `sharedExpertNum`, `sharedExpertRankNum`, `globalBs`, and `commAlg` parameters and the `HCCL_BUFFSIZE` value used when calling the operator must be the same on all devices and must remain the same across different layers in the network. They must also match the corresponding parameters of the `CombineV2` series operators.

- Shape format description in the parameter description:
    - `A`: Maximum number of tokens that the current device may receive. The value ranges are as follows:
        - For shared experts, let `ranksPerSharedExpert = sharedExpertRankNum / sharedExpertNum` and `maxBS = (globalBs == 0 ? BS : globalBs / epWorldSize)`. Then `A >= maxBS * ((epWorldSize + ranksPerSharedExpert - 1) / ranksPerSharedExpert)` must be satisfied.
        - For MoE experts, if `globalBs` is 0, `A` >= `BS` *`epWorldSize`* min(`localExpertNum`, `K`) must be satisfied; if `globalBs` is not 0, `A` >= `globalBs` * min(`localExpertNum`, `K`) must be satisfied.
    - `K`: Number of selected topK experts. The value range is 0 < `K` ≤ 16, and 0 < `K` ≤ `moeExpertNum` + `zeroExpertNum` + `copyExpertNum` + `constExpertNum` must also be satisfied.
    - `localExpertNum`: Number of experts on the current device.
        - For a shared expert device, `localExpertNum` = 1
        - For a MoE expert device, `localExpertNum` = `moeExpertNum` / (`epWorldSize` - `sharedExpertRankNum`); TP domain communication is not supported in the current version.

- Attribute constraints:
    - `zeroExpertNum`: Value range: [0, MAX_INT32), where MAX_INT32 = 2^31 - 1. The valid zero expert IDs are in the range of [`moeExpertNum`, `moeExpertNum` + `zeroExpertNum`).
    - `copyExpertNum`: Value range: [0, MAX_INT32), where MAX_INT32 = 2^31 - 1. The valid copy expert IDs are in the range of [`moeExpertNum` + `zeroExpertNum`, `moeExpertNum` + `zeroExpertNum` + `copyExpertNum`).
    - `constExpertNum`: Value range: [0, MAX_INT32), where MAX_INT32 = 2^31 - 1. The valid constant expert IDs are in the range of [`moeExpertNum` + `zeroExpertNum` + `copyExpertNum`, `moeExpertNum` + `zeroExpertNum` + `copyExpertNum` + `constExpertNum`).

- In the formulas in this document, "/" indicates integer division.
- Communicator usage constraints:
    - The `CombineV2` series operators and `MoeDistributeDispatchV2` in a model support only the same EP communicator, and no other operators are allowed in this communicator.
    - TP domain communication is not supported currently.

- Communication mode constraints:
    - <term>Ascend 950DT products</term>: Only UB Memory communication is supported.

- <term>Atlas A2 products</term>:
    - Parameter constraints:
        - `commAlg`: The current version supports four input methods: nullptr, "", "fullmesh", and "hierarchy". If "hierarchy" is configured, it is recommended to use a driver of version 25.0.RC1.1 or later.
            - nullptr and "": Only in this scenario do the `HCCL_INTRA_PCIE_ENABLE` and `HCCL_INTRA_ROCE_ENABLE` settings take effect. When `HCCL_INTRA_PCIE_ENABLE`=1&&`HCCL_INTRA_ROCE_ENABLE`=0, the "hierarchy" algorithm is used; otherwise, the "fullmesh" algorithm is used. This method is not recommended.
            - "fullmesh": Token data is directly sent through RDMA to the devices where the topk target experts reside.
            - "hierarchy": Token data is sent twice, across machines and within a machine. RDMA communication is used only between devices with the same ID on different servers, and HCCS communication is used within a server.
        - `epWorldSize`: Depending on the value of commAlg, "fullmesh" supports 2, 3, 4, 5, 6, 7, 8, 16, 32, 64, 128, 192, 256, and 384; "hierarchy" supports 16, 32, and 64.
        - `moeExpertNum`: Depending on the value of commAlg, "fullmesh" supports (0, 1024], and "hierarchy" supports (0, 512].
        - `epRecvCountsOut`: The shape must be (`moeExpertNum` + 2 * `globalBs` * `K` * `serverNum`,). The first `moeExpertNum` values indicate the number of tokens received from each device in the EP communicator. The 2 * `globalBs` * `K` * `serverNum` values store the number of tokens that can be reduced in advance before communication during inter-server and intra-server combines and the offsets of tokens in the communication buffer. If 0 is passed to `globalBs`, calculate it as `BS` * `epWorldSize` here.
        - `performanceInfoOptional`: You can pass valid data or a null pointer. A null pointer indicates that the communication time recording function is disabled. When valid data is passed, it must be a 1D tensor with shape (ep\_world\_size,), the data type supports int64, and the data format must be ND.
    - `HCCL_INTRA_PCIE_ENABLE` and `HCCL_INTRA_ROCE_ENABLE`: Using these environment variables to control the communication algorithm is not recommended. The original scenario of `HCCL_INTRA_PCIE_ENABLE`=1&&`HCCL_INTRA_ROCE_ENABLE`=0 is replaced by `commAlg` = "hierarchy" hereafter, and the default scenario is replaced by `commAlg` = "fullmesh".
    - When `commAlg` is configured as "hierarchy", `scalesOptional`, `xActiveMaskOptional`, `oriXOptional`, `zeroExpertNum`, and `copyExpertNum` are not supported.
    - `quantMode` supports 0 (non-quantization) and 2 (pertoken dynamic quantization).
    - Shape format description in the parameter description:
        - `H`: Hidden size. The value range is (0, 10240] and the value must be a multiple of 32.
        - `BS`: Batch sequence size, that is, the number of tokens finally output by the current device.
            - `commAlg` = "fullmesh": The value range is (0, 256].
            - `commAlg` = "hierarchy": The value range is (0, 512].
    - `HCCL_BUFFSIZE`: Before calling this operator, check whether the value of the `HCCL_BUFFSIZE` environment variable is reasonable. This environment variable indicates the memory size occupied by a single communicator, in MB. If it is not configured, the default value is 200MB.
        - `commAlg` = "fullmesh": The value must be >= (`BS` *`epWorldSize`* min(`localExpertNum`, `K`) *`H`* 4B + 4MB).
        - `commAlg` = "hierarchy": The value must be >= (`moeExpertNum` + `epWorldSize` / 4) *Align512(`maxBS`* (`H` *2 + 16* Align8(`K`))) *1B + 8MB, where Align8(x) = ((x + 8 - 1) / 8)* 8 and Align512(x) = ((x + 512 - 1) / 512) * 512.
    - Networking constraints: In the multi-node scenario, only switch-based networking is supported; direct connection between two nodes is not supported.

- <term>Atlas A3 products</term>  :
    - In this scenario, a single device contains dual dies (referred to as dies), so "the current device" in the parameter description refers to a single die.
    - Parameter constraints:
        - `elasticInfoOptional`: Not supported in the current version. Pass a null pointer.
        - `epWorldSize`: Value range: [2, 768].
        - `moeExpertNum`: Value range: (0, 1024].
        - `groupTp`: Reserved parameter. TP domain communication is no longer supported.
        - `sharedExpertNum`: Supported value range: [0, 4].
        - `commAlg`: The current version supports the following input methods: "", "fullmesh_v1", "fullmesh_v2", and "hierarchy".
            - "": Default value. Enables the fullmesh_v1 template.
            - "fullmesh_v1": Enables the fullmesh_v1 template.
            - "fullmesh_v2": Enables the fullmesh_v2 template, which cannot be enabled when `BS` values differ across devices, when xActiveMask is used as input, or in special expert scenarios.
            - "hierarchy": Enables the hierarchical ROCE direct-drive capability. Set the environment variable `HCCL_LOGIC_SUPERPOD_ID` based on different logical supernodes. For example, set `export HCCL_LOGIC_SUPERPOD_ID=0` on one machine and `export HCCL_LOGIC_SUPERPOD_ID=1` on the other machine.
        - `epRecvCountsOut`: The shape must be (`epWorldSize` * `localExpertNum`,).
        - `performanceInfoOptional`: Reserved parameter, not supported in the current version. Pass a null pointer.
    - Shape format description in the parameter description:
        - `H`: Hidden size. The value range is [1024, 8192].
        - `BS`: Batch sequence size, that is, the number of tokens finally output by the current device.
            - `commAlg` = "fullmesh_v2" or "hierarchy": The value range is (0, 256].
            - `commAlg` = "fullmesh_v1" or "": The value range is (0, 512].
    - `HCCL_BUFFSIZE`: Before calling this operator, check whether the value of the `HCCL_BUFFSIZE` environment variable is reasonable. This environment variable indicates the memory size occupied by a single communicator, in MB. If it is not configured, the default value is 200MB. The value must be >= 2 and satisfy >= 2 *(`localExpertNum`* `maxBS` *`epWorldSize`* Align512(Align32(2 *`H`) + 64) + (`K` + `sharedExpertNum`)* `maxBS` *Align512(2* `H`)). For `localExpertNum`, use the number of experts on the current MoE expert device, where Align512(x) = ((x + 512 - 1) / 512) *512 and Align32(x) = ((x + 32 - 1) / 32)* 32.

- <term>Ascend 950DT products</term>:
    - Parameter constraints:
        - `elasticInfoOptional`: Not supported in the current version. Pass a null pointer.
        - `epWorldSize`: Value range: [2, 1024].
        - `moeExpertNum`: Value range: (0, 2048].
        - `sharedExpertNum`: Supported value range: [0, 4].
        - `commAlg`: The current version supports three input methods: "", "fullmesh_v1", and "fullmesh_v2".
            - "": Default value. Enables the fullmesh_v1 template.
            - "fullmesh_v1": Enables the fullmesh_v1 template.
            - "fullmesh_v2": Enables the fullmesh_v2 template, which cannot be enabled when `BS` values differ across devices, when xActiveMask is used as input, or in special expert scenarios.
        - `epRecvCountsOut`: The shape must be (`epWorldSize` * `localExpertNum`,).
        - `performanceInfoOptional`: Reserved parameter, not supported in the current version. Pass a null pointer.
        - Currently only 0 is supported for `expertShardType`, indicating that shared expert devices are placed before MoE expert devices.
        - `quantMode` supports 0 (non-quantization), 1 (static quantization), 2 (pertoken dynamic quantization), 3 (pergroup dynamic quantization), and 4 (mx dynamic quantization).
    - Shape format description in the parameter description:
        - `H`: Hidden size. The value range is [1024, 8192].
        - `BS`: Batch sequence size, that is, the number of tokens finally output by the current device. The value range is (0, 512].
    - `HCCL_BUFFSIZE`: Before calling this operator, check whether the value of the `HCCL_BUFFSIZE` environment variable is reasonable. This environment variable indicates the memory size occupied by a single communicator, in MB. If it is not configured, the default value is 200MB. The value must be >= 2 and satisfy:
        - `commAlg` = "fullmesh_v1" or "": >= 2 * (`localExpertNum` * `maxBS` * `epWorldSize` * Align512(Align32(2 * `H`) + 64) + (`K` + `sharedExpertNum`) * `maxBS` * 480Align512(2 * `H`)).
        - `commAlg` = "fullmesh_v2": >= 2 * (`localExpertNum` * `maxBS` * `epWorldSize` * 480Align512(Align32(2 * `H`) + 64) + (`K` + `sharedExpertNum`) * `maxBS` * 480Align512(2 * `H`)).
        - Where `480Align512(x) = ((x + 480 - 1) / 480) * 512`, `Align512(x) = ((x + 512 - 1) / 512) * 512`, and `Align32(x) = ((x + 32 - 1) / 32) * 32`.
        - For <term>Ascend 950DT products</term>, when `quantMode` is 0 (non-quantization) and `scalesOptional` is not empty, the length of each token in the formula is `max(Align32(2 * H) + 64, Align32(Align32(2 * H) + 4 * H) + 16)`, and the result is aligned to 512B for fullmesh_v1 or 480B for fullmesh_v2.

## Invocation Instructions

| Invocation Method  | Sample Code                                  | Description                                                     |
| :--------: | :----------------------------------------: | :-------------------------------------------------------: |
| aclnn API | [test_aclnn_moe_distribute_dispatch_v2.cpp](./examples/test_aclnn_moe_distribute_dispatch_v2.cpp) | Invokes the moe_distribute_dispatch_v2 operator through the [aclnnMoeDistributeDispatchV2](docs/aclnnMoeDistributeDispatchV2_en.md) API. |
