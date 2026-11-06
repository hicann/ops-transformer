# MoeDistributeDispatch

English | [简体中文](./README.md)

## Product Support

| Product                                                         | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950DT products</term>                             |    √     |
| <term>Atlas A3 products</term>       |    √     |
| <term>Atlas A2 products</term> |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                               |    ×     |
| <term>Atlas training products</term>                              |    ×     |

## Function Description

- Operator function: Quantizes token data (optional) and performs AllToAllV communication in the EP (Expert Parallelism) domain. TP (Tensor Parallelism) domain communication is no longer supported in the current version, and TP-related parameters are reserved.
- Calculation formula:

    - Case 1: If quaneMode=0 (non-quantization scenario):

    $$
    expandXOut = AllToAllV(X)
    $$

    - Case 2: If quaneMode=2 (pertoken dynamic quantization scenario):

    $$
    xFp32 = CastToFp32(X) \times scales \\
    dynamicScales = dstTypeMax/Max(Abs(xFp32)) \\
    quantOut = CastToInt8(xFp32 \times dynamicScalesValue) \\
    allToAllXOut = AllToAllV(quantOut) \\
    allToAllDynamicScalesOut = AllToAllV(1.0/dynamicScales) \\
    expandXOut = AllToAllV(quantOut) \\
    dynamicScalesOut = allToAllDynamicScalesOut
    $$

    Note that this operator must be used together with `MoeDistributeCombine`.

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
<td>Token data sent by the current card.</td>
<td>FLOAT16, BFLOAT16</td>
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
<td>scales</td>
<td>Optional input</td>
<td>Smoothing weight of each expert, quantization coefficient, or quantization coefficient of the fused quantization smoothing weight.</td>
<td>FLOAT32</td>
<td>ND</td>
</tr>
<tr>
<td>xActiveMask</td>
<td>Optional input</td>
<td>Reserved parameter. This parameter is not supported in the current version. Pass a null pointer.</td>
<td>-</td>
<td>ND</td>
</tr>
<tr>
<td>expertScales</td>
<td>Optional input</td>
<td>TopK expert weights of each token.</td>
<td>FLOAT32</td>
<td>ND</td>
</tr>
<tr>
<td>groupEp</td>
<td>Attribute</td>
<td>Name of the EP communicator (expert parallelism communicator). String length range: [1, 128).</td>
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
<td>ID of the current card in the EP domain. Value range: [0, epWorldSize). The epRankId values of cards in the same EP communicator must be unique.</td>
<td>INT64</td>
<td>ND</td>
</tr>
<tr>
<td>moeExpertNum</td>
<td>Attribute</td>
<td>Number of MoE experts. Value range: (0, 512]. The value must satisfy moeExpertNum % (epWorldSize - sharedExpertRankNum) = 0.</td>
<td>INT64</td>
<td>ND</td>
</tr>
<tr>
<td>groupTp</td>
<td>Optional attribute</td>
<td><li>Name of the TP communicator (reserved parameter, not supported in the current version).</li><li>Default value: "".</li></td>
<td>STRING</td>
<td>ND</td>
</tr>
<tr>
<td>tpWorldSize</td>
<td>Optional attribute</td>
<td><li>Size of the TP communicator (reserved parameter, not supported in the current version, only 0 or 1 is supported).</li><li>Default value: 0.</li></td>
<td>INT64</td>
<td>ND</td>
</tr>
<tr>
<td>tpRankId</td>
<td>Optional attribute</td>
<td><li>ID of the current card in the TP domain (reserved parameter, not supported in the current version, only 0 is supported).</li><li>Default value: 0.</li></td>
<td>INT64</td>
<td>ND</td>
</tr>
<tr>
<td>expertShardType</td>
<td>Optional attribute</td>
<td><li>Indicates the distribution type of shared expert cards. Currently, only 0 is supported, indicating that shared expert cards are placed before MoE expert cards.</li><li>Default value: 0.</li></td>
<td>INT64</td>
<td>ND</td>
</tr>
<tr>
<td>sharedExpertNum</td>
<td>Optional attribute</td>
<td><li>Indicates the number of shared experts (one shared expert can be replicated and deployed on multiple cards). Current value range: [0, 1]. 0 indicates no shared expert, and 1 indicates one shared expert. Only 1 is supported in the current version.</li><li>Default value: 1.</li></td>
<td>INT64</td>
<td>ND</td>
</tr>
<tr>
<td>sharedExpertRankNum</td>
<td>Optional attribute</td>
<td><li>Indicates the number of shared expert cards. Current value range: [0, epWorldSize). If the value is not 0, epWorldSize % sharedExpertRankNum = 0 must be satisfied.</li><li>Default value: 0.</li></td>
<td>INT64</td>
<td>ND</td>
</tr>
<tr>
<td>quantMode</td>
<td>Optional attribute</td>
<td><li>Indicates the quantization mode. Supported values: 0: no quantization, 1: static quantization, 2: pertoken dynamic quantization, 3: pergroup dynamic quantization, 4: mx quantization.</li><li>Default value: 0.</li></td>
<td>INT64</td>
<td>ND</td>
</tr>
<tr>
<td>globalBs</td>
<td>Optional attribute</td>
<td><li>Global batch size in the EP domain. If the BS values of all ranks are the same, globalBs = BS * epWorldSize or 0. If the BS values of ranks are different, globalBs = maxBS * epWorldSize (maxBS is the maximum BS of a single rank).</li><li>Default value: 0.</li></td>
<td>INT64</td>
<td>ND</td>
</tr>
<tr>
<td>expertTokenNumsType</td>
<td>Optional attribute</td>
<td><li>Semantic type of values in the output expertTokenNums. Supported values: 0: the output in expertTokenNums is the prefix sum of the number of tokens processed by each expert; 1: the output in expertTokenNums is the number of tokens processed by each expert.</li><li>Default value: 1.</li></td>
<td>INT64</td>
<td>ND</td>
</tr>
<tr>
<td>expandX</td>
<td>Output</td>
<td>Token features expanded based on expertIds.</td>
<td>FLOAT16, BFLOAT16, INT8, FLOAT8_E5M2, FLOAT8_E4M3FN, HIFLOAT8</td>
<td>ND</td>
</tr>
<tr>
<td>dynamicScales</td>
<td>Output</td>
<td>In the quantization scenario, indicates the quantization coefficient of the tokens output by the current card.</td>
<td>FLOAT32, FLOAT8_E8M0</td>
<td>ND</td>
</tr>
<tr>
<td>expandIdx</td>
<td>Output</td>
<td>Indicates the number of tokens sent to the same expert (corresponding to expandIdx in the MoeDistributeCombine operator).</td>
<td>INT32</td>
<td>ND</td>
</tr>
<tr>
<td>expertTokenNums</td>
<td>Output</td>
<td>Indicates the number of tokens received by each expert.</td>
<td>INT64</td>
<td>ND</td>
</tr>
<tr>
<td>epRecvCounts</td>
<td>Output</td>
<td>Number of tokens received from each card in the EP communicator (corresponding to epSendCounts in MoeDistributeCombine).</td>
<td>INT32</td>
<td>ND</td>
</tr>
<tr>
<td>tpRecvCounts</td>
<td>Output</td>
<td>Number of tokens received from each card in the TP communicator (reserved output, not supported in the current version, pass a null pointer). If there is no TP domain communication, this output does not exist.</td>
<td>INT32</td>
<td>ND</td>
</tr>
<tr>
<td>expandScales</td>
<td>Output</td>
<td>Indicates the weights of the tokens output by the current card (corresponding to expandScales in MoeDistributeCombine).</td>
<td>FLOAT32</td>
<td>ND</td>
</tr>
</tbody>
</table>

* <term>Atlas A2 products</term>:
    * `FLOAT16` and `BFLOAT16` data types are supported.
    * The `quantMode` attribute supports only 0 and 2.
    * The shared expert scenario is not supported. The `expertShardType`, `sharedExpertNum`, and `sharedExpertRankNum` attributes are not supported.
    * Only the EP domain is supported. TP domain communication is not supported. The `groupTp`, `tpWorldSize`, and `tpRankId` attributes are not supported, and `tpRecvCounts` is invalid.
    * The content of `expandScales` is valid only when the environment variables `HCCL_INTRA_PCIE_ENABLE` = 1 and `HCCL_INTRA_ROCE_ENABLE` = 0 are set.

* <term>Atlas A3 products</term>:
    * The `FLOAT8_E4M3FN`, `FLOAT8_E5M2`, `HIFLOAT8`, and `FLOAT8_E8M0` data types are not supported.
    * The `quantMode` attribute supports only 0 and 2.
    * `expandScales` is not supported.

* <term>Ascend 950DT products</term>:
    * `expandScales` is not supported.
    * TP domain communication is currently not supported, and the `groupTp`, `tpWorldSize`, and `tpRankId` attributes are not supported. `tpSendCounts` is invalid.

## Constraints

- The `MoeDistributeDispatch` operator must be used together with the `MoeDistributeCombine` operator. For details, see the invocation example.

- The driver versions of all nodes in the operator communicator must be the same.

- In different product models, communication algorithms, or versions, the element values in the Tensor outputs `expandIdx`, `epRecvCounts`, `tpRecvCounts`, and `expandScales` of `MoeDistributeDispatch` may differ. When using these Tensors, pass them directly to the corresponding parameters of `MoeDistributeCombine`. Other business logic of the model should not depend on these values.

- The values of the `groupEp`, `epWorldSize`, `moeExpertNum`, `groupTp`, `tpWorldSize`, `expertShardType`, `sharedExpertNum`, `sharedExpertRankNum`, and `globalBs` attributes used during operator invocation must be the same on all cards, the same in different layers of the network, and the same as the corresponding parameters of `MoeDistributeCombine`.

- Shape format description in the parameter description:
    - `A`: Indicates the maximum number of tokens that the current card may receive. The value range is as follows:
        - For shared experts, `A` = `BS` *`epWorldSize`* `sharedExpertNum` / `sharedExpertRankNum` must be satisfied.
        - For MoE experts, if `globalBs` is 0, `A` >= `BS` *`epWorldSize`* min(`localExpertNum`, `K`) must be satisfied. If `globalBs` is not 0, `A` >= `globalBs` * min(`localExpertNum`, `K`) must be satisfied.
    - `localExpertNum`: Indicates the number of experts on the current card.
        - For a shared expert card, `localExpertNum` = 1
        - For a MoE expert card, `localExpertNum` = `moeExpertNum` / (`epWorldSize` - `sharedExpertRankNum`). TP domain communication is currently not supported.

- "/" in the formulas in this document indicates integer division.

- Communicator usage constraints:
    - In a model, `MoeDistributeCombine` and `MoeDistributeDispatch` support only the same EP communicator, and no other operators are allowed in the communicator.
    - TP domain communication is currently not supported.

- Communication mode constraints:
  - <term>Ascend 950DT products</term>: Only UB Memory communication is supported.

- <term>Atlas A2 products</term>:
    - Shape format description in the parameter description:
        - `H`: Indicates the hidden size. Value range: (0, 7168], and the value must be a multiple of 32.
        - `BS`: Indicates the batch sequence size, that is, the number of tokens finally output by the current card. Value range: [1, 256].
        - `K`: Indicates that topK experts are selected. The value must satisfy 0 < `K` ≤ moeExpertNum. Value range: [1, 16].
    - `HCCL_BUFFSIZE`: Before invoking this operator, check whether the value of the `HCCL_BUFFSIZE` environment variable is reasonable. This environment variable indicates the memory size occupied by a single communicator, in MB. If not configured, the default value is 200MB. The value must be >= (`BS` *`epWorldSize`* min(`localExpertNum`, `K`) *`H`* 4B + 4MB).
    - `HCCL_INTRA_PCIE_ENABLE` and `HCCL_INTRA_ROCE_ENABLE`: Setting the environment variables `HCCL_INTRA_PCIE_ENABLE` = 1 and `HCCL_INTRA_ROCE_ENABLE` = 0 reduces the amount of data transmitted across machines and may improve operator performance. In this case, `HCCL_BUFFSIZE` >= `moeExpertNum` *`BS`* (`H` *2 + 16* Align8(`K`))B + 104MB is required. In addition, for the input parameter `moeExpertNum`, only `moeExpertNum` % `epWorldSize` = 0 is required, and `moeExpertNum` / `epWorldSize` <= 24 is not required. However, the `scales` feature is not supported. Align8(x) = ((x + 8 - 1) / 8) * 8.
    - `epWorldSize`: Supported values: 16, 32, 64.
    - `quantMode` constraints:
        - When `quantMode` is 2, the scenario is pertoken dynamic quantization, and the data type of `expandX` supports `INT8`.
            - The input `scales` can be a null pointer.
            - If valid data is passed for the input `scales`, the shape is (`moeExpertNum`, `H`).
    - Networking constraint: In the multi-machine scenario, only switch-based networking is supported, and two-machine direct connection networking is not supported.

- <term>Atlas A3 products</term>:
    - In this scenario, a single card contains dual DIEs (also referred to as dies). Therefore, the "current card" in the parameter description refers to a single DIE.
    - Shape format description in the parameter description:
        - `H`: Indicates the hidden size. Value: 7168.
        - `BS`: Indicates the batch sequence size, that is, the number of tokens finally output by the current card. Value range: [1, 512].
        - `K`: Indicates that topK experts are selected. The value must satisfy 0 < `K` ≤ moeExpertNum. Value range: [1, 8].
    - `HCCL_BUFFSIZE`: Before invoking this operator, check whether the value of the `HCCL_BUFFSIZE` environment variable is reasonable. This environment variable indicates the memory size occupied by a single communicator, in MB. If not configured, the default value is 200MB. The value must be >= 2 and satisfy 1024 ^ 2 *(`HCCL_BUFFSIZE` - 2) / 2 >= `BS`* 2 *(`H` + 128)* (`epWorldSize` * `localExpertNum` + `K` + 1). `localExpertNum` must be the number of local experts on a MoE expert card.
    - `epWorldSize`: Supported values: 8, 16, 32, 64, 128, 144, 256, 288.
    - `quantMode` constraints:
        - When `quantMode` is 2, the scenario is pertoken dynamic quantization, and the data type of `expandX` supports `INT8`.
            - The input `scales` can be a null pointer.
            - If valid data is passed for the input `scales` and shared expert cards exist, the shape is (`sharedExpertNum` + `moeExpertNum`, `H`).
            - If valid data is passed for the input `scales` and no shared expert card exists, the shape is (`moeExpertNum`, `H`).

- <term>Ascend 950DT products</term>:
    - Shape format description in the parameter description:
        - `H`: Indicates the hidden size. Value: 7168.
        - `BS`: Indicates the batch sequence size, that is, the number of tokens finally output by the current card. Value range: [1, 512].
        - `K`: Indicates that topK experts are selected. The value must satisfy 0 < `K` ≤ moeExpertNum. Value range: [1, 8].
    - `epWorldSize`: Supported values: 2, 4, 8, 16, 32, 64, 128, 144, 256, 288.
    - `HCCL_BUFFSIZE`: Before invoking this operator, check whether the value of the `HCCL_BUFFSIZE` environment variable is reasonable. This environment variable indicates the memory size occupied by a single communicator, in MB. If not configured, the default value is 200MB. The value must be >= 2 and satisfy 1024 ^ 2 *(`HCCL_BUFFSIZE` - 2) / 2 >= `BS`* 2 *(`H` + 128)* (`epWorldSize` * `localExpertNum` + `K` + 1). `localExpertNum` must be the number of local experts on a MoE expert card.
    - `quantMode` constraints:
        - When `quantMode` is 2, the scenario is pertoken dynamic quantization, and the data type of `expandX` supports `INT8`.
            - The input `scales` can be a null pointer.
            - If valid data is passed for the input `scales` and shared expert cards exist, the shape is (`sharedExpertNum` + `moeExpertNum`, `H`).
            - If valid data is passed for the input `scales` and no shared expert card exists, the shape is (`moeExpertNum`, `H`).

## Invocation Description

| Invocation Method | Sample Code                                  | Description                                                     |
| :--------: | :----------------------------------------: | :-------------------------------------------------------: |
| aclnn API | [test_aclnn_moe_distribute_dispatch.cpp](./examples/test_aclnn_moe_distribute_dispatch.cpp) | The moe_distribute_dispatch operator is invoked through the [aclnnMoeDistributeDispatch](docs/aclnnMoeDistributeDispatch_en.md) API. |
