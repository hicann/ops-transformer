# MoeDistributeCombine

English | [简体中文](./README.md)

## Product Support Status

| Product                                                         | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950DT products</term>                             |    √     |
| <term>Atlas A3 products</term>       |    √     |
| <term>Atlas A2 products</term> |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                               |    ×     |
| <term>Atlas training products</term>                              |    ×     |

## Function Description

Operator function: performs AllToAllV communication and finally aggregates the received data (multiplied by weights and then added). TP-domain communication is not supported.

- Computation formulas:

    $$
    ataOut = AllToAllV(expandX)\\
    xOut = Sum(expertScales * ataOut) + sharedExpertX
    $$

Note that this operator must be used together with MoeDistributeDispatch, which is equivalent to returning the data along the same path over which the MoeDistributeDispatch operator collected it.

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
<td>Token features expanded according to expertIds.</td>
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
<td>expandIdx</td>
<td>Input</td>
<td>Indicates the number of tokens received by the same expert, corresponding to the expandIdx output in MoeDistributeDispatch.</td>
<td>INT32</td>
<td>ND</td>
</tr>
<tr>
<td>epSendCounts</td>
<td>Input</td>
<td>Number of tokens received from each card in the EP communicator, corresponding to the epRecvCounts output in MoeDistributeDispatch.</td>
<td>INT32</td>
<td>ND</td>
</tr>
<tr>
<td>expertScales</td>
<td>Input</td>
<td>Weights of the topK experts of each token.</td>
<td>FLOAT32</td>
<td>ND</td>
</tr>
<tr>
<td>tpSendCounts</td>
<td>Optional input</td>
<td>Reserved parameter. TP-domain communication is no longer supported; pass a null pointer.</td>
<td>INT32</td>
<td>ND</td>
</tr>
<tr>
<td>xActiveMask</td>
<td>Optional input</td>
<td>Reserved parameter. Not supported in the current version; pass a null pointer.</td>
<td>-</td>
<td>ND</td>
</tr>
<tr>
<td>activationScale</td>
<td>Optional input</td>
<td>Reserved parameter. Not supported in the current version; pass a null pointer.</td>
<td>-</td>
<td>ND</td>
</tr>
<tr>
<td>weightScale</td>
<td>Optional input</td>
<td>Reserved parameter. Not supported in the current version; pass a null pointer.</td>
<td>-</td>
<td>ND</td>
</tr>
<tr>
<td>groupList</td>
<td>Optional input</td>
<td>Reserved parameter. Not supported in the current version; pass a null pointer.</td>
<td>-</td>
<td>ND</td>
</tr>
<tr>
<td>expandScales</td>
<td>Optional input</td>
<td>Indicates the weights of the tokens input on the local card, corresponding to the expandScales output in MoeDistributeDispatch.</td>
<td>FLOAT32</td>
<td>ND</td>
</tr>
<tr>
<td>groupEp</td>
<td>Attribute</td>
<td>EP communicator name (expert parallel communicator); the string length range is [1, 128).</td>
<td>STRING</td>
<td>ND</td>
</tr>
<tr>
<td>epWorldSize</td>
<td>Attribute</td>
<td>EP communicator size.</td>
<td>INT64</td>
<td>ND</td>
</tr>
<tr>
<td>epRankId</td>
<td>Attribute</td>
<td>ID of the local card in the EP domain; the value range is [0, epWorldSize), and the epRankId values of the cards in the same EP communicator must be unique.</td>
<td>INT64</td>
<td>ND</td>
</tr>
<tr>
<td>moeExpertNum</td>
<td>Attribute</td>
<td>Number of MoE experts; the value range is (0, 512], and moeExpertNum % (epWorldSize - sharedExpertRankNum) = 0 must be satisfied.</td>
<td>INT64</td>
<td>ND</td>
</tr>
<tr>
<td>groupTp</td>
<td>Optional attribute</td>
<td><li>TP communicator name (data parallel communicator). Reserved parameter; TP-domain communication is no longer supported.</li><li>The default value is "".</li></td>
<td>STRING</td>
<td>ND</td>
</tr>
<tr>
<td>tpWorldSize</td>
<td>Optional attribute</td>
<td><li>TP communicator size. Reserved parameter; TP-domain communication is no longer supported; only 0 or 1 is allowed.</li><li>The default value is 0.</li></td>
<td>INT64</td>
<td>ND</td>
</tr>
<tr>
<td>tpRankId</td>
<td>Optional attribute</td>
<td><li>ID of the local card in the TP domain. Reserved parameter; TP-domain communication is no longer supported; pass 0.</li><li>The default value is 0.</li></td>
<td>INT64</td>
<td>ND</td>
</tr>
<tr>
<td>expertShardType</td>
<td>Optional attribute</td>
<td><li>Indicates the distribution type of shared expert cards. Currently only 0 can be passed, indicating that shared expert cards are placed before MoE expert cards.</li><li>The default value is 0.</li></td>
<td>INT64</td>
<td>ND</td>
</tr>
<tr>
<td>sharedExpertNum</td>
<td>Optional attribute</td>
<td><li>Indicates the number of shared experts (one shared expert can be replicated and deployed on multiple cards).</li><li>The default value is 1.</li></td>
<td>INT64</td>
<td>ND</td>
</tr>
<tr>
<td>sharedExpertRankNum</td>
<td>Optional attribute</td>
<td><li>Indicates the number of shared expert cards.</li><li>The default value is 0.</li></td>
<td>INT64</td>
<td>ND</td>
</tr>
<tr>
<td>globalBs</td>
<td>Optional attribute</td>
<td><li>Global batch size in the EP domain.</li><li>The default value is 0.</li></td>
<td>INT64</td>
<td>ND</td>
</tr>
<tr>
<td>outDtype</td>
<td>Optional attribute</td>
<td><li>Specifies the data type of the output x. Reserved parameter; not supported in the current version; pass 0.</li><li>The default value is 0.</li></td>
<td>INT64</td>
<td>ND</td>
</tr>
<tr>
<td>commQuantMode</td>
<td>Optional attribute</td>
<td><li>Communication quantization type; the value range is 0 or 2. 0 indicates that communication is not quantized, and 2 indicates int8 quantization for communication.</li><li>The default value is 0.</li></td>
<td>INT64</td>
<td>ND</td>
</tr>
<tr>
<td>groupListType</td>
<td>Optional attribute</td>
<td><li>Format of groupList. Reserved parameter; not supported in the current version; pass 0.</li><li>The default value is 0.</li></td>
<td>INT64</td>
<td>ND</td>
</tr>
<tr>
<td>x</td>
<td>Output</td>
<td>Indicates the processed tokens; the data type and data format are consistent with expandX.</td>
<td>FLOAT16, BFLOAT16</td>
<td>ND</td>
</tr>
</tbody>
</table>

* <term>Atlas A2 products</term>:
    * The shared expert scenario is not supported, and the `expertShardType`, `sharedExpertNum`, and `sharedExpertRankNum` attributes are not supported.
    * TP-domain communication is currently not supported, the `groupTp`, `tpWorldSize`, and `tpRankId` attributes are not supported, and `tpSendCounts` is invalid content.
    * `expandScales` must be passed in only when the environment variables `HCCL_INTRA_PCIE_ENABLE` = 1 and `HCCL_INTRA_ROCE_ENABLE` = 0 are set.

* <term>Atlas A3 products</term>:
    * `expandScales` is not supported.
    * TP-domain communication is currently not supported, the `groupTp`, `tpWorldSize`, and `tpRankId` attributes are not supported, and `tpSendCounts` is invalid content.

* <term>Ascend 950DT products</term>:
    * `expandScales` is not supported.
    * TP-domain communication is currently not supported, the `groupTp`, `tpWorldSize`, and `tpRankId` attributes are not supported, and `tpSendCounts` is invalid content.

## Constraints

- The `MoeDistributeDispatch` operator and the `MoeDistributeCombine` operator must be used together. For details, refer to the invocation example.

- The driver versions of all nodes in the operator communicator must be the same.

- Across different product models, different communication algorithms, or different versions, the element values in the Tensor outputs `expandIdx`, `epRecvCounts`, `tpRecvCounts`, and `expandScales` of `MoeDistributeDispatch` may differ. When using them, directly pass the preceding Tensors to the corresponding parameters of `MoeDistributeCombine`; other business logic of the model should not depend on them.

- The values of the `groupEp`, `epWorldSize`, `moeExpertNum`, `groupTp`, `tpWorldSize`, `expertShardType`, `sharedExpertNum`, `sharedExpertRankNum`, and `globalBs` attributes used during operator invocation must be consistent across all cards, must be consistent across different layers in the network, and must also be consistent with the corresponding parameters of `MoeDistributeDispatch`.

- Shape format description in the parameter description:
    - `A`: indicates the maximum number of tokens that the local card may receive; the value range is as follows:
        - For shared experts, `A` = `BS` *`epWorldSize`* `sharedExpertNum` / `sharedExpertRankNum` must be satisfied.
        - For MoE experts, when `globalBs` is 0, `A` >= `BS` *`epWorldSize`* min(`localExpertNum`, `K`) must be satisfied; when `globalBs` is not 0, `A` >= `globalBs` * min(`localExpertNum`, `K`) must be satisfied.
    - `localExpertNum`: indicates the number of experts on the local card.
        - For shared expert cards, `localExpertNum` = 1
        - For MoE expert cards, `localExpertNum` = `moeExpertNum` / (`epWorldSize` - `sharedExpertRankNum`). TP-domain communication is not supported in the current version.

- In the formulas in this document, "/" denotes integer division.

- Communicator usage constraints:
    - In one model, `MoeDistributeCombine` and `MoeDistributeDispatch` support only the same EP communicator, and no other operators are allowed in that communicator.
    - TP-domain communication is currently not supported.

- Communication mode constraints:
    - <term>Ascend 950DT products</term>: only UB Memory communication is supported.

- <term>Atlas A2 products</term>:
    - Shape format description in the parameter description:
        - `H`: indicates the hidden size (hidden layer size); the value range is (0, 7168], and H must be a multiple of 32.
        - `BS`: indicates the batch sequence size, that is, the number of tokens finally output by the local card; the value range is [1, 256].
        - `K`: indicates selecting topK experts; 0 < `K` ≤ `moeExpertNum` must be satisfied; the value range is [1, 16].
    - `HCCL_BUFFSIZE`: before invoking this operator, check whether the value of the `HCCL_BUFFSIZE` environment variable is reasonable. This environment variable indicates the memory size occupied by a single communicator, in MB; the default value is 200 MB when it is not configured, and it must satisfy >= (`BS` *`epWorldSize`* min(`localExpertNum`, `K`) *`H`* 4B + 4MB).
    - `HCCL_INTRA_PCIE_ENABLE` and `HCCL_INTRA_ROCE_ENABLE`: setting the environment variables `HCCL_INTRA_PCIE_ENABLE` = 1 and `HCCL_INTRA_ROCE_ENABLE` = 0 can reduce the volume of cross-machine communication data and may improve operator performance. In this case, `HCCL_BUFFSIZE` >= `moeExpertNum` *`BS`* (`H` *2 + 16* Align8(`K`))B + 104MB is required. In addition, for the input parameter `moeExpertNum`, only `moeExpertNum` % `epWorldSize` = 0 is required, and `moeExpertNum` / `epWorldSize` <= 24 is not required, where Align8(x) = ((x + 8 - 1) / 8) * 8.
    - Parameter constraints:
        - `epWorldSize`: supported values are 16, 32, and 64.
        - `moeExpertNum`: `moeExpertNum` / `epWorldSize` <= 24 must be satisfied.
            - When the environment variables `HCCL_INTRA_PCIE_ENABLE` = 1 and `HCCL_INTRA_ROCE_ENABLE` = 0 are set, the preceding constraint does not apply.
        - `globalBs`: when the `BS` values of all ranks are the same, `globalBs` = `BS` *`epWorldSize` or `globalBs` = 0; when the `BS` values of ranks differ, `globalBs` = `maxBS`* `epWorldSize` or `globalBs` = 256 *`epWorldSize`, where `maxBS` indicates the maximum single-rank `BS` value. Passing in `maxBS`* `epWorldSize` is recommended; always passing in 256 * `epWorldSize` cannot support scenarios where BS supports values greater than 256 in later versions.
        - The value range of `commQuantMode` is 0 or 2. 0 indicates that communication is not quantized, and 2 indicates int8 quantization for communication (2 is supported only when HCCL_INTRA_PCIE_ENABLE=1, HCCL_INTRA_ROCE_ENABLE=0, and the driver version is ≥ 25.0.RC1.1).
    - Networking constraint: the multi-machine scenario supports only switch-based networking; direct connection between two machines is not supported.

- <term>Atlas A3 products</term>:
    - In this scenario, a single card contains dual dies (also known as bare dies), so "the local card" in the parameter description refers to a single die.
    - Shape format description in the parameter description:
        - `H`: indicates the hidden size (hidden layer size); the value is 7168.
        - `BS`: indicates the batch sequence size, that is, the number of tokens finally output by the local card; the value range is [1, 512].
        - `K`: indicates selecting topK experts; 0 < `K` ≤ moeExpertNum must be satisfied; the value range is [1, 8].
    - Parameter constraints:
        - `epWorldSize`: supported values are 8, 16, 32, 64, 128, 144, 256, and 288.
        - `groupTp`: reserved parameter; TP-domain communication is no longer supported.
        - `tpWorldSize`: reserved parameter; TP-domain communication is no longer supported; only 0 or 1 is allowed.
        - `tpRankId`: reserved parameter; TP-domain communication is no longer supported; pass 0.
        - `sharedExpertRankNum`: the current value range is [0, `epWorldSize`); when it is not 0, `epWorldSize` % `sharedExpertRankNum` = 0 must be satisfied.
        - `globalBs`: when the `BS` values of all ranks are the same, `globalBs` = `BS` *`epWorldSize` or `globalBs` = 0; when the `BS` values of ranks differ, `globalBs` = `maxBS`* `epWorldSize`, where `maxBS` indicates the maximum single-card `BS` value.
    - `HCCL_BUFFSIZE`: before invoking this operator, check whether the value of the `HCCL_BUFFSIZE` environment variable is reasonable. This environment variable indicates the memory size occupied by a single communicator, in MB; the default value is 200 MB when it is not configured. It must satisfy >= 2 and 1024 ^ 2 *(`HCCL_BUFFSIZE` - 2) / 2 >= `BS`* 2 *(`H` + 128)* (`epWorldSize` * `localExpertNum` + `K` + 1), where `localExpertNum` must use the number of local experts on MoE expert cards.

- <term>Ascend 950DT products</term>:
    - Parameter constraints:
        - `epWorldSize`: supported values are 2, 4, 8, 16, 32, 64, 128, 144, 256, and 288.
        - `sharedExpertRankNum`: the current value range is [0, `epWorldSize`); when it is not 0, `epWorldSize` % `sharedExpertRankNum` = 0 must be satisfied.
        - `globalBs`: when the `BS` values of all ranks are the same, `globalBs` = `BS` *`epWorldSize` or `globalBs` = 0; when the `BS` values of ranks differ, `globalBs` = `maxBS`* `epWorldSize`, where `maxBS` indicates the maximum single-card `BS` value.
        - The value range of `commQuantMode` is 0 or 2. 0 indicates that communication is not quantized, and 2 indicates int8 quantization for communication.
    - `HCCL_BUFFSIZE`: before invoking this operator, check whether the value of the `HCCL_BUFFSIZE` environment variable is reasonable. This environment variable indicates the memory size occupied by a single communicator, in MB; the default value is 200 MB when it is not configured. It must satisfy >= 2 and 1024 ^ 2 *(`HCCL_BUFFSIZE` - 2) / 2 >= `BS`* 2 *(`H` + 128)* (`epWorldSize` * `localExpertNum` + `K` + 1), where `localExpertNum` must use the number of local experts on MoE expert cards.

## Invocation Description

| Invocation Method  | Sample Code                                  | Description                                                     |
| :--------: | :----------------------------------------: | :-------------------------------------------------------: |
| aclnn interface | [test_aclnn_moe_distribute_combine.cpp](./examples/test_aclnn_moe_distribute_combine.cpp) | Invoke the moe_distribute_combine operator through the [aclnnMoeDistributeCombine](docs/aclnnMoeDistributeCombine_en.md) interface. |
