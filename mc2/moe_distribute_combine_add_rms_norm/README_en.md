# MoeDistributeCombineAddRmsNorm

English | [简体中文](./README.md)

## Product Support Status

| Product                                                         | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950DT products</term>     |    √     |
| <term>Atlas A3 products</term>       |    √     |
| <term>Atlas A2 products</term> |    ×     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                               |    ×     |
| <term>Atlas training products</term>                              |    ×     |

## Function Description

**Operator function: performs AllToAllV communication, then aggregates the received data (multiplied by weights and then added), and then completes the fused Add + RmsNorm. TP (Tensor Parallelism) domain communication is no longer supported in the current version, and TP-related parameters are reserved parameters.

**Computation formulas**:

$$
ataOut = AllToAllV(expandX)\\
combineOut = Sum(expertScales * ataOut + expertScales * sharedExpertX)\\
x = combineOut + residualX\\
y = \frac{x}{RMS(x)} * gamma,\quad\text{where}RMS(x) = \sqrt{\frac{1}{H}\sum_{i=1}^{H}x_{i}^{2}+normEps}
$$

Note that this interface must be used together with aclnnMoeDistributeDispatchV2, which is equivalent to returning the data along the same path over which the MoeDistributeDispatchV2 operator collected it.

## Parameter Description

<table style="undefined;table-layout: fixed; width: 1550px"> <colgroup>
<col style="width: 180px">
<col style="width: 120px">
<col style="width: 280px">
<col style="width: 320px">
<col style="width: 250px">
<col style="width: 120px">
<col style="width: 140px">
<col style="width: 140px">
</colgroup>
<thead>
<tr>
<th>Parameter Name</th>
<th>Input/Output</th>
<th>Description</th>
<th>Usage Instructions</th>
<th>Data Type</th>
<th>Data Format</th>
<th>Dimension</th>
<th>Non-contiguous</th>
</tr>
</thead>
<tbody>
<tr>
<td>expandX</td>
<td>Input</td>
<td>Token features expanded according to expertIds.</td>
<td>Must be a 2D tensor with shape (A , H).</td>
<td>BFLOAT16</td>
<td>ND</td>
<td>2</td>
<td>√</td>
</tr>
<tr>
<td>expertIds</td>
<td>Input</td>
<td>TopK expert indices of each token.</td>
<td>Must be a 2D tensor with shape (Bs, K).</td>
<td>INT32</td>
<td>ND</td>
<td>2</td>
<td>√</td>
</tr>
<tr>
<td>assistInfoForCombine</td>
<td>Input</td>
<td>Corresponds to the assistInfoForCombineOut output in aclnnMoeDistributeDispatchV2.</td>
<td>Must be a 1D tensor with shape (A * 128, ).</td>
<td>INT32</td>
<td>ND</td>
<td>1</td>
<td>√</td>
</tr>
<tr>
<td>epSendCounts</td>
<td>Input</td>
<td>Corresponds to the epRecvCounts output in aclnnMoeDistributeDispatchV2.</td>
<td>Must be a 1D tensor with shape (epWorldSize * localExpertNum, ).</td>
<td>INT32</td>
<td>ND</td>
<td>1</td>
<td>√</td>
</tr>
<tr>
<td>expertScales</td>
<td>Input</td>
<td>Weights of the topK experts of each token.</td>
<td>Must be a 2D tensor with shape (Bs, K).</td>
<td>FLOAT32</td>
<td>ND</td>
<td>2</td>
<td>√</td>
</tr>
<tr>
<td>residualX</td>
<td>Input</td>
<td>Right matrix of the Add in AddRmsNorm.</td>
<td>Must be a 3D tensor with shape (Bs, 1, H).</td>
<td>BFLOAT16</td>
<td>ND</td>
<td>3</td>
<td>√</td>
</tr>
<tr>
<td>gamma</td>
<td>Input</td>
<td>gamma input in RmsNorm.</td>
<td>Must be a 1D tensor with shape (H, ).</td>
<td>BFLOAT16</td>
<td>ND</td>
<td>1</td>
<td>√</td>
</tr>
<tr>
<td>tpSendCountsOptional</td>
<td>Input</td>
<td>Reserved parameter; TP-domain communication is no longer supported.</td>
<td>Not supported in the current version; pass a null pointer.</td>
<td>INT32</td>
<td>ND</td>
<td>-</td>
<td>√</td>
</tr>
<tr>
<td>xActiveMaskOptional</td>
<td>Input</td>
<td>Indicates whether a token participates in communication.</td>
<td><ul><li>Valid data or a null pointer can be passed; by default, all tokens participate in communication. When 1D, the shape is (BS, ); when 2D, the shape is (BS, K).</li></ul></td>
<td>BOOL</td>
<td>ND</td>
<td>-</td>
<td>√</td>
</tr>
<tr>
<td>activationScaleOptional</td>
<td>Input</td>
<td>Reserved parameter.</td>
<td>Not supported in the current version; pass a null pointer.</td>
<td>-</td>
<td>ND</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td>weightScaleOptional</td>
<td>Input</td>
<td>Reserved parameter.</td>
<td>Not supported in the current version; pass a null pointer.</td>
<td>-</td>
<td>ND</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td>groupListOptional</td>
<td>Input</td>
<td>Reserved parameter.</td>
<td>Not supported in the current version; pass a null pointer.</td>
<td>-</td>
<td>ND</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td>expandScalesOptional</td>
<td>Input</td>
<td>Corresponds to the expandScales output in aclnnMoeDistributeDispatchV2; reserved parameter.</td>
<td>Not supported in the current version; pass a null pointer.</td>
<td>-</td>
<td>ND</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td>sharedExpertXOptional</td>
<td>Input</td>
<td>Indicates the tokens computed by the shared experts.</td>
<td>Valid data or a null pointer can be passed; when 2D, the shape is (Bs, H), and when 3D, the shape is (Bs, 1, H))</td>
<td>BFLOAT16</td>
<td>ND</td>
<td>-</td>
<td>√</td>
</tr>
<tr>
<td>groupEp</td>
<td>Input</td>
<td>EP communicator name (expert parallel communicator).</td>
<td>The string length range is [1, 128).</td>
<td>STRING</td>
<td>ND</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td>epWorldSize</td>
<td>Input</td>
<td>EP communicator size.</td>
<td>-</td>
<td>INT64</td>
<td>ND</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td>epRankId</td>
<td>Input</td>
<td>ID of the local card in the EP domain.</td>
<td>The value range is [0, epWorldSize), and the epRankId values of the cards in the same EP communicator must be unique.</td>
<td>INT64</td>
<td>ND</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td>moeExpertNum</td>
<td>Input</td>
<td>Number of MoE experts.</td>
<td>moeExpertNum % (epWorldSize - sharedExpertRankNum) = 0 must be satisfied.</td>
<td>INT64</td>
<td>ND</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td>groupTp</td>
<td>Input</td>
<td>TP communicator name (reserved parameter; not supported in the current version).</td>
<td>The default value is "".</td>
<td>STRING</td>
<td>ND</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td>tpWorldSize</td>
<td>Input</td>
<td>TP communicator size (reserved parameter; not supported in the current version).</td>
<td>Only 0 or 1 is allowed; the default value is 0.</td>
<td>INT64</td>
<td>ND</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td>tpRankId</td>
<td>Input</td>
<td>ID of the local card in the TP domain (reserved parameter; not supported in the current version).</td>
<td>Pass 0; the default value is 0.</td>
<td>INT64</td>
<td>ND</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td>expertShardType</td>
<td>Input</td>
<td>Indicates the distribution type of shared expert cards.</td>
<td>Currently only 0 can be passed, indicating that shared expert cards are placed before MoE expert cards</td>
<td>INT64</td>
<td>ND</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td>sharedExpertNum</td>
<td>Input</td>
<td>Indicates the number of shared experts.</td>
<td>Not supported in the current version; pass 0.</td>
<td>INT64</td>
<td>ND</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td>sharedExpertRankNum</td>
<td>Input</td>
<td>Indicates the number of shared expert cards.</td>
<td>Not supported in the current version; pass 0.</td>
<td>INT64</td>
<td>ND</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td>globalBs</td>
<td>Input</td>
<td>Global batch size in the EP domain.</td>
<td><ul><li>When the Bs values of all ranks are the same, globalBs = Bs * epWorldSize or 0.</li><li>When the Bs values of ranks differ, globalBs = maxBs * epWorldSize (maxBs is the maximum single-card Bs value).</li></ul></td>
<td>INT64</td>
<td>ND</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td>outDtype</td>
<td>Input</td>
<td>Specifies the data type of the output x; reserved parameter</td>
<td>Not supported in the current version; pass 0.</td>
<td>INT64</td>
<td>ND</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td>commQuantMode</td>
<td>Input</td>
<td>Communication quantization type.</td>
<td>Not supported in the current version; pass 0.</td>
<td>INT64</td>
<td>ND</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td>groupListType</td>
<td>Input</td>
<td>Format of group List; reserved parameter.</td>
<td>Not supported in the current version; pass 0.</td>
<td>INT64</td>
<td>ND</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td>commAlg</td>
<td>Input</td>
<td>Indicates the communication-affinity memory layout algorithm.</td>
<td>Reserved field; not supported in the current version; pass a null pointer.</td>
<td>STRING</td>
<td>ND</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td>normEps</td>
<td>Input</td>
<td>Used to prevent a divide-by-zero error in AddRmsNorm.</td>
<td>The value can be 1e-6.</td>
<td>FLOAT</td>
<td>ND</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td>yOut</td>
<td>Output</td>
<td>Output result after RmsNorm.</td>
<td>Must be a 3D tensor.</td>
<td>BFLOAT16</td>
<td>ND</td>
<td>3</td>
<td>-</td>
</tr>
<tr>
<td>rstdOut</td>
<td>Output</td>
<td>Output result after RmsNorm.</td>
<td>Must be a 3D tensor with shape (Bs, 1, 1).</td>
<td>FLOAT32</td>
<td>ND</td>
<td>3</td>
<td>√</td>
</tr>
<tr>
<td>xOut</td>
<td>Output</td>
<td>Output result after Add.</td>
<td>Must be a 3D tensor with shape (Bs, 1, H).</td>
<td>BFLOAT16</td>
<td>ND</td>
<td>3</td>
<td>-</td>
</tr>
<tr>
<td>workspaceSize</td>
<td>Output</td>
<td>Returns the workspace size that needs to be requested on the Device.</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td>executor</td>
<td>Output</td>
<td>Returns the op executor, which contains the computation flow of the operator.</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
</tr>
</tbody>
</table>

## Constraints

- The aclnnMoeDistributeDispatchV2 interface and the aclnnMoeDistributeCombineAddRmsNorm interface must be used together. For details, refer to the invocation example.

- The values of the groupEp, epWorldSize, moeExpertNum, groupTp, tpWorldSize, expertShardType, sharedExpertNum, sharedExpertRankNum, and globalBs parameters used during interface invocation must be consistent across all cards, must be consistent across different layers in the network, and must also be consistent with the corresponding parameters of aclnnMoeDistributeDispatchV2.

- <term>Atlas A3 products</term>: in this scenario, a single card contains dual dies (also known as bare dies), so "the local card" in the parameter description refers to a single die.

- Shape format description in the parameter description:
    - A: indicates the maximum number of tokens that the local card needs to distribute; the value range is as follows:
        - For shared experts, A = Bs *epWorldSize* sharedExpertNum / sharedExpertRankNum must be satisfied.
        - For MoE experts, when globalBs is 0, A >= Bs *epWorldSize* min(localExpertNum, K) must be satisfied; when globalBs is not 0, A >= globalBs * min(localExpertNum, K) must be satisfied.
    - H: indicates the hidden size (hidden layer size); the value range is [1024, 8192].
    - Bs: indicates the batch sequence size, that is, the number of tokens finally output by the local card; the value range is 0 < Bs ≤ 512.
    - K: indicates selecting topK experts; the value range is 0 < K ≤ 16, and 0 < K ≤ moeExpertNum must also be satisfied.
    - localExpertNum: indicates the number of experts on the local card.
        - For shared expert cards, localExpertNum = 1
        - For MoE expert cards, localExpertNum = moeExpertNum / (epWorldSize - sharedExpertRankNum). TP-domain communication is not supported in the current version.

- HCCL_BUFFSIZE:
    Before invoking this interface, check whether the value of the HCCL_BUFFSIZE environment variable is reasonable. This environment variable indicates the memory size occupied by a single communicator, in MB; the default value is 200 MB when it is not configured. It must satisfy >= 2 and 1024 ^ 2 *(HCCL_BUFFSIZE - 2) / 2 >= (BS* 2 *(H + 128)* (epWorldSize * localExpertNum + K + 1)), where localExpertNum must use the number of local experts on MoE expert cards.

- Communicator usage constraints:
    - In one model, aclnnMoeDistributeCombineAddRmsNorm and aclnnMoeDistributeDispatchV2 support only the same EP communicator, and no other operators are allowed in that communicator.
    - TP-domain communication is currently not supported.

- Communication mode constraints:
    - <term>Ascend 950DT products</term>: only UB Memory communication is supported.

## Invocation Description

| Invocation Method  | Sample Code                                  | Description                                                     |
| :--------: | :----------------------------------------: | :-------------------------------------------------------: |
| aclnn interface | [test_aclnn_moe_distribute_combine_add_rms_norm.cpp](./examples/test_aclnn_moe_distribute_combine_add_rms_norm.cpp) | Invoke the moe_distribute_combine_add_rms_norm operator through the [aclnnMoeDistributeCombineAddRmsNorm](docs/aclnnMoeDistributeCombineAddRmsNorm_en.md) interface. |
