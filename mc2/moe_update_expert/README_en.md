# MoeUpdateExpert

English | [简体中文](./README.md)

## Product Support Status

| Product                                                      |  Supported   |
| :----------------------------------------------------------- |:-------:|
| <term>Ascend 950PR&950DT products</term>                     |    √    |
| <term>Atlas A3 products</term>                               |    √    |
| <term>Atlas A2 products</term>                               |    ×    |
| <term>Atlas 200I/500 A2 inference products</term>            |    ×    |
| <term>Atlas inference products</term>                        |    ×    |
| <term>Atlas training products</term>                         |    ×    |

## Description

**Operator function**:

* To solve load imbalance scenarios, this operator maps the topK logical expert IDs of each token to physical card IDs.
* Supports pruning the topK experts to which a token is sent based on a threshold.

## Parameters

<table style="undefined;table-layout: fixed; width: 1392px"> <colgroup>
 <col style="width: 120px">
 <col style="width: 120px">
 <col style="width: 160px">
 <col style="width: 150px">
 <col style="width: 80px">
 </colgroup>
 <thead>
  <tr>
   <th>Parameter</th>
   <th>Input/Output/Attribute</th>
   <th>Description</th>
   <th>Data Type</th>
   <th>Data Format</th>
  </tr>
 </thead>
 <tbody>
  <tr>
   <td>expertIds</td>
   <td>Input</td>
   <td>TopK expert indexes of each token. An aclTensor on the device side, which must be a 2D tensor with the shape (BS, K); non-contiguous tensors are supported.</td>
   <td>INT32, INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>eplbTable</td>
   <td>Input</td>
   <td>Mapping table from logical experts to physical experts. The external caller must ensure that the values of the input tensor are correct. There are worldSize cards, and each card deploys place_per_rank routing expert instances, for a total of worldSize*place_per_rank instances. The first column of each row is the number of instances (count) deployed for the logical expert corresponding to the row number (value range: [1, worldSize]), and the [1, count] columns of each row are the corresponding instance IDs (value range: [0, worldSize*place_per_rank); valid instance IDs must not be duplicated). An aclTensor on the device side, which must be a 2D tensor with the shape (moeExpertNum, F); non-contiguous tensors are supported.</td>
   <td>INT32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>expertScalesOptional</td>
   <td>Input</td>
   <td>Scale weights of the topK experts of each token. You must ensure that the scales are arranged in descending order within a token. Valid data or a null pointer can be passed. When valid data is passed, valid data must also be passed for pruningThresholdOptional. An aclTensor on the device side, which must be a 2D tensor with the shape (BS, K); non-contiguous tensors are supported.</td>
   <td>FLOAT16, BFLOAT16, FLOAT</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>pruningThresholdOptional</td>
   <td>Input</td>
   <td>Minimum threshold of the expert scale weights. When the scale of a topK expert corresponding to a token is less than the threshold, the token is pruned for this expert (not sent to this expert for processing). Valid data or a null pointer can be passed. When valid data is passed, valid data must also be passed for expertScalesOptional. An aclTensor on the device side, which must be a 1D or 2D tensor with the shape (K,) or (1, K); non-contiguous tensors are supported.</td>
   <td>FLOAT</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>activeMaskOptional</td>
   <td>Input</td>
   <td>Indicates whether a token participates in communication. Valid data or a null pointer can be passed. When valid data is passed, valid data must also be passed for expertScalesOptional and pruningThresholdOptional. true indicates that the token participates in communication, and true values must be arranged before false values (for example, {true, false, true} is invalid). Passing a null pointer indicates that all tokens participate in communication. An aclTensor on the device side, which must be a 1D tensor with the shape (BS,); non-contiguous tensors are supported.</td>
   <td>BOOL</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>localRankId</td>
   <td>Input</td>
   <td>ID of the current card. When balanceMode is 0, the value range is [0, worldSize). The localRankId values of the cards in the same communicator must be unique.</td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>worldSize</td>
   <td>Input</td>
   <td>Size of the communicator. When balanceMode is 0, the value range is [2, 768].</td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>balanceMode</td>
   <td>Input</td>
   <td>Balancing rule. 0 indicates distribution by rank, and 1 indicates distribution by token. The default value is 0; value range: [0, 1].</td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>balancedExpertIds</td>
   <td>Output</td>
   <td>Instance IDs of the physical experts of the topK experts of each token after mapping. An aclTensor on the device side, which must be a 2D tensor with the shape (BS, K); the data type and data format are the same as those of expertIds.</td>
   <td>INT32, INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>balancedActiveMask</td>
   <td>Output</td>
   <td>Balanced activeMask after pruning. This output is valid when valid data is passed for expertScalesOptional and pruningThresholdOptional. An aclTensor on the device side, which must be a 2D tensor with the shape (BS, K); non-contiguous tensors are supported.</td>
   <td>BOOL</td>
   <td>ND</td>
  </tr>
 </tbody>
</table>

## Constraints

- The aclnnMoeUpdateExpert API must be used together with the aclnnMoeDistributeDispatchV2 API and the aclnnMoeDistributeCombineV2 or aclnnMoeDistributeCombineAddRmsNorm API. The calling sequence is aclnnMoeUpdateExpert, aclnnMoeDistributeDispatchV2, and then aclnnMoeDistributeCombineV2 or aclnnMoeDistributeCombineAddRmsNorm. For details, see the calling instructions.

- The values of the worldSize and moeExpertNum parameters used during API calling must be the same on all cards, the same in different layers of the network, and consistent with the corresponding parameters of aclnnMoeDistributeDispatchV2, aclnnMoeDistributeCombineV2, or aclnnMoeDistributeCombineAddRmsNorm.

- <term>Atlas A3 products</term>: In this scenario, a single card contains dual dies (referred to as "grains" or "bare dies" for short). Therefore, "the current card" in the parameter descriptions refers to a single die.

- Shape format descriptions in the parameter descriptions:
    - BS: Indicates the batch sequence size, that is, the number of tokens finally output by the current card. Value range: 0 < BS ≤ 512.
    - K: Indicates the selection of topK experts. The value range is 0 < K ≤ 16, and 0 < K ≤ moeExpertNum must also be satisfied.
    - moeExpertNum: Indicates the number of MoE experts. Value range: (0, 1024].
    - F: Indicates the number of columns of the mapping table. Value range: [2, worldSize + 1]. The first column is the number of instances deployed for the MoE expert corresponding to each row number (value > 0), and the last F-1 columns are the IDs of the physical cards to which the MoE expert is deployed.
    - The total number of MoE expert instances deployed on all cards is at most 1024, that is, place_per_rank * worldSize ≤ 1024.
    - The number of instances deployed on each card must be the same.

## Calling Instructions

| Calling Method           | Sample Code                                                      | Description                                                                                             |
| :--------: | :----------------------------------------: | :-------------------------------------------------------: |
| aclnn API | [test_aclnn_moe_update_expert.cpp](./examples/test_aclnn_moe_update_expert.cpp) | Calls the moe_update_expert operator in the form of the [aclnnMoeUpdateExpert](docs/aclnnMoeUpdateExpert_en.md) API. |
