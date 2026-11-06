# DistributeBarrierExtend

English | [简体中文](./README.md)

## Product Support

| Product                                                         | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                             |    √     |
| <term>Atlas A3 products</term>       |    ×     |
| <term>Atlas A2 products</term> |    ×     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                               |    ×     |
| <term>Atlas training products</term>                              |    ×     |

## Description

 - Operator function: Synchronizes all cards in a communicator. xRef is used only to build a Tensor dependency, and the API performs no operation on xRef.

    Compared with the DistributeBarrier operator, this operator has the following changes:
    - The `context` input parameter is added to store communicator information.
    For details, see the parameter description below.

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
   <th>Input/Output</th>
   <th>Description</th>
   <th>Data Type</th>
   <th>Data Format</th>
  </tr></thead>
 <tbody>
  <tr>
   <td>context</td>
   <td>Input</td>
   <td>Communicator information data of the current card.</td>
   <td>INT32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>xRef</td>
   <td>Input</td>
   <td>Has no service semantics. It is used only to build a Tensor dependency, and the API performs no operation on it.</td>
   <td>BFLOAT16, FLOAT16, FLOAT32, BOOL, INT8, INT16, INT32, INT64, UINT8, UINT16, UINT32, UINT64, FLOAT8_E5M2, FLOAT8_E4M3FN, FLOAT4_E1M2, FLOAT4_E2M1, HIFLOAT8, INT4</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>timeOut</td>
   <td>Input</td>
   <td>Timeout setting. If there is no response within this period, the operation is considered to time out.</td>
   <td>INT32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>elasticInfo</td>
   <td>Input</td>
   <td>Dynamic scale-in information of the EP communicator: <br><term>Atlas A2 products</term>: Not supported. Pass a null pointer. <br><term>Atlas A3 products</term>: 1D Tensor (shape <code>4 + 2 * epWorldSize,</code>) of the INT32 type. The first 4 values are scale-in configurations, and the last 2*epWorldSize values are the rank mapping table.</td>
   <td>INT32</td>
   <td>ND (non-contiguous Tensors supported)</td>
  </tr>
  <tr>
   <td>group</td>
   <td>Input</td>
   <td>Communicator name, which specifies the communicator used to synchronize all cards.</td>
   <td>STRING</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>worldSize</td>
   <td>Input</td>
   <td>Communicator size.</td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
 </tbody></table>

  - <term>Ascend 950PR&950DT products</term>: The timeout unit of the timeOut parameter is us. A value of 5000000us is recommended. The minimum timeout may vary depending on the actual environment.

## Constraints

- Communicator usage constraints:
    - aclnnDistributeBarrier in a model must use a dedicated communicator, and no other operators are allowed in this communicator.

- Usage scenario description:
    - Call this operator in network models that require synchronization of all cards. It can shield performance fluctuations caused by fast and slow cards and helps analyze performance.
    - The operator supports consecutive calls. When building a graph, pass the input of the previous operator and the output of the next operator to the API as input parameters.
    - When elasticInfo is enabled, ensure that Dispatch/Combine also receives this parameter, and that this elasticInfo is consistent with the elasticInfo in Dispatch/Combine.

## Calling Instructions

| Calling Method  | Sample Code                                  | Description                                                     |
| :--------: | :----------------------------------------: | :-------------------------------------------------------: |
| aclnn API | [test_aclnn_distribute_barrier.cpp](../distribute_barrier/examples/test_aclnn_distribute_barrier.cpp) | Calls the distribute_barrier_extend operator through the [aclnnDistributeBarrier](../distribute_barrier/docs/aclnnDistributeBarrier_en.md) API. |
