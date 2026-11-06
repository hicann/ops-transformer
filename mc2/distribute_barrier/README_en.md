# DistributeBarrier

English | [简体中文](./README.md)

## Product Support

| Product                                                         | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950DT products</term>                             |    √     |
| <term>Atlas A3 products</term>       |    √     |
| <term>Atlas A2 products</term> |    ×     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                               |    ×     |
| <term>Atlas training products</term>                              |    ×     |

## Description

Operator function: Performs synchronization across all cards within the communicator. xRef is used only to build tensor dependencies; the API does not perform any operation on xRef.

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
   <td>xRef</td>
   <td>Input</td>
   <td>No business semantics. Used only as the input tensor dependency; the API does not perform any operation on it.</td>
   <td>BFLOAT16, FLOAT16, FLOAT32, BOOL, INT8, INT16, INT32, INT64, UINT8, UINT16, UINT32, UINT64, FLOAT8_E5M2, FLOAT8_E4M3FN, FLOAT4_E1M2, FLOAT4_E2M1, HIFLOAT8, INT4</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>timeOut</td>
   <td>Input</td>
   <td>Timeout setting. If there is no response within this time, a timeout occurs.</td>
   <td>INT32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>elasticInfo</td>
   <td>Input</td>
   <td>Dynamic scale-in information of the EP communicator:<br><term>Atlas A2 products</term>: not supported; pass a null pointer;<br><term>Atlas A3 products</term>: a 1D tensor (shape <code>4 + 2 * epWorldSize,</code>) of the INT32 type. The first 4 elements are the scale-in configuration, and the last 2*epWorldSize elements are the rank mapping table.</td>
   <td>INT32</td>
   <td>ND (non-contiguous tensors supported)</td>
  </tr>
  <tr>
   <td>group</td>
   <td>Input</td>
   <td>Communicator name. It is the communicator in which synchronization across all cards is performed.</td>
   <td>STRING</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>worldSize</td>
   <td>Input</td>
   <td>Size of the communicator.</td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
 </tbody></table>

  - <term>Atlas A3 products</term>: The FLOAT8_E5M2, FLOAT8_E4M3FN, FLOAT4_E1M2, FLOAT4_E2M1, HIFLOAT8, and INT4 types are not supported.
  - <term>Ascend 950DT products</term>: The unit of the timeout in the timeOut parameter is us. A value of 5000000us is recommended. The lower limit of the timeout may differ depending on the actual environment.

## Constraints

- Communicator usage constraint:
    - aclnnDistributeBarrier in a model needs to use a dedicated communicator, and other operators are not allowed in this communicator.

- Communication method constraint:
    - <term>Ascend 950DT products</term>: Only UB Memory communication is supported.

- Usage scenario description:
    - Call this operator in a network model that requires synchronization across all cards. It can shield performance fluctuations caused by fast and slow cards and help analyze performance.
    - The operator can be called consecutively. When the operator is added to a graph, the input of the previous operator and the output of the next operator must be passed to the API as input parameters.
    - When elasticInfo is enabled, ensure that Dispatch/Combine also passes in this parameter and that this elasticInfo is consistent with the elasticInfo in Dispatch/Combine.

## Calling Instructions

| Calling Method  | Sample Code                                  | Description                                                     |
| :--------: | :----------------------------------------: | :-------------------------------------------------------: |
| aclnn API | [test_aclnn_distribute_barrier.cpp](./examples/test_aclnn_distribute_barrier.cpp) | Call the distribute_barrier operator through the [aclnnDistributeBarrier](docs/aclnnDistributeBarrier_en.md) API. |
