# BatchMatMulReduceScatterAlltoAll

English | [简体中文](./README.md)

## Product Support

| Product                                                         | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                             |    ×     |
| <term>Atlas A3 products</term>       |    √     |
| <term>Atlas A2 products</term> |    ×     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                               |    ×     |
| <term>Atlas training products</term>                              |    ×     |

## Description

Operator function: BatchMatMulReduceScatterAllToAll is a communication-computation fusion operator that implements BatchMatMul computation in parallel with ReduceScatter and AllToAll collective communication.

Computation formula: The general computation process is as follows: BatchMatMul computation --> transposition (required when yShardType is 0) --> ReduceScatter collective communication --> Add --> AllToAll collective communication. The computation logic is as follows, where y is the output:

$$
temp1 = BatchMatMul(x, weight)
$$

$$
temp2 = ReduceScatter(temp1)
$$

$$
temp3 = Add(temp2, bias)
$$

$$
y = AllToAll(temp3)
$$

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
   <td>x</td>
   <td>Input</td>
   <td>Left matrix of the BatchMatMul computation. The tensor must be 3-dimensional.</td>
   <td>FLOAT16, BFLOAT16</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>weight</td>
   <td>Input</td>
   <td>Right matrix of the BatchMatMul computation. The data type must be the same as x, and the tensor must be 3-dimensional.</td>
   <td>FLOAT16, BFLOAT16</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>biasOptional</td>
   <td>Input</td>
   <td>bias for the Add computation. The Add operation is performed after ReduceScatter communication. When x is FLOAT16, biasOptional must be FLOAT16. When x is BFLOAT16, biasOptional must be FLOAT32. Two or three dimensions are supported, and a null pointer can be passed.</td>
   <td>FLOAT16, FLOAT32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>groupEp</td>
   <td>Input</td>
   <td>Communicator name for expert parallelism. The string length must be greater than 0 and less than 128.</td>
   <td>STRING</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>groupTp</td>
   <td>Input</td>
   <td>Communicator name for tensor parallelism. The string length must be greater than 0 and less than 128.</td>
   <td>STRING</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>epWorldSize</td>
   <td>Input</td>
   <td>Size of the EP communicator. Supported values: 2, 4, 8, 16, and 32.</td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>tpWorldSize</td>
   <td>Input</td>
   <td>Size of the TP communicator. Supported values: 2, 4, 8, 16, and 32.</td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>yShardType</td>
   <td>Input</td>
   <td>Integer. 0 indicates that ReduceScatter is performed by tp on the H dimension (the second dimension of the BatchMatMul computation result; the result has three dimensions in total, with dimension indexes 0, 1, and 2 in sequence). 1 indicates that ReduceScatter is performed by tp on the C dimension (the first dimension of the BatchMatMul computation result).</td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>out</td>
   <td>Output</td>
   <td>aclTensor on the Device. The tensor is the result of the batch_matmul computation + reduce_scatter computation + all_to_all communication. The data type must be the same as input x, and the tensor must be 3-dimensional.</td>
   <td>FLOAT16, BFLOAT16</td>
   <td>ND</td>
  </tr>
 </tbody></table>

## Constraints

Due to the requirements of collective communication and the BatchMatMul computation, the input and output shapes must satisfy the following mathematical relationships (where ep = epWorldSize and tp = tpWorldSize):

- Scenario where ReduceScatter is performed on the H axis, that is, yShardType is 0:
  - x: (E/ep, ep*C, M/tp)
  - weight: (E/ep, M/tp, H)
  - biasOptional: When the pointer is not null, the shape is (E/ep, 1, H/tp) for three dimensions and (E/ep, H/tp) for two dimensions.
  - y: (E, C, H/tp)

- Scenario where ReduceScatter is performed on the C axis, that is, yShardType is 1:
  - x: (E/ep, ep*tp*C/tp, M/tp)
  - weight: (E/ep, M/tp, H)
  - biasOptional: When the pointer is not null, the shape is (E/ep, 1, H) for three dimensions and (E/ep, H) for two dimensions.
  - y: (E, C/tp, H)

- Data relationship description:
  - For example, if x.size(0) equals E/tp and y.size(0) equals E, then y.size(0) = ep*x.size(0), that is, y.size(0) is an integer multiple of ep. Other relationships are similar.
  - The value range of E is [2, 512], and E is an integer multiple of ep.
  - The value range of H is [1, 65535]. When yShardType is 0, H is an integer multiple of tp.
  - The value range of M/tp is [1, 65535].
  - The value range of E/ep is [1, 32].
  - Both ep and tp support only 2, 4, 8, 16, and 32.
  - The names of groupEp and groupTp cannot be the same.
  - C is greater than 0, and the upper limit is the device memory limit of the operator. When yShardType is 1, C is an integer multiple of tp.
  - Communication-computation fusion operators do not support concurrent calls, and different communication-computation fusion operators do not support concurrent calls either.
  - Cross-supernode communication is not supported; only communication within a supernode is supported.

## Calling Instructions

| Calling Method  | Sample Code                                  | Description                                                     |
| :--------: | :----------------------------------------: | :-------------------------------------------------------: |
| aclnn API | [test_aclnn_batch_mat_mul_reduce_scatter_allto_all.cpp](./examples/test_aclnn_batch_mat_mul_reduce_scatter_allto_all.cpp) | Call the batch_mat_mul_reduce_scatter_allto_all operator through the [aclnnBatchMatMulReduceScatterAlltoAll](docs/aclnnBatchMatMulReduceScatterAlltoAll_en.md) API. |
