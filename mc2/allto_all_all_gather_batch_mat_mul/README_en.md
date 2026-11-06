# AlltoAllAllGatherBatchMatMul

English | [简体中文](./README.md)

## Product Support Status

| Product | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                             |    ×     |
| <term>Atlas A3 products</term>       |    √     |
| <term>Atlas A2 products</term> |    ×     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                             |    ×     |
| <term>Atlas training products</term>                              |    ×     |

## Function Description

Operator function: fuses and parallelizes the AllToAll and AllGather collective communications with the BatchMatMul computation.

Computation formula:
The computation logic is as follows, where y1, y2, and y3 are outputs:

$$
x1 = AllToAll(x)
$$

$$
y2 = AllGather(x1)
$$

$$
y3 = BatchMatMul(y2, weight, bias)
$$

$$
y1 = ActivationFunction(y3)
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
   <td>The result after communication serves as the left matrix of the BatchMatMul computation. This input performs the AllToAll and AllGather collective communications and must be three-dimensional.</td>
   <td>FLOAT16, BFLOAT16</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>weight</td>
   <td>Input</td>
   <td>Right matrix of the BatchMatMul computation. The data type must be the same as that of x, and the tensor must be three-dimensional.</td>
   <td>FLOAT16, BFLOAT16</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>biasOptional</td>
   <td>Input</td>
   <td>bias of the BatchMatMul computation. When x is FLOAT16, biasOptional must be FLOAT16; when x is BFLOAT16, biasOptional must be FLOAT32. Supports two or three dimensions. Supports passing a null pointer.</td>
   <td>FLOAT16, FLOAT32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>groupEp</td>
   <td>Input</td>
   <td>ep communicator name, the communicator for expert parallelism. The string length must be greater than 0 and less than 128.</td>
   <td>STRING</td>
   <td>-</td>
  </tr>
  <tr>
   <td>groupTp</td>
   <td>Input</td>
   <td>tp communicator name, the communicator for tensor parallelism. The string length must be greater than 0 and less than 128.</td>
   <td>STRING</td>
   <td>-</td>
  </tr>
  <tr>
   <td>epWorldSize</td>
   <td>Input</td>
   <td>Size of the ep communicator. Supports 2/4/8/16/32.</td>
   <td>INT64</td>
   <td>-</td>
  </tr>
  <tr>
   <td>tpWorldSize</td>
   <td>Input</td>
   <td>Size of the tp communicator. Supports 2/4/8/16/32.</td>
   <td>INT64</td>
   <td>-</td>
  </tr>
  <tr>
   <td>xShardType</td>
   <td>Input</td>
   <td>0 indicates that allgather is performed along the tp domain on the H dimension (that is, the second dimension of x; x is three-dimensional, with dimensions 0, 1, and 2), and 1 indicates that allgather is performed along the tp domain on the C dimension (that is, the first dimension of x).</td>
   <td>INT64</td>
   <td>-</td>
  </tr>
  <tr>
   <td>actType</td>
   <td>Input</td>
   <td>Activation function type. Supports inputs of 0/1/2/3/4, where 0 indicates no activation function. The mapping is [0: None, 1: GELU, 2: Silu, 3: Relu, 4: FastGELU].</td>
   <td>INT64</td>
   <td>-</td>
  </tr>
  <tr>
   <td>y1Out</td>
   <td>Output</td>
   <td>Final computation result. If an activation function exists, this is the output of the activation function; otherwise, this is the output of BatchMatMul. Supports three dimensions, and the data type is the same as that of the input x.</td>
   <td>FLOAT16, BFLOAT16</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>y2OutOptional</td>
   <td>Output</td>
   <td>Optional output, the output of AllGather, which may be required for the backward pass. Supports three dimensions, and the data type is the same as that of the input x. A null pointer indicates that this output is not required.</td>
   <td>FLOAT16, BFLOAT16</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>y3OutOptional</td>
   <td>Output</td>
   <td>Optional output, the output of BatchMatMul when an activation function exists. Supports three dimensions, and the data type is the same as that of the input x. A null pointer indicates that this output is not required.</td>
   <td>FLOAT16, BFLOAT16</td>
   <td>ND</td>
  </tr>
 </tbody></table>

## Constraints

Due to the collective communication and the BatchMatMul computation, the input and output shapes must satisfy the following mathematical relationships: (where ep=epWorldSize and tp=tpWorldSize)

AllGather along the H axis, that is, the scenario where xShardType is 0:

  - x: (E, C, H/tp)
  - weight: (E/ep, H, M/tp)
  - biasOptional: if not a null pointer, (E/ep, 1, M/tp) for three dimensions and (E/ep, M/tp) for two dimensions
  - y1Out: (E/ep, ep*C, M/tp)
  - y2OutOptional: (E/ep, ep*C, H)
  - y3OutOptional: (E/ep, ep*C, M/tp)

AllGather along the C axis, that is, the scenario where xShardType is 1:

  - x: (E, C/tp, H)
  - weight: (E/ep, H, M/tp)
  - biasOptional: if not a null pointer, (E/ep, 1, M/tp) for three dimensions and (E/ep, M/tp) for two dimensions
  - y1Out: (E/ep, ep*tp\*C/tp, M/tp)
  - y2OutOptional: (E/ep, ep*tp\*C/tp, H)
  - y3OutOptional: (E/ep, ep*tp\*C/tp, M/tp)

Data relationship description:

  - For example, if x.size(0) equals E and weight.size(0) equals E/ep, this means x.size(0) = ep*weight.size(0), and x.size(0) is an integer multiple of ep; other relationships are similar.
  - The value range of E is [2, 512], and E must be an integer multiple of ep.
  - The value range of H is [1, 65535]. When xShardType is 0, H must be an integer multiple of tp.
  - The value range of M/tp is [1, 65535].
  - The value range of E/ep is [1, 32].
  - Both ep and tp support only 2, 4, 8, 16, and 32.
  - The groupEp and groupTp names must not be the same.
  - C must be greater than 0, with an upper limit of the operator device memory limit. When xShardType is 1, C must be an integer multiple of tp.
  - Communication-computation fused operators do not support concurrent calls, and different communication-computation fused operators do not support concurrent calls either.
  - Cross-supernode is not supported; only intra-supernode is supported.

## Calling Instructions

| Calling Method  | Sample Code                                  | Description                                                     |
| :--------: | :----------------------------------------: | :-------------------------------------------------------: |
| aclnn interface | [test_aclnn_allto_all_all_gather_batch_mat_mul.cpp](./examples/test_aclnn_allto_all_all_gather_batch_mat_mul.cpp) | Calls the allto_all_all_gather_batch_mat_mul operator through the [aclnnAlltoAllAllGatherBatchMatMul](docs/aclnnAlltoAllAllGatherBatchMatMul_en.md) interface. |
