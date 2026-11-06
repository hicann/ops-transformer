# GroupedMatMulAllReduce

English | [简体中文](./README.md)

> Note:
> This document is only a brief introduction to the operator function. Direct user calling is not supported, because direct kernel invocation is currently not supported. The document will be completed when such support becomes available!!!!!!

**This API will be deprecated in a later version. Do not use this API.**

## Product Support

| Product                                                         | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                             |    ×     |
| <term>Atlas A3 products</term>       |    ×     |
| <term>Atlas A2 products</term> |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                               |    ×     |
| <term>Atlas training products</term>                              |    ×     |

## Description

Operator function: Implements multi-card parallel AllReduce on top of grouped_matmul, performing grouped matrix multiplication where each group can have different matrix multiplication dimension sizes. Based on the number of Tensors in x, weight, and y, the following four scenarios are supported:

- The numbers of Tensors in x, weight, and y equal the number of groups, that is, the Tensors corresponding to the data of each group are independent.
- The number of Tensors in x is 1, and the numbers of Tensors in weight/y equal the number of groups. In this case, the optional parameter group_list describes how x is grouped along rows. For example, group_list[0]=10 indicates that the first 10 rows of x participate in the first group matrix multiplication calculation.
- The numbers of Tensors in x and weight equal the number of groups, and the number of Tensors in y is 1. In this case, the results of the group matrix multiplications are stored consecutively in the same Tensor.
- The numbers of Tensors in x and y are 1, and the number of Tensors in weight equals the number of groups. This is a combination of the first two scenarios.

Calculation formula:
Non-quantization scenario:

$$
y_i=x_i\times weight_i + bias_i
$$

## Parameter Description

<table style="undefined;table-layout: fixed; width: 1576px"> <colgroup>
 <col style="width: 170px">
 <col style="width: 170px">
 <col style="width: 800px">
 <col style="width: 800px">
 <col style="width: 200px">
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
   <td>Required parameter. aclTensorList on the Device, corresponding to input x in the formula. The maximum supported length is 64.</td>
   <td>FLOAT16, BFLOAT16 (data types of Tensors in the list)</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>weight</td>
   <td>Input</td>
   <td>Required parameter. aclTensorList on the Device, corresponding to weight in the formula. The maximum supported length is 64.</td>
   <td>FLOAT16, BFLOAT16 (data types of Tensors in the list)</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>bias</td>
   <td>Input</td>
   <td>Optional parameter. aclTensorList on the Device, corresponding to bias in the formula. The maximum supported length is 64.</td>
   <td>FLOAT16, FLOAT32 (data types of Tensors in the list)</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>groupListOptional</td>
   <td>Input</td>
   <td>Optional parameter. aclIntArray on the Host, representing the distribution of matmul sizes along the M dimension of inputs and outputs. The maximum supported length is 64.</td>
   <td>INT64 (array element type)</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>splitItem</td>
   <td>Input</td>
   <td>Optional attribute, indicating whether to split tensors for inputs and outputs: 0 (default) = neither inputs nor outputs are split; 1 = inputs are split and outputs are not split; 2 = inputs are not split and outputs are split; 3 = both inputs and outputs are split.</td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>group</td>
   <td>Input</td>
   <td>String on the Host that identifies the column group, that is, the communicator name. Obtain commName through the Hccl API HcclGetCommName and pass it as this parameter.</td>
   <td>STRING</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>reduceOp</td>
   <td>Input</td>
   <td>Reduce operation type. The current version supports only the input "sum".</td>
   <td>STRING</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>commTurn</td>
   <td>Input</td>
   <td>Integer on the Host, the number of communication data splits (total data volume/communication volume per time). The current version supports only the input 0.</td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>streamMode</td>
   <td>Input</td>
   <td>Integer on the Host, the enumeration of acl stream modes. Currently, only the value 1 is supported.</td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>y</td>
   <td>Output</td>
   <td>aclTensorList on the Device, corresponding to output y in the formula. The maximum supported length is 64.</td>
   <td>FLOAT16, BFLOAT16 (data types of Tensors in the list)</td>
   <td>ND</td>
  </tr>
 </tbody></table>

## Constraints

- Data type combination constraints: The supported data type combinations of x, weight, and bias are:
  - "x-FLOAT16, weight-FLOAT16, bias-FLOAT16"
  - "x-BFLOAT16, weight-BFLOAT16, bias-FLOAT32"
- Dimension constraints:
  - When splitItem is 0, x supports 2 to 6 dimensions, and y supports 2 to 6 dimensions.
  - When splitItem is 1/2/3, x supports 2 dimensions, and y supports 2 dimensions.
  - Regardless of the value of splitItem, weight supports 2 dimensions.
- Device quantity constraint: Deployment on 2, 4, or 8 cards is supported.
- Tensor dimension size constraints:
  - The size of the last dimension of each group of tensors in x and weight must be smaller than 65536 (for the last dimension of x: the K axis when transpose_x is false, and the M axis when transpose_x is true; for the last dimension of weight: the N axis when transpose_weight is false, and the K axis when transpose_weight is true).
  - After 32-byte alignment, the size of each dimension of each group of tensors in x and weight must be smaller than the maximum value of int32 (2147483647).

## Calling Instructions

| Calling Method  | Sample Code                                  | Description                                                     |
| :--------: | :----------------------------------------: | :-------------------------------------------------------: |
| aclnn API | [test_grouped_mat_mul_all_reduce.cpp](tests/ut/op_kernel/test_grouped_mat_mul_all_reduce.cpp) | Calls the grouped_mat_mul_all_reduce operator through the [aclnnGroupedMatMulAllReduce](docs/aclnnGroupedMatMulAllReduce_en.md) API. |
