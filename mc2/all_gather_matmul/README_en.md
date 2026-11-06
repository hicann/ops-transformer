# AllGatherMatmul

English | [简体中文](./README.md)

## Product Support Status

| Product | Supported |
| ---- | :----: |
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term>   | √ |
| <term>Atlas A2 products</term> | √ |

## Functional Description

- Operator function: Fuses AllGather communication with MatMul computation.
- Computation formula:

    $$
    y=AllGather(x1)@x2+bias
    $$

    $$
    gatherOut=AllGather(x1)
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
      <th>Parameter</th>
      <th>Input/Output/Attribute</th>
      <th>Description</th>
      <th>Data Type</th>
      <th>Data Format</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>x1</td>
      <td>Input</td>
      <td>Input x1 in the formula.</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>Input</td>
      <td>Input x2 in the formula.</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>bias</td>
      <td>Optional input</td>
      <td>Input bias in the formula.</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Output y in the formula.</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>gather_out</td>
      <td>Output</td>
      <td>Output gatherOut in the formula.</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>group</td>
      <td>Attribute</td>
      <td><li>String on the Host side that identifies the communicator, that is, the communicator name.</li><li>Obtained through the Hccl API "extern HcclResult HcclGetCommName(HcclComm comm, char* commName);", where commName is the group.</li></td>
      <td>CHAR*, STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>is_trans_a</td>
      <td>Optional attribute</td>
      <td><li>Determines whether x1 is transposed before matrix multiplication.</li><li>The default value is false.</li></td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>is_trans_b</td>
      <td>Optional attribute</td>
      <td><li>Determines whether x2 is transposed before matrix multiplication.</li><li>The default value is false.</li></td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>gather_index</td>
      <td>Optional attribute</td>
      <td><li>Identifies the gather target. The current version supports only input 0, indicating that the target is x1.</li><li>The default value is 0.</li></td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>commTurn</td>
      <td>Optional attribute</td>
      <td><li>Number of communication data splits, that is, the total data volume divided by the volume of each communication.</li><li>The default value is 0.</li></td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>rank_size</td>
      <td>Optional attribute</td>
      <td><li>Number of cards in the communicator.</li><li>The default value is 0.</li></td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>is_gather_out</td>
      <td>Optional attribute</td>
      <td><li>If true, gather_out is output.</li><li>The default value is true.</li></td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
  </tbody></table>

## Constraints

* In the current version, input x1 is 2D with shape (m, k). x2 must be 2D with shape (k, n). The axes must satisfy the input requirements of the MM operator, the k axes must be equal, and the value range of the k axis is [256, 65535).
* For the empty tensor scenarios supported by x1/x2, m and n can be empty, k cannot be empty, and the following conditions must be satisfied:
    * m is empty, k is not empty, and n is not empty;
    * m is not empty, k is not empty, and n is empty;
    * m is empty, k is not empty, and n is empty.
* The data types of the x1 computation input, x2 computation input, output computation output, and gather_out computation output must be consistent.
* The x2 matrix supports both the transposed and non-transposed scenarios, the x2 matrix supports non-contiguous tensors constructed through transposition, and the x1 matrix supports only the non-transposed scenario.
* bias is optional and can be empty. When it is not empty, the current version supports only 1D input, and the scenario where the bias input is non-0 is not yet supported.
* The output is 2D with shape (m*rank_size, n), where rank_size is the number of cards.
* gather_index supports only input 0 in the current version.
* commTurn supports only input 0 in the current version.
* <term>Atlas A2 products</term>:
    * Supports 2, 4, and 8 cards, and supports only the all mesh networking over HCCS links.
    * The communication-computation fusion MC2 operators in one model support only the same communicator.
* <term>Atlas A3 products</term>:
    * Supports 2, 4, 8, 16, and 32 cards, and supports only the double ring networking over HCCS links.

## Invocation Description

| Invocation Method   | Sample Code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn API  | [test_aclnn_all_gather_matmul](./examples/test_aclnn_all_gather_matmul.cpp) | Calls the AllGatherMatmul operator through the [aclnnAllGatherMatmul](docs/aclnnAllGatherMatmul_en.md) API. |
