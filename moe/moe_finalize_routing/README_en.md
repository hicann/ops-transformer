# MoeFinalizeRouting

English | [简体中文](./README.md)

## Product Support Status

|Product      | Supported |
|:----------------------------|:-----------:|
|<term>Ascend 950PR&950DT products</term>|      ×     |
|<term>Atlas A3 products</term>|      √     |
|<term>Atlas A2 products</term>|      √     |
|<term>Atlas 200I/500 A2 inference products</term>|      ×     |
|<term>Atlas inference products</term>|      ×     |
|<term>Atlas training products</term>|      ×     |
|<term>Kirin X90 processor products</term> | √ |
|<term>Kirin 9030 processor products</term> | √ |

## Function Description

- **Operator function**: In MoE computation, the output of the MoE FFN is finally processed and merged.
- **Calculation Formula**:

  $$
  expertid=expandedExpertIdx[i,k]
  $$

  $$
  out(i,j)=x1_{i,j}+x2Optional_{i,j}+\sum_{k=0}^{K-1}(scales_{i,k}*(expandedX_{expandedRowIdx_{i+k*num_rows},j}+bias_{expertid,j}))
  $$

## Parameter Description

  <table style="table-layout: auto; width: 100%">
    <thead>
      <tr>
        <th style="white-space: nowrap">Parameter Name</th>
        <th style="white-space: nowrap">Input/Output/Attribute</th>
        <th style="white-space: nowrap">Description</th>
        <th style="white-space: nowrap">Data Type</th>
        <th style="white-space: nowrap">Data Format</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>expandedX</td>
        <td>Input</td>
        <td>`expandedX` in the formula, that is, the FFN output of MoE.</td>
        <td>FLOAT16, BFLOAT16, FLOAT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>x1</td>
        <td>Input</td>
        <td>`x1` in the formula.</td>
        <td>FLOAT16, BFLOAT16, FLOAT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>x2Optional</td>
        <td>Input</td>
        <td>`x2Optional` in the formula.</td>
        <td>FLOAT16, BFLOAT16, FLOAT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>bias</td>
        <td>Input</td>
        <td>`bias` in the formula.</td>
        <td>FLOAT16, BFLOAT16, FLOAT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>scales</td>
        <td>Input</td>
        <td>`scales` in the formula.</td>
        <td>FLOAT16, BFLOAT16, FLOAT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>expandedRowIdx</td>
        <td>Input</td>
        <td>`expandedRowIdx` in the formula.</td>
        <td>INT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>expandedExpertIdx</td>
        <td>Input</td>
        <td>`expandedExpertIdx` in the formula.</td>
        <td>INT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>out</td>
        <td>Output</td>
        <td>`out` in the formula.</td>
        <td>FLOAT16, BFLOAT16, FLOAT32</td>
        <td>ND</td>
      </tr>
    </tbody>
  </table>

## Constraints

- The inputs expandedX, x1, x2Optional, bias, scales, and expandedExpertIdx are 2D tensors, and expandedRowIdx is a 1D tensor, where x2Optional is an optional input.
- The size of dimension 0 of x1, x2Optional, scales, and expandedExpertIdx is the same.
- The size of dimension 1 of x1, x2Optional, expandedX, and bias is the same.
- The size of dimension 1 of scales is the same as that of expandedExpertIdx, which is K in the formula.
- The size of dimension 0 of expandedX is the same as that of expandedRowIdx.
- The data types of expandedRowIdx and expandedExpertIdx are INT32, and the data types of the remaining inputs and the output out are FLOAT16, BFLOAT16, or FLOAT32.
- <term>Kirin X90 processor products</term>, <term>Kirin 9030 processor products</term>: BFLOAT16 is not supported.

## Calling Instructions

| Calling Method   | Sample Code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn API  | [test_aclnn_moe_finalize_routing.cpp](examples/test_aclnn_moe_finalize_routing.cpp) | Calls the MoeFinalizeRouting operator through the [aclnnMoeFinalizeRouting](docs/aclnnMoeFinalizeRouting_en.md) API. |
