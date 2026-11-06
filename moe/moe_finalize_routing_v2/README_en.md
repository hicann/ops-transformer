# MoeFinalizeRoutingV2

English | [简体中文](./README.md)

## Product Support Status

| Product                                                      | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                      |     √    |
| <term>Atlas A3 products</term>      |    √    |
| <term>Atlas A2 products</term>      |    √    |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                             |    √     |
| <term>Atlas training products</term>                              |    ×     |
| <term>Kirin X90 processor products</term> | √ |
| <term>Kirin 9030 processor products</term> | √ |

## Function Description

- **Operator function**: In MoE (Mixture of Experts) computation, the output of the MoE (Mixture of Experts) FFN (Feed-Forward Network) is finally processed and merged.
- **Calculation Formula**:

  $$
  expertid=expertIdx[i,k]
  $$

  $$
  out(i,j)=x1_{i,j}+x2Optional_{i,j}+\sum_{k=0}^{K}(scales_{i,k}*(expandedX_{expandedRowIdx_{i+k*num_rows},j}+bias_{expertid,j}))
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
        <td>`expandedX` in the formula, that is, the FFN (Feed-Forward Network) output of MoE (Mixture of Experts).</td>
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
        <td>x1Optional</td>
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
        <td>biasOptional</td>
        <td>Input</td>
        <td>`bias` in the formula.</td>
        <td>FLOAT16, BFLOAT16, FLOAT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>scalesOptional</td>
        <td>Input</td>
        <td>`scales` in the formula.</td>
        <td>FLOAT16, BFLOAT16, FLOAT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>expertIdxOptional</td>
        <td>Input</td>
        <td>`expertIdx` in the formula.</td>
        <td>INT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>dropPadMode</td>
        <td>Attribute</td>
        <td>Indicates whether the drop mode is supported and describes the arrangement of expandedRowIdx.</td>
        <td>INT64</td>
        <td>-</td>
      </tr>
      <tr>
        <td>k</td>
        <td>Attribute</td>
        <td>Number of top-K experts selected for each token. When scalesOptional is not empty and k > 1, the system checks that k is consistent with the second dimension of the scalesOptional shape. When scalesOptional is empty, k is directly used to derive the output shape. The default value is 1.</td>
        <td>INT64</td>
        <td>-</td>
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

- <term>Kirin X90 processor products</term>, <term>Kirin 9030 processor products</term>: BFLOAT16 is not supported.

## Constraints

None.

## Calling Instructions

| Calling Method   | Sample Code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn API  | [test_aclnn_moe_finalize_routing_v2.cpp](examples/test_aclnn_moe_finalize_routing_v2.cpp) | Calls the MoeFinalizeRoutingV2 operator through the [aclnnMoeFinalizeRoutingV2](docs/aclnnMoeFinalizeRoutingV2_en.md) API. |
| PyTorch API | - | Calls the moe_finalize_routing operator through the [moe_finalize_routing](docs/torchapi_moe_finalize_routing_en.md) API. |
