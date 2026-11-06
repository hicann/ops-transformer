# MoeFinalizeRoutingV2Grad

English | [简体中文](./README.md)

## Product Support

| Product                                                      |  Supported   |
| :----------------------------------------------------------- |:-------:|
| <term>Ascend 950PR&950DT products</term>                     |     √    |
| <term>Atlas A3 products</term>                               |    √    |
| <term>Atlas A2 products</term>                               |    √    |
| <term>Atlas 200I/500 A2 inference products</term>            |    ×    |
| <term>Atlas inference products</term>                        |    ×    |
| <term>Atlas training products</term>                         |    ×    |

## Description

- **Operator function**: Backpropagation of aclnnMoeFinalizeRoutingV2.
- **Formula**:
  R: batch * sequence

  H: hidden

  K: topk

  gradY: (R, H)

  expandedRowIdx: (R * K)

  expandedXOptional: (R * K, H) or (activeNum, H) or (expertNum, expertCapacity, H)

  scalesOptional: (R, K)

  expertIdxOptional: (R, K)

  biasOptional: (E, H)

  i : 0 ~ R * K - 1

  j : 0 ~ H

  (1) scalesOptional is a null pointer:

  $$
  gradExpandedXOut[expandedRowIdx[i]][j] = gradY[i / K][j]
  $$

  (2) scalesOptional is not a null pointer and biasOptional is a null pointer:

  $$
  gradExpandedXOut[expandedRowIdx[i]][j] = gradY[i / K][j] * scalesOptional[i]
  $$

  $$
  gradScalesOut[i] = sum(expandedXOptional[expandedRowIdx[i]][j] * gradY[i / K][j])
  $$

  (3) scalesOptional is not a null pointer and biasOptional is not a null pointer:

  $$
  gradExpandedXOut[expandedRowIdx[i]][j] = gradY[i / K][j] * scalesOptional[i]
  $$

  $$
  gradScalesOut[i] = sum((expandedXOptional[expandedRowIdx[i]][j] + biasOptional[expertIdxOptional[i]][j]) * gradY[i / K][j])
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
        <td>gradY</td>
        <td>Input</td>
        <td>`gradY` in the formula, indicating the derivative of the forward output y of MoeFinalizeRoutingV2.</td>
        <td>FLOAT16, BFLOAT16, FLOAT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>expandedRowIdx</td>
        <td>Input</td>
        <td>`expandedRowIdx` in the formula, indicating the index of tokens sorted by expert order.</td>
        <td>INT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>expandedXOptional</td>
        <td>Input</td>
        <td>`expandedXOptional` in the formula, indicating the features expanded according to expertIdx.</td>
        <td>FLOAT16, BFLOAT16, FLOAT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>scalesOptional</td>
        <td>Input</td>
        <td>`scalesOptional` in the formula, indicating the scaling of the features.</td>
        <td>FLOAT16, BFLOAT16, FLOAT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>expertIdxOptional</td>
        <td>Input</td>
        <td>`expertIdxOptional` in the formula, indicating the index of the expert that processes each feature.</td>
        <td>INT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>biasOptional</td>
        <td>Input</td>
        <td>`biasOptional` in the formula, indicating the offset of the features.</td>
        <td>FLOAT16, BFLOAT16, FLOAT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>dropPadMode</td>
        <td>Attribute</td>
        <td>Indicates the scenario to use.</td>
        <td>INT64</td>
        <td>-</td>
      </tr>
      <tr>
        <td>activeNum</td>
        <td>Attribute</td>
        <td>`activeNum` in the formula, indicating the maximum number of output rows of gradExpandedXOut.</td>
        <td>INT64</td>
        <td>-</td>
      </tr>
      <tr>
        <td>expertNum</td>
        <td>Attribute</td>
        <td>`expertNum` in the formula, indicating the number of experts.</td>
        <td>INT64</td>
        <td>-</td>
      </tr>
      <tr>
        <td>expertCapacity</td>
        <td>Attribute</td>
        <td>`expertCapacity` in the formula.</td>
        <td>INT64</td>
        <td>-</td>
      </tr>
      <tr>
        <td>gradExpandedXOut</td>
        <td>Output</td>
        <td>`gradExpandedXOut` in the formula, indicating the derivative of the forward input expandedX of MoeFinalizeRoutingV2.</td>
        <td>FLOAT16, BFLOAT16, FLOAT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>gradScalesOut</td>
        <td>Output</td>
        <td>`gradScalesOut` in the formula, indicating the derivative of the forward input scales of MoeFinalizeRoutingV2.</td>
        <td>FLOAT16, BFLOAT16, FLOAT32</td>
        <td>ND</td>
      </tr>
    </tbody>
  </table>

## Constraints

None.

## Calling Instructions

| Calling Method   | Sample Code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn API  | [test_aclnn_moe_finalize_routing_v2_grad.cpp](examples/test_aclnn_moe_finalize_routing_v2_grad.cpp) | Call the MoeFinalizeRoutingV2Grad operator through the [aclnnMoeFinalizeRoutingV2Grad](docs/aclnnMoeFinalizeRoutingV2Grad_en.md) API. |
| PyTorch API | - | Call the moe_finalize_routing_grad operator through the [moe_finalize_routing_grad](docs/torchapi_moe_finalize_routing_grad_en.md) API. |
