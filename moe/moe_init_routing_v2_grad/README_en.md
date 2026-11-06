# MoeInitRoutingV2Grad

English | [简体中文](./README.md)

## Product Support

|Product          |  Supported  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     √    |
|  <term>Atlas A3 products</term>   |     √    |
|  <term>Atlas A2 products</term>     |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     ×    |
|  <term>Atlas inference products</term>    |     ×    |
|  <term>Atlas training products</term>    |     ×    |

## Functional Description

- **Operator function**: Backpropagation of [aclnnMoeInitRoutingV2](../moe_init_routing_v2/docs/aclnnMoeInitRoutingV2_en.md), which performs the weighted summation of tokens.
- **Computation formulas**:

  $$
  gradX_i=\sum_{t=0}^{topK}gradExpandedX[expandedRowIdx[i * topK + t]]
  $$

## Parameter Description

  <table style="table-layout: auto; width: 100%">
  <thead>
    <tr>
      <th style="white-space: nowrap">Parameter</th>
      <th style="white-space: nowrap">Input/Output/Attribute</th>
      <th style="white-space: nowrap">Description</th>
      <th style="white-space: nowrap">Data Type</th>
      <th style="white-space: nowrap">Data Format</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>gradExpandedX</td>
      <td>Input</td>
      <td>Target tensor after routing, corresponding to `gradExpandedX` in the formula.</td>
      <td>FLOAT16, BFLOAT16, FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>expandedRowIdx</td>
      <td>Input</td>
      <td>Index of tokens sorted in expert order, corresponding to `expandedRowIdx` in the formula.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>topK</td>
      <td>Attribute</td>
      <td>Value of topk, corresponding to `topK` in the formula.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dropPadMode</td>
      <td>Attribute</td>
      <td>Indicates whether the scenario is a Drop type.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>activeNum</td>
      <td>Attribute</td>
      <td>Indicates whether the Active scenario is used.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out</td>
      <td>Output</td>
      <td>Output of the routing backward calculation, corresponding to `gradX` in the formula.</td>
      <td>FLOAT16, BFLOAT16, FLOAT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

  None.

## Calling Instructions

| Calling Method   | Sample Code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn interface  | [test_aclnn_moe_init_routing_v2_grad](examples/test_aclnn_moe_init_routing_v2_grad.cpp) | Calls the MoeInitRoutingV2Grad operator through the [aclnnMoeInitRoutingV2Grad](docs/aclnnMoeInitRoutingV2Grad_en.md) interface. |
| PyTorch API | - | Calls the moe_init_routing_grad operator through the [moe_init_routing_grad](docs/torchapi_moe_init_routing_grad_en.md) interface. |
