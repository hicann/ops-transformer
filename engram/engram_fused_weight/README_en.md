# EngramFusedWeight

English | [简体中文](./README.md)

## Applicable Products

| Product                                                     | Supported |
| :------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                   |    √    |
| <term>Atlas A3 products</term>                             |    ×    |
| <term>Atlas A2 products</term>                             |    ×    |
| <term>Atlas 200I/500 A2 inference products</term>                    |    ×    |
| <term>Atlas inference products</term>                            |    ×    |
| <term>Atlas training products</term>                            |    ×    |

## Description

- **Operator function**: fuses weights for Engram gating. Multiplies two BF16/FP16 weights element-wise, upcasts them, and outputs the FP32 result: `weight_fused = fp32(weight_hidden) * fp32(weight_embed)`.

- **Computation formula**: for each flattened element $i$ ($i = 0, 1, \dots, hcMult \times hiddenSize - 1$):

  $$
  y_i = \mathrm{float}(x1_i) \times \mathrm{float}(x2_i)
  $$

  In the kernel, BF16/FP16 values are first upcast to FP32 and then multiplied, ensuring FP32 precision.

## Parameters

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
        <td>weight_hidden</td>
        <td>Input</td>
        <td>The first weight of Engram gating (q_weight), with shape (hcMult, hiddenSize).</td>
        <td>BFLOAT16 and FLOAT16</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>weight_embed</td>
        <td>Input</td>
        <td>The second weight of Engram gating (k_weight), whose shape and data type are the same as those of weight_hidden.</td>
        <td>BFLOAT16 and FLOAT16</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>weight_fused</td>
        <td>Output</td>
        <td>The fused weight, whose shape is the same as that of the inputs.</td>
        <td>FLOAT</td>
        <td>ND</td>
      </tr>
    </tbody>
  </table>

## Constraints

- <term>Ascend 950PR&950DT products</term>:
  - The inputs support only the 2-D format with shape (hcMult, hiddenSize), and the two inputs must have the same shape and data type.
  - The total number of elements (hcMult × hiddenSize) must be aligned to 256.

## Calling Instructions

| Calling Method      | Sample Code                 | Description                                                         |
|--------------|-------------------------|--------------------------------------------------------------|
| aclnn API | [test_aclnn_engram_fused_weight](examples/test_aclnn_engram_fused_weight.cpp) | Calls the EngramFusedWeight operator through the [aclnnEngramFusedWeight](docs/aclnnEngramFusedWeight.md) API. |
| PyTorch API | - | Calls the EngramFusedWeight operator through the [torchapi_engram_fused_weight](docs/torchapi_engram_fused_weight.md) API. |
