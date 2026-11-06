# MoeInitRoutingQuant

English | [简体中文](./README.md)

## Product Support

|Product | Supported |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     ×    |
|  <term>Atlas A3 products</term>   |     ×    |
|  <term>Atlas A2 products</term>     |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     ×    |
|  <term>Atlas inference products</term>    |     ×    |
|  <term>Atlas training products</term>    |     ×    |

## Function Description

- **Operator function**: performs the routing computation of MoE based on the computation result of [aclnnMoeGatingTopKSoftmax](../moe_gating_top_k_softmax/docs/aclnnMoeGatingTopKSoftmax_en.md) and quantizes the result.
- **Computation formula**:

  $$
  expandedExpertIdx,sortedRowIdx=keyValueSort(expertIdx,rowIdx)
  $$

  $$
  expandedRowIdx[sortedRowIdx[i]]=i
  $$

  $$
  expandedX[i]=quant[x[sortedRowIdx[i]\%numRows]]
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
      <td>x</td>
      <td>Input</td>
      <td>Input of MOE, that is, the token feature input, corresponding to `x` in the formula.</td>
      <td>FLOAT16, BFLOAT16, FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>rowIdx</td>
      <td>Input</td>
      <td>Indicates the original row position corresponding to each position, corresponding to `rowIdx` in the formula.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>expertIdx</td>
      <td>Input</td>
      <td>The K experts processed for each row of features, output by aclnnMoeInitRoutingQuantSoftmax, corresponding to `expertIdx` in the formula.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>activeNum</td>
      <td>Attribute</td>
      <td>Indicates the total maximum number of rows to process and must be greater than or equal to 0. Only this number of rows in expandedXOut is valid.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scale</td>
      <td>Attribute</td>
      <td>Required for quantization computation.</td>
      <td>DOUBLE</td>
      <td>-</td>
    </tr>
    <tr>
      <td>offset</td>
      <td>Attribute</td>
      <td>Required for quantization computation.</td>
      <td>DOUBLE</td>
      <td>-</td>
    </tr>
    <tr>
      <td>expandedXOut</td>
      <td>Output</td>
      <td>Features expanded according to expertIdx, corresponding to `expandedX` in the formula.</td>
      <td>INT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>expandedRowIdxOut</td>
      <td>Output</td>
      <td>Mapping between expandedX and x, corresponding to `expandedRowIdx` in the formula.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>expandedExpertIdxOut</td>
      <td>Output</td>
      <td>Outputs the sorted result of expertIdx, corresponding to `expandedExpertIdx` in the formula.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

  None.

## Calling Instructions

| Calling Method | Sample Code | Description |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn API | [test_aclnn_moe_init_routing_quant](examples/test_aclnn_moe_init_routing_quant.cpp) | Calls the MoeInitRoutingQuant operator through the [aclnnMoeInitRoutingQuant](docs/aclnnMoeInitRoutingQuant_en.md) API. |
