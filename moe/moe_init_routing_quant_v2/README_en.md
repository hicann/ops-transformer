# MoeInitRoutingQuantV2

English | [简体中文](./README.md)

## Product Support

|Product | Supported |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     √    |
|  <term>Atlas A3 products</term>   |     √    |
|  <term>Atlas A2 products</term>     |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     ×    |
|  <term>Atlas inference products</term>    |     ×    |
|  <term>Atlas training products</term>    |     ×    |

## Function Description

- **Operator function**: performs the routing computation of MoE based on the computation result of [aclnnMoeGatingTopKSoftmaxV2](../moe_gating_top_k_softmax_v2/docs/aclnnMoeGatingTopKSoftmaxV2_en.md).

  This interface has the following functional changes compared with [aclnnMoeInitRoutingQuant](../moe_init_routing_quant/docs/aclnnMoeInitRoutingQuant_en.md). Choose an appropriate interface according to the actual scenario:

  - Adds the drop mode. In this mode, the output is processed according to the expertCapacity of each expert; tokens beyond expertCapacity are not processed, and the insufficient part is padded with 0s.
  - Adds the optional output expertTokensCountOrCumsumOutOptional in dropless mode, and the optional output expertTokensBeforeCapacityOutOptional in drop mode.
  - Removes the rowIdx input.
  - Adds the dynamic quant computation mode.

- **Computation formula**:

  1. Sort the input expertIdx to obtain the sorted result sortedExpertIdx and the corresponding indices sortedRowIdx:

    $$
    sortedExpertIdx, sortedRowIdx=keyValueSort(expertIdx)
    $$

  2. Use sortedRowIdx for position mapping to obtain expandedRowIdxOut:

    $$
    expandedRowIdxOut[sortedRowIdx[i]]=i
    $$

  3. In dropless mode, compute the histogram result of each expert in sortedExpertIdx and then perform Cumsum to obtain expertTokensCountOrCumsumOutOptional:

    $$
    expertTokensCountOrCumsumOutOptional[i]=Cumsum(Histogram(sortedExpertIdx))
    $$

  4. In drop mode, compute the histogram result of each expert in sortedExpertIdx to obtain expertTokensBeforeCapacityOutOptional:

    $$
    expertTokensBeforeCapacityOutOptional[i]=Histogram(sortedExpertIdx)
    $$

  5. Compute the quant result:
    - Static quant:

        $$
        quantResult = round((x * scaleOptional) + offsetOptional)
        $$

    - Dynamic quant:
        - If scale is not entered:

            $$
            dynamicQuantScaleOutOptional = row\_max(abs(x)) / 127
            $$

            $$
            quantResult = round(x / dynamicQuantScaleOutOptional)
            $$

        - If scale is entered:

            $$
            dynamicQuantScaleOutOptional = row\_max(abs(x * scaleOptional)) / 127
            $$

            $$
            quantResult = round(x * scaleOptional / dynamicQuantScaleOutOptional)
            $$

  6. Take the values at the corresponding positions of the first NUM\_ROWS elements of sortedRowIdx from quantResult to obtain expandedXOut:

    $$
    expandedXOut[i]=quantResult[sortedRowIdx[i]\%NUM\_ROWS]
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
      <td>expertIdx</td>
      <td>Input</td>
      <td>The K experts processed for each row of features, output by aclnnMoeGatingTopKSoftmaxV2, corresponding to `expertIdx` in the formula.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scaleOptional</td>
      <td>Optional input</td>
      <td>Parameter used to compute the quant result. It must be entered in static quant scenarios, corresponding to `scaleOptional` in the formula.</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>offsetOptional</td>
      <td>Optional input</td>
      <td>Offset value used to compute the quant result. It must be entered in static quant scenarios, corresponding to `offsetOptional` in the formula.</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>activeNum</td>
      <td>Attribute</td>
      <td>Indicates whether the scenario is Active. This attribute takes effect when dropPadMode is 0, and the value must be greater than or equal to 0.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>expertCapacity</td>
      <td>Attribute</td>
      <td>Indicates the number of tokens that each expert can process. The value must be greater than or equal to 0.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>expertNum</td>
      <td>Attribute</td>
      <td>Indicates the number of experts. The value must be greater than or equal to 0.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dropPadMode</td>
      <td>Attribute</td>
      <td>Indicates whether the scenario is Drop/Pad. The value is 0 or 1. 0: the scenario is not Drop/Pad, and expertCapacity is not checked in this scenario.
1: the scenario is Drop/Pad. expertNum and expertCapacity need to be checked, and the values processed by each expert that exceed or fall short of expertCapacity are processed accordingly.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>expertTokensCountOrCumsumFlag</td>
      <td>Attribute</td>
      <td>0: expertTokensCountOrCumsumOutOptional is not output. 1: the output value is the cumulative number of tokens processed by each expert. 2: the output value is the number of tokens processed by each expert.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>expertTokensBeforeCapacityFlag</td>
      <td>Attribute</td>
      <td>The value is false or true. false: expertTokensBeforeCapacityOutOptional is not output.
true: the output value is the number of tokens processed by each expert before drop.</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>quantMode</td>
      <td>Attribute</td>
      <td>The value is 0 or 1. 0: static quant scenario; 1: dynamic quant scenario.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>expandedXOut</td>
      <td>Output</td>
      <td>Features expanded according to expertIdx, corresponding to `expandedXOut` in the formula.</td>
      <td>INT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>expandedRowIdxOut</td>
      <td>Output</td>
      <td>Mapping between expandedX and x, corresponding to `expandedRowIdxOut` in the formula.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>expertTokensCountOrCumsumOutOptional</td>
      <td>Optional output</td>
      <td>Outputs the statistics and cumulative value of the number of tokens processed by each expert. Whether to output is controlled by the expertTokensCountOrCumsumFlag parameter. This value is output only in non-Drop/Pad scenarios, corresponding to `expertTokensCountOrCumsumOutOptional` in the formula.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>expertTokensBeforeCapacityOutOptional</td>
      <td>Optional output</td>
      <td>Outputs the statistics of the number of tokens processed by each expert before drop. Whether to output is controlled by the expertTokensBeforeCapacityFlag parameter. This value is output only in Drop/Pad scenarios, corresponding to `expertTokensBeforeCapacityOutOptional` in the formula.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dynamicQuantScaleOutOptional</td>
      <td>Optional output</td>
      <td>Outputs the intermediate value in the dynamic quant computation. This value is output only in dynamic quant scenarios, corresponding to `dynamicQuantScaleOutOptional` in the formula.</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

  None.

## Calling Instructions

| Calling Method | Sample Code | Description |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn API | [test_aclnn_moe_init_routing_quant_v2](examples/test_aclnn_moe_init_routing_quant_v2.cpp) | Calls the MoeInitRoutingQuantV2 operator through the [aclnnMoeInitRoutingQuantV2](docs/aclnnMoeInitRoutingQuantV2_en.md) API. |
