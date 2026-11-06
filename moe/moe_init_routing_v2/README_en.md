# MoeInitRoutingV2

English | [简体中文](./README.md)

## Product Support

|Product          |  Supported  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     √    |
|  <term>Atlas A3 products</term>   |     √    |
|  <term>Atlas A2 products</term>     |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     ×    |
|  <term>Atlas inference products</term>    |     √    |
|  <term>Atlas training products</term>    |     ×    |

## Functional Description

- Operator function: This operator corresponds to the **routing calculation** in MoE (Mixture of Experts). It takes the outputs x and expertIdx of the MoeGatingTopKSoftmax operator as input, and outputs the routing matrix expanded_x and other results for subsequent calculation. This interface provides the following function changes compared with the V1 interface (MoeInitRouting, source code not open-sourced). Select an appropriate interface based on actual requirements:

  - Adds the Drop mode. In this mode, the output aligns the number of tokens that each expert needs to process to expertCapacity. Tokens exceeding expertCapacity are dropped, and insufficient tokens are padded with 0.
  - Adds the optional output expertTokensCountOrCumsumOut in the Dropless mode, which outputs the cumulative number of tokens (cumsum) that each expert needs to process, or the number of tokens (count) that each expert needs to process.
  - Adds the optional output expertTokensBeforeCapacityOut in the Drop mode, which outputs the number of tokens that each expert should process before dropping.
  - Deletes the rowIdx input.

  **Note:**
  Routing calculation is a step in the MoE model. A MoE model mainly consists of a set of expert models and a gating model. During calculation, the input data first passes through the gating network (Gating Network, which includes the MoeGatingTopKSoftmax operator) to compute the k experts with the highest weights for each data element. The result is then input to the MoeInitRouting operator to generate the routing matrix. Afterwards, each expert in the model processes its assigned data based on the routing matrix and produces the corresponding output. Finally, the outputs of all experts are weighted and summed to form the final prediction result.

- Computation formulas:

  1. Sort the input expertIdx to obtain the sorted result sortedExpertIdx and the corresponding sequence numbers sortedRowIdx:

  $$
  sortedExpertIdx, sortedRowIdx=keyValueSort(expertIdx)
  $$

  2. Perform position mapping based on sortedRowIdx to obtain expandedRowIdxOut:

  $$
  expandedRowIdxOut[sortedRowIdx[i]]=i
  $$

  3. Take the values in x at the positions corresponding to the first numRows elements of sortedRowIdx to obtain expandedXOut:

  $$
  expandedXOut[i]=x[sortedRowIdx[i]\%numRows]
  $$

  4. Count the histogram result of each expert in sortedExpertIdx and then perform cumsum to obtain expertTokensCountOrCumsumOut:

  $$
  expertTokensCountOrCumsumOut[i]=Cumsum(Histogram(sortedExpertIdx))
  $$

  5. Count the histogram result of each expert in sortedExpertIdx to obtain expertTokensBeforeCapacityOut:

  $$
  expertTokensBeforeCapacityOut[i]=Histogram(sortedExpertIdx)
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
      <td>x</td>
      <td>Input</td>
      <td>Input of MoE, that is, the token feature input.</td>
      <td>FLOAT16, BFLOAT16, FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>expertIdx</td>
      <td>Input</td>
      <td>Sequence numbers of the k experts that process each token.</td>
      <td>Ascend 950PR&950DT products: INT32, INT64/Other processors: INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>activeNum</td>
      <td>Attribute</td>
      <td>Indicates whether the Active scenario is used.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>expertCapacity</td>
      <td>Attribute</td>
      <td>Indicates the number of tokens that each expert can process.</td>
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
      <td>Indicates whether the Drop/Pad scenario is used.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>expertTokensCountOrCumsumFlag</td>
      <td>Attribute</td>
      <td>The value is 0, 1, or 2.
        0: expertTokensCountOrCumsumOut is not output.
        1: the output value is the cumulative number of tokens processed by each expert.
        2: the output value is the number of tokens processed by each expert.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>expertTokensBeforeCapacityFlag</td>
      <td>Attribute</td>
      <td>The value is false or true.
        false: expertTokensBeforeCapacityOut is not output.
        true: the output value is the number of tokens processed by each expert before the drop operation.</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>expandedXOut</td>
      <td>Output</td>
      <td>Features expanded based on expertIdx. In the Dropless/Active scenario, it must be a 2D tensor.</td>
      <td>FLOAT16, BFLOAT16, FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>expandedRowIdxOut</td>
      <td>Output</td>
      <td>Index mapping between expandedXOut and x. It must be a 1D tensor.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>expertTokensCountOrCumsumOut</td>
      <td>Output</td>
      <td>Outputs the statistics and cumulative value of the number of tokens processed by each expert. Output is controlled by the expertTokensCountOrCumsumFlag parameter.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>expertTokensBeforeCapacityOut</td>
      <td>Output</td>
      <td>Outputs the statistics of the number of tokens processed by each expert before the drop operation.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

- The input x must be 2D, and the numRows of x must equal the numRows of expertIdx.
- When dropPadMode is 1, expertNum and expertCapacity must be greater than 0.
- The numRows axis of x must be greater than expertCapacity.

## Calling Instructions

| Calling Method   | Sample Code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn interface  | [test_aclnn_moe_init_routing_v2](examples/test_aclnn_moe_init_routing_v2.cpp) | Calls the MoeInitRoutingV2 operator through the [aclnnMoeInitRoutingV2](docs/aclnnMoeInitRoutingV2_en.md) interface. |
