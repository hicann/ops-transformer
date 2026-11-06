# MoeComputeExpertTokens

English | [简体中文](./README.md)

## Product Support Status

| Product                                                      | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                             |    √     |
| <term>Atlas A3 products</term>     |    √     |
| <term>Atlas A2 products</term> |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                             |    ×     |
| <term>Atlas training products</term>                              |    ×     |
| <term>Kirin X90 processor products</term> | √ |
| <term>Kirin 9030 processor products</term> | √ |

## Function Description

- **Operator function**: In MoE computation, binary search is used to find the position of the last row processed by each expert.
- **Calculation Formula**:

  $$
  out_{i}=BinarySearch(sortedExperts,numExperts)
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
        <td>sortedExperts</td>
        <td>Input</td>
        <td>sortedExperts in the formula.</td>
        <td>INT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>numExperts</td>
        <td>Attribute</td>
        <td>Total number of experts.</td>
        <td>INT64</td>
        <td>-</td>
      </tr>
      <tr>
        <td>out</td>
        <td>Output</td>
        <td>Output in the formula.</td>
        <td>INT32</td>
        <td>ND</td>
      </tr>
    </tbody>
  </table>

## Constraints

- The shape size of sortedExperts must be less than $2^(24)$.
- The input constant of numExperts must be greater than 0 and cannot exceed 2048.
- The input shape size must not exceed the maximum memory that can be allocated on the device. Otherwise, abnormal termination occurs.

## Calling Instructions

| Calling Method      | Calling Sample                 | Description                                                         |
|--------------|-------------------------|--------------------------------------------------------------|
| aclnn calling | [test_aclnn_moe_compute_expert_tokens](examples/test_aclnn_moe_compute_expert_tokens.cpp) | Calls the [MoeComputeExpertTokens](docs/aclnnMoeComputeExpertTokens_en.md) operator through the API. |
