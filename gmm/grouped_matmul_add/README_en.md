# GroupedMatmulAdd

English | [简体中文](./README.md)

## Product Support

| Product                                                      | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                             |    √     |
| <term>Atlas A3 products</term>     |    √     |
| <term>Atlas A2 products</term> |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                             |    ×     |
| <term>Atlas training products</term>                              |    ×     |

## Function Description

- Operator function: performs grouped matrix multiplication, and the dimension sizes of matrix multiplication can differ between groups. The basic function is matrix multiplication, for example $yRef_i[m_i,n_i]=x_i[m_i,k_i] \times weight_i[k_i,n_i]+y_i[m_i,n_i], i=1...g$, where g is the number of groups, and $m_i/k_i/n_i$ are the corresponding shapes. Currently, only grouping along the K axis is supported.

  - Grouping along the K axis: the $k_i$ values differ, but $m_i/n_i$ are the same for each group.
- Calculation formula:

  $$
  yRef_i=x_i\times weight_i + y_i
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
      <td style="white-space: nowrap">x</td>
      <td style="white-space: nowrap">Input</td>
      <td style="white-space: nowrap">Input x in the formula.</td>
      <td style="white-space: nowrap">FLOAT16, BFLOAT16</td>
      <td style="white-space: nowrap">ND</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">weight</td>
      <td style="white-space: nowrap">Input</td>
      <td style="white-space: nowrap">weight in the formula.</td>
      <td style="white-space: nowrap">FLOAT16, BFLOAT16</td>
      <td style="white-space: nowrap">ND</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">groupList</td>
      <td style="white-space: nowrap">Input</td>
      <td style="white-space: nowrap">Indicates the cumsum result (cumulative sum) of the matmul size distribution of the input along the K axis.</td>
      <td style="white-space: nowrap">INT64</td>
      <td style="white-space: nowrap">ND</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">y</td>
      <td style="white-space: nowrap">Input</td>
      <td style="white-space: nowrap">Indicates the output matrix for in-place accumulation.</td>
      <td style="white-space: nowrap">FLOAT32</td>
      <td style="white-space: nowrap">ND</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">transposeX</td>
      <td style="white-space: nowrap">Attribute</td>
      <td style="white-space: nowrap">Specifies whether the x matrix is transposed.</td>
      <td style="white-space: nowrap">BOOL</td>
      <td style="white-space: nowrap">-</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">transposeWeight</td>
      <td style="white-space: nowrap">Attribute</td>
      <td style="white-space: nowrap">Specifies whether the weight matrix is transposed.</td>
      <td style="white-space: nowrap">BOOL</td>
      <td style="white-space: nowrap">-</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">groupType</td>
      <td style="white-space: nowrap">Attribute</td>
      <td style="white-space: nowrap">Indicates the grouping type.</td>
      <td style="white-space: nowrap">INT64</td>
      <td style="white-space: nowrap">-</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">groupListType</td>
      <td style="white-space: nowrap">Attribute</td>
      <td style="white-space: nowrap">Indicates the format of the grouping groupList.</td>
      <td style="white-space: nowrap">INT64</td>
      <td style="white-space: nowrap">-</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">yRef</td>
      <td style="white-space: nowrap">Output</td>
      <td style="white-space: nowrap">Indicates the output matrix for in-place accumulation.</td>
      <td style="white-space: nowrap">FLOAT32</td>
      <td style="white-space: nowrap">ND</td>
    </tr>
  </tbody>
</table>

## Constraint Description

- For each group of tensors in x and weight, the size of each dimension after 32-byte alignment must be less than the maximum int32 value 2147483647.

## Interface Invocation

| Invocation Method | Sample | Description |
|--------------|-------------------------|--------------------------------------------------------------|
| aclnn invocation | [test_aclnn_grouped_matmul_add](examples/test_aclnn_grouped_matmul_add.cpp) | Calls the [GroupedMatmulAdd](docs/aclnnGroupedMatmulAdd_en.md) operator through the API. |
