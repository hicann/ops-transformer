# AddExample

English | [简体中文](./README.md)

## Product Support

| Product | Supported |
| ---- | :----:|
|Atlas A3 products|√|
|Atlas A2 products|√|

## Function Description

- Operator function: Performs the addition computation.

- Computation formula:

$$
y = x1 + x2
$$

## Parameter Description

<table style="undefined;table-layout: fixed; width: 980px"><colgroup>
  <col style="width: 100px">
  <col style="width: 150px">
  <col style="width: 280px">
  <col style="width: 330px">
  <col style="width: 120px">
  </colgroup>
  <thead>
    <tr>
      <th>Parameter</th>
      <th>Input/Output/Attribute</th>
      <th>Description</th>
      <th>Data Type</th>
      <th>Data Format</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>x1</td>
      <td>Input</td>
      <td>Input parameter for the add_example computation, which is x1 in the formula.</td>
      <td>FLOAT, INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>Input</td>
      <td>Input parameter for the add_example computation, which is x2 in the formula.</td>
      <td>FLOAT, INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Output parameter for the add_example computation, which is y in the formula.</td>
      <td>FLOAT, INT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

None

## Invocation Description

<table><thead>
  <tr>
    <th>Invocation Method</th>
    <th>Invocation Sample</th>
    <th>Description</th>
  </tr></thead>
<tbody>
  <tr>
    <td>aclnn invocation</td>
    <td><a href="./examples/test_aclnn_add_example.cpp">test_aclnn_add_example</a></td>
    <td rowspan="2">A quick invocation sample is shown below. It is for reference only. For details about how to compile and run an operator, see <a href="../../docs/zh/invocation/quick_op_invocation.md">Operator Invocation</a>.</td>
  </tr>
</tbody>
</table>

```bash
## Quick Start
# Run the following commands in the root directory of the ops-transformer repository
# Assume that the environment variables are ready
bash build.sh --soc=${soc_version} --ops=add_example
bash build.sh --run_example add_example eager cus
```
