# MhcPostBackward

English | [简体中文](./README.md)

## Product Support

|Product      | Supported |
|:----------------------------|:-----------:|
|<term>Ascend 950PR&950DT products</term>|      √     |
|<term>Atlas A3 products</term>|      √     |
|<term>Atlas A2 products</term>|      √     |
|<term>Atlas 200I/500 A2 inference products</term>|      ×     |
|<term>Atlas inference products</term>|      ×     |
|<term>Atlas training products</term>|      ×     |

## Function Description

- Operator function: mhc_post performs Post Mapping on the previous layer output $h_{t}^{out}$ in the MHC (Manifold-Constrained Hyper-Connection) architecture through a series of computations, performs Res Mapping on the previous layer input $x_j$, and then applies a residual connection to the two results to obtain the input $x_{l+1}$ of the next layer. This operator implements the backward function of the preceding process.

- Computation formula:
  $$
  grad\_x = H_{l}^{res} \times grad\_y\\
  grad\_h\_res = x_{l} \times {grad\_y}^{T}
  $$
  $$
  grad\_h\_out=({grad\_y} * (H_{l}^{post}.unsqueeze(-1))).sum(dim=-2)\\
  grad\_h\_post=({grad\_y} * (h_{l}^{out}.unsqueeze(-2))).sum(dim=-1)
  $$

## Parameter Description

<table style="undefined;table-layout: fixed; width: 952px"><colgroup>
<col style="width: 106px">
<col style="width: 87px">
<col style="width: 445px">
<col style="width: 209px">
<col style="width: 105px">
</colgroup>
<thead>
  <tr>
    <th>Parameter</th>
    <th>Input/Output</th>
    <th>Description</th>
    <th>Data Type</th>
    <th>Data Format</th>
  </tr></thead>
<tbody>
   <tr>
    <td>grad_y</td>
    <td>Input</td>
    <td>Data to be computed, representing the input data of the MHC layer in the network</td>
    <td>FLOAT16, BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>x</td>
    <td>Input</td>
    <td>Data to be computed, representing the input data of the MHC layer in the network</td>
    <td>FLOAT16, BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>h_res</td>
    <td>Input</td>
    <td>h_res transformation matrix of mHC</td>
    <td>FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>h_out</td>
    <td>Input</td>
    <td>Output of the Atten/MLP layer</td>
    <td>FLOAT16, BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>h_post</td>
    <td>Input</td>
    <td>h_post transformation matrix of MHC</td>
    <td>FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>grad_x</td>
    <td>Output</td>
    <td>Gradient of x, the input data of the MHC layer in the network</td>
    <td>FLOAT16, BFLOAT16</td>
    <td>ND</td>
  </tr>
   <tr>
    <td>grad_h_res</td>
    <td>Output</td>
    <td>Gradient of h_res, the input data of the MHC layer in the network</td>
    <td>FLOAT32</td>
    <td>ND</td>
  </tr>
   <tr>
    <td>grad_h_out</td>
    <td>Output</td>
    <td>Gradient of h_out, the input data of the MHC layer in the network</td>
    <td>FLOAT16, BFLOAT16</td>
    <td>ND</td>
  </tr>
   <tr>
    <td>grad_h_post</td>
    <td>Output</td>
    <td>Gradient of h_post, the input data of the MHC layer in the network</td>
    <td>FLOAT32</td>
    <td>ND</td>
  </tr>
</tbody>
</table>

## Constraints

None

## Calling Instructions

| Calling Method      | Calling Sample                 | Description                                                         |
|--------------|-------------------------|--------------------------------------------------------------|
| aclnn invocation | [test_aclnn_mhc_post_backward](examples/test_aclnn_mhc_post_backward.cpp) | Calls the operator through the [aclnnMhcpostBackward](docs/aclnnMhcPostBackward_en.md) API.|
