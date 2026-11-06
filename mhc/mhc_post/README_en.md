# MhcPost

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

- Operator function: MhcPost performs Post Mapping on the previous layer output $h_{t}^{out}$ in the mHC architecture through a series of computations, performs Res Mapping on the previous layer input $x_l$, and then applies a residual connection to the two results to obtain the input $x_{l+1}$ of the next layer.

- Computation formula:

  When h_res is provided:

  $$
  x_{l+1} = (H_{l}^{res})^{T} \times x_l + h_{l}^{out} \otimes H_{t}^{post}
  $$

  When h_res is omitted (supported only by Ascend 950):

  $$
  x_{l+1} = x_l + h_{l}^{out} \otimes H_{t}^{post}
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
      <td>x</td>
      <td>Input</td>
      <td>Tensor to be computed, representing the input data of the mHC layer in the network.</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>h_res</td>
      <td>Input (optional)</td>
      <td>h_res transformation matrix of mHC, which is the doubly stochastic matrix after the sinkhorn transformation. When omitted, the operator degrades to a direct residual connection (supported only by Ascend 950).</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>h_out</td>
      <td>Input</td>
      <td>Output of the Atten/MLP layer.</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>h_post</td>
      <td>Input</td>
      <td>h_post transformation matrix of mHC.</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>out</td>
      <td>Output</td>
      <td>Output data of the mHC layer in the network, used as the input of the next layer.</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody>
  </table>

## Constraints

- <term>Ascend 950PR&950DT products</term>: h_res accepts nullptr. In this case, the operator degrades to a direct residual connection.
- <term>Atlas A2 products</term> and <term>Atlas A3 products</term>: h_res is a required parameter and does not accept nullptr.

  | Specification Item   | Specification               | Description                                |
  | :------- | :----------------- | :------------------------------------- |
  | n        | 4                  | Fixed at 4                              |
  | d        | Range 1 to 100000       | Multiple of 128                             |

## Calling Instructions

| Calling Method      | Calling Sample                 | Description                                                         |
|--------------|-------------------------|--------------------------------------------------------------|
| aclnn invocation | [test_aclnn_mhc_post](examples/test_aclnn_mhc_post.cpp) | Calls the MhcPost operator through the [aclnnMhcPost](docs/aclnnMhcPost_en.md) API. |
| PyTorch API | - | Calls the MhcPost operator through the [mhc_post](docs/torchapi_mhc_post_en.md) API. |
