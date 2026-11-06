# MoeGatingTopKBackward

English | [简体中文](./README.md)

## Product Support

| Product | Supported |
| :----------------------------------------- | ------|
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term> | √ |
| <term>Atlas A2 products</term> | √ |
| <term>Atlas 200I/500 A2 inference products</term> | × |
| <term>Atlas inference products</term> | × |
| <term>Atlas training products</term> | × |

## Function Description

- Operator function: performs backward gradient computation for the MoE (Mixture of Experts) gating Top-K selection. This operator is the backward operator of MoeGatingTopK. Based on the normalized scores (xNorm) output by the forward operator, the upstream gradient (gradY), and the expert indices (expertIdx), it computes the gradient (gradX) of the input score matrix. The softmax mode (normType=0) and the sigmoid mode (normType=1) are supported.
- Computation formula:

  1. Scaling the gradient

    $$
    gradYScaled_{ip} = routedScalingFactor \cdot gradY_{ip}
    $$

  2. Backpropagation of renorm (skipped in softmax mode when renorm=0, in which case $gradWPrime_{ip} = gradYScaled_{ip}$ is set directly; executed in all other cases)

    $$
    wPrime_{ip} = xNorm_{i,\ expertIdx_{ip}}
    $$

    $$
    D_i = \sum_{p} wPrime_{ip} + eps
    $$

    $$
    w_{ip} = \frac{wPrime_{ip}}{D_i}
    $$

    $$
    beta_i = \sum_{p} w_{ip} \cdot gradYScaled_{ip}
    $$

    $$
    gradWPrime_{ip} = \frac{gradYScaled_{ip} - beta_i}{D_i}
    $$

  3. Scattering to the full dimensions

    $$
    gradNormX_{ij} = \sum_{p:\ expertIdx_{ip}=j} gradWPrime_{ip}
    $$

  4. Backpropagation

    - Softmax mode (normType=0, full N-dimensional Jacobian-vector product):

      $$
      dot_i = \sum_{j} xNorm_{ij} \cdot gradNormX_{ij}
      $$

      $$
      gradX_{ij} = xNorm_{ij} \cdot (gradNormX_{ij} - dot_i)
      $$

    - Sigmoid mode (normType=1):

      $$
      gradX_{ij} = xNorm_{ij} \cdot (1 - xNorm_{ij}) \cdot gradNormX_{ij}
      $$

## Parameter Description

<table style="undefined;table-layout: fixed; width: 1110px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 300px">
  <col style="width: 300px">
  <col style="width: 170px">
  </colgroup>
  <thead>
  <tr>
    <th>Parameter Name</th>
    <th>Input/Output/Attribute</th>
    <th>Description</th>
    <th>Data Type</th>
    <th>Data Format</th>
  </tr></thead>
<tbody>
  <tr>
    <td>x_norm</td>
    <td>Input</td>
    <td>Input of the computation, corresponding to xNorm in the formula. It must be a 2D tensor with dimensions [M,N]. The last dimension (the number of experts N) must be greater than or equal to 1 and less than or equal to 2048.</td>
    <td>FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>grad_y</td>
    <td>Input</td>
    <td>Upstream gradient of yOut output by the forward operator, corresponding to gradY in the formula. It must be a 2D tensor with dimensions [M,K], where K must be greater than or equal to 1 and less than or equal to N.</td>
    <td>FLOAT16, BFLOAT16, FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>expert_idx</td>
    <td>Input</td>
    <td>Top-k expert indices output by the forward operator, corresponding to expertIdx in the formula. The shape must be the same as that of grad_y, with dimensions [M,K].</td>
    <td>INT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>renorm</td>
    <td>Optional attribute</td>
    <td>renorm flag of the forward operator in softmax mode. 0: renorm is not performed; 1: renorm is performed. It takes effect only in the norm_type=0 (softmax) mode. When norm_type=1 (sigmoid), renorm is always applied.</td>
    <td>INT64</td>
    <td>-</td>
  </tr>
  <tr>
    <td>norm_type</td>
    <td>Optional attribute</td>
    <td>Type of the norm function. 0 means the Softmax function, and 1 means the Sigmoid function.</td>
    <td>INT64</td>
    <td>-</td>
  </tr>
  <tr>
    <td>routed_scaling_factor</td>
    <td>Optional attribute</td>
    <td>The routed_scaling_factor coefficient used in the forward computation, corresponding to routedScalingFactor in the formula. Default value: 1.0.</td>
    <td>FLOAT32</td>
    <td>-</td>
  </tr>
  <tr>
    <td>eps</td>
    <td>Optional attribute</td>
    <td>Constant used in the forward computation to prevent division by zero, corresponding to eps in the formula. Default value: 1e-20.</td>
    <td>FLOAT32</td>
    <td>-</td>
  </tr>
  <tr>
    <td>grad_x</td>
    <td>Output</td>
    <td>Gradient of the input parameter x of the forward operator, corresponding to gradX in the formula. The data type must be the same as that of grad_y. The shape must be the same as that of x_norm, with dimensions [M,N].</td>
    <td>FLOAT16, BFLOAT16, FLOAT32</td>
    <td>ND</td>
  </tr>
</tbody></table>

## Constraints

- <term>Atlas A3 products</term> and <term>Atlas A2 products</term>:
  - norm_type supports only the value 1 (sigmoid mode).

## Calling Instructions

| Calling Method | Sample Code | Description |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn API | [test_aclnn_moe_gating_top_k_backward](examples/test_aclnn_moe_gating_top_k_backward.cpp) | Calls the MoeGatingTopKBackward operator through the [aclnnMoeGatingTopKBackward](docs/aclnnMoeGatingTopKBackward_en.md) API. |
