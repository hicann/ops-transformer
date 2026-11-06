# MhcSinkhornBackward

English | [简体中文](./README.md)

## Product Support Status

|Product      | Supported |
|:----------------------------|:-----------:|
|<term>Ascend 950PR&950DT products</term>|      √     |
|<term>Atlas A3 products</term>|      ×     |
|<term>Atlas A2 products</term>|      ×     |
|<term>Atlas 200I/500 A2 inference products</term>|      ×     |
|<term>Atlas inference products</term>|      ×     |
|<term>Atlas training products</term>|      ×     |

## Function Description

- Operator function: MhcSinkhornBackward is the backward operator of MhcSinkhorn. In the mHC (Manifold-Constrained Hyper-Connections) architecture, the MhcSinkhorn operator performs the Sinkhorn transformation on the input matrix to obtain the doubly stochastic matrix $\mathbf{H}_{\text{res}}$. All elements of the output doubly stochastic matrix are ≥ 0, the sum of each row is 1, and the sum of each column is 1 (it has three major characteristics: norm preservation, composition closure, and geometric interpretation of convex combinations). The backward computation of the Sinkhorn transformation is performed on the gradient of the doubly stochastic matrix $\mathbf{H}_{\text{res}}$ in the mHC architecture to obtain the gradient of the input $\mathbf{H}'_{\text{res}}$.

- Calculation Formula:

  - The Sinkhorn-Knopp algorithm achieves doubly stochastic projection through $\mathbf{num\_iters}$ iterative normalizations in the forward computation. In the iterative computation of backward propagation:

  - First $\mathbf{num\_iters}-1$ iterations:

    $$
    \begin{aligned}
      \mathbf{dot\_prod}_{2i+1} &= \sum_{\dim=-2,\text{keepdim}=\text{True}} (\mathbf{grad}_{curr}\ {⋅}\ \mathbf{norm\_out}_{2i+1}),  \\
      \mathbf{grad}_{curr} &← \frac{\mathbf{grad}_{curr} - \mathbf{dot\_prod}_{2i+1}}{\mathbf{sum\_out}_{2i+1}}, \\
      \mathbf{dot\_prod}_{2i} &= \sum_{\dim=-1,\text{keepdim}=\text{True}} (\mathbf{grad}_{curr}\ {⋅}\ \mathbf{norm\_out}_{2i}),  \\
      \mathbf{grad}_{curr} &← \frac{\mathbf{grad}_{curr} - \mathbf{dot\_prod}_{2i}}{\mathbf{sum\_out}_{2i}}, \\
    \end{aligned}
    $$

  - Last iteration:

    $$
    \begin{aligned}
      \mathbf{dot\_prod}_{1} &= \sum_{\dim=-2,\text{keepdim}=\text{True}} (\mathbf{grad}_{curr}\ {⋅}\ \mathbf{norm\_out}_{1}),  \\
      \mathbf{grad}_{curr} &← \frac{\mathbf{grad}_{curr} - \mathbf{dot\_prod}_{1}}{\mathbf{sum\_out}_{1}}, \\
      \mathbf{dot\_prod}_{0} &= \sum_{\dim=-1,\text{keepdim}=\text{True}} (\mathbf{grad}_{curr}\  {⋅}\  \mathbf{norm\_out}_{0}),  \\
      \mathbf{grad}_{input} &← ({\mathbf{grad}_{curr} - \mathbf{dot\_prod}_{0}})\  {⋅}\  \mathbf{norm\_out}_{0} \\
    \end{aligned}
    $$

  - where $\mathbf{grad}_\text{curr}$ is the initial gradient, $\mathbf{grad}_\text{input}$ is the output gradient, $\mathbf{norm\_out}_\text{k}$ is the normalization direction vector of the $k$-th iteration, and $\mathbf{sum\_out}_\text{k}$ is the corresponding scaling coefficient.

## Parameter Description

<table style="table-layout: auto; width: 100%"><colgroup>
<col style="white-space: nowrap">
<col style="white-space: nowrap">
<col style="white-space: nowrap">
<col style="white-space: nowrap">
<col style="white-space: nowrap">
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
    <td>grad_y</td>
    <td>Input</td>
    <td>Gradient of H_res output by the Sinkhorn transformation.</td>
    <td>FLOAT32</td>
    <td>ND</td>
</tr>
<tr>
    <td>norm</td>
    <td>Input</td>
    <td>Intermediate norm result saved during the forward computation of the Sinkhorn transformation.</td>
    <td>FLOAT32</td>
    <td>ND</td>
</tr>
<tr>
    <td>sum</td>
    <td>Input</td>
    <td>Intermediate sum result saved during the forward computation of the Sinkhorn transformation.</td>
    <td>FLOAT32</td>
    <td>ND</td>
</tr>
<tr>
    <td>grad_input</td>
    <td>Output</td>
    <td>Gradient of the H_res input to the Sinkhorn transformation.</td>
    <td>FLOAT32</td>
    <td>ND</td>
</tr>
</tbody>
</table>

## Constraints

- The input grad_y supports only 3D (T,n,n) or 4D (B,S,n,n).
- The input norm supports only 1D (2\*num_iters\*n\*align_n\*B\*S) or (2\*num_iters\*n\*align_n\*T).
- The input sum supports only 1D (2\*num_iters\*align_n\*B\*S) or (2\*num_iters\*align_n\*T).
- num_iters: The value range is 1~100. If the value is out of this range, an invalid parameter error is reported.
- n: Only 4, 6, or 8 is supported.
- align_n: The fixed value is 8.

## Calling Instructions

| Calling Method      | Calling Sample                 | Description                                                         |
|--------------|-------------------------|--------------------------------------------------------------|
| aclnn calling | [test_aclnn_mhc_sinkhorn_backward](examples/test_aclnn_mhc_sinkhorn_backward.cpp) | Calls the MhcSinkhornBackward operator through the [aclnnMhcSinkhornBackward](docs/aclnnMhcSinkhornBackward_en.md) API. |
