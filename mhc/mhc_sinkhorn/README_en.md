# MhcSinkhorn

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

- Operator function: MhcSinkhorn is the core operator of the mHC architecture. It performs the Sinkhorn iterative normalization transformation on the initial mixing matrix H'_res of the mHC layer, projects it onto the doubly stochastic matrix manifold (Birkhoff polytope), and generates the doubly stochastic matrix H_res (output y) whose row sums and column sums are all 1. This matrix serves as a key input for the MhcPost operator, stabilizes signal propagation in deep networks, and prevents gradient vanishing or explosion. It also supports outputting the intermediate normalization results (norm_out) and summation results (sum_out) during the iteration for backward gradient computation.

- Calculation Formula:

  The Sinkhorn transformation performs a fixed number of num_iters iterations, and there is no early-stop judgment based on the convergence error. The first iteration first performs softmax normalization on the input along the last dimension, and then column (dim=-1) and row (dim=-2) normalizations are performed alternately. The anti-zero-division parameter eps is added to the sum at each step:

  $$
  \begin{aligned}
      \mathbf{norm\_out}[0] &= \text{softmax}(\mathbf{x}, \dim=-1) + \epsilon, \\
      \mathbf{sum\_out}[1] &= \sum_{\dim=-2,\text{keepdim}=\text{True}} \mathbf{norm\_out}[0] + \epsilon, \\
      \mathbf{norm\_out}[1] &= \frac{\mathbf{norm\_out}[0]}{\mathbf{sum\_out}[1]}, \\
  \end{aligned}
  $$

  The $i$-th iteration ($i = 1, 2, \dots, num\_iters-1$):

  $$
  \begin{aligned}
      \mathbf{sum\_out}[2i] &= \sum_{\dim=-1,\text{keepdim}=\text{True}} \mathbf{norm\_out}[2i-1] + \epsilon, \\
      \mathbf{norm\_out}[2i] &= \frac{\mathbf{norm\_out}[2i-1]}{\mathbf{sum\_out}[2i]}, \\
      \mathbf{sum\_out}[2i+1] &= \sum_{\dim=-2,\text{keepdim}=\text{True}} \mathbf{norm\_out}[2i] + \epsilon, \\
      \mathbf{norm\_out}[2i+1] &= \frac{\mathbf{norm\_out}[2i]}{\mathbf{sum\_out}[2i+1]}, \\
  \end{aligned}
  $$

  Final output:

  $$
  y = \mathbf{norm\_out}[2 \times num\_iters - 1]
  $$

  where $\mathbf{x}$ is the input initial mixing matrix H'_res, $\epsilon$ is the anti-zero-division parameter (attribute eps), $\text{softmax}(\cdot, \dim=-1)$ indicates softmax normalization performed along the last dimension, $\sum_{\dim=d,\text{keepdim}=\text{True}}$ indicates summation along dimension d with the dimension kept, and num_iters is the number of iterations (attribute num_iters).

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
 <th>Parameter Name</th>
 <th>Input/Output/Attribute</th>
 <th>Description</th>
 <th>Data Type</th>
 <th>Data Format</th>
 </tr></thead>
<tbody>
 <tr>
 <td>h_res</td>
 <td>Input</td>
 <td>Initial mixing matrix H'_res of the mHC layer to be transformed, that is, x in the formula. Both the 3D (T, n, n) and 4D (B, S, n, n) batch layouts are supported.</td>
 <td>FLOAT32</td>
 <td>ND</td>
 </tr>
 <tr>
 <td>eps</td>
 <td>Optional attribute</td>
 <td>Anti-zero-division parameter for normalization. This value is added after each summation step to ensure numerical stability. It is ε in the formula. Default value: 1e-6.</td>
 <td>FLOAT32</td>
 <td>Scalar</td>
 </tr>
 <tr>
 <td>num_iters</td>
 <td>Optional attribute</td>
 <td>Number of Sinkhorn iterations. The iteration is performed a fixed number of times based on this value, without early convergence stop. Value range: 1~100. Default value: 20.</td>
 <td>INT64</td>
 <td>Scalar</td>
 </tr>
 <tr>
 <td>out_flag</td>
 <td>Optional attribute</td>
 <td>Whether to output the intermediate iteration results norm_out/sum_out (used for backward gradient computation). The value is 0 or 1. Default value: 0.</td>
 <td>INT64</td>
 <td>Scalar</td>
 </tr>
 <tr>
 <td>y</td>
 <td>Output</td>
 <td>Doubly stochastic matrix H_res after the Sinkhorn transformation. Its shape is the same as that of the input h_res, and it is used as the h_res input of the MhcPost operator.</td>
 <td>FLOAT32</td>
 <td>ND</td>
 </tr>
 <tr>
 <td>norm_out</td>
 <td>Output</td>
 <td>Intermediate normalization results during the iteration, used for backward gradient computation. Output only when out_flag=1.</td>
 <td>FLOAT32</td>
 <td>ND</td>
 </tr>
 <tr>
 <td>sum_out</td>
 <td>Output</td>
 <td>Intermediate summation results during the iteration, used for backward gradient computation. Output only when out_flag=1.</td>
 <td>FLOAT32</td>
 <td>ND</td>
 </tr>
</tbody>
</table>

## Constraints

- The input h_res must be a 3D (T, n, n) or 4D (B, S, n, n) matrix, the last two dimensions must be equal, and n supports only 4, 6, or 8. 2D matrix input is not supported.
- The value range of num_iters is 1~100 (recommended value: 20). The iteration is performed a fixed number of times, without an early stop mechanism based on the error. eps is the anti-zero-division parameter (recommended value: 1e-6), not the convergence threshold.
- When out_flag=1, norm_out/sum_out are 1D tensors, and the numbers of elements are 2\*num\_iters\*n\*n\_align\*T and 2\*num\_iters\*n\_align\*T respectively (n_align is n aligned up to a multiple of 8, and T is dimension 0 of the 3D input or B\*S of the 4D input); when out_flag=0, the shapes of norm_out/sum_out must be {1}.
- Only the hardware environment of Ascend 950PR&950DT products is supported. Other Atlas products are not supported yet.

## Calling Instructions

| Calling Method      | Calling Sample                 | Description                                                         |
|--------------|-------------------------|--------------------------------------------------------------|
| aclnn calling | [test_aclnn_mhc_sinkhorn](examples/test_aclnn_mhc_sinkhorn.cpp) | Calls the operator through [aclnnMhcSinkhorn](docs/aclnnMhcSinkhorn_en.md).|
