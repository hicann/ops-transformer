# MhcPreSinkhornBackward

English | [简体中文](./README.md)

## Product Support

| Product                                                         | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                             |    √     |
| <term>Atlas A3 products</term>     |    √     |
| <term>Atlas A2 products</term> |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                             |    ×     |
| <term>Atlas training products</term>                              |    ×     |

## Function Description

- **Operator function**: `MhcPreSinkhornBackward` is the backward operator of `MhcPreSinkhorn` and computes backward gradient propagation of the Sinkhorn transformation in the mHC (Manifold-Constrained Hyper-Connections) structure.
  - <term>Atlas A3 products</term>, <term>Atlas A2 products</term>: The API uses the HF32 computation mode in the Cube.
  - <term>Ascend 950PR&950DT products</term>: The API uses the FP32 computation mode in the Cube.

- **Main outputs**: `gradX`, `gradPhi`, `gradAlpha`, and `gradBias`.
- **Forward cache dependency**: `hPre`, `hcBeforeNorm`, `invRms`, `sumOut`, and `normOut`.
- **Computation formula**:

  $$
  \begin{aligned}
  gradX &= \nabla_{x}(\text{MhcPreSinkhorn}(x, \phi, \alpha, \text{bias})) \\
  gradPhi &= \nabla_{\phi}(\text{MhcPreSinkhorn}(x, \phi, \alpha, \text{bias})) \\
  gradAlpha &= \nabla_{\alpha}(\text{MhcPreSinkhorn}(x, \phi, \alpha, \text{bias})) \\
  gradBias &= \nabla_{\text{bias}}(\text{MhcPreSinkhorn}(x, \phi, \alpha, \text{bias}))
  \end{aligned}
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
        <td>gradHin</td>
        <td>Input</td>
        <td>Gradient of the output h_in.</td>
        <td>FLOAT16, BFLOAT16</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>gradHPost</td>
        <td>Input</td>
        <td>Gradient of the output h_post.</td>
        <td>FLOAT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>gradHRes</td>
        <td>Input</td>
        <td>Gradient of the output h_res.</td>
        <td>FLOAT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>x</td>
        <td>Input</td>
        <td>Forward input x.</td>
        <td>FLOAT16, BFLOAT16</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>phi</td>
        <td>Input</td>
        <td>Forward parameter phi.</td>
        <td>FLOAT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>alpha</td>
        <td>Input</td>
        <td>Forward parameter alpha.</td>
        <td>FLOAT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>bias</td>
        <td>Input</td>
        <td>Forward parameter bias.</td>
        <td>FLOAT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>hPre</td>
        <td>Input</td>
        <td>Intermediate result h_pre saved in the forward pass.</td>
        <td>FLOAT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>hcBeforeNorm</td>
        <td>Input</td>
        <td>Intermediate result hc_before_norm saved in the forward pass.</td>
        <td>FLOAT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>invRms</td>
        <td>Input</td>
        <td>Intermediate result inv_rms saved in the forward pass.</td>
        <td>FLOAT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>sumOut</td>
        <td>Input</td>
        <td>Intermediate sum result saved during the forward computation of the Sinkhorn transformation.</td>
        <td>FLOAT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>normOut</td>
        <td>Input</td>
        <td>Intermediate norm result saved during the forward computation of the Sinkhorn transformation.</td>
        <td>FLOAT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>hcEps</td>
        <td>Attribute</td>
        <td>Numerical stability parameter. The default value is 1e-6.</td>
        <td>DOUBLE</td>
        <td>-</td>
      </tr>
      <tr>
        <td>gradX</td>
        <td>Output</td>
        <td>Gradient of input x.</td>
        <td>FLOAT16, BFLOAT16</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>gradPhi</td>
        <td>Output</td>
        <td>Gradient of parameter phi.</td>
        <td>FLOAT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>gradAlpha</td>
        <td>Output</td>
        <td>Gradient of parameter alpha.</td>
        <td>FLOAT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>gradBias</td>
        <td>Output</td>
        <td>Gradient of parameter bias.</td>
        <td>FLOAT32</td>
        <td>ND</td>
      </tr>
    </tbody>
  </table>

## Constraints

- **Dimension Format Constraints**

  - <term>Ascend 950PR&950DT products</term>: Input supports two dimension formats: BSND (4D) and TND (3D). T = B × S indicates the cumulative sum of the sequence lengths of all batches. All inputs with B/S dimensions must use the same dimension format (either all 4D BSND or all 3D TND).
  - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>: Input supports only the BSND (4D) format.

  The shape mapping of each input/output in the two formats is as follows:

  | Parameter | BSND (4D) | TND (3D) |
  | :--- | :--- | :--- |
  | gradHin | (B,S,C) | (T,C) |
  | gradHPost | (B,S,N) | (T,N) |
  | gradHRes | (B,S,N*N) or (B,S,N,N) | (T,N*N) or (T,N,N) |
  | x | (B,S,N,C) | (T,N,C) |
  | hPre | (B,S,N) | (T,N) |
  | hcBeforeNorm | (B,S,N*N+2*N) | (T,N*N+2*N) |
  | invRms | (B,S,1) | (T,1) |
  | sumOut | (2*sk_iter_count,B,S,N) | (2*sk_iter_count,T,N) |
  | normOut | (2*sk_iter_count,B,S,N,N) | (2*sk_iter_count,T,N,N) |
  | gradX | (B,S,N,C) | (T,N,C) |

  The shapes of phi and gradPhi are (2N+N^2, N*C) and are not affected by the dimension format.

- **Specification Constraints**

  - N (for Ascend 950PR&950DT products): The N dimension supports values greater than 0 and not greater than 8.
  - N (for <term>Atlas A2 products</term>, <term>Atlas A3 products</term>): The N dimension supports only 4.
  - C: must be greater than 0, less than 100000, and divisible by 128.
  - sk_iter_count: The number of Sinkhorn iterations currently supports only 20.

- **Data Type and Shape Constraints**

  - The data type of gradHin must be the same as that of x. FLOAT16 and BFLOAT16 are supported.
  - The shape of gradX must be the same as that of x, and the data type must be the same as that of gradHin.
  - The shapes of gradPhi, gradAlpha, and gradBias must be the same as those of phi, alpha, and bias, respectively.
  - The data types of the other input and output tensors must be FLOAT32.

## Calling Instructions

| Calling Method | Calling Sample | Description |
| :--- | :--- | :--- |
| aclnn invocation | [test_aclnn_mhc_pre.cpp](examples/test_aclnn_mhc_pre_sinkhorn_backward.cpp) | Calls the MhcPreSinkhornBackward operator through the [aclnnMhcPreSinkhornBackward](docs/aclnnMhcPreSinkhornBackward_en.md) API. |
