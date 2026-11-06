# MhcPreBackward

English | [简体中文](./README.md)

## Product Support

| Product                                                     | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                             |    √     |
| <term>Atlas A3 products</term>     |    √     |
| <term>Atlas A2 products</term> |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                             |    ×     |
| <term>Atlas training products</term>                              |    ×     |

## Function Description

- **Operator function**: `MhcPreBackward` is the backward operator of `MhcPre` and is used to compute backward gradients in the mHC (Manifold-Constrained Hyper-Connections) structure.
- **Main outputs**: `gradX`, `gradPhi`, `gradAlpha`, and `gradBias`, and `gradGamma` when `gamma != nullptr`.
- **Forward cache dependency**: `invRms`, `hMix`, `hPre`, and `hPost`.
- **Optional inputs**: `gamma` and `gradXPostOptional`.
- **Computation formula**:

  $$
  \begin{aligned}
  gradX &= \nabla_{x}(\text{MhcPre}(x, \phi, \alpha, \gamma)) \\
  gradPhi &= \nabla_{\phi}(\text{MhcPre}(x, \phi, \alpha, \gamma)) \\
  gradAlpha &= \nabla_{\alpha}(\text{MhcPre}(x, \phi, \alpha, \gamma)) \\
  gradBias &= \nabla_{bias}(\text{MhcPre}(x, \phi, \alpha, \gamma))
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
        <td>x</td>
        <td>Input</td>
        <td>Input data of the mHC layer.</td>
        <td>BFLOAT16, FLOAT16</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>phi</td>
        <td>Input</td>
        <td>Parameter matrix of mHC.</td>
        <td>FLOAT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>alpha</td>
        <td>Input</td>
        <td>Scaling parameter of mHC.</td>
        <td>FLOAT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>grad_h_in</td>
        <td>Input</td>
        <td>Gradient of h_in.</td>
        <td>BFLOAT16, FLOAT16</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>grad_h_post</td>
        <td>Input</td>
        <td>Gradient of h_post.</td>
        <td>FLOAT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>grad_h_res</td>
        <td>Input</td>
        <td>Gradient of h_res.</td>
        <td>FLOAT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>inv_rms</td>
        <td>Input</td>
        <td>inv_rms cached in the forward pass.</td>
        <td>FLOAT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>h_mix</td>
        <td>Input</td>
        <td>h_mix cached in the forward pass.</td>
        <td>FLOAT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>h_pre</td>
        <td>Input</td>
        <td>h_pre cached in the forward pass.</td>
        <td>FLOAT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>h_post</td>
        <td>Input</td>
        <td>h_post cached in the forward pass.</td>
        <td>FLOAT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>gamma</td>
        <td>Optional input</td>
        <td>Scaling factor of RMSNorm.</td>
        <td>FLOAT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>grad_x_post</td>
        <td>Optional input</td>
        <td>Accumulated grad_x term from the subsequent path.</td>
        <td>BFLOAT16, FLOAT16</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>hc_eps</td>
        <td>Attribute</td>
        <td>eps parameter used after the sigmoid of h_pre.</td>
        <td>FLOAT32</td>
        <td>-</td>
      </tr>
      <tr>
        <td>grad_x</td>
        <td>Output</td>
        <td>Gradient of x.</td>
        <td>BFLOAT16, FLOAT16</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>grad_phi</td>
        <td>Output</td>
        <td>Gradient of phi.</td>
        <td>FLOAT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>grad_alpha</td>
        <td>Output</td>
        <td>Gradient of alpha.</td>
        <td>FLOAT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>grad_bias</td>
        <td>Output</td>
        <td>Overall gradient of bias.</td>
        <td>FLOAT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>grad_gamma</td>
        <td>Optional output</td>
        <td>Gradient of gamma.</td>
        <td>FLOAT32</td>
        <td>ND</td>
      </tr>
    </tbody>
  </table>

## Constraints

- <term>Ascend 950PR&950DT products</term>:
  - N currently supports 4, 6, and 8.
  - D must be in the range of 1 to 16384 and aligned to 64 elements.
- <term>Atlas A2 products</term>:
  - N currently supports 4.
  - D must be in the range of 1 to 100000 and aligned to 128 elements.

## Calling Instructions

| Calling Method | Calling Sample | Description |
| :--- | :--- | :--- |
| aclnn invocation | [test_aclnn_mhc_pre_backward.cpp](examples/test_aclnn_mhc_pre_backward.cpp) | Calls the MhcPreBackward operator through the [aclnnMhcPreBackward](docs/aclnnMhcPreBackward_en.md) API. |
| aclnn invocation | [test_aclnn_mhc_pre_backward_v2.cpp](examples/test_aclnn_mhc_pre_backward_v2.cpp) | Calls the MhcPreBackward operator through the [aclnnMhcPreBackwardV2](docs/aclnnMhcPreBackwardV2_en.md) API and specifies the Cube computation mode. |
