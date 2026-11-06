# MhcPre

English | [简体中文](./README.md)

## Product Support

| Product                                                     | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                      |    √     |
| <term>Atlas A3 products</term>     |    √     |
| <term>Atlas A2 products</term>     |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                               |    ×     |
| <term>Atlas training products</term>                               |    ×     |

## Function Description

- **Operator function**: Obtains the $H^{res}$ and $H^{post}$ projection matrices of the hidden layer in the MHC architecture and the input matrix $h^{in}$ of the Attention or MLP layer through a series of computations.

- **Computation formula**:

  Here, xFlat denotes treating the last two dimensions n and D of x as a vector of length nD, gammaFlat denotes treating gamma as a vector of length nD, @ denotes matrix multiplication, and $\odot$ denotes element-wise multiplication.

  $$
  \begin{aligned}
  invRms &= \left(mean(xFlat^{2}) + normEps\right)^{-\frac{1}{2}}\\
  xGamma &= \begin{cases}
      xFlat \odot gammaFlat, & gamma \ne null \\
      xFlat, & gamma = null
  \end{cases}\\
  hMix &= xGamma @ phi^{T}\\
  w &= hMix \odot invRms\\
  (pPre, pPost, pRes) &= \begin{cases}
      split(w, (n, n, n^{2})), & alpha.shape=(3) \\
      (split(w, (n, n)), 0), & alpha.shape=(2)
  \end{cases}\\
  hPre &= \sigma(pPre \odot alpha0 + bias0) + hcEps\\
  hPost &= \begin{cases}
      2\sigma(pPost \odot alpha1 + bias1), & alpha.shape=(3) \\
      2\sigma(pPost \odot alpha1 + bias1) + hcEps, & alpha.shape=(2)
  \end{cases}\\
  hRes &= \begin{cases}
      pRes \odot alpha2 + bias2, & alpha.shape=(3) \\
      0, & alpha.shape=(2)
  \end{cases}\\
  hIn_{d} &= \sum_{i=0}^{n-1} hPre_{i} x_{i,d}
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
        <td>Data to be computed, representing the input data of the mHC layer in the network, corresponding to x in the formulas.</td>
        <td>BFLOAT16, FLOAT16</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>phi</td>
        <td>Input</td>
        <td>Parameter matrix of mHC, corresponding to phi in the formulas.</td>
        <td>FLOAT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>alpha</td>
        <td>Input</td>
        <td>Scaling parameter of mHC, corresponding to alpha in the formulas.</td>
        <td>FLOAT32</td>
        <td>-</td>
      </tr>
      <tr>
        <td>bias</td>
        <td>Input</td>
        <td>Bias parameter of mHC, corresponding to bias in the formulas.</td>
        <td>FLOAT32</td>
        <td>-</td>
      </tr>
      <tr>
        <td>gamma</td>
        <td>Optional input</td>
        <td>Scaling factor for the RmsNorm computation, corresponding to gamma in the formulas.</td>
        <td>FLOAT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>out_flag</td>
        <td>Optional input</td>
        <td>Specifies whether to output invRms, hMix, and hPre in the formulas. The default value 0 indicates no output, and 1 indicates that all values are output.</td>
        <td>INT64</td>
        <td>-</td>
      </tr>
      <tr>
        <td>norm_eps</td>
        <td>Optional input</td>
        <td>Division-by-zero prevention parameter of RmsNorm, corresponding to normEps in the formulas.</td>
        <td>FLOAT</td>
        <td>-</td>
      </tr>
      <tr>
        <td>hc_eps</td>
        <td>Optional input</td>
        <td>eps parameter after the sigmoid of h_pre, and the eps parameter after the sigmoid of h_post when alpha.shape=(2), corresponding to hcEps in the formulas.</td>
        <td>FLOAT</td>
        <td>-</td>
      </tr>
      <tr>
        <td>op_impl_mode</td>
        <td>Optional input</td>
        <td>Specifies the computation mode of the MhcPre operator. 0 indicates that the Cube uses FP32 mode for computation, and 1 indicates that the Cube uses HF32 mode for computation. The default value is 0.</td>
        <td>INT64</td>
        <td>-</td>
      </tr>
      <tr>
        <td>h_in</td>
        <td>Output</td>
        <td>The output h_in is used as the input of the Attention/MLP layer, corresponding to hIn in the formulas.</td>
        <td>BFLOAT16, FLOAT16</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>h_post</td>
        <td>Output</td>
        <td>Output h_post transformation matrix of mHC, corresponding to hPost in the formulas.</td>
        <td>FLOAT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>h_res</td>
        <td>Output</td>
        <td>Output h_res transformation matrix of mHC (without the sinkhorn transformation), corresponding to hRes in the formulas.</td>
        <td>FLOAT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>inv_rms</td>
        <td>Optional output</td>
        <td>1/r obtained from the RmsNorm computation, corresponding to invRms in the formulas.</td>
        <td>FLOAT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>h_mix</td>
        <td>Optional output</td>
        <td>Result of the matrix multiplication of xGamma and phi, corresponding to hMix in the formulas.</td>
        <td>FLOAT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>h_pre</td>
        <td>Optional output</td>
        <td>h_pre matrix after the sigmoid computation, corresponding to hPre in the formulas.</td>
        <td>FLOAT32</td>
        <td>ND</td>
      </tr>
    </tbody>
  </table>

## Constraints

- <term>Ascend 950PR&950DT products</term>:
  - n currently supports 4, 6, and 8.
  - D must be in the range of 1 to 16384 and aligned to 16.
- <term>Atlas A3 products</term> and <term>Atlas A2 products</term>:
  - The parameter `op_impl_mode` supports only the value 0.
  - n currently supports 4.
  - D must be within 100000 and aligned to 128.

## Calling Instructions

| Calling Method      | Calling Sample                 | Description                                                         |
|--------------|-------------------------|--------------------------------------------------------------|
| aclnn invocation | [test_aclnn_mhc_pre](examples/test_aclnn_mhc_pre.cpp) | Calls the MhcPre operator through the [aclnnMhcPre](docs/aclnnMhcPre_en.md) API. |
| aclnn invocation | [test_aclnn_mhc_pre_v2](examples/test_aclnn_mhc_pre_v2.cpp) | Calls the MhcPre operator through the [aclnnMhcPreV2](docs/aclnnMhcPreV2_en.md) API, and selects the FP32 or HF32 computation mode of the Cube through opImplMode. |
