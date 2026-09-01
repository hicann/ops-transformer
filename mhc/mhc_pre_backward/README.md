# MhcPreBackward

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×     |

## 功能说明

- **算子功能**：`MhcPreBackward`是`MhcPre`的反向算子，用于计算mHC（Manifold-Constrained Hyper-Connections）结构中的反向梯度。
- **主要输出**：`gradX`、`gradPhi`、`gradAlpha`、`gradBias`，以及在`gamma != nullptr`时输出`gradGamma`。
- **前向缓存依赖**：`invRms`、`hMix`、`hPre`、`hPost`。
- **可选输入**：`gamma`、`gradXPostOptional`。
- **计算公式**：

  $$
  \begin{aligned}
  gradX &= \nabla_{x}(\text{MhcPre}(x, \phi, \alpha, \gamma)) \\
  gradPhi &= \nabla_{\phi}(\text{MhcPre}(x, \phi, \alpha, \gamma)) \\
  gradAlpha &= \nabla_{\alpha}(\text{MhcPre}(x, \phi, \alpha, \gamma)) \\
  gradBias &= \nabla_{bias}(\text{MhcPre}(x, \phi, \alpha, \gamma))
  \end{aligned}
  $$

## 参数说明

  <table style="table-layout: auto; width: 100%">
    <thead>
      <tr>
        <th style="white-space: nowrap">参数名</th>
        <th style="white-space: nowrap">输入/输出/属性</th>
        <th style="white-space: nowrap">描述</th>
        <th style="white-space: nowrap">数据类型</th>
        <th style="white-space: nowrap">数据格式</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>x</td>
        <td>输入</td>
        <td>mHC层输入数据。</td>
        <td>BFLOAT16、FLOAT16</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>phi</td>
        <td>输入</td>
        <td>mHC参数矩阵。</td>
        <td>FLOAT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>alpha</td>
        <td>输入</td>
        <td>mHC缩放参数。</td>
        <td>FLOAT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>grad_h_in</td>
        <td>输入</td>
        <td>对h_in的梯度。</td>
        <td>BFLOAT16、FLOAT16</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>grad_h_post</td>
        <td>输入</td>
        <td>对h_post的梯度。</td>
        <td>FLOAT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>grad_h_res</td>
        <td>输入</td>
        <td>对h_res的梯度。</td>
        <td>FLOAT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>inv_rms</td>
        <td>输入</td>
        <td>前向缓存的inv_rms。</td>
        <td>FLOAT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>h_mix</td>
        <td>输入</td>
        <td>前向缓存的h_mix。</td>
        <td>FLOAT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>h_pre</td>
        <td>输入</td>
        <td>前向缓存的h_pre。</td>
        <td>FLOAT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>h_post</td>
        <td>输入</td>
        <td>前向缓存的h_post。</td>
        <td>FLOAT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>gamma</td>
        <td>可选输入</td>
        <td>RMSNorm缩放因子。</td>
        <td>FLOAT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>grad_x_post</td>
        <td>可选输入</td>
        <td>来自后续路径的grad_x累加项。</td>
        <td>BFLOAT16、FLOAT16</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>hc_eps</td>
        <td>属性</td>
        <td>h_pre sigmoid后使用的eps参数。</td>
        <td>FLOAT32</td>
        <td>-</td>
      </tr>
      <tr>
        <td>grad_x</td>
        <td>输出</td>
        <td>x的梯度。</td>
        <td>BFLOAT16、FLOAT16</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>grad_phi</td>
        <td>输出</td>
        <td>phi的梯度。</td>
        <td>FLOAT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>grad_alpha</td>
        <td>输出</td>
        <td>alpha的梯度。</td>
        <td>FLOAT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>grad_bias</td>
        <td>输出</td>
        <td>bias整体梯度。</td>
        <td>FLOAT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>grad_gamma</td>
        <td>可选输出</td>
        <td>gamma的梯度。</td>
        <td>FLOAT32</td>
        <td>ND</td>
      </tr>
    </tbody>
  </table>

## 约束说明

- <term>Ascend 950PR/Ascend 950DT</term>：
  - N目前支持4、6、8。
  - D支持1~16384，需满足64元素对齐。
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：
  - N目前支持4。
  - D支持1~100000，需满足128元素对齐。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
| :--- | :--- | :--- |
| aclnn调用 | [test_aclnn_mhc_pre_backward.cpp](examples/test_aclnn_mhc_pre_backward.cpp) | 通过[aclnnMhcPreBackward](docs/aclnnMhcPreBackward.md)接口方式调用MhcPreBackward算子。 |
