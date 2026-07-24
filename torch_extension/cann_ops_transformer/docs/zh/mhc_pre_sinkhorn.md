# mhc_pre_sinkhorn

## 产品支持情况

- <term>Ascend 950PR/Ascend 950DT</term>：支持
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：支持
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：支持
- <term>Atlas 200I/500 A2 推理产品</term>：不支持
- <term>Atlas 推理系列产品</term>：不支持
- <term>Atlas 训练系列产品</term>：不支持

## 功能说明

- 接口功能：

  基于一系列计算得到MHC架构中hidden层的$\mathbf{H}'_{\text{res}}$和$\mathbf{H}_{\text{post}}$投影矩阵以及Attention或MLP层的输入矩阵$\mathbf{h}_{\text{in}}$。对$\mathbf{H}'_{\text{res}}$矩阵执行Sinkhorn迭代归一化变换，最终得到双随机矩阵$\mathbf{H}_{\text{res}}$；支持输出中间计算结果，用于反向梯度计算。

- 计算公式：

  $$
  \begin{aligned}
  \vec{x^{'}_{l}} &= \frac{1}{\sqrt{\frac{1}{d} \sum_{\dim=-2,\text{keepdim}=\text{True}} x_i^2 + \epsilon_{norm}}}\\
  H^{pre}_l &= \alpha^{pre}_{l} \cdot(\vec{x^{'}_{l}}\varphi^{pre}_{l}) + b^{pre}_{l}\\
  H^{post}_l &= \alpha^{post}_{l} \cdot(\vec{x^{'}_{l}}\varphi^{post}_{l}) + b^{post}_{l}\\
  H^{res}_l &= \alpha^{res}_{l} \cdot(\vec{x^{'}_{l}}\varphi^{res}_{l}) + b^{res}_{l}\\
  H^{pre}_l &= \sigma (H^{pre}_{l})\\
  H^{post}_l &= 2\sigma (H^{post}_{l})\\
  h_{in} &=\vec{x_{l}}H^{pre}_l
  \end{aligned}
  $$

  将$\mathbf{H^{res}_l}$作为输入，Sinkhorn变换共执行$\mathbf{numIters}$次迭代，迭代过程中生成中间归一化结果$\mathbf{normOut}[k]$和求和结果$\mathbf{sumOut}[k]$，最终输出最后一次迭代的$\mathbf{normOut}$作为变换结果。

  第一次迭代（初始化）：

  $$
  \begin{aligned}
      \mathbf{normOut}[0] &= \text{softmax}(\mathbf{H^{res}_l},  \dim=-1) + \epsilon_{hc}, \\
      \mathbf{sumOut}[1] &= \sum_{\dim=-2,\text{keepdim}=\text{True}} \mathbf{normOut}[0] + \epsilon_{hc}, \\
      \mathbf{normOut}[1] &= \frac{\mathbf{normOut}[0]}{\mathbf{sum\_out}[1]}, \\
  \end{aligned}
  $$

  第$i$次迭代（$i = 1, 2, \dots, \mathbf{num\_iters}-1$）：

  $$
  \begin{aligned}
      \mathbf{sumOut}[2i] &= \sum_{\dim=-1,\text{keepdim}=\text{True}} \mathbf{normOut}[2i-1] + \epsilon_{hc}, \\
      \mathbf{normOut}[2i] &= \frac{\mathbf{normOut}[2i-1]}{\mathbf{sum\_out}[2i]}, \\
      \mathbf{sumOut}[2i+1] &= \sum_{\dim=-2,\text{keepdim}=\text{True}} \mathbf{normOut}[2i] + \epsilon_{hc}, \\
      \mathbf{normOut}[2i+1] &= \frac{\mathbf{normOut}[2i]}{\mathbf{sum\_out}[2i+1]}, \\
  \end{aligned}
  $$

  最终输出：

  $$
  \mathbf{normOut}[2 \times \mathbf{num\_iters} - 1]
  $$

  $$
  \mathbf{sumOut}[2 \times \mathbf{num\_iters} - 1]
  $$

## 函数原型

```python
cann_ops_transformer.mhc_pre_sinkhorn(x, phi, alpha, bias, hcMult, numIters, hcEps, normEps, outFlag) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)
```

## 参数说明

<table style="undefined;table-layout: fixed; width:1200px"><colgroup>
<col style="width: 120px">
<col style="width: 120px">
<col style="width: 100px">
<col style="width: 300px">
<col style="width: 120px">
<col style="width: 200px">
</colgroup>
<thead>
<tr>
    <th>参数名</th>
    <th>参数类型</th>
    <th>可选/必选</th>
    <th>描述</th>
    <th>数据类型</th>
    <th>维度(shape)</th>
</tr>
</thead>
<tbody>
    <tr>
        <td>x</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>MHC层的输入数据。对应公式中x。</td>
        <td>bfloat16</td>
        <td>(B, S, N, C)</td>
    </tr>
    <tr>
        <td>phi</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>MHC的参数矩阵。对应公式中ψ<sup>pre</sup>、ψ<sup>post</sup>、ψ<sup>res</sup>。使用时对N轴进行拆分，分别是(N, N*C), (N, N*C), (N*N, N*C)。</td>
        <td>float32</td>
        <td>(N*N+2*N, N*C)</td>
    </tr>
    <tr>
        <td>alpha</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>MHC的缩放参数，包含3个元素分别对应H<sup>pre</sup>、H<sup>post</sup>、H<sup>res</sup>的缩放系数。对应公式中α<sup>pre</sup>、α<sup>post</sup>、α<sup>res</sup>。</td>
        <td>float32</td>
        <td>(3)</td>
    </tr>
    <tr>
        <td>bias</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>MHC的bias参数，包含2N+N²个元素。对应公式中β<sup>pre</sup>、β<sup>post</sup>、β<sup>res</sup>，分别对应N, N, N*N。</td>
        <td>float32</td>
        <td>(2N+N²)</td>
    </tr>
    <tr>
        <td>hcMult</td>
        <td>int</td>
        <td>必选</td>
        <td>残差流数量，HC维度大小。</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>numIters</td>
        <td>int</td>
        <td>必选</td>
        <td>Sinkhorn算法迭代次数。</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>hcEps</td>
        <td>float</td>
        <td>可选</td>
        <td>H<sup>pre</sup>的sigmoid后的eps参数。对应公式中ε<sub>hc</sub>。默认值1e-6。</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>normEps</td>
        <td>float</td>
        <td>可选</td>
        <td>RmsNorm的防除零参数。对应公式中ε<sub>norm</sub>。默认值1e-6。</td>
        <td>-</td>
        <td>-</td>
    </tr>
    </tbody>
    </table>

## 输出说明

<table style="undefined;table-layout: fixed; width:1200px"><colgroup>
<col style="width: 120px">
<col style="width: 120px">
<col style="width: 100px">
<col style="width: 300px">
<col style="width: 120px">
<col style="width: 200px">
</colgroup>
<thead>
<tr>
    <th>参数名</th>
    <th>参数类型</th>
    <th>可选/必选</th>
    <th>描述</th>
    <th>数据类型</th>
    <th>维度(shape)</th>
</tr>
</thead>
<tbody>
    <tr>
        <td>hin</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>输出的h_in，作为Attention/MLP层的输入。对应公式中h<sub>in</sub>。</td>
        <td>bfloat16</td>
        <td>(B, S, C)</td>
    </tr>
    <tr>
        <td>hPost</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>输出的MHC的h_post变换矩阵。对应公式中H<sup>post</sup>。</td>
        <td>float32</td>
        <td>(B, S, N)</td>
    </tr>
    <tr>
        <td>hRes</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>输出的MHC的h_res变换矩阵（Sinkhorn归一化后的双随机矩阵）。对应公式中H<sup>res</sup>。</td>
        <td>float32</td>
        <td>(B, S, N*N)</td>
    </tr>
</tbody>
</table>

## 约束说明

- 该接口支持训练、推理场景下使用
- 该接口支持单算子模式调用
- 规格约束：

  | 规格项   | 规格               | 规格说明                               |
  | :------- | :----------------- | :------------------------------------- |
  | numIters | 20                 | 迭代次数超出该范围会返回参数无效错误。 |
  | N        | 4                  | 目前只支持4。                          |
  | C        | A2/A3：128对齐，[1, 100000]；A5：4096、7168 | 尾轴C的取值与芯片版本相关：<term>Atlas A2/A3</term>需满足128对齐且取值范围为[1, 100000]；<term>Ascend 950PR/Ascend 950DT</term>仅支持4096、7168。 |

- 参数约束：
  - x不支持空Tensor。
  - phi不支持空Tensor。
  - alpha不支持空Tensor。
  - bias不支持空Tensor。

## 确定性计算

默认支持确定性计算。

## 调用说明

- 单算子模式调用：

  ```python
  import torch
  import torch_npu
  from cann_ops_transformer.ops import mhc_pre_sinkhorn

  B = 1
  S = 1024
  N = 4
  C = 4096

  x = torch.randn(B, S, N, C, dtype=torch.bfloat16).npu()
  phi = torch.randn(N * N + 2 * N, N * C, dtype=torch.float32).npu()
  alpha = torch.randn(3, dtype=torch.float32).npu()
  bias = torch.randn(N * N + 2 * N, dtype=torch.float32).npu()

  hcMult = 4
  numIters = 20
  hcEps = 1e-6
  normEps = 1e-6

  hin, hPost, hRes = mhc_pre_sinkhorn(
    x, phi, alpha, bias, hcMult, numIters, hcEps, normEps
  )
  ```
