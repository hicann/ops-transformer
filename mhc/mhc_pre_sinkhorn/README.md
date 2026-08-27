# MhcPreSinkhorn

## 产品支持情况

| 产品                                                     | 是否支持 |
| :------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                   |    √    |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> |    √    |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √    |
| <term>Atlas 200I/500 A2 推理产品</term>                  |    ×    |
| <term>Atlas 推理系列产品</term>                          |    ×    |
| <term>Atlas 训练系列产品</term>                          |    ×    |

## 功能说明

- **算子功能**：基于一系列计算得到MHC架构中hidden层的$\mathbf{H}'_{\text{res}}$和$\mathbf{H}_{\text{post}}$投影矩阵以及Attention或MLP层的输入矩阵$\mathbf{h}_{\text{in}}$。对$\mathbf{H}'_{\text{res}}$矩阵执行Sinkhorn迭代归一化变换，最终得到双随机矩阵$\mathbf{H}_{\text{res}}$；支持输出中间计算结果，用于反向梯度计算。包括sigmoid计算之后的$\mathbf{H^{pre}_l}$矩阵、$\vec{x^{'}_{l}}$与$\mathbf{\varphi}$矩阵乘的结果，输入x的RmsNorm结果$\mathbf{\vec{x^{'}_{l}}}$、迭代过程中的中间归一化结果和$\mathbf{normOut}$和求和结果$\mathbf{sumOut}$。

- **计算公式**：

  $$
  \begin{aligned}
  \vec{x^{'}_{l}} &= \frac{1}{\sqrt{\frac{1}{d} \sum_{\dim=-2,\text{keepdim}=\text{True}} x_i^2 + \epsilon}}\\
  H^{pre}_l &= \alpha^{pre}_{l} ·(\vec{x^{'}_{l}}\varphi^{pre}_{l}) + b^{pre}_{l}\\
  H^{post}_l &= \alpha^{post}_{l} ·(\vec{x^{'}_{l}}\varphi^{post}_{l}) + b^{post}_{l}\\
  H^{res}_l &= \alpha^{res}_{l} ·(\vec{x^{'}_{l}}\varphi^{res}_{l}) + b^{res}_{l}\\
  H^{pre}_l &= \sigma (H^{pre}_{l})\\
  H^{post}_l &= 2\sigma (H^{post}_{l})\\
  h_{in} &=\vec{x_{l}}H^{pre}_l
  \end{aligned}
  $$

  - 将$\mathbf{H^{res}_l}$作为输入，Sinkhorn变换共执行$\mathbf{numIters}$次迭代，迭代过程中生成中间归一化结果$\mathbf{normOut}[k]$和求和结果$\mathbf{sumOut}[k]$，最终输出最后一次迭代的$\mathbf{normOut}$作为变换结果。

    第一次迭代（初始化）：

    $$
    \begin{aligned}
        \mathbf{normOut}[0] &= \text{softmax}(\mathbf{H^{res}_l},  \dim=-1) + \epsilon, \\
        \mathbf{sumOut}[1] &= \sum_{\dim=-2,\text{keepdim}=\text{True}} \mathbf{normOut}[0] + \epsilon, \\
        \mathbf{normOut}[1] &= \frac{\mathbf{normOut}[0]}{\mathbf{sum\_out}[1]}, \\
    \end{aligned}
    $$

    第$i$次迭代（$i = 1, 2, \dots, \mathbf({num\_iters}-1)$）：

    $$
    \begin{aligned}
        \mathbf{sumOut}[2i] &= \sum_{\dim=-1,\text{keepdim}=\text{True}} \mathbf{normOut}[2i-1] + \epsilon, \\
        \mathbf{normOut}[2i] &= \frac{\mathbf{normOut}[2i-1]}{\mathbf{sum\_out}[2i]}, \\
        \mathbf{sumOut}[2i+1] &= \sum_{\dim=-2,\text{keepdim}=\text{True}} \mathbf{normOut}[2i] + \epsilon, \\
        \mathbf{normOut}[2i+1] &= \frac{\mathbf{normOut}[2i]}{\mathbf{sum\_out}[2i+1]}, \\
    \end{aligned}
    $$

  - 最终输出

  $$
  \mathbf{normOut}[2 \times \mathbf{num\_iters} - 1]
  $$

  $$
  \mathbf{sumOut}[2 \times \mathbf{num\_iters} - 1]
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
        <td>待计算数据，表示网络中mHC层的输入数据。</td>
        <td>BFLOAT16, FLOAT16</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>phi</td>
        <td>输入</td>
        <td>mHC的参数矩阵。</td>
        <td>FLOAT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>alpha</td>
        <td>输入</td>
        <td>mHC的缩放参数。</td>
        <td>FLOAT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>bias</td>
        <td>输入</td>
        <td>mHC的bias参数。</td>
        <td>FLOAT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>hcMult</td>
        <td>可选属性</td>
        <td>残差流数量，HC维度大小，当前仅支持4。</td>
        <td>INT32</td>
        <td>-</td>
      </tr>
      <tr>
        <td>numIters</td>
        <td>可选属性</td>
        <td>表示sinkhorn算法迭代次数，当前仅支持20。</td>
        <td>INT32</td>
        <td>-</td>
      </tr>
      <tr>
        <td>hcEps</td>
        <td>可选属性</td>
        <td>h_pre的sigmoid后的eps参数。</td>
        <td>DOUBLE</td>
        <td>-</td>
      </tr>
      <tr>
        <td>normEps</td>
        <td>可选属性</td>
        <td>RmsNorm的防除零参数。</td>
        <td>DOUBLE</td>
        <td>-</td>
      </tr>
      <tr>
        <td>needBackward</td>
        <td>可选属性</td>
        <td>是否需要输出额外属性。</td>
        <td>BOOL</td>
        <td>-</td>
      </tr>
      <tr>
        <td>hIn</td>
        <td>输出</td>
        <td>输出的h_in作为Attention/MLP层的输入。</td>
        <td>BFLOAT16, FLOAT16</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>hPost</td>
        <td>输出</td>
        <td>输出的mHC的h_post变换矩阵。</td>
        <td>FLOAT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>hRes</td>
        <td>输出</td>
        <td>输出的mHC的h_res变换矩阵。</td>
        <td>FLOAT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>hPre</td>
        <td>输出</td>
        <td>需要反向时输出，做完sigmoid计算之后的hPre矩阵。needBackward为false时，输出无效。</td>
        <td>FLOAT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>hcBeforeNorm</td>
        <td>输出</td>
        <td>需要反向时输出，x与phi矩阵乘的结果。needBackward为false时，输出无效。</td>
        <td>FLOAT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>invRms</td>
        <td>输出</td>
        <td>需要反向时输出，RmsNorm计算得到的1/r。needBackward为false时，输出无效。</td>
        <td>FLOAT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>sumOut</td>
        <td>输出</td>
        <td>需要反向时输出，每一次迭代的colSum/rowSum结果。needBackward为false时，输出无效。</td>
        <td>FLOAT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>normOut</td>
        <td>输出</td>
        <td>需要反向时输出，每一次colSum/rowSum迭代后的comb结果。needBackward为false时，输出无效。</td>
        <td>FLOAT32</td>
        <td>ND</td>
      </tr>
    </tbody>
  </table>

## 约束说明

- n目前支持4。
- numIters目前仅支持20。

- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：
  - 输入x仅支持4维BSND格式，shape为(bs, seq_len, n, c)。
  - 输入x的最后一维c需满足128对齐，且取值范围为[1, 100000]。
- <term>Ascend 950PR/Ascend 950DT</term>：
  - 输入x支持4维BSND格式和3维TND格式，shape分别为(bs, seq_len, n, c)和(t, n, c)。
  - TND格式下，t表示所有Batch序列长度的累加和。
  - 输入x的最后一维c仅支持4096、7168。
  - 输入phi的数据范围限定在$\pm \frac{1}{\sqrt{nc}}$范围内，eps限定在1e-6，此范围内具有较好的数值稳定性。

## 调用说明

| 调用方式      | 调用样例                 | 说明                                                         |
|--------------|-------------------------|--------------------------------------------------------------|
| aclnn调用 | [test_aclnn_mhc_pre_sinkhorn](examples/test_aclnn_mhc_pre_sinkhorn.cpp) | 通过[aclnnMhcPreSinkhorn](docs/aclnnMhcPreSinkhorn.md)接口方式调用MhcPreSinkhorn算子。 |
| PyTorch API | - | 通过[cann_ops_transformer.mhc_pre_sinkhorn](../../torch_extension/cann_ops_transformer/docs/zh/mhc_pre_sinkhorn.md)接口方式调用MhcPreSinkhorn算子。 |
