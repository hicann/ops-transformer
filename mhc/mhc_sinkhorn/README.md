# MhcSinkhorn

## 产品支持情况

|产品      | 是否支持 |
|:----------------------------|:-----------:|
|<term>Ascend 950PR&950DT系列产品</term>|      √     |
|<term>Atlas A3系列产品</term>|      ×     |
|<term>Atlas A2系列产品</term>|      ×     |
|<term>Atlas 200I/500 A2推理产品</term>|      ×     |
|<term>Atlas推理系列产品</term>|      ×     |
|<term>Atlas训练系列产品</term>|      ×     |

## 功能说明

- 算子功能：MhcSinkhorn是mHC架构的核心算子，对mHC层初始混合矩阵H'_res执行Sinkhorn迭代归一化变换，将其投影到双随机矩阵流形（Birkhoff多胞形），生成满足行和、列和均为1的双随机矩阵H_res（输出y），为MhcPost算子提供关键输入，稳定深度网络信号传播、解决梯度消失/爆炸问题。同时支持输出迭代过程中的中间归一化结果（norm_out）和求和结果（sum_out），用于反向梯度计算。

- 计算公式：

  Sinkhorn变换共执行num_iters次固定迭代，不存在按收敛误差提前停止的判断。第一次迭代先对输入按最后一维执行softmax归一化，之后交替执行列（dim=-1）、行（dim=-2）归一化，每步求和均累加防除零参数eps：

  $$
  \begin{aligned}
      \mathbf{norm\_out}[0] &= \text{softmax}(\mathbf{x}, \dim=-1) + \epsilon, \\
      \mathbf{sum\_out}[1] &= \sum_{\dim=-2,\text{keepdim}=\text{True}} \mathbf{norm\_out}[0] + \epsilon, \\
      \mathbf{norm\_out}[1] &= \frac{\mathbf{norm\_out}[0]}{\mathbf{sum\_out}[1]}, \\
  \end{aligned}
  $$

  第$i$次迭代（$i = 1, 2, \dots, num\_iters-1$）：

  $$
  \begin{aligned}
      \mathbf{sum\_out}[2i] &= \sum_{\dim=-1,\text{keepdim}=\text{True}} \mathbf{norm\_out}[2i-1] + \epsilon, \\
      \mathbf{norm\_out}[2i] &= \frac{\mathbf{norm\_out}[2i-1]}{\mathbf{sum\_out}[2i]}, \\
      \mathbf{sum\_out}[2i+1] &= \sum_{\dim=-2,\text{keepdim}=\text{True}} \mathbf{norm\_out}[2i] + \epsilon, \\
      \mathbf{norm\_out}[2i+1] &= \frac{\mathbf{norm\_out}[2i]}{\mathbf{sum\_out}[2i+1]}, \\
  \end{aligned}
  $$

  最终输出：

  $$
  y = \mathbf{norm\_out}[2 \times num\_iters - 1]
  $$

  其中：$\mathbf{x}$为输入的初始混合矩阵H'_res，$\epsilon$为防除零参数（属性eps），$\text{softmax}(\cdot, \dim=-1)$表示在最后一维执行softmax归一化，$\sum_{\dim=d,\text{keepdim}=\text{True}}$表示在维度d上求和并保持维度，num_iters为迭代次数（属性num_iters）。

## 参数说明

<table style="undefined;table-layout: fixed; width: 952px"><colgroup>
<col style="width: 106px">
<col style="width: 87px">
<col style="width: 445px">
<col style="width: 209px">
<col style="width: 105px">
</colgroup>
<thead>
 <tr>
 <th>参数名</th>
 <th>输入/输出/属性</th>
 <th>描述</th>
 <th>数据类型</th>
 <th>数据格式</th>
 </tr></thead>
<tbody>
 <tr>
 <td>h_res</td>
 <td>输入</td>
 <td>待变换的mHC层初始混合矩阵H'_res，即公式中的x。支持3D (T, n, n) 或 4D (B, S, n, n) 批量布局。</td>
 <td>FLOAT32</td>
 <td>ND</td>
 </tr>
 <tr>
 <td>eps</td>
 <td>可选属性</td>
 <td>归一化防除零参数，每步求和后累加该值以保证数值稳定，即公式中的ε。默认1e-6。</td>
 <td>FLOAT32</td>
 <td>标量</td>
 </tr>
 <tr>
 <td>num_iters</td>
 <td>可选属性</td>
 <td>Sinkhorn迭代次数，按该值固定执行迭代，不做收敛提前停止。取值范围1~100，默认20。</td>
 <td>INT64</td>
 <td>标量</td>
 </tr>
 <tr>
 <td>out_flag</td>
 <td>可选属性</td>
 <td>是否输出迭代中间结果norm_out/sum_out（用于反向梯度计算）。取值0或1，默认0。</td>
 <td>INT64</td>
 <td>标量</td>
 </tr>
 <tr>
 <td>y</td>
 <td>输出</td>
 <td>经Sinkhorn变换后的双随机矩阵H_res，shape与输入h_res一致，作为MhcPost算子的h_res输入。</td>
 <td>FLOAT32</td>
 <td>ND</td>
 </tr>
 <tr>
 <td>norm_out</td>
 <td>输出</td>
 <td>迭代过程中的归一化中间结果，用于反向梯度计算。仅out_flag=1时输出。</td>
 <td>FLOAT32</td>
 <td>ND</td>
 </tr>
 <tr>
 <td>sum_out</td>
 <td>输出</td>
 <td>迭代过程中的求和中间结果，用于反向梯度计算。仅out_flag=1时输出。</td>
 <td>FLOAT32</td>
 <td>ND</td>
 </tr>
</tbody>
</table>

## 约束说明

- 输入h_res需为3D (T, n, n) 或 4D (B, S, n, n) 矩阵，最后两维须相等，且n仅支持4、6或8；不支持二维矩阵输入。
- num_iters取值范围1~100（建议值20），迭代次数固定执行，无按误差提前停止机制；eps为防除零参数（建议值1e-6），非收敛阈值。
- out_flag=1时，norm_out/sum_out为一维Tensor，元素个数分别为2\*num\_iters\*n\*n\_align\*T和2\*num\_iters\*n\_align\*T（n_align为n按8向上对齐，T为3D输入的第0维或4D输入的B\*S）；out_flag=0时，norm_out/sum_out的shape须为{1}。
- 仅支持Ascend 950PR&950DT系列产品硬件环境，其他Atlas系列产品暂不支持。

## 调用说明

| 调用方式      | 调用样例                 | 说明                                                         |
|--------------|-------------------------|--------------------------------------------------------------|
| aclnn调用 | [test_aclnn_mhc_sinkhorn](examples/test_aclnn_mhc_sinkhorn.cpp) | 通过[aclnnMhcSinkhorn](docs/aclnnMhcSinkhorn.md)方式调用算子。|
