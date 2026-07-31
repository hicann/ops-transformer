# MhcSinkhorn

## 产品支持情况

|产品      | 是否支持 |
|:----------------------------|:-----------:|
|<term>Ascend 950PR/Ascend 950DT</term>|      √     |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>|      ×     |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>|      ×     |
|<term>Atlas 200I/500 A2 推理产品</term>|      ×     |
|<term>Atlas 推理系列产品</term>|      ×     |
|<term>Atlas 训练系列产品</term>|      ×     |

## 功能说明

- 算子功能：MhcSinkhorn是mHC架构的核心算子，通过Sinkhorn-Knopp迭代算法将mHC层初始混合矩阵投影到双随机矩阵流形（Birkhoff多胞形），生成满足行和、列和均为1的双随机矩阵h_res，为MhcPost算子提供关键输入，稳定深度网络信号传播、解决梯度消失/爆炸问题。

- 计算公式：

  $$
  \begin{align}
  M^{(k+1)} &= M^{(k)} \oslash (1_n \cdot (M^{(k)})^T 1_n) \\
  M^{(k+2)} &= M^{(k+1)} \oslash ((M^{(k+1)} 1_n) \cdot 1_n^T) \\
  h_{res} &= M^{(k+2)}
  \end{align}
  $$

  其中：$M$为初始混合矩阵，$\oslash$为元素级除法，$1_n$为n维全1向量，$k$为迭代次数，迭代至矩阵满足双随机特性时停止。

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
 <th>输入/输出</th>
 <th>描述</th>
 <th>数据类型</th>
 <th>数据格式</th>
 </tr></thead>
<tbody>
 <tr>
 <td>init_matrix</td>
 <td>输入</td>
 <td>待变换的mHC层初始混合矩阵，为超连接原始矩阵。即公式中的M。</td>
 <td>FLOAT32</td>
 <td>ND</td>
 </tr>
 <tr>
 <td>max_iter</td>
 <td>输入</td>
 <td>Sinkhorn-Knopp迭代最大次数，控制迭代收敛过程。即公式中的k。</td>
 <td>INT32</td>
 <td>标量</td>
 </tr>
 <tr>
 <td>epsilon</td>
 <td>输入</td>
 <td>收敛阈值，矩阵行/列和与1的误差小于该值时停止迭代。</td>
 <td>FLOAT32</td>
 <td>标量</td>
 </tr>
 <tr>
 <td>h_res</td>
 <td>输出</td>
 <td>经Sinkhorn变换后的双随机矩阵，作为MhcPost算子的h_res输入。即公式中的Hres。</td>
 <td>FLOAT32</td>
 <td>ND</td>
 </tr>
</tbody>
</table>

## 约束说明

- 输入init_matrix需为二维非负矩阵（N×N），确保迭代后可形成双随机矩阵。
- max_iter建议取值范围50\~200，epsilon建议取值范围1e-6~1e-4，平衡收敛效果与计算效率。
- 仅支持Ascend 950PR/Ascend 950DT硬件环境，其他Atlas系列产品暂不支持。

## 调用说明

| 调用方式      | 调用样例                 | 说明                                                         |
|--------------|-------------------------|--------------------------------------------------------------|
| aclnn调用 | [test_aclnn_mhc_sinkhorn](examples/test_aclnn_mhc_sinkhorn.cpp) | 通过[aclnnMhcSinkhorn](docs/aclnnMhcSinkhorn.md)方式调用算子。|
  