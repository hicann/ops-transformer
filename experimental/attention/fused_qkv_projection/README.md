# AclnnFusedQkvProjection

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    ×     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                               |    ×     |
| <term>Atlas 训练系列产品</term>                               |    ×     |


## 功能说明

- **算子功能**：将 QKV 三次矩阵乘法融合为一次，通过单次 Matmul 计算 `hidden_states @ weight` 得到融合投影，再拆分为独立的 Q、K、V 输出张量，可选加 bias。

- **计算公式**：

$$
\begin{aligned}
\text{fused} &= \text{hidden\_states} \cdot \text{weight} + \text{bias} \quad (\text{bias 可选}) \\
Q &= \text{fused}[:, :, 0 : qDim] \\
K &= \text{fused}[:, :, qDim : qDim + kDim] \\
V &= \text{fused}[:, :, qDim + kDim : qDim + kDim + vDim]
\end{aligned}
$$

其中 $qDim + kDim + vDim = \text{fusedDim}$，即 weight 矩阵的列数。

- **实现方式**：使用 Cube 单元（Matmul API）完成 `hidden_states @ weight` 的矩阵乘法，从 LCM 取回结果后按偏移拆分写入 Q、K、V 的 GM 输出区域。

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
      <td>hidden_states</td>
      <td>输入</td>
      <td>输入的 hidden states，shape 为 <code>[batch, seqLen, hiddenSize]</code>。</td>
      <td>FLOAT, FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>weight</td>
      <td>输入</td>
      <td>融合的 QKV 投影权重，shape 为 <code>[hiddenSize, fusedDim]</code>，其中 <code>fusedDim = qDim + kDim + vDim</code>。</td>
      <td>FLOAT, FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>bias</td>
      <td>可选输入</td>
      <td>融合投影的偏置，shape 为 <code>[fusedDim]</code>。不填为无偏置。</td>
      <td>FLOAT, FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>query</td>
      <td>输出</td>
      <td>Q 投影结果，shape 为 <code>[batch, seqLen, qDim]</code>。</td>
      <td>FLOAT, FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>key</td>
      <td>输出</td>
      <td>K 投影结果，shape 为 <code>[batch, seqLen, kDim]</code>。</td>
      <td>FLOAT, FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>value</td>
      <td>输出</td>
      <td>V 投影结果，shape 为 <code>[batch, seqLen, vDim]</code>。</td>
      <td>FLOAT, FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>q_output_dim</td>
      <td>属性</td>
      <td>Q 的输出维度（qDim）。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>k_output_dim</td>
      <td>属性</td>
      <td>K 的输出维度（kDim）。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>v_output_dim</td>
      <td>属性</td>
      <td>V 的输出维度（vDim）。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
  </tbody>
</table>

## 约束与限制

- **M 约束**：$M = batch \times seqLen \ge 16$，满足 Cube 最小粒度要求。
- **维度对齐**：$qDim$、$kDim$、$vDim$ 建议为 8 的倍数（32B 对齐），非对齐维度会走标量保底路径，性能较低。
- **数据排布**：$weight$ 按 $[q_0, \dots, q_{qDim-1}, k_0, \dots, k_{kDim-1}, v_0, \dots, v_{vDim-1}]$ 顺序在最后一维拼接。
