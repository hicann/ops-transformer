# ChunkGatedDeltaRule

## 产品支持情况

|产品      | 是否支持 |
|:----------------------------|:-----------:|
|<term>Ascend 950PR/Ascend 950DT</term>|      √     |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>|      √     |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>|      √     |
|<term>Atlas 200I/500 A2 推理产品</term>|      ×     |
|<term>Atlas 推理系列产品</term>|      ×     |
|<term>Atlas 训练系列产品</term>|      ×     |

## 功能说明

- 算子功能：完成chunk版的Gated Delta Rule计算。

- 计算公式：

  Gated Delta Rule（门控Delta规则，GDR）是一种应用于循环神经网络的算子，也被应用于一种线性注意力机制中。在每个时间步 $t$，GDR根据当前的输入 $q_t$、$k_t$、$v_t$、上一个隐藏状态 $S_{t-1}$、衰减系数 $\alpha_t$ 以及更新强度 $\beta_t$，计算当前的注意力输出 $o_t$ 和新的隐藏状态 $S_t$，其计算公式如下：
  $$
  S_t := S_{t-1}(\alpha_t(I - \beta_t k_t k_t^T)) + \beta_t v_t k_t^T = \alpha_t S_{t-1} + \beta_t (v_t - \alpha_t S_{t-1}k_t)k_t^T
  $$
  $$
  o_t := S_t (q_t \cdot scale)
  $$

  其中，$S_{t-1},S_t \in \mathbb{R}^{D_v \times D_k}$，$q_t, k_t \in \mathbb{R}^{D_k}$，$v_t \in \mathbb{R}^{D_v}$，$\alpha_t \in \mathbb{R}$，$\beta_t \in \mathbb{R}$，$o_t \in \mathbb{R}^{D_v}$。

  Chunked Gated Delta Rule是GDR的chunk版实现([参考论文](https://arxiv.org/abs/2412.06464))，它通过将输入序列切块，实现了一定的并行效果，在长上下文场景其计算效率相对Recurrent Gated Delta Rule更高，适用于prefill阶段。输入一个长度为L的序列，该算子可以计算出每一步的输出 $o_t, t \in \{1, 2, .., L\}$ 以及最终的状态矩阵 $S_L$。

## 参数说明

<table style="undefined;table-layout: fixed; width: 900px"><colgroup>
<col style="width: 180px">
<col style="width: 120px">
<col style="width: 200px">
<col style="width: 300px">
<col style="width: 100px">
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
    <td>query</td>
    <td>输入</td>
    <td>公式中的q。</td>
    <td>BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>key</td>
    <td>输入</td>
    <td>公式中的输入k。</td>
    <td>BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>value</td>
    <td>输入</td>
    <td>公式中的输入v。</td>
    <td>BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>beta</td>
    <td>输入</td>
    <td>公式中的β。</td>
    <td>BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>initial_state</td>
    <td>输入</td>
    <td>初始状态矩阵，公式中的输入S_0。</td>
    <td>BFLOAT16、FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>actual_seq_lengths</td>
    <td>输入</td>
    <td>每个batch的序列长度。</td>
    <td>INT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>g</td>
    <td>输入</td>
    <td>衰减系数，公式中的α=e^g</td>
    <td>FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>out</td>
    <td>输出</td>
    <td>每一步的attention结果，公式中的o_t。</td>
    <td>BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>final_state</td>
    <td>输出</td>
    <td>最终的状态矩阵，公式中的S_L。</td>
    <td>BFLOAT16、FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>scale_value</td>
    <td>可选属性</td>
    <td>query的缩放因子，公式中的scale。默认为1.0</td>
    <td>FLOAT</td>
    <td>-</td>
  </tr>
</tbody>
</table>

## 约束说明

- 为方便理解后续排布格式（如BNSD、TND等），统一说明各缩写维度含义：
  * B：输入样本批量大小（Batch）。
  * T：设 $L_i$ 为第 $i$ 个序列长度，则 $T=\sum_i^B L_i$ 表示累积序列长度。
  * Nk：Query和Key头数。
  * Nv：Value头数。
  * Dk：Query和Key隐藏层维度。
  * Dv：Value隐藏层维度。

- 当前仅支持TND布局：
  - query、key形状：$(T, Nk, Dk)$
  - value、out形状：$(T, Nv, Dv)$
  - beta、g形状：$(T, Nv)$
  - actual_seq_lengths形状：$(B,)$
  - initial_state、final_state形状：$(B, Nv, Dv, Dk)$

  维度需满足以下约束：
  - $0 \lt Nv \le 64，0 \lt Nk \le 64$，且 $Nv \bmod Nk = 0$
  - $0 \lt Dv \le 128，0 \lt Dk \le 128$
  - $B \gt 0，T \gt 0$

- 受算法数值特性限制，需满足以下取值约束，否则易出现数值溢出、精度异常：
  - 张量元素：$-1 \le query[i][j][k] \le 1$
  - 张量元素：$-1 \le key[i][j][k] \le 1$
  - 张量元素：$-1 \le g[i][j] \le 0$
  - 张量元素：$0 < beta[i][j] < 1$

## 调用说明

| 调用方式  | 样例代码                                                     | 说明                                                         |
| --------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| aclnn | [test_aclnn_chunk_gated_delta_rul.cpp](./examples/test_aclnn_chunk_gated_delta_rule.cpp) | 通过[aclnnChunkGatedDeltaRule](./docs/aclnnChunkGatedDeltaRule.md)调用aclnnChunkGatedDeltaRule算子 |
