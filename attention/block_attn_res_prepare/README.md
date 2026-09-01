# BlockAttnResPrepare

## 产品支持情况

| 产品 | 是否支持 |
| ---- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term> | √ |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | × |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | × |
| <term>Atlas 200I/500 A2 推理产品</term> | × |
| <term>Atlas 推理系列产品</term> | × |
| <term>Atlas 训练系列产品</term> | × |

## 功能说明

- 算子功能：完成 Attention Residuals 历史残差注意力两阶段计算的第一阶段，一次并行计算全部 $S$ 个目标 slot 的块间注意力，并返回 softmax 加权分子及统计量 $(O,M,L)$，供下一阶段融合当前残差。

- 计算公式：

$$
\operatorname{block\_attn\_res\_prepare}
\left(V,Q,\operatorname{valid\_blocks};\epsilon\right)
\longrightarrow (O,M,L)
$$

对 token $t$ 的第 $n$ 个有效历史来源计算 RMS 归一化因子：

$$
R_{t,n}=\sqrt{\frac{1}{D}\sum_{d=0}^{D-1}V_{t,n,d}^{2}+\epsilon}
$$

对每个目标 slot、token 和有效历史来源计算 logit：

$$
Z_{s,t,n}=R_{t,n}^{-1}\sum_{d=0}^{D-1}Q_{s,d}V_{t,n,d}
$$

设 $N_v$ 为 `valid_blocks` 指定的有效历史来源数，则：

$$
M_{s,t}=\max_{0\le n<N_v}Z_{s,t,n}
$$

$$
E_{s,t,n}=\exp\left(Z_{s,t,n}-M_{s,t}\right)
$$

$$
L_{s,t}=\sum_{n=0}^{N_v-1}E_{s,t,n}
$$

$$
O_{s,t,d}=\sum_{n=0}^{N_v-1}E_{s,t,n}V_{t,n,d}
$$

当 `valid_blocks[0] == 0` 时，不执行上述 softmax 计算，直接返回：

$$
O=\mathbf{0}_{S\times T\times D},\qquad
M=m_{\min}\mathbf{1}_{S\times T},\qquad
L=\mathbf{0}_{S\times T},\qquad
m_{\min}=-3.4028234663852886\times10^{38}
$$

即 `numerator` 为 shape `[S, T, D]` 的全 0 Tensor，`logit_max` 为 shape `[S, T]` 且所有元素均为 FLOAT32 最小有限值 $m_{\min}$ 的 Tensor，`exp_sum` 为 shape `[S, T]` 的全 0 Tensor。该结果表示 online softmax 的空状态。

其中：

- $V\in\mathbb{R}^{T\times N\times D}$ 表示输入 `block_res`；$V_{t,n,d}$ 表示第 $t$ 个 token、第 $n$ 个历史残差块在第 $d$ 个隐藏特征上的值。
- $Q\in\mathbb{R}^{S\times D}$ 表示输入 `pseudo_query`（伪 Query）；每个目标 slot 对应一个 Query 向量，$Q_{s,d}$ 表示第 $s$ 个目标 slot 在第 $d$ 个隐藏特征上的值。
- $T$ 表示 token 数，$N$ 表示 `block_res` 第 1 维可容纳的历史残差块数，$S$ 表示目标 slot 数，$D$ 表示HiddenSize。
- $N_v=\min(\texttt{valid\_blocks[0]},N)$，表示实际参与计算的历史残差块数。
- $R$ 表示 RMS 归一化因子，$Z$ 表示归一化注意力 logit，$E$ 表示 max-shift 后的指数值。
- $O$、$M$、$L$ 分别表示 softmax 加权分子、logit 最大值和指数和，对应输出 `numerator`、`logit_max`、`exp_sum`。
- $\epsilon$ 表示 RMS 归一化的数值稳定项，对应属性 `eps`。

## 参数说明

<table style="undefined;table-layout: fixed; width: 980px"><colgroup>
  <col style="width: 120px">
  <col style="width: 150px">
  <col style="width: 490px">
  <col style="width: 120px">
  <col style="width: 100px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出/属性</th>
      <th>描述</th>
      <th>数据类型</th>
      <th>数据格式</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>block_res</td>
      <td>输入</td>
      <td>历史残差块，对应公式中的V，shape为[T, N, D]。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>valid_blocks</td>
      <td>输入</td>
      <td>当前有效历史来源数，shape为[1]。值为0时，numerator返回全0 Tensor、logit_max返回所有元素均为FLOAT32最小有限值的Tensor、exp_sum返回全0 Tensor；大于N时按N处理。</td>
      <td>UINT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>pseudo_query</td>
      <td>输入</td>
      <td>伪 Query，对应公式中的Q；每个目标slot对应一个Query向量，shape为[S, D]。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>numerator</td>
      <td>输出</td>
      <td>softmax加权分子，对应公式中的O，shape为[S, T, D]。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>logit_max</td>
      <td>输出</td>
      <td>有效历史来源logit的最大值，对应公式中的M，shape为[S, T]。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>exp_sum</td>
      <td>输出</td>
      <td>max-shift后的指数和，对应公式中的L，shape为[S, T]。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>eps</td>
      <td>属性</td>
      <td>RMS归一化的数值稳定项，可选，默认值为1e-6，必须为有限正数。</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
  </tbody>
</table>

## 约束说明

- `T >= 0`，`S >= 0`，`1 <= N <= 64`，`1 <= D <= 8192`。
- `block_res` 的 shape 为 `[T, N, D]`，`pseudo_query` 的 shape 为 `[S, D]`；`numerator`、`logit_max` 和 `exp_sum` 的 shape 分别为 `[S, T, D]`、`[S, T]` 和 `[S, T]`。
- `valid_blocks` 的 shape 必须为 `[1]`；`valid_blocks[0] == 0` 时，`numerator` 返回全 0 Tensor、`logit_max` 返回所有元素均为 FLOAT32 最小有限值的 Tensor、`exp_sum` 返回全 0 Tensor；`valid_blocks[0] > N` 时按 `N` 处理。
- 所有输入和输出 Tensor 均必须为连续 Tensor，不支持非连续 Tensor。
- `eps` 必须为有限正数。
- `T == 0` 或 `S == 0` 时，返回对应 shape 的空输出。

## 调用说明

<table><thead>
  <tr>
    <th>调用方式</th>
    <th>调用样例</th>
    <th>说明</th>
  </tr></thead>
<tbody>
  <tr>
    <td>aclnn调用</td>
    <td><a href="./examples/test_aclnn_block_attn_res_prepare.cpp">test_aclnn_block_attn_res_prepare</a></td>
    <td>通过<a href="./docs/aclnnBlockAttnResPrepare.md">aclnnBlockAttnResPrepare.md</a>调用算子，算子编译block_attn_res_prepare。详细的算子编译运行方法参见<a href="../../docs/zh/invocation/quick_op_invocation.md">算子调用</a>。</td>
  </tr>
</tbody>
</table>

```bash
# 在ops-transformer仓库根目录执行
bash build.sh --pkg --soc=ascend950 --ops=block_attn_res_prepare
bash build.sh --run_example block_attn_res_prepare eager cust --soc=ascend950 --vendor_name=custom
```
