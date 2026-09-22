# KdaInputProj

## 产品支持情况

| 产品 | 是否支持 |
| :---------------------------- | :-----------: |
| <term>Ascend 950PR&950DT系列产品</term> | √ |
| <term>Atlas A3系列产品</term> | × |
| <term>Atlas A2系列产品</term> | × |
| <term>Atlas 200I/500 A2推理产品</term> | × |
| <term>Atlas推理系列产品</term> | × |
| <term>Atlas训练系列产品</term> | × |

## 功能说明

- 算子功能：推理场景下 Recurrent KDA 的前处理算子，对隐藏状态 $X$ 分别投影得到 `qkv`、`beta`、`gate`、`g`。整体计算流程如下：

  ![KdaInputProj Flow](docs/figures/kda_input_proj_flow.png)

  - Stage 1：AIC 与 AIV 并行执行 `Matmul (beta/gate/g)` 与 `DynamicMxQuant`。`Matmul (beta/gate/g)` 的 BlockMmad 任务由全部 AIC 核并行处理，以降低计算轮次、提高利用率。
  - Stage 2：AIC 与 AIV 并行执行 `QuantMatmul (qkv)` 与 `Sigmoid`。`QuantMatmul` 的激活来自 Stage 1 的 `DynamicMxQuant` 输出，`weight_qkv` 为离线 MX 量化权重。

- **计算公式**：

  beta / gate / g 投影（Cube，BF16 输入、FP32 累加）：

  $$
  \mathbf{beta}_{raw} = X W_{\beta}^{\mathrm{T}},\quad
  \mathbf{gate} = X W_{gate}^{\mathrm{T}},\quad
  \mathbf{g} = X W_{g}^{\mathrm{T}}
  $$

  beta 经 Sigmoid（AIV，输出 FP32）：

  $$
  \mathbf{beta} = \sigma(\mathbf{beta}_{raw}) = \frac{1}{1 + e^{-\mathbf{beta}_{raw}}}
  $$

  qkv 量化矩阵乘（激活为 DynamicMxQuant($X$)）：

  $$
  \mathbf{qkv} = \mathrm{QuantMatmul}\bigl(\mathrm{DynamicMxQuant}(X),\; W_{qkv},\; \mathrm{weight\_qkv\_scale}\bigr)
  $$

  其中 $X$ 为 $[T,K]$ BF16；`gate`、`g` 为 BF16；`beta` 为 FP32。

## 参数说明

<table style="undefined;table-layout: fixed; width: 1427px"><colgroup>
<col style="width: 194px">
<col style="width: 146px">
<col style="width: 721px">
<col style="width: 230px">
<col style="width: 136px">
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
    <td>x</td>
    <td>输入</td>
    <td>隐藏层输入，对应公式中的 $X$，shape 为 $[T,K]$</td>
    <td>BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>weightQkv</td>
    <td>输入</td>
    <td>qkv 投影权重 $W_{qkv}$，matmul RHS，shape 为 $[K,N_{qkv}]$。K 维须与 x 的 K 一致</td>
    <td>FLOAT8_E4M3FN</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>weightBeta</td>
    <td>输入</td>
    <td>beta 投影权重 $W_{\beta}$，shape 为 $[K,N_{beta}]$</td>
    <td>BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>weightGate</td>
    <td>输入</td>
    <td>gate 投影权重 $W_{gate}$，shape 为 $[K,N_{gate}]$</td>
    <td>BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>weightG</td>
    <td>输入</td>
    <td>g 投影权重 $W_{g}$，shape 为 $[K,N_{g}]$</td>
    <td>BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>weightQkvScale</td>
    <td>输入</td>
    <td>weightQkv 的 MX 量化缩放因子，最后一维固定为 2（高低位打包），MX block 大小为 64。weightQkv 为转置 view 时 shape 为 $[N_{qkv},\lceil K/64\rceil,2]$，否则为 $[\lceil K/64\rceil,N_{qkv},2]$</td>
    <td>FLOAT8_E8M0</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>qkvOut</td>
    <td>输出</td>
    <td>qkv 投影输出，shape 为 $[T,N_{qkv}]$</td>
    <td>BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>betaOut</td>
    <td>输出</td>
    <td>beta 投影后经 Sigmoid 的输出，shape 为 $[T,N_{beta}]$，调用方预分配</td>
    <td>FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>gateOut</td>
    <td>输出</td>
    <td>gate 投影输出，shape 为 $[T,N_{gate}]$</td>
    <td>BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>gOut</td>
    <td>输出</td>
    <td>g 投影输出，shape 为 $[T,N_{g}]$</td>
    <td>BFLOAT16</td>
    <td>ND</td>
  </tr>
</tbody></table>

维度符号：$T$ 为 token 个数；$K$ 为隐藏维；$N_{qkv}$、$N_{beta}$、$N_{gate}$、$N_{g}$ 为各路输出特征维。

典型 shape 示例（参考模型配置）：$x$ 为 `[T, 7168]`，`weight_qkv` 为 `[7168, 4608]`，`weight_beta` 为 `[7168, 12]`，`weight_gate` / `weight_g` 为 `[7168, 1536]`，`weight_qkv_scale` 为 `[112, 4608, 2]`。

## 约束说明

- 确定性计算：aclnnKdaInputProj 默认为确定性实现。
- shape 约束
  - $T$、$K$、$N_{qkv}$、$N_{beta}$、$N_{gate}$、$N_{g}$ 均大于 0。
  - 各权重矩阵的 K 维必须与 `x` 的 K 维一致。
  - `weightQkv`、`weightBeta`、`weightGate`、`weightG`、`weightQkvScale` 仅支持转置非连续。
- 数据类型约束
  - `x`、`weightBeta`、`weightGate`、`weightG`、`gateOut`、`gOut`、`qkvOut`：BF16。
  - `weightQkv`：FLOAT8_E4M3FN。
  - `weightQkvScale`：FLOAT8_E8M0。
  - `betaOut`：FP32。

## 调用说明

| 调用方式 | 样例代码 / 文档 | 说明 |
| -------- | ---------------- | ---- |
| aclnn 接口 | [test_aclnn_kda_input_proj.cpp](./examples/test_aclnn_kda_input_proj.cpp) | 通过 [aclnnKdaInputProj](./docs/aclnnKdaInputProj.md) 两段式接口调用；|
