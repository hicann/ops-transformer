# TextGatedResidual

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :---: |
| <term>Ascend 950PR&950DT系列产品</term> | √ |
| <term>Atlas A3系列产品</term> | √ |
| <term>Atlas A2系列产品</term> | √ |
| <term>Atlas 200I/500 A2推理产品</term> | × |
| <term>Atlas推理系列产品</term> | × |
| <term>Atlas训练系列产品</term> | × |

## 功能说明

- **算子功能**：融合分组 RMSNorm、低秩 down/SiLU/up、sigmoid 混合和可选 inject，输出 mixed；needBackward=true 时保存三个反向缓存。

- **计算公式**：

### 分组 RMS 归一化

设输入 `x` 的 shape 为 $(B,S,C\times H)$，按通道拆分为 $x_{b,s,c,h}$。分组 RMS 归一化定义为：

$$
invRms_{b,s,c}=\frac{1}{\sqrt{\frac{1}{H}\sum_{j=0}^{H-1}x_{b,s,c,j}^{2}+\epsilon}}
$$

$$
xNorm_{b,s,c,h}=x_{b,s,c,h}\cdot invRms_{b,s,c}\cdot(1+\gamma_{cH+h})
$$

未提供 `gamma` 时缩放因子为 1；传入的 gamma 不应提前加 1。

### 读融合路径

将 $xNorm$ 展平为 $(B,S,C\times H)$ 后计算：

$$
downPreAct_{b,s,r}=\sum_{d=0}^{C\times H-1}xNorm_{b,s,d}\,wDown_{r,d}
$$

$$
gate_{b,s,c,h}=\operatorname{sigmoid}\left(\sum_{r=0}^{R-1}\operatorname{SiLU}\left(\frac{downPreAct_{b,s,r}}{C}\right)\,wUp_{c\times H+h,r}\right)
$$

$$
mixed_{b,s,h}=\frac{1}{C}\sum_{c=0}^{C-1}gate_{b,s,c,h}\,xNorm_{b,s,c,h}
$$

### 反向缓存和使用方式

定义 D=C×H，upPreActOptional 缓存为：

$$
upPreAct_{b,s,d}=\sum_{r=0}^{R-1}\operatorname{SiLU}\left(\frac{downPreAct_{b,s,r}}{C}\right)\,wUp_{d,r}
$$

downPreActOptional 为未除以 C 的下投影结果，upPreActOptional 为 sigmoid 之前的上投影结果，二者 dtype 均与 x 一致。
needBackward=true 时必须提供 invRmsOptional、downPreActOptional 和 upPreActOptional；该要求不受 wGateOptional 是否传入影响。needBackward=false 时三个缓存输出可传入 nullptr，其内容无效，不应读取。

### 注入路径

当 `wGateOptional` 非空 时，注入路径为：

$$
inject_{b,s,c}=2\cdot\operatorname{sigmoid}\left(\frac{1}{C}\sum_{d=0}^{C\times H-1}xNorm_{b,s,d}\,wGate_{c,d}\right)
$$

### 符号说明

其中 $B,S$ 为 batch 和序列长度，$C$ 为通道数，$H$ 为每通道隐藏维度，$R$ 为瓶颈维度；`gamma` 为 shape $(C\times H)$ 的可选原始权重，缩放因子为 $1+\gamma_{cH+h}$。

`needBackward=true` 时，前向另输出 `invRmsOptional`、`downPreActOptional` 和 `upPreActOptional` 供反向计算；`wGateOptional` 非空 时输出 `injectOptional`，反向由 `injectOptional / 2` 还原注入门控值。

## 参数说明

<table style="table-layout: auto; width: 100%">
<thead><tr><th>参数名</th><th>输入/输出/属性</th><th>描述</th><th>数据类型</th><th>数据格式</th></tr></thead>
<tbody>
<tr><td>x</td><td>输入</td><td>前向残差输入，shape 为 (B,S,D)，D=C×H。</td><td>FLOAT16、BFLOAT16</td><td>ND</td></tr>
<tr><td>wDown</td><td>输入</td><td>下投影权重，shape 为 (R,D)，dtype 与 x 一致。</td><td>FLOAT16、BFLOAT16</td><td>ND</td></tr>
<tr><td>wUp</td><td>输入</td><td>上投影权重，shape 为 (D,R)，dtype 与 x 一致。</td><td>FLOAT16、BFLOAT16</td><td>ND</td></tr>
<tr><td>wGateOptional</td><td>可选输入</td><td>注入权重，shape 为 (C,D)，为空时不计算 inject。</td><td>FLOAT16、BFLOAT16</td><td>ND</td></tr>
<tr><td>gammaOptional</td><td>可选输入</td><td>shape 为 (D)，缩放因子为 1+gamma，为空时缩放因子为 1。</td><td>FLOAT32</td><td>ND</td></tr>
<tr><td>channels</td><td>必选属性</td><td>通道数 C，须大于 0 且 D 可被 C 整除。</td><td>INT64</td><td>-</td></tr>
<tr><td>eps</td><td>可选属性</td><td>RMS 数值稳定性参数，非负有限值，默认 1e-6；ACLNN 调用须显式传值。</td><td>DOUBLE</td><td>-</td></tr>
<tr><td>needBackward</td><td>可选属性</td><td>是否保存反向缓存，默认 false；ACLNN 调用须显式传值。</td><td>BOOL</td><td>-</td></tr>
<tr><td>mixed</td><td>输出</td><td>融合结果，shape 为 (B,S,H)，dtype 与 x 一致。</td><td>FLOAT16、BFLOAT16</td><td>ND</td></tr>
<tr><td>injectOptional</td><td>输出</td><td>注入权重，shape 为 (B,S,C)；仅 wGateOptional 非空时有效。</td><td>FLOAT16、BFLOAT16</td><td>ND</td></tr>
<tr><td>invRmsOptional</td><td>输出</td><td>逐通道 RMS 归一化系数，shape 为 (B,S,C)；仅 needBackward=true 时有效。</td><td>FLOAT32</td><td>ND</td></tr>
<tr><td>downPreActOptional</td><td>输出</td><td>shape 为 (B,S,R)，保存未除以 C 的 down 投影；仅 needBackward=true 时有效。</td><td>FLOAT16、BFLOAT16</td><td>ND</td></tr>
<tr><td>upPreActOptional</td><td>输出</td><td>shape 为 (B,S,D)，保存 sigmoid 前的 up 投影；仅 needBackward=true 时有效。</td><td>FLOAT16、BFLOAT16</td><td>ND</td></tr>
</tbody>
</table>

## 约束说明

- B、S 大于等于 0，D、R 大于 0，D=C×H；输入和计算输出 dtype 须一致，gamma 和 invRms 为 FLOAT32。
- 输入限制：通道数 C 为整数，满足 2≤C≤10；每通道隐藏维度 H 满足 256≤H≤4096 且 H%256=0；低秩维度 R 满足 64≤R≤896 且 R%64=0。
- 输入 x 的 shape 为 [B,S,D]，D=C×H，H=D/C；wDown、wUp 的 shape 分别为 [R,D]、[D,R]，可选 wGateOptional 的 shape 为 [C,D]，可选 gammaOptional 的 shape 为 [D]。
- wGateOptional 非空时必须提供 injectOptional；needBackward=true 时必须提供三个缓存输出。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
| --- | --- | --- |
| aclnn调用 | [test_aclnn_text_gated_residual](examples/test_aclnn_text_gated_residual.cpp) | 通过 [aclnnTextGatedResidual](docs/aclnnTextGatedResidual.md) 两段接口调用算子。 |
