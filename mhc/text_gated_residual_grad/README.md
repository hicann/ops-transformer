# TextGatedResidualGrad

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

`TextGatedResidualGrad` 是 `TextGatedResidual` 的反向算子。根据混合输出和注入输出的上游梯度，以及前向保存的张量，计算输入 `x` 和权重 `wDown`、`wUp`、`wGate`、`gamma` 的梯度。

记 $B$ 为批大小，$S$ 为序列长度，$C$ 为通道数，$H$ 为每通道的特征数，$R$ 为下投影维度，$D=C\times H$。$\odot$ 表示逐元素乘法，$@$ 表示矩阵乘法，$\sigma$ 表示 Sigmoid 函数。

对应的前向计算为：

$$
\begin{aligned}
\mathrm{invRms}_{b,s,c} &= \left(\frac{1}{H}\sum_{j=0}^{H-1}x_{b,s,c,j}^{2}+\epsilon\right)^{-1/2} \\
\mathrm{xNorm} &= x\odot\mathrm{invRms}\odot(1+\gamma) \\
\mathrm{downPreAct} &= \mathrm{xNorm}@\mathrm{wDown}^{T} \\
\mathrm{act} &= \mathrm{SiLU}(\mathrm{downPreAct}/C) \\
\mathrm{up} &= \mathrm{act}@\mathrm{wUp}^{T} \\
\mathrm{mixed}_{b,s,j} &= \frac{1}{C}\sum_{c=0}^{C-1}\mathrm{xNorm}_{b,s,c,j}\,\sigma(\mathrm{up}_{b,s,c,j}) \\
\mathrm{inject} &= 2\sigma\!\left((\mathrm{xNorm}@\mathrm{wGate}^{T})/C\right)
\end{aligned}
$$

其中，归一化及通道求和时将最后一维 $D$ 视为 $[C,H]$，矩阵乘法时仍使用最后一维 $D$。`invRms` 在每个通道的 $H$ 个特征上广播，`gamma` 在 $B$、$S$ 维上广播。$\epsilon$ 是前向归一化使用的参数，本接口通过 `invRms` 接收对应的前向结果。

给定 `gradMixed` 和 `gradInject`，定义

$$
L=\sum_{b,s,j}\mathrm{gradMixed}_{b,s,j}\,\mathrm{mixed}_{b,s,j}
 +\sum_{b,s,c}\mathrm{gradInject}_{b,s,c}\,\mathrm{inject}_{b,s,c}.
$$

本算子的输出为：

$$
dx=\frac{\partial L}{\partial x},\quad
dwDown=\frac{\partial L}{\partial wDown},\quad
dwUp=\frac{\partial L}{\partial wUp},\quad
dwGate=\frac{\partial L}{\partial wGate},\quad
dgamma=\frac{\partial L}{\partial\gamma}.
$$

`wGate` 缺省时不包含注入分支及其梯度；`gamma` 缺省时归一化缩放系数为 1，不输出 `dgamma`。`invRms`、`inject`、`downPreAct`、`up` 必须来自同一次前向计算，并与本次传入的输入、权重对应。

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 | shape |
| :--- | :--- | :--- | :--- | :--- | :--- |
| gradMixed | 输入 | 混合输出 mixed 的上游梯度。 | FLOAT16、BFLOAT16 | ND | [B,S,H] |
| x | 输入 | 前向输入。 | FLOAT16、BFLOAT16 | ND | [B,S,D] |
| wDown | 输入 | 下投影权重。 | FLOAT16、BFLOAT16 | ND | [R,D] |
| wUp | 输入 | 上投影权重。 | FLOAT16、BFLOAT16 | ND | [D,R] |
| invRms | 输入 | 前向保存的各通道均方根倒数。 | FLOAT32 | ND | [B,S,C] |
| downPreAct | 输入 | 前向保存的下投影结果 xNorm@wDownᵀ。 | FLOAT16、BFLOAT16 | ND | [B,S,R] |
| up | 输入 | 前向保存的上投影结果 act@wUpᵀ。 | FLOAT16、BFLOAT16 | ND | [B,S,D] |
| gradInject | 可选输入 | 注入输出 inject 的上游梯度。 | FLOAT16、BFLOAT16 | ND | [B,S,C] |
| wGate | 可选输入 | 注入分支权重。 | FLOAT16、BFLOAT16 | ND | [C,D] |
| gamma | 可选输入 | 归一化缩放权重，实际缩放系数为 1+gamma。 | FLOAT32 | ND | [D] |
| inject | 可选输入 | 前向保存的注入输出 2*sigmoid((xNorm@wGateᵀ)/C)。 | FLOAT16、BFLOAT16 | ND | [B,S,C] |
| channels | 属性 | 通道数 C。 | INT64 | - | 标量 |
| dx | 输出 | x 的梯度。 | 同 x | ND | [B,S,D] |
| dwDown | 输出 | wDown 的梯度。 | FLOAT16、BFLOAT16 | ND | [R,D] |
| dwUp | 输出 | wUp 的梯度。 | FLOAT16、BFLOAT16 | ND | [D,R] |
| dwGate | 可选输出 | wGate 的梯度。 | FLOAT16、BFLOAT16 | ND | [C,D] |
| dgamma | 可选输出 | gamma 的梯度。 | FLOAT32 | ND | [D] |

## 约束说明

- `C`、`H`、`R` 为正整数；`H`、`R` 均须为 16 的整数倍。

通道数须满足 `C ≤ 20`。

`H` 的取值范围按产品确定。令

<!-- npu="950" id7 -->
- <term>Ascend 950PR&950DT系列产品</term>：

  $$
  K = 62448-16\left\lceil C/8\right\rceil
  $$

<!-- end id7 -->
<!-- npu="A3,910b" id8 -->
- <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：

  $$
  K = 39664-16\left\lceil C/8\right\rceil
  $$

<!-- end id8 -->

`H`、`R` 须同时满足：

$$
\boxed{
\begin{aligned}
13H &\le K,\\
3H+8R &\le K,\\
7H+6R &\le K.
\end{aligned}
}
$$

因此，给定 `C` 后，`H` 的上限为

$$
H_{\max}=16\left\lfloor\frac{K}{208}\right\rfloor.
$$

`R` 须同时满足上述关系式和 `R ≤ 640`。

其中，$\lceil\cdot\rceil$ 表示向上取整，$\lfloor\cdot\rfloor$ 表示向下取整。上述关系适用于 FLOAT16、BFLOAT16，以及所有合法的可选参数组合。

例如，`C=4、H=2560、R=320` 适用于上述三类产品。其他 `C/H/R` 组合须满足上述约束。调用时仍需通过第一段接口 `aclnnTextGatedResidualGradGetWorkspaceSize` 完成参数校验，并按返回的 `workspaceSize` 申请内存。

- 所有 FLOAT16/BFLOAT16 输入以及 `dx`、`dwDown`、`dwUp`、`dwGate` 的数据类型必须一致；各权重梯度与对应权重的数据类型相同。`invRms`、`gamma` 和 `dgamma` 为 FLOAT32。
- 支持非连续输入 Tensor。输入输出的 shape 必须符合参数表，同名维度的大小必须一致。
- 支持 `B=0` 或 `S=0`：`dx` 为空，已提供的非空权重梯度和 `dgamma` 全部置零，其余参数仍须满足上述约束。
- 输出梯度覆盖写入，调用前无需初始化，也不会累加到输出 Tensor 的原有数据中。
- 默认执行确定性计算。

ACLNN 调用时，可选参数须满足以下要求：

| 功能 | 提供时 | 缺省时 |
| :--- | :--- | :--- |
| 注入分支 | `wGate`、`gradInject`、`inject`、`dwGate` 四者必须同时提供 | 四者必须全部传入 `nullptr` |
| 归一化缩放 | `gamma`、`dgamma` 必须同时提供 | 两者必须全部传入 `nullptr`，缩放系数取 1 |

两组可选参数相互独立。`nullptr` 表示省略该功能，不能用 shape 为 `[0]` 的占位 Tensor 代替。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
| :--- | :--- | :--- |
| aclnn 调用 | [接口调用示例](./docs/aclnnTextGatedResidualGrad.md#调用示例) | 创建输入输出 Tensor，调用两段式接口获取并执行计算。 |
