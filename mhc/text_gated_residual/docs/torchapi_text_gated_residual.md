# text_gated_residual

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT系列产品</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3系列产品</term>：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2系列产品</term>：支持
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2推理产品</term>：不支持
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas推理系列产品</term>：不支持
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas训练系列产品</term>：不支持
<!-- end id6 -->

## 功能说明

- **接口功能**：对输入执行分组 RMS 归一化、低秩变换和门控混合，返回 `mixed`；提供 `w_gate` 时同时返回注入系数 `inject`。梯度模式开启且任一输入需要梯度时，自动绑定一阶反向。

- **计算公式**：

    设输入为 $x\in\mathbb{R}^{B\times S\times D}$，通道数为 $C$，每通道隐藏维度为 $H$，$D=C\times H$，低秩维度为 $R$。

    $$
    \begin{aligned}
    v_{b,s,c,h} &= \frac{x_{b,s,c,h}}{\sqrt{\frac{1}{H}\sum_{j=0}^{H-1}x_{b,s,c,j}^{2}+\epsilon}},\\
    z_{b,s,c,h} &= v_{b,s,c,h}(1+\gamma_{cH+h}),\\
    a &= \operatorname{SiLU}\left((zW_{down}^{T})/C\right),\\
    u &= aW_{up}^{T},\\
    mixed_{b,s,h} &= \frac{1}{C}\sum_{c=0}^{C-1}z_{b,s,c,h}\,\sigma(u_{b,s,c,h}),\\
    inject &= 2\sigma\left((zW_{gate}^{T})/C\right).
    \end{aligned}
    $$

    矩阵乘法时将 $z$ 的最后两维展平为 $D$。$\sigma$ 表示 Sigmoid，$\operatorname{SiLU}(t)=t\sigma(t)$。未提供 `gamma` 时缩放因子为 1；未提供 `w_gate` 时不计算 `inject`。

## 函数原型

```python
text_gated_residual(
    x,
    w_down,
    w_up,
    w_gate=None,
    channels=4,
    gamma=None,
    eps=1e-6,
) -> Union[Tensor, Tuple[Tensor, Tensor]]
```

## 参数说明

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
| --- | --- | --- | --- | --- | --- |
| x | Tensor | 必选 | 输入特征，数据格式为 ND，$D=C\times H$。 | float16、bfloat16 | [B, S, D] |
| w_down | Tensor | 必选 | 下投影权重，数据格式为 ND，dtype 与 `x` 一致。 | 同 x | [R, D] |
| w_up | Tensor | 必选 | 上投影权重，数据格式为 ND，dtype 与 `x` 一致。 | 同 x | [D, R] |
| w_gate | Tensor | 可选 | 注入分支权重，数据格式为 ND，dtype 与 `x` 一致；默认值为 `None`。 | 同 x | [C, D] |
| channels | int | 可选 | 通道数 $C$，默认值为 4。 | int | - |
| gamma | Tensor | 可选 | RMS 缩放权重，使用 `1 + gamma` 缩放，数据格式为 ND；默认值为 `None`。 | float32 | [D] |
| eps | float | 可选 | RMS 归一化的数值稳定项，默认值为 `1e-6`。 | float | - |

## 返回值说明

未提供 `w_gate` 时返回 `mixed`；提供 `w_gate` 时返回 `(mixed, inject)`。

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
| --- | --- | --- | --- | --- | --- |
| mixed | Tensor | 必选 | 门控混合结果，数据格式为 ND。 | 同 x | [B, S, H] |
| inject | Tensor | 可选 | 注入系数，仅提供 `w_gate` 时返回，数据格式为 ND。 | 同 x | [B, S, C] |

## 约束说明

- 支持单算子模式调用，暂不支持 TorchAir 图模式调用。
- `2 <= C <= 10`；`256 <= H <= 4096` 且 `H` 为 256 的倍数；`64 <= R <= 896` 且 `R` 为 64 的倍数。
- `x` 为三维 Tensor，最后一维必须可被 `channels` 整除；各输入 shape 必须符合参数表，同名维度的大小必须一致。
- `x`、`w_down`、`w_up`、`w_gate` 的 dtype 必须一致；所有已提供的 Tensor 必须位于同一 NPU 设备。
- `eps` 必须为非负有限数，且可表示为 float32。
- 支持部分输入求导，不要求所有输入都设置 `requires_grad=True`。求导时还要求 $R\le640$，且 $13H\le K$、$3H+8R\le K$、$7H+6R\le K$。Ascend 950PR&950DT 系列产品取 $K=62448-16\lceil C/8\rceil$；Atlas A2、Atlas A3 系列产品取 $K=39664-16\lceil C/8\rceil$。
- `torch.no_grad()` 下仅执行前向。不支持高阶求导。

## 确定性/Batch一致性

- 确定性说明：默认支持确定性计算。
- Batch一致性说明：默认非Batch一致性实现。

## 调用示例

- 单算子模式调用：

    ```python
    import torch
    import torch_npu
    from cann_ops_transformer.ops.mhc.text_gated_residual import text_gated_residual

    torch_npu.npu.set_device(0)
    b, s, c, h, r = 1, 32, 4, 256, 64
    d = c * h
    x = torch.randn(b, s, d, dtype=torch.float16, device="npu", requires_grad=True)
    w_down = (torch.randn(r, d, dtype=torch.float16, device="npu") * 0.01).requires_grad_()
    w_up = (torch.randn(d, r, dtype=torch.float16, device="npu") * 0.01).requires_grad_()
    w_gate = (torch.randn(c, d, dtype=torch.float16, device="npu") * 0.01).requires_grad_()
    gamma = torch.zeros(d, dtype=torch.float32, device="npu", requires_grad=True)

    mixed, inject = text_gated_residual(
        x, w_down, w_up, w_gate=w_gate, channels=c, gamma=gamma,
    )
    (mixed.float().sum() + inject.float().sum()).backward()
    torch_npu.npu.synchronize()
    print(mixed.shape, inject.shape, x.grad.shape)
    ```

- TorchAir 图模式调用：暂不支持。
