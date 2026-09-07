# block_attention_residuals

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR/Ascend 950DT</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：支持
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2 推理产品</term>：不支持
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas 推理系列产品</term>：不支持
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas 训练系列产品</term>：不支持
<!-- end id6 -->

## 功能说明

- **接口功能**：将 `partial_block` 与 `block_res` 按 block 维拼接后，完成 RMS 归一化、`norm_weight ⊙ proj_weight` 投影打分、Softmax 加权融合，输出 `hidden_states`。

- **计算公式**：

    $$
    \operatorname{block\_attention\_residuals}
    \left(partial\_block, block\_res, proj\_weight, norm\_weight; valid\_block\_num, norm\_eps\right)
    \longrightarrow hidden\_states.
    $$

    记 token 数为 $T$，block 数为 $N$，hidden size 为 $H$，拼接后逻辑行数 $B = N + 1$。行序与 golden 一致：前 $N$ 行来自 `block_res`，最后一行来自 `partial_block`。

    $$
    v_{t,i,h} =
    \begin{cases}
    block\_res_{t,i,h}, & 0 \le i < N \\
    partial\_block_{t,h}, & i = N
    \end{cases}
    $$

    $$
    variance_{t,i} = \frac{1}{H}\sum_{h=0}^{H-1} v_{t,i,h}^{2}
    $$

    $$
    inv\_rms_{t,i} = \frac{1}{\sqrt{variance_{t,i} + norm\_eps}}
    $$

    $$
    k_{t,i,h} = v_{t,i,h} \cdot inv\_rms_{t,i}
    $$

    $$
    score\_weight_{h} = norm\_weight_{h} \cdot proj\_weight_{0,h}
    $$

    $$
    s_{t,i} = \sum_{h=0}^{H-1} k_{t,i,h} \cdot score\_weight_{h}
    $$

    $$
    probs_{t,i} = \frac{e^{s_{t,i}}}{\sum_{j=0}^{N} e^{s_{t,j}}}
    $$

    $$
    hidden\_states_{t,h} = \sum_{i=0}^{N} probs_{t,i} \cdot v_{t,i,h}
    $$

    加权融合使用**原始** $v$（非 RMS 后的 $k$）。Torch 接口当前仅返回 $hidden\_states$。

    其中：

    - $partial\_block \in \mathbb{R}^{T \times H}$ 表示输入 `partial_block`，作为拼接后的第 $N$ 行 value。
    - $block\_res \in \mathbb{R}^{T \times N \times H}$ 表示输入 `block_res`，作为拼接后的前 $N$ 行 value。
    - $proj\_weight$ 表示输入 `proj_weight`，shape 为 $[1, H]$ 或 $[H]$；$proj\_weight_{0,h}$ 表示其第 $h$ 个元素。
    - $norm\_weight \in \mathbb{R}^{H}$ 表示输入 `norm_weight`。
    - $valid\_block\_num$ 表示输入 `valid_block_num`。Torch 适配层可不传，默认等于 $N$；当前仅支持该默认值。
    - $norm\_eps$ 表示输入 `norm_eps`，是计算 RMS 归一化因子时使用的数值稳定项。
    - $t \in [0, T)$、$i \in [0, N]$、$h \in [0, H)$ 分别表示 token、逻辑行和 hidden dimension 的索引。
    - $v_{t,i,h}$ 表示拼接后的 value；$inv\_rms_{t,i}$ 表示逐行 RMS 归一化系数；$score\_weight_{h}$ 表示 `norm_weight` 与 `proj_weight` 逐元素相乘结果。
    - $s_{t,i}$ 表示投影得分；$probs_{t,i}$ 表示 Softmax 概率；$hidden\_states_{t,h}$ 表示加权融合输出。
    - $\sum$、$\exp$ 分别表示求和和自然指数运算，$\mathbb{R}$ 表示实数域。

## 函数原型

```python
cann_ops_transformer.block_attention_residuals(
    partial_block,
    block_res,
    proj_weight,
    norm_weight,
    valid_block_num=None,
    norm_eps=1.0e-6,
) -> Tensor
```

## 参数说明

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
| --- | --- | --- | --- | --- | --- |
| partial_block | Tensor | 必选 | 拼接后的第 $N$ 行 value，对应公式中的 $partial\_block$。数据格式为 ND。 | float16、bfloat16、float32 | [T, H] |
| block_res | Tensor | 必选 | 前 $N$ 行 value，对应公式中的 $block\_res$。数据格式为 ND；dtype 必须与 `partial_block` 一致。 | 同 partial_block | [T, N, H] |
| proj_weight | Tensor | 必选 | 投影权重，与 `norm_weight` 共同构成 $score\_weight$。数据格式为 ND；dtype 必须与 `partial_block` 一致。 | 同 partial_block | [H] 或 [1, H] |
| norm_weight | Tensor | 必选 | RMS 缩放权重。数据格式为 ND；dtype 必须与 `partial_block` 一致。 | 同 partial_block | [H] |
| valid_block_num | int | 可选 | 不传或传入 `None` / `-1` 时默认使用 $N$；当前仅支持等于 $N$，其它取值报错。 | int | - |
| norm_eps | float | 可选 | RMS 归一化的数值稳定项，必须为有限正数，默认值为 `1.0e-6`。 | float | - |

## 返回值说明

始终返回 `hidden_states`。反向算子未上库，不返回 `inv_norm` / `probs`。

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
| --- | --- | --- | --- | --- | --- |
| hidden_states | Tensor | 必选 | 加权融合结果，对应公式中的 $hidden\_states$。数据格式为 ND。 | 同 partial_block | [T, H] |

## 约束说明

- 当前反向算子未上库，Torch 接口仅支持正向调用。
- 该接口支持单算子模式调用，暂不支持 TorchAir 图模式调用。
- 主路径 dtype 必须一致，支持 float16 / bfloat16 / float32。
- shape 要求：`T >= 0`，`H >= 1`，`1 <= N <= 100`。`T == 0` 时返回对应 shape 的空输出。
- `block_res.shape[0]`、`block_res.shape[2]` 必须分别等于 `partial_block` 的 $T$、$H$；`proj_weight` 最后一维与 `norm_weight` 长度必须等于 $H$。
- `valid_block_num` 为可选参数：Torch 不传或传入 `None` / `-1` 时默认等于 $N$；ACLNN 传 `-1` 时同样回落到 $N$。当前仅支持该默认值（等于 $N$）。
- `norm_eps` 必须为有限正数。
- 输入 Tensor 支持非连续布局，接口内部转为连续后再计算。
- `partial_block` 或 `block_res` 含 `NaN`、`Inf`，或平方、乘法及累加发生溢出时，结果可能包含 `NaN` 或 `Inf`。

## 确定性/Batch一致性

- 确定性说明：默认支持确定性计算。

- Batch一致性说明：默认Batch一致性实现。

## 调用示例

- 单算子模式调用：

    ```python
    import torch
    import torch_npu
    import cann_ops_transformer

    torch_npu.npu.set_device(0)

    partial_block = torch.tensor(
        [[1.0, 2.0, 3.0, 4.0]],
        dtype=torch.bfloat16,
        device="npu",
    )
    block_res = torch.tensor(
        [
            [
                [1.0, 2.0, 3.0, 4.0],
                [2.0, 0.0, -1.0, 1.0],
            ]
        ],
        dtype=torch.bfloat16,
        device="npu",
    )
    proj_weight = torch.tensor([0.5, 0.25, 0.25, 0.0], dtype=torch.bfloat16, device="npu")
    norm_weight = torch.ones(4, dtype=torch.bfloat16, device="npu")

    # valid_block_num / norm_eps 为可选参数；省略时分别使用 N、1e-6。反向未上库，仅返回 hidden_states。
    hidden_states = cann_ops_transformer.block_attention_residuals(
        partial_block,
        block_res,
        proj_weight,
        norm_weight,
    )
    torch_npu.npu.synchronize()

    print("hidden_states:", hidden_states)
    ```

- TorchAir 图模式调用：暂不支持。
