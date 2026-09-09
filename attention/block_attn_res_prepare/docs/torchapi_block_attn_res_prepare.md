# block_attn_res_prepare

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR/Ascend 950DT</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：不支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：不支持
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

- **接口功能**：完成 Attention Residuals 历史残差注意力两阶段计算的第一阶段，一次并行计算全部 $S$ 个目标 slot 的块间注意力，并返回 softmax 加权分子及统计量 $(O,M,L)$，供下一阶段融合当前残差。

- **计算公式**：

    $$
    \operatorname{block\_attn\_res\_prepare}
    \left(V,Q,\operatorname{valid\_blocks};\epsilon\right)
    \longrightarrow (O,M,L).
    $$

    对 token $t$ 的第 $n$ 个历史来源，首先计算 RMS 归一化因子：

    $$
    R_{t,n}=\sqrt{\frac{1}{D}\sum_{d=0}^{D-1}V_{t,n,d}^{2}+\epsilon}.
    $$

    对每个 slot、token 和有效历史来源计算 logits：

    $$
    Z_{s,t,n}=R_{t,n}^{-1}\sum_{d=0}^{D-1}Q_{s,d}V_{t,n,d}.
    $$

    设 $N_v$ 为 `valid_blocks` 指定的有效历史来源数，则：

    $$
    M_{s,t}=\max_{0\le n<N_v}Z_{s,t,n},
    $$

    $$
    E_{s,t,n}=\exp\left(Z_{s,t,n}-M_{s,t}\right),
    $$

    $$
    L_{s,t}=\sum_{n=0}^{N_v-1}E_{s,t,n},
    $$

    $$
    O_{s,t,d}=\sum_{n=0}^{N_v-1}E_{s,t,n}V_{t,n,d}.
    $$

    当 `valid_blocks[0] == 0` 时，不执行上述 softmax 计算，直接返回：

    $$
    O=\mathbf{0}_{S\times T\times D},\qquad
    M=m_{\min}\mathbf{1}_{S\times T},\qquad
    L=\mathbf{0}_{S\times T},\qquad
    m_{\min}=-3.4028234663852886\times10^{38}.
    $$

    即 `numerator` 为 shape `[S, T, D]` 的全 0 Tensor，`logit_max` 为 shape `[S, T]` 且所有元素均为 float32 最小有限值 $m_{\min}$ 的 Tensor，`exp_sum` 为 shape `[S, T]` 的全 0 Tensor。该结果表示 online softmax 的空状态。

    其中：

    - $V\in\mathbb{R}^{T\times N\times D}$ 表示输入 `block_res`，用于存放历史残差块；$V_{t,n,d}$ 表示第 $t$ 个 token、第 $n$ 个历史来源在第 $d$ 个 hidden dimension 上的值。
    - $Q\in\mathbb{R}^{S\times D}$ 表示输入 `pseudo_query`；$Q_{s,d}$ 表示第 $s$ 个目标 slot 在第 $d$ 个 hidden dimension 上的值。
    - $\operatorname{valid\_blocks}$ 表示输入 `valid_blocks`，其唯一元素与 $N$ 的较小值得到当前参与计算的有效历史来源数 $N_v$。
    - $\epsilon$ 表示输入 `eps`，是计算 RMS 归一化因子时使用的数值稳定项。
    - $T$ 表示 token 数，$N$ 表示历史来源缓冲区容量，$S$ 表示目标 slot 数，$D$ 表示 hidden dimension。
    - $t\in[0,T)$、$n\in[0,N_v)$、$s\in[0,S)$、$d\in[0,D)$ 分别表示 token、有效历史来源、目标 slot 和 hidden dimension 的索引。
    - $R_{t,n}$ 表示第 $t$ 个 token 的第 $n$ 个有效历史来源对应的 RMS 归一化因子，$R_{t,n}^{-1}$ 表示其倒数。
    - $Z_{s,t,n}$ 表示第 $s$ 个目标 slot、第 $t$ 个 token 与第 $n$ 个有效历史来源之间的归一化注意力 logit。
    - $M_{s,t}$ 表示 $Z_{s,t,n}$ 在所有有效历史来源上的最大值，用于执行数值稳定的 max-shift。
    - $E_{s,t,n}$ 表示 max-shift 后的指数值，即 $\exp(Z_{s,t,n}-M_{s,t})$。
    - $L_{s,t}$ 表示所有有效历史来源的指数和，是后续计算 softmax 归一化时使用的分母。
    - $O_{s,t,d}$ 表示所有有效历史来源按 $E_{s,t,n}$ 加权求和得到的 softmax 加权分子；$O$、$M$、$L$ 分别对应返回值 `numerator`、`logit_max`、`exp_sum`。
    - $\sum$、$\max$ 和 $\exp$ 分别表示求和、取最大值和自然指数运算，$\mathbb{R}$ 表示实数域。

## 函数原型

```python
cann_ops_transformer.block_attn_res_prepare(
    block_res,
    valid_blocks,
    pseudo_query,
    *,
    eps=1.0e-6,
) -> Tuple[Tensor, Tensor, Tensor]
```

## 参数说明

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
| --- | --- | --- | --- | --- | --- |
| block_res | Tensor | 必选 | 历史残差块，对应公式中的 $V$。数据格式为 ND；$T$ 为 token 数，$N$ 为可容纳的历史残差块数，$D$ 为隐藏特征维度。 | float32 | [T, N, D] |
| valid_blocks | Tensor | 必选 | 当前有效历史残差块数 $N_v$。数据格式为 ND。值为0时，`numerator` 返回全0 Tensor、`logit_max` 返回所有元素均为float32最小有限值的Tensor、`exp_sum` 返回全0 Tensor；值大于 $N$ 时按 $N$ 处理。 | uint64 | [1] |
| pseudo_query | Tensor | 必选 | 伪 Query，对应公式中的 $Q$；每个目标 slot 对应一个 Query 向量，最后一维必须与 `block_res` 的最后一维相同。数据格式为 ND。 | float32 | [S, D] |
| eps | float | 可选 | RMS 归一化的数值稳定项，必须为有限正数，默认值为 `1.0e-6`。 | float | - |

## 返回值说明

返回由 `numerator`、`logit_max` 和 `exp_sum` 组成的三元组。当 `valid_blocks[0] == 0` 时，三者组成 online softmax 的空状态：`numerator` 为全 0 Tensor，`logit_max` 的所有元素均为 float32 最小有限值，`exp_sum` 为全 0 Tensor。

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
| --- | --- | --- | --- | --- | --- |
| numerator | Tensor | 必选 | softmax 加权分子，对应公式中的 $O$。数据格式为 ND。 | float32 | [S, T, D] |
| logit_max | Tensor | 必选 | 有效历史来源 logits 的最大值，对应公式中的 $M$。数据格式为 ND。 | float32 | [S, T] |
| exp_sum | Tensor | 必选 | max-shift 后的指数和，对应公式中的 $L$。数据格式为 ND。 | float32 | [S, T] |

## 约束说明

- 该接口支持训练、推理场景下使用。
- 该接口支持单算子模式调用，暂不支持 TorchAir 图模式调用。
- shape要求：`T >= 0`，`S >= 0`，`1 <= N <= 64`，`1 <= D <= 8192`。
- `block_res.shape[2]` 必须等于 `pseudo_query.shape[1]`。
- 所有输入和输出 Tensor 均必须为连续 Tensor，不支持非连续 Tensor。
- `valid_blocks[0] == 0` 时，`numerator` 返回全 0 Tensor、`logit_max` 返回所有元素均为 float32 最小有限值的 Tensor、`exp_sum` 返回全 0 Tensor；`valid_blocks[0] > N` 时按 `N` 处理。
- `T == 0` 或 `S == 0` 时，返回对应 shape 的空输出。
- `block_res` 或 `pseudo_query` 含 `NaN`、`Inf`，或 FP32 平方、乘法及累加发生溢出时，结果可能包含 `NaN` 或 `Inf`。

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

    block_res = torch.tensor(
        [
            [
                [1.0, 2.0, 3.0, 4.0],
                [2.0, 0.0, -1.0, 1.0],
            ]
        ],
        dtype=torch.float32,
        device="npu",
    )
    valid_blocks = torch.tensor([2], dtype=torch.uint64, device="npu")
    pseudo_query = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
        device="npu",
    )

    # eps 为可选参数；省略时使用默认值 1.0e-6。
    numerator, logit_max, exp_sum = cann_ops_transformer.block_attn_res_prepare(
        block_res,
        valid_blocks,
        pseudo_query,
    )
    torch_npu.npu.synchronize()

    print("numerator:", numerator)
    print("logit_max:", logit_max)
    print("exp_sum:", exp_sum)
    ```

- TorchAir 图模式调用：暂不支持。
