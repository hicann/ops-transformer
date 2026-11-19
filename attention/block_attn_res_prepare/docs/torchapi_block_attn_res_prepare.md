# block_attn_res_prepare

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT系列产品</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3系列产品</term>：不支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2系列产品</term>：不支持
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

- 接口功能：完成Attention Residuals历史残差注意力两阶段计算的第一阶段，批量计算全部 $S$ 个目标slot的历史在线Softmax状态 $(O,M,L)$，供下一阶段与当前残差合并。

- 计算公式：

  设 $V$ 为历史残差`block_res`的FP32表示，$Q$ 为有效Query`pseudo_query`，$N_v=\min(\texttt{valid\_blocks[0]},N)$ 为有效历史残差块数，$\epsilon=\operatorname{FP32}(\texttt{eps})$。

  当 $N_v>0$ 时，先计算历史残差的均方根归一化（Root Mean Square Normalization，RMSNorm）和注意力logit：

  $$
  \widehat V_{t,n}=\operatorname{RMSNorm}(V_{t,n},\mathbf{1},\epsilon),\qquad
  Z_{s,t,n}=\left\langle Q_s,\widehat V_{t,n}\right\rangle
  $$

  再沿有效历史残差块维度计算在线Softmax状态：

  $$
  M_{s,t}=\max_{0\le n<N_v}Z_{s,t,n},\qquad
  E_{s,t,n}=\exp(Z_{s,t,n}-M_{s,t})
  $$

  $$
  L_{s,t}=\sum_{n=0}^{N_v-1}E_{s,t,n},\qquad
  O_{s,t,d}=\sum_{n=0}^{N_v-1}E_{s,t,n}V_{t,n,d}
  $$

  其中：

  - $T$、$N$、$S$、$D$ 分别表示token数、历史残差块容量、目标slot数和隐藏维度；$t\in[0,T)$、$n\in[0,N_v)$、$s\in[0,S)$、$d\in[0,D)$ 为对应索引，$V_{t,n}$ 和 $Q_s$ 为 $D$ 维向量。
  - RMSNorm沿 $D$ 维计算，缩放参数为全1向量；打分所需的缩放参数可预先与原始Query逐元素相乘，合并到 $Q$ 中。
  - $O$、$M$、$L$ 分别对应输出`numerator`、`logit_max`和`exp_sum`。$O$ 是未除以 $L$ 的加权分子。

  当 $N_v=0$ 时，返回在线Softmax的空状态：

  $$
  O=\mathbf{0}_{S\times T\times D},\qquad
  M=m_{\min}\mathbf{1}_{S\times T},\qquad
  L=\mathbf{0}_{S\times T},\qquad
  m_{\min}=-3.4028234663852886\times10^{38}
  $$

  $m_{\min}$ 为FLOAT32最小有限值。

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
| block_res | Tensor | 必选 | 历史残差块，对应公式中的 $V$。数据格式为ND。bfloat16输入转换为float32进行计算。 | torch.bfloat16、torch.float32 | [T, N, D] |
| valid_blocks | Tensor | 必选 | ND格式的历史残差块计数。0表示空历史，公式中的 $N_v$ 取该计数与 $N$ 的较小值。 | torch.uint64 | [1] |
| pseudo_query | Tensor | 必选 | 伪Query，对应公式中的 $Q$；每个目标slot对应一个Query向量。数据格式为ND。 | torch.float32 | [S, D] |
| eps | float | 可选 | RMS归一化的数值稳定项，对应公式中的 $\epsilon$，经校验后作为FLOAT32属性参与计算。必须为有限值，且位于float32正规正数范围`[FLT_MIN, FLT_MAX]`（约`[1.1754943508222875e-38, 3.4028234663852886e+38]`），不支持次正规数。默认值为`1.0e-6`。 | float64 | - |

## 返回值说明

返回`(numerator, logit_max, exp_sum)`。三个输出均为在输入设备上新分配的连续Tensor。

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
| --- | --- | --- | --- | --- | --- |
| numerator | Tensor | 必选 | softmax加权分子，对应公式中的 $O$。数据格式为ND。 | torch.float32 | [S, T, D] |
| logit_max | Tensor | 必选 | 有效历史来源logit的最大值，对应公式中的 $M$。数据格式为ND。 | torch.float32 | [S, T] |
| exp_sum | Tensor | 必选 | max-shift后的指数和，对应公式中的 $L$。数据格式为ND。 | torch.float32 | [S, T] |

## 约束说明

- 该接口支持训练中的前向计算和推理。
- 该接口支持单算子模式调用，暂不支持TorchAir图模式调用。
- shape要求：`T >= 0`，`S >= 0`，`1 <= N <= 64`，`1 <= D <= 8192`。
- 非空计算（`T > 0`且`S > 0`）时，`T`和`S`不能超过`4294967295`（`UINT32_MAX`），`T * S`不能超过`9223372036854775807`（`INT64_MAX`）。
- `block_res.shape[2]`必须等于`pseudo_query.shape[1]`。
- 所有输入和输出Tensor均须连续。
- 所有输入Tensor必须位于同一设备。
- `T == 0`或`S == 0`时，在参数校验和形状推导通过后返回对应shape的空输出，不发起AICore计算；上述非空计算的上限不适用于该路径，`eps`有效范围及其他接口校验仍适用。
- `block_res`或`pseudo_query`含`NaN`、`Inf`，或FP32平方、乘法及累加发生溢出时，结果可能包含`NaN`或`Inf`。

## 确定性/Batch一致性

- 确定性说明：默认支持确定性计算。

- Batch一致性说明：默认不支持Batch一致性。

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
      dtype=torch.bfloat16,
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

  # 省略eps，使用默认值1.0e-6。
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

- TorchAir图模式调用：暂不支持。
