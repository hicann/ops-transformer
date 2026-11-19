# block_attn_res_update

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

- 接口功能：将`delta`原地累加到`partial_block`，计算更新后残差的归一化score，与历史在线Softmax状态合并得到注意力输出，并支持对BF16混合结果执行输出均方根归一化（Root Mean Square Normalization，RMSNorm）。

- 版本演进：本接口统一调用[aclnnBlockAttnResUpdateV2](aclnnBlockAttnResUpdateV2.md)，通过`output_norm_weight`和独立的`output_norm_eps`启用输出RMSNorm；权重为`None`时保留原有计算行为。

- 计算公式：

  对每个token $t$，先更新局部残差并原地写回：

  $$
  p_t=P_t+\operatorname{FP32}(\Delta_t),\qquad
  P^{\mathrm{out}}_t=p_t
  $$

  通过RMSNorm和Query点积得到当前残差的score：

  $$
  C_t=\left\langle Q,\operatorname{RMSNorm}(p_t,\mathbf{1},\epsilon)\right\rangle
  $$

  将当前残差与历史在线Softmax状态合并：

  $$
  m_t=\max(M_t,C_t),\qquad
  \alpha_t=\exp(M_t-m_t),\qquad
  \beta_t=\exp(C_t-m_t)
  $$

  $$
  H^{(0)}_t=\operatorname{BF16}\left(
  \frac{\alpha_t O_t+\beta_t p_t}{\alpha_t L_t+\beta_t}
  \right)
  $$

  `output_norm_weight`为`None`时，输出 $H=H^{(0)}$；否则对BF16混合结果执行输出RMSNorm：

  $$
  H_t=\operatorname{BF16}\left(
  \operatorname{RMSNorm}\left(\operatorname{FP32}(H^{(0)}_t),\operatorname{FP32}(W),\epsilon_{\mathrm{out}}\right)
  \right)
  $$

  其中：

  - $T$ 和 $D$ 分别表示token数和隐藏维度，$t\in[0,T)$；$P_t$、$p_t$、$\Delta_t$、$O_t$、$H_t$ 和 $Q$ 均为 $D$ 维向量。
  - $P$ 和 $P^{\mathrm{out}}$ 分别表示`partial_block`调用前后的值，$\Delta$ 表示`delta`，$Q$ 表示`pseudo_query`；$p$ 为FP32残差累加结果。
  - $O$、$M$、$L$ 分别表示历史状态`numerator`、`logit_max`、`exp_sum`，$H$ 表示输出`h`。$W$ 表示`output_norm_weight`。
  - $\epsilon=\operatorname{FP32}(\texttt{eps})$，$\epsilon_{\mathrm{out}}=\operatorname{FP32}(\texttt{output\_norm\_eps})$，两者分别用于打分和输出归一化。
  - RMSNorm沿 $D$ 维计算，打分时的缩放参数为全1向量；打分所需的缩放参数可预先与原始Query逐元素相乘，合并到 $Q$ 中。

## 函数原型

```python
cann_ops_transformer.block_attn_res_update(
    partial_block,
    delta,
    pseudo_query,
    numerator,
    logit_max,
    exp_sum,
    *,
    eps=1e-6,
    output_norm_weight=None,
    output_norm_eps=1e-6,
) -> Tensor
```

## 参数说明

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
| --- | --- | --- | --- | --- | --- |
| `partial_block` | Tensor | 必选 | 局部残差 $P$，调用后原地保存FP32累加结果 $P^{\mathrm{out}}$。 | `torch.float32` | `[T, D]` |
| `delta` | Tensor | 必选 | 累加到局部残差的增量 $\Delta$。 | `torch.bfloat16` | `[T, D]` |
| `pseudo_query` | Tensor | 必选 | 计算当前残差归一化打分的有效Query $Q$。 | `torch.float32` | `[D]` |
| `numerator` | Tensor | 必选 | 历史在线Softmax状态的加权分子 $O$，只读。 | `torch.float32` | `[T, D]` |
| `logit_max` | Tensor | 必选 | 每个token的历史最大logit $M$，只读。 | `torch.float32` | `[T]` |
| `exp_sum` | Tensor | 必选 | 每个token的历史指数和 $L$，只读。 | `torch.float32` | `[T]` |
| `eps` | float | 可选 | 残差打分的稳定项 $\epsilon$，须为有限正数，默认值为`1e-6`。 | float64 | - |
| `output_norm_weight` | Optional[Tensor] | 可选 | 输出RMSNorm的缩放权重 $W$，默认值为`None`，表示禁用输出RMSNorm。 | `torch.bfloat16`、`torch.float32` | `[D]` |
| `output_norm_eps` | float | 可选 | 输出RMSNorm的稳定项 $\epsilon_{\mathrm{out}}$，默认值为`1e-6`。传入权重时取值范围为`[FLT_MIN, FLT_MAX]`，否则忽略。 | float64 | - |

## 返回值说明

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
| --- | --- | --- | --- | --- | --- |
| `h` | Tensor | 必选 | 当前层输出 $H$，为连续Tensor。传入`output_norm_weight`时返回输出RMSNorm结果。 | `torch.bfloat16` | `[T, D]` |

## 约束说明

- 该接口支持推理场景下使用。
- 该接口支持单算子模式和aclgraph模式调用。
- 所有Tensor输入和输出必须连续，且位于同一设备。
- `T >= 0`，`0 <= D <= 8192`。`partial_block`、`delta`、`numerator`、`h`的shape必须为`[T, D]`；`pseudo_query`和非空`output_norm_weight`为`[D]`；`logit_max`、`exp_sum`为`[T]`。
- `T == 0`或`D == 0`时，完成参数校验后返回对应shape的空输出；传入权重时，仍须满足其dtype、shape、布局及epsilon约束。
- `eps`以double传入ACLNN，再转换为`float32`参与计算；它必须为有限正数，非空计算时转换后的值也必须有限且大于0。
- 传入`output_norm_weight`时，`output_norm_eps`转换为`float32`前的值必须在`[FLT_MIN, FLT_MAX]`范围内；权重为`None`时，禁用输出RMSNorm并忽略`output_norm_eps`。`FLT_MIN = 2^-126 ≈ 1.1754943508222875e-38`，`FLT_MAX ≈ 3.4028234663852886e38`；不支持次正规epsilon，以免默认Sqrt的FTZ模式将其清零。
- 非空计算时，调用者须保证每个token的历史在线Softmax状态非空：`numerator`和`logit_max`中的元素有限，`exp_sum`中的元素有限且严格大于0，且三者来自同一份历史状态。这三个输入只读，不支持以零`numerator`、`logit_max=-inf`和`exp_sum=0`表示的空历史状态。
- 其他底层约束请参见[aclnnBlockAttnResUpdateV2](aclnnBlockAttnResUpdateV2.md)。

## 确定性计算

- 确定性说明：默认支持确定性计算。

## 调用示例

- 单算子模式调用（启用输出RMSNorm）：

  ```python
  import torch
  import torch_npu
  import cann_ops_transformer

  T = 32
  D = 7168

  # 创建输入
  partial_block = torch.rand((T, D), dtype=torch.float32).npu()
  delta = torch.rand((T, D), dtype=torch.bfloat16).npu()
  pseudo_query = torch.rand((D,), dtype=torch.float32).npu()
  # 用值和logit均为0的单个历史项构造同源的O/M/L状态。
  numerator = torch.zeros((T, D), dtype=torch.float32).npu()
  logit_max = torch.zeros((T,), dtype=torch.float32).npu()
  exp_sum = torch.ones((T,), dtype=torch.float32).npu()

  output_norm_weight = torch.ones((D,), dtype=torch.float32).npu()

  # 调用单算子接口
  h = cann_ops_transformer.block_attn_res_update(
      partial_block,
      delta,
      pseudo_query,
      numerator,
      logit_max,
      exp_sum,
      eps=1e-6,
      output_norm_weight=output_norm_weight,
      output_norm_eps=1e-6,
  )
  ```

- aclgraph模式调用（禁用输出RMSNorm）：

  ```python
  import torch
  import torch_npu
  import cann_ops_transformer

  # 定义待编译的网络
  class OneOp(torch.nn.Module):
      def forward(
          self,
          partial_block,
          delta,
          pseudo_query,
          numerator,
          logit_max,
          exp_sum,
      ):
          return cann_ops_transformer.block_attn_res_update(
              partial_block,
              delta,
              pseudo_query,
              numerator,
              logit_max,
              exp_sum,
              eps=1e-6,
          )

  # 使用npugraph_ex后端编译网络
  compiled_op = torch.compile(
      OneOp(),
      backend="npugraph_ex",
      fullgraph=True,
      dynamic=False,
      options={
          "static_kernel_compile": True,
      },
  )

  T = 8
  D = 7168

  # 创建输入
  partial_block = torch.rand((T, D), dtype=torch.float32).npu()
  delta = torch.rand((T, D), dtype=torch.bfloat16).npu()
  pseudo_query = torch.rand((D,), dtype=torch.float32).npu()
  # 用值和logit均为0的单个历史项构造同源的O/M/L状态。
  numerator = torch.zeros((T, D), dtype=torch.float32).npu()
  logit_max = torch.zeros((T,), dtype=torch.float32).npu()
  exp_sum = torch.ones((T,), dtype=torch.float32).npu()

  # 执行编译后的网络
  h = compiled_op(
      partial_block,
      delta,
      pseudo_query,
      numerator,
      logit_max,
      exp_sum,
  )
  ```
