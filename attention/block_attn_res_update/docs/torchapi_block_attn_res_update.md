# block_attn_res_update

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

- **接口功能**：

  Attention Residuals的历史残差注意力两段式计算的阶段二：将`delta`原地累加到`partial_block`，计算更新后残差的RMSNorm score；随后与阶段一`block_attn_res_prepare`返回的softmax统计量`numerator`、`logit_max`、`exp_sum`（分别对应O、M、L）计算得到输出`h`，同时将更新后的残差原地写回`partial_block`。
- **计算公式**：

  设`T`表示token数，`D`表示hidden size。主要计算过程如下：

  1. 更新当前`partial_block`：

     $$
     p[t,d] = partial\_block[t,d] + \operatorname{FP32}(delta[t,d])
     $$
  2. 计算RMSNorm分母和当前score：

     $$
     r[t] = \sqrt{\frac{1}{D}\sum_{d=0}^{D-1}p[t,d]^2 + eps}
     $$

     $$
     score[t] = \frac{\sum_{d=0}^{D-1}p[t,d] \times pseudo\_query[d]}{r[t]}
     $$
  3. 将当前score与历史online softmax状态合并：

     $$
     m[t] = \max(logit\_max[t], score[t])
     $$

     $$
     alpha[t] = \exp(logit\_max[t] - m[t]), \qquad beta[t] = \exp(score[t] - m[t])
     $$

     $$
     ell[t] = exp\_sum[t] \times alpha[t] + beta[t]
     $$

     $$
     updated\_numerator[t,d] = numerator[t,d] \times alpha[t] + p[t,d] \times beta[t]
     $$
  4. 生成输出：

     $$
     partial\_block[t,d] \leftarrow p[t,d]
     $$

     $$
     h[t,d] = \operatorname{Cast}_{delta.dtype}\left(\frac{updated\_numerator[t,d]}{ell[t]}\right)
     $$

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
) -> Tensor
```

## 参数说明

| 参数名          | 参数类型  | 可选/必选 | 描述                                                       | 数据类型           | 维度(shape) |
| --------------- | --------- | --------- | ---------------------------------------------------------- | ------------------ | ----------- |
| `partial_block` | Tensor    | 必选      | 当前已累计的`partial_block`。调用完成后被原地更新。         | `torch.float32`    | `(T, D)`    |
| `delta`         | Tensor    | 必选      | 本次需要累加的`delta`。                                    | `torch.bfloat16`   | `(T, D)`    |
| `pseudo_query`  | Tensor    | 必选      | 用于计算当前`partial_block` logit的`pseudo_query`。         | `torch.float32`    | `(D,)`      |
| `numerator`     | Tensor    | 必选      | `block_attn_res_prepare`输出的历史online softmax加权和。    | `torch.float32`    | `(T, D)`    |
| `logit_max`     | Tensor    | 必选      | `block_attn_res_prepare`输出的历史最大logit。               | `torch.float32`    | `(T,)`      |
| `exp_sum`       | Tensor    | 必选      | `block_attn_res_prepare`输出的历史softmax分母累积值。       | `torch.float32`    | `(T,)`      |
| `eps`           | `float`   | 可选      | RMSNorm计算中的数值稳定项，默认值为`1e-6`。                 | `float64`          | -           |

## 返回值说明

| 参数名 | 参数类型 | 可选/必选 | 描述                                                                   | 数据类型         | 维度(shape) |
| ------ | -------- | --------- | ---------------------------------------------------------------------- | ---------------- | ----------- |
| `h`    | Tensor   | 必选      | `block_attn_res_update`输出。数据类型和shape与`delta`一致，为连续Tensor。 | `torch.bfloat16` | `(T, D)`    |

## 约束说明

- 该接口支持推理场景下使用。
- 该接口支持单算子模式和aclgraph模式调用。
- 所有 Tensor 输入和输出必须连续。
- `T`满足`T >= 0`，`D`满足`0 <= D <= 8192`。
- 当`T == 0`或`D == 0`时支持空Tensor返回。
- `partial_block`、`delta`和`numerator`的shape必须一致，均为`(T, D)`。
- `pseudo_query`的shape必须为`(D,)`，`logit_max`和`exp_sum`的shape必须为`(T,)`。
- `eps`必须为有限正数；省略时使用默认值`1e-6`。Torch C++ bridge在调用ACLNN前将其显式转换为float32。
- Tensor数据类型组合必须满足：

  | partial_block     | delta              | pseudo_query      | numerator         | logit_max         | exp_sum           | h                  |
  | ----------------- | ------------------ | ----------------- | ----------------- | ----------------- | ----------------- | ------------------ |
  | `torch.float32`   | `torch.bfloat16`   | `torch.float32`   | `torch.float32`   | `torch.float32`   | `torch.float32`   | `torch.bfloat16`   |

## 确定性计算

默认支持确定性计算。

## 调用示例

- 单算子模式调用：

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
  numerator = torch.rand((T, D), dtype=torch.float32).npu()
  logit_max = torch.rand((T,), dtype=torch.float32).npu()
  exp_sum = torch.rand((T,), dtype=torch.float32).npu()

  # 调用单算子接口
  h = cann_ops_transformer.block_attn_res_update(
      partial_block,
      delta,
      pseudo_query,
      numerator,
      logit_max,
      exp_sum,
  )
  ```

- aclgraph模式调用：

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
  numerator = torch.rand((T, D), dtype=torch.float32).npu()
  logit_max = torch.rand((T,), dtype=torch.float32).npu()
  exp_sum = torch.rand((T,), dtype=torch.float32).npu()

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
