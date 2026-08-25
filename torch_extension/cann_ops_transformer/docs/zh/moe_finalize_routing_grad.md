# moe_finalize_routing_grad

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

- 接口功能：`moe_finalize_routing_grad`是`moe_finalize_routing`的反向算子，封装`aclnnMoeFinalizeRoutingV2Grad`接口。该接口根据正向输出的`expanded_row_idx`，将`grad_y`（即正向输出`y`的梯度）按路由权重`scales`映射回`expanded_x`的梯度`grad_expanded_x`和`scales`的梯度`grad_scales`。

- 计算公式：

  设`i`为扁平索引，范围为`[0, R * K)`，`R`为token行数，`K`为top-K专家个数：

  - `scales`未传入时（K必须为1）：

    $$
    grad\_expanded\_x[expanded\_row\_idx[i]][j] = grad\_y[i / K][j]
    $$

  - `scales`传入、`bias`未传入时：

    $$
    grad\_expanded\_x[expanded\_row\_idx[i]][j] = grad\_y[i / K][j] \cdot scales[i / K][i \% K]
    $$

    $$
    grad\_scales[i] = \sum_{j} expanded\_x[expanded\_row\_idx[i]][j] \cdot grad\_y[i / K][j]
    $$

  - `scales`和`bias`均传入时：

    $$
    grad\_expanded\_x[expanded\_row\_idx[i]][j] = grad\_y[i / K][j] \cdot scales[i / K][i \% K]
    $$

    $$
    grad\_scales[i] = \sum_{j} (expanded\_x[expanded\_row\_idx[i]][j] + bias[expert\_idx[i]][j]) \cdot grad\_y[i / K][j]
    $$

- 说明：

  - 该反向算子仅计算`grad_expanded_x`和`grad_scales`，不计算`x1`、`x2`、`bias`的梯度（在自动反向下返回`None`）。
  - `expanded_row_idx`的排布布局与正向不同：正向使用`(K, R)`布局（`idx[k * R + row]`），本反向算子使用`(R, K)`布局（`idx[row * K + k]`）。通过自动反向调用时，框架会自动完成布局转置；手动调用时需自行确保布局正确。

## 函数原型

```python
cann_ops_transformer.moe_finalize_routing_grad(
    grad_y: torch.Tensor,
    expanded_row_idx: torch.Tensor,
    expanded_x: Optional[torch.Tensor] = None,
    scales: Optional[torch.Tensor] = None,
    expert_idx: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    drop_pad_mode: int = 0,
    active_num: int = 0,
    expert_num: int = 0,
    expert_capacity: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]
```

## 参数说明

|  参数名   | 参数类型 |  可选/必选 |    描述    |     数据类型    |   维度(shape)   |
|---------|----------|----------|---------------|---------------|------------|
|  grad_y  |  Tensor |  必选  |  正向输出`y`的梯度。  |  float16、bfloat16、float32   | `(R, H)` |
|  expanded_row_idx  |  Tensor |  必选  |  行索引映射，采用`(R, K)`布局。注意与正向的`(K, R)`布局不同。Dropless场景下值域为`[0, R * K - 1]`且不重复；DropPad场景下值域为`[-1, E * C - 1]`，除-1外不重复（-1表示丢弃）。当`scales`未传入时K必须为1。  |  int32   | `(R * K,)` |
|  expanded_x  |  Tensor |  可选  |  正向输入`expanded_x`，用于计算`grad_scales`。传入`scales`时必须同时传入。数据类型需与`grad_y`一致。  |  与grad_y一致   | `(NUM_ROWS * K, H)`或`(E, C, H)` |
|  scales  |  Tensor |  可选  |  正向输入`scales`，即路由权重系数。传入`scales`时必须同时传入`expanded_x`。  |  float16、bfloat16、float32   | `(R, K)` |
|  expert_idx  |  Tensor |  可选  |  正向输入`expert_idx`，即专家索引。传入`bias`时必须同时传入。值域为`[0, E-1]`。  |  int32   | `(R, K)` |
|  bias  |  Tensor |  可选  |  正向输入`bias`，即偏置量。  |  与grad_y一致   | `(E, H)` |
|  drop_pad_mode  |  int |  可选  |  丢弃/填充模式，需与正向保持一致。`0`：drop less；`1`：drop pad。默认值为0。  |  -   | - |
|  active_num  |  int |  可选  |  Dropless场景下`grad_expanded_x`的最大输出行数，仅当大于0且小于`R * K`时生效；DropPad场景下不生效。默认值为0。  |  -   | - |
|  expert_num  |  int |  可选  |  专家数。Dropless场景下不生效；DropPad场景下必须大于0，且当`bias`传入时必须等于`bias`的第0维大小（E）。默认值为0。  |  -   | - |
|  expert_capacity  |  int |  可选  |  每个专家能够处理的tokens数，`drop_pad_mode=1`时必须传入且大于0。默认值为0。  |  -   | - |

## 返回值说明

|  参数名   | 参数类型 |  可选/必选 |    描述    |     数据类型    |   维度(shape)   |
|---------|----------|----------|---------------|---------------|------------|
|  grad_expanded_x  |  Tensor |  必选  |  `expanded_x`的梯度。Dropless场景下shape为`(dim0, H)`，其中`dim0`为`expanded_row_idx.numel()`（当`active_num=0`时）或`min(active_num, expanded_row_idx.numel())`（当`active_num > 0`时）；DropPad场景下shape为`(expert_num, expert_capacity, H)`。数据类型与`grad_y`一致。  |  与grad_y一致   | `(dim0, H)`或`(E, C, H)` |
|  grad_scales  |  Tensor |  必选  |  `scales`的梯度。shape为`(R, K)`，其中`K`为`scales`的第二维（当`scales`传入时）或1（当`scales`未传入时）。数据类型：当`scales`传入时与`scales`一致，否则与`grad_y`一致。  |  float16、bfloat16、float32   | `(R, K)` |

## 约束说明

- 该接口支持训练场景下使用。
- 该接口支持单算子模式和图模式。
- `grad_y`必须是2维张量。
- `expanded_row_idx`必须是1维张量。
- `drop_pad_mode`仅支持取值0或1，不支持2或3（行排列模式）。
- 传入`bias`时，`expert_idx`必须同时传入。
- 传入`scales`时，`expanded_x`必须同时传入。
- `drop_pad_mode=1`时，`expert_num`必须大于0（当`bias`传入时必须等于`bias`的第0维大小E），`expert_capacity`必须大于0。
- `expanded_x`、`bias`的数据类型必须与`grad_y`一致。`scales`的数据类型在<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>上必须与`grad_y`一致；在<term>Ascend 950PR/Ascend 950DT</term>上可以与`grad_y`不一致。
- 该反向算子仅支持常规专家场景，不支持aclnnMoeFinalizeRoutingV4特有特性（正向的`x`、`alpha1`、`alpha2`、`v`以及有效的`zero_expert_range`、`copy_expert_range`、`constant_expert_range`）。当正向使用了这些特性时，调用自动反向会抛出`NotImplementedError`。
- 该算子为[moe\_finalize\_routing](./moe_finalize_routing.md)的反向算子，各参数需与正向调用保持一致。
- `expanded_row_idx`布局差异：正向`moe_finalize_routing`的`expanded_row_idx`采用`(K, R)`布局（`drop_pad_mode`为0或1时），本反向算子采用`(R, K)`布局。自动反向下框架会自动转置；手动调用时需注意提供正确布局。

## 确定性计算

默认支持确定性计算。

## 配套接口

该算子为[moe\_finalize\_routing](./moe_finalize_routing.md)的反向算子。

> **说明**：当正向 `moe_finalize_routing` 中 `expanded_x`、`scales` 等可微输入的 `requires_grad` 为True，且未使用aclnnMoeFinalizeRoutingV4特有特性（`x`、`alpha1`、`alpha2`、`v`、`zero_expert_range`等）时，`loss.backward()` 会自动触发本算子，无需手动调用。仅在需要显式控制梯度的场景下保留手动调用路径。注意：`x1`、`x2`、`bias`的梯度不会被计算，如需对`x1`/`x2`求梯度，建议使用外部残差加法替代将其作为正向输入。

## 调用示例

- 单算子模式调用：

  ```python
  import torch
  import torch_npu
  from cann_ops_transformer.ops import moe_finalize_routing, moe_finalize_routing_grad

  torch_npu.npu.set_device(0)

  NUM_ROWS = 2
  K = 2
  H = 8
  E = 4

  expanded_x = torch.randn(NUM_ROWS * K, H, dtype=torch.float32, device="npu")
  # 正向expanded_row_idx为(K, R)布局：idx[k*R + row]
  forward_row_idx = torch.tensor([0, 1, 2, 3], dtype=torch.int32, device="npu")
  scales = torch.randn(NUM_ROWS, K, dtype=torch.float32, device="npu")
  expert_idx = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32, device="npu")
  bias = torch.randn(E, H, dtype=torch.float32, device="npu")

  # 正向
  y = moe_finalize_routing(expanded_x, forward_row_idx, bias=bias,
                           scales=scales, expert_idx=expert_idx, k=K)

  # 手动调用反向：需将expanded_row_idx从(K, R)布局转置为(R, K)布局
  grad_row_idx = forward_row_idx.reshape(K, NUM_ROWS).t().contiguous().reshape(-1)
  grad_y = torch.randn_like(y)
  grad_expanded_x, grad_scales = moe_finalize_routing_grad(
      grad_y, grad_row_idx, expanded_x=expanded_x, scales=scales,
      expert_idx=expert_idx, bias=bias, drop_pad_mode=0,
  )
  print(grad_expanded_x.shape)  # torch.Size([4, 8])
  print(grad_scales.shape)      # torch.Size([2, 2])
  ```

- 自动反向调用：

  ```python
  import torch
  import torch_npu
  from cann_ops_transformer.ops import moe_finalize_routing

  torch_npu.npu.set_device(0)

  NUM_ROWS = 4
  K = 2
  H = 8

  expanded_x = torch.randn(NUM_ROWS * K, H, dtype=torch.float32, device="npu",
                           requires_grad=True)
  row_idx = torch.arange(NUM_ROWS * K, dtype=torch.int32, device="npu")
  scales = torch.randn(NUM_ROWS, K, dtype=torch.float32, device="npu",
                       requires_grad=True)

  # 正向（expanded_x.requires_grad=True时自动启用autograd）
  y = moe_finalize_routing(expanded_x, row_idx, scales=scales, k=K)

  # 反向自动触发moe_finalize_routing_grad
  loss = y.sum()
  loss.backward()
  print(expanded_x.grad.shape)  # torch.Size([8, 8])
  print(scales.grad.shape)      # torch.Size([4, 2])
  ```
