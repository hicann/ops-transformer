# moe_init_routing_grad

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

- 接口功能：`moe_init_routing_grad`是`moe_init_routing`的反向算子，封装`aclnnMoeInitRoutingV2Grad`接口。该接口根据正向输出的`expanded_row_idx`，将`grad_expanded_x`（即`expanded_x`的梯度）按行映射回原始输入`x`的梯度`grad_x`。

- 计算公式：

  $$
  grad\_x[i] = \sum_{t=0}^{K-1} grad\_expanded\_x[expanded\_row\_idx[i \cdot K + t]]
  $$

  其中`i`范围为`[0, NUM_ROWS)`，`K`为`top_k`，`expanded_row_idx`为正向`moe_init_routing`的输出。DropPad场景下`expanded_row_idx`值为-1的项不参与累加。

## 函数原型

```python
cann_ops_transformer.moe_init_routing_grad(
    grad_expanded_x: torch.Tensor,
    expanded_row_idx: torch.Tensor,
    top_k: int,
    drop_pad_mode: int = 0,
    active_num: int = 0,
) -> torch.Tensor
```

## 参数说明

|  参数名   | 参数类型 |  可选/必选 |    描述    |     数据类型    |   维度(shape)   |
|---------|----------|----------|---------------|---------------|------------|
|  grad_expanded_x  |  Tensor |  必选  |  正向输出`expanded_x`的梯度。Dropless场景下为2维张量，shape为`(active_num, H)`或`(NUM_ROWS * K, H)`；DropPad场景下为3维张量，shape为`(expert_num, expert_capacity, H)`。  |  float16、bfloat16、float32   | ND |
|  expanded_row_idx  |  Tensor |  必选  |  正向`moe_init_routing`的输出`expanded_row_idx`，表示`expanded_x`与`x`的行映射关系。为1维张量，shape为`(NUM_ROWS * K,)`。  |  int32   | ND |
|  top_k  |  int |  必选  |  正向输入`expert_idx`的第二维大小K，即每个token选出的top-K专家个数。取值需大于0。  |  -   | - |
|  drop_pad_mode  |  int |  可选  |  默认值为0，表示丢弃/填充模式，需与正向保持一致。0表示Dropless场景；1表示DropPad场景。  |  -   | - |
|  active_num  |  int |  可选  |  默认值为0，表示正向`moe_init_routing`的`active_num`参数值。取值需大于等于0，当`drop_pad_mode=0`时生效：0表示非Active场景，大于0表示Active场景（此时`grad_expanded_x`的第0维大小必须等于`active_num`）。  |  -   | - |

## 返回值说明

|  参数名   | 参数类型 |  可选/必选 |    描述    |     数据类型    |   维度(shape)   |
|---------|----------|----------|---------------|---------------|------------|
|  grad_x  |  Tensor |  必选  |  输入`x`的梯度。shape为`(NUM_ROWS, H)`，其中`NUM_ROWS = expanded_row_idx.numel() // top_k`，`H`为`grad_expanded_x`的最后一维（DropPad场景下为第三维）。数据类型与`grad_expanded_x`一致。  |  与grad_expanded_x一致   | ND |

## 约束说明

- 该接口支持训练场景下使用。
- 该接口支持单算子模式和图模式。
- `grad_expanded_x`在Dropless场景下必须是2维张量，在DropPad场景下必须是3维张量。
- `expanded_row_idx`必须是1维张量，其长度必须能被`top_k`整除。
- `drop_pad_mode`仅支持取值0或1。
- `top_k`必须大于0。
- `active_num`必须大于等于0。
- Dropless场景且`active_num=0`时，`grad_expanded_x`的第0维大小必须与`expanded_row_idx`的长度一致；Dropless场景且`active_num>0`时，`grad_expanded_x`的第0维大小必须等于`active_num`。
- `grad_expanded_x`的最后一维（DropPad场景下为第三维）必须与输出`grad_x`的第二维一致。
- 该算子为[moe\_init\_routing](./moe_init_routing.md)的反向算子，`expanded_row_idx`、`top_k`、`drop_pad_mode`、`active_num`需与正向调用保持一致。
- 该反向算子仅支持非量化场景（正向`quant_mode=-1`），不支持aclnnMoeInitRoutingV4特有特性（`scale`、`offset`、`topk_weight`、`x_dtype`、`row_idx_type`非0）。当正向使用了这些特性时，调用自动反向会抛出`NotImplementedError`。

## 确定性计算

默认支持确定性计算。

## 配套接口

该算子为[moe\_init\_routing](./moe_init_routing.md)的反向算子。

> **说明**：当正向 `moe_init_routing` 中 `x.requires_grad` 为True，且未使用aclnnMoeInitRoutingV4特有特性（量化、`scale`、`offset`、`topk_weight`等）时，`loss.backward()` 会自动触发本算子，无需手动调用。仅在需要显式控制梯度的场景下保留手动调用路径。

## 调用示例

- 单算子模式调用：

  ```python
  import torch
  import torch_npu
  from cann_ops_transformer.ops import moe_init_routing, moe_init_routing_grad

  torch_npu.npu.set_device(0)

  n = 4
  h = 8
  k = 2
  expert_num = 8
  drop_pad_mode = 0
  expert_tokens_num_type = 1
  expert_tokens_num_flag = True
  quant_mode = -1
  active_expert_range = [0, 4]
  row_idx_type = 0

  x = torch.randn((n, h), dtype=torch.float32, device="npu")
  expert_idx = torch.randint(0, expert_num, (n, k), dtype=torch.int32, device="npu")

  # 正向
  expanded_x, expanded_row_idx, expert_tokens_count, expanded_scale, expanded_topk_weight = \
      moe_init_routing(x, expert_idx, active_num=-1, expert_capacity=-1,
                       expert_num=expert_num, drop_pad_mode=drop_pad_mode,
                       expert_tokens_num_type=expert_tokens_num_type,
                       expert_tokens_num_flag=expert_tokens_num_flag,
                       quant_mode=quant_mode, active_expert_range=active_expert_range,
                       row_idx_type=row_idx_type)

  # 手动调用反向
  grad_expanded_x = torch.randn_like(expanded_x)
  grad_x = moe_init_routing_grad(grad_expanded_x, expanded_row_idx, top_k=k,
                                 drop_pad_mode=drop_pad_mode, active_num=0)
  print(grad_x.shape)  # torch.Size([4, 8])
  ```

- 自动反向调用：

  ```python
  import torch
  import torch_npu
  from cann_ops_transformer.ops import moe_init_routing

  torch_npu.npu.set_device(0)

  n = 4
  h = 8
  k = 2
  expert_num = 8

  x = torch.randn((n, h), dtype=torch.float32, device="npu", requires_grad=True)
  expert_idx = torch.randint(0, expert_num, (n, k), dtype=torch.int32, device="npu")

  # 正向（x.requires_grad=True时自动启用autograd）
  expanded_x, expanded_row_idx, expert_tokens_count, expanded_scale, expanded_topk_weight = \
      moe_init_routing(x, expert_idx, active_num=-1, expert_capacity=-1,
                       expert_num=expert_num, drop_pad_mode=0,
                       expert_tokens_num_type=1, expert_tokens_num_flag=True,
                       quant_mode=-1, active_expert_range=[0, 4], row_idx_type=0)

  # 反向自动触发moe_init_routing_grad
  loss = expanded_x.sum()
  loss.backward()
  print(x.grad.shape)  # torch.Size([4, 8])
  ```
