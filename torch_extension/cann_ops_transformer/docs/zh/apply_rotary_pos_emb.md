# apply_rotary_pos_emb

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
- <term>Atlas 推理系列产品</term>：支持
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas 训练系列产品</term>：不支持
<!-- end id6 -->

## 功能说明

- **接口功能**：

    为提升网络性能，将query和key两路旋转位置编码融合为一次kernel调用，返回旋转位置编码后的query和key输出张量，输入张量不被修改。

- **计算公式**：

    `rotary_mode`为`"half"`时，将`query`、`key`沿最后一维二等分后执行旋转位置编码：

    $$query\_q1 = query[..., : query.shape[-1] // 2]$$

    $$query\_q2 = query[..., query.shape[-1] // 2 :]$$

    $$query\_rotate = cat((-query\_q2, query\_q1), dim=-1)$$

    $$q\_embed = (query \cdot cos) + query\_rotate \cdot sin$$

    `key`的计算方式与`query`相同，得到$k\_embed$。`rotary_mode`为`"interleave"`时，相邻两元素为一组配对旋转；`rotary_mode`为`"quarter"`时，将最后一维四等分后两两配对旋转。

- **说明**：

  - `layout`为`"TND"`时输入为3维Tensor，其他`layout`下输入为4维Tensor。其中B（Batch）表示批量大小，S（Seq-Length）表示序列长度，N（Head-Num）表示多头数，D（Head-Dim）表示每个头的隐藏维度大小，T表示B和S合轴。
  - `"BSH"`与`"BSND"`共用底层布局，按`"BSND"`的维度语义处理。
  - `rotary_mode`为`"half"`时，该接口支持自动微分，反向自动调用[apply_rotary_pos_emb_grad](./apply_rotary_pos_emb_grad.md)。

## 函数原型

```python
cann_ops_transformer.apply_rotary_pos_emb(query, key, cos, sin, layout="BSND", rotary_mode="half") -> (Tensor, Tensor)
```

## 参数说明

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
|--------|----------|-----------|------|----------|-------------|
| query | Tensor | 必选 | 待执行旋转位置编码的第一个张量。 | bfloat16、float16、float32 | `layout`为"TND"时3维，其他`layout`下4维 |
| key | Tensor | 必选 | 待执行旋转位置编码的第二个张量。 | 同query | 同query |
| cos | Tensor | 必选 | 旋转位置编码余弦值张量，N维度必须等于1。 | 同query | 同query |
| sin | Tensor | 必选 | 旋转位置编码正弦值张量，shape需与cos一致。 | 同query | 同cos |
| layout | str | 可选 | 输入张量布局格式，支持"BSND"、"BSH"、"SBND"、"BNSD"、"TND"。默认值为"BSND"。 | - | - |
| rotary_mode | str | 可选 | 旋转编码模式，支持"half"、"quarter"、"interleave"。默认值为"half"。 | - | - |

## 返回值说明

- **query_out**（`Tensor`）：旋转位置编码后的query输出张量，为新分配张量，shape和数据类型与输入`query`一致，输入`query`不被修改。
- **key_out**（`Tensor`）：旋转位置编码后的key输出张量，为新分配张量，shape和数据类型与输入`key`一致，输入`key`不被修改。

<!-- npu="950" id7 -->
> **自动微分说明**：仅支持<term>Ascend 950PR/Ascend 950DT</term>。当`query`、`key`、`cos`、`sin`中任一输入`requires_grad=True`且`rotary_mode="half"`时，该接口支持自动微分，对loss执行`.backward()`时自动调用`apply_rotary_pos_emb_grad`计算`query`、`key`、`cos`、`sin`四路梯度。`rotary_mode`为`"quarter"`或`"interleave"`时不支持自动微分。
<!-- end id7 -->

## 约束说明

- 该接口支持推理、训练场景下使用。
- 该接口支持单算子模式和图模式调用。
- 不支持空Tensor。
- 输入张量`query`、`key`、`cos`、`sin`的数据类型必须相同。
- `cos`、`sin`的shape必须相同，且N维度必须等于1。
- `rotary_mode`为"half"和"interleave"时，输入shape最后一维（D）必须被2整除；`rotary_mode`为"quarter"时，输入shape最后一维（D）必须被4整除。
<!-- npu="A3,910b,310p" id8 -->
- <term>Atlas 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：
  - `layout`仅支持"BSND"的4维Tensor、"TND"的3维Tensor。
  - `layout`为"BSND"时，`query`、`key`、`cos`、`sin`输入shape的前2维（B、S）必须相等；`layout`为"TND"时，第1维（T）必须相等。
  - `query`、`key`输入shape的最后一维（D）必须相等且等于128或64，`cos`、`sin`输入shape的最后一维（D）必须与之相等。
<!-- end id8 -->
<!-- npu="950" id9 -->
- <term>Ascend 950PR/Ascend 950DT</term>：
  - 训练场景自动微分仅支持`rotary_mode="half"`。
  - `layout`支持"BSND"、"SBND"、"BNSD"的4维Tensor，"TND"的3维Tensor。
  - 对于任意`layout`，`query`与`key`除N维度外其他维度必须相同。
  - `query`、`key`输入shape的最后一维（D）必须相等且小于等于1024；`cos`、`sin`输入shape的最后一维（D）必须相等，且小于等于`query`、`key`输入shape的最后一维（D）。
  - `layout`为"BSND"时，`cos`、`sin`的B维度可以等于1，也可以与`query`的B维度一致。
<!-- end id9 -->
<!-- npu="310p" id10 -->
- <term>Atlas 推理系列产品</term>：不支持`bfloat16`。
<!-- end id10 -->

## 确定性计算

默认支持确定性计算。

## 调用示例

- 单算子模式调用

    ```python
    import torch
    import torch_npu
    from cann_ops_transformer.ops import apply_rotary_pos_emb

    torch_npu.npu.set_device(0)

    B = 1
    S = 64
    N = 8
    D = 128

    query = torch.randn(B, S, N, D, device="npu", dtype=torch.float16)
    key = torch.randn(B, S, N, D, device="npu", dtype=torch.float16)
    cos = torch.randn(B, S, 1, D, device="npu", dtype=torch.float16)
    sin = torch.randn(B, S, 1, D, device="npu", dtype=torch.float16)

    query_out, key_out = apply_rotary_pos_emb(
        query,
        key,
        cos,
        sin,
        layout="BSND",
        rotary_mode="half",
    )

    print(f"Output query shape: {query_out.shape}")
    print(f"Output key shape: {key_out.shape}")
    ```
<!-- npu="950" id11 -->
- <term>Ascend 950PR/Ascend 950DT</term>训练模式调用（自动微分）

    ```python
    import torch
    import torch_npu
    from cann_ops_transformer.ops import apply_rotary_pos_emb

    torch_npu.npu.set_device(0)

    B, S, N, D = 1, 64, 8, 128

    query = torch.randn(B, S, N, D, device="npu", dtype=torch.float16, requires_grad=True)
    key = torch.randn(B, S, N, D, device="npu", dtype=torch.float16, requires_grad=True)
    cos = torch.randn(B, S, 1, D, device="npu", dtype=torch.float16, requires_grad=True)
    sin = torch.randn(B, S, 1, D, device="npu", dtype=torch.float16, requires_grad=True)

    # 正向：返回q_embed、k_embed，自动追踪计算图
    query_out, key_out = apply_rotary_pos_emb(
        query,
        key,
        cos,
        sin,
        layout="BSND",
        rotary_mode="half",  # 自动微分仅支持half模式
    )

    loss = query_out.sum() + key_out.sum()
    loss.backward()  # 自动调用apply_rotary_pos_emb_grad

    print(query.grad.shape)  # query梯度
    print(key.grad.shape)    # key梯度
    print(cos.grad.shape)    # cos梯度
    print(sin.grad.shape)    # sin梯度
    ```
<!-- end id11 -->

- 图模式调用

    ```python
    import torch
    import torch_npu
    import torchair
    from cann_ops_transformer.ops import apply_rotary_pos_emb

    torch_npu.npu.set_device(0)

    B, S, N, D = 1, 64, 8, 128

    class ApplyRotaryPosEmbModel(torch.nn.Module):
        def forward(self, query, key, cos, sin):
            return apply_rotary_pos_emb(query, key, cos, sin, layout="BSND", rotary_mode="half")

    model = ApplyRotaryPosEmbModel().npu()
    npu_backend = torchair.get_npu_backend()
    model = torch.compile(model, backend=npu_backend, dynamic=False)

    query = torch.randn(B, S, N, D, device="npu", dtype=torch.float16)
    key = torch.randn(B, S, N, D, device="npu", dtype=torch.float16)
    cos = torch.randn(B, S, 1, D, device="npu", dtype=torch.float16)
    sin = torch.randn(B, S, 1, D, device="npu", dtype=torch.float16)

    query_out, key_out = model(query, key, cos, sin)
    ```
