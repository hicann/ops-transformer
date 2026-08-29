# apply_rotary_pos_emb_grad

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

    执行双路旋转位置编码[apply_rotary_pos_emb](./apply_rotary_pos_emb.md)的反向计算，将query和key两路梯度计算融合为一次kernel调用。

- **计算公式**：

    设正向计算中`cos`、`sin`发生广播的轴列表为`dims`。`rotary_mode`为`"half"`时：

    $$
    grad\_q_1, grad\_q_2 = chunk(grad\_query\_embed, chunks=2, dim=-1)
    $$

    $$
    grad\_k_1, grad\_k_2 = chunk(grad\_key\_embed, chunks=2, dim=-1)
    $$

    $$
    cos_1, cos_2 = chunk(cos, chunks=2, dim=-1)
    $$

    $$
    sin_1, sin_2 = chunk(sin, chunks=2, dim=-1)
    $$

    $$
    grad\_query = cat(cos_1 \cdot grad\_q_1 + sin_2 \cdot grad\_q_2,\ cos_2 \cdot grad\_q_2 - sin_1 \cdot grad\_q_1,\ dim=-1)
    $$

    $$
    grad\_key = cat(cos_1 \cdot grad\_k_1 + sin_2 \cdot grad\_k_2,\ cos_2 \cdot grad\_k_2 - sin_1 \cdot grad\_k_1,\ dim=-1)
    $$

    当同时传入`query`和`key`时，令：

    $$
    query\_rotate = cat((-query_2, query_1), dim=-1)
    $$

    $$
    key\_rotate = cat((-key_2, key_1), dim=-1)
    $$

    则`cos`和`sin`的梯度为：

    $$
    grad\_cos = sum(grad\_query\_embed \cdot query + grad\_key\_embed \cdot key, dims)
    $$

    $$
    grad\_sin = sum(grad\_query\_embed \cdot query\_rotate + grad\_key\_embed \cdot key\_rotate, dims)
    $$

- **说明**：

  - `layout=4`时输入为3维Tensor，其他支持的`layout`下输入为4维Tensor。其中B（Batch）表示批量大小，S（Seq-Length）表示序列长度，N（Head-Num）表示多头数，D（Head-Dim）表示每个头的隐藏维度大小，T表示B和S合轴。
  - `query`和`key`必须同时传入或同时不传入。两者均不传入时，仅计算`grad_query`和`grad_key`，返回的`grad_cos`和`grad_sin`为`None`。

## 函数原型

```python
cann_ops_transformer.apply_rotary_pos_emb_grad(
    grad_query_embed,
    grad_key_embed,
    cos,
    sin,
    *,
    query=None,
    key=None,
    rotary_mode="half",
    layout=1,
) -> (Tensor, Tensor, Optional[Tensor], Optional[Tensor])
```

## 参数说明

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
|--------|----------|-----------|------|----------|-------------|
| grad_query_embed | Tensor | 必选 | 正向输出`query_out`的梯度。 | bfloat16、float16、float32 | `layout=4`时为(T, Nq, D)，其他支持的`layout`下为4维Tensor |
| grad_key_embed | Tensor | 必选 | 正向输出`key_out`的梯度。除N维度外，shape需与`grad_query_embed`一致。 | 同grad_query_embed | `layout=4`时为(T, Nk, D)，其他支持的`layout`下为4维Tensor |
| cos | Tensor | 必选 | 正向计算输入的余弦值张量，N维度必须等于1。 | 同grad_query_embed | 与输入布局对应的3维或4维Tensor |
| sin | Tensor | 必选 | 正向计算输入的正弦值张量，shape需与`cos`一致。 | 同grad_query_embed | 同cos |
| query | Tensor | 可选 | 正向计算输入`query`。传入时计算`grad_cos`和`grad_sin`；必须与`key`同时传入或同时不传入。默认值为`None`。 | 同grad_query_embed | 与grad_query_embed一致 |
| key | Tensor | 可选 | 正向计算输入`key`。传入时计算`grad_cos`和`grad_sin`；必须与`query`同时传入或同时不传入。默认值为`None`。 | 同grad_query_embed | 与grad_key_embed一致 |
| rotary_mode | str | 可选 | 旋转编码模式，仅支持`"half"`，默认值为`"half"`。 | - | - |
| layout | int | 可选 | 输入张量布局格式，1表示BSND，2表示SBND，4表示TND，默认值为1。 | - | - |

## 返回值说明

- **grad_query**（`Tensor`）：正向输入`query`的梯度，shape和数据类型与`grad_query_embed`一致。
- **grad_key**（`Tensor`）：正向输入`key`的梯度，shape和数据类型与`grad_key_embed`一致。
- **grad_cos**（`Optional[Tensor]`）：正向输入`cos`的梯度。当`query`和`key`均传入时，shape和数据类型与`cos`一致；否则为`None`。
- **grad_sin**（`Optional[Tensor]`）：正向输入`sin`的梯度。当`query`和`key`均传入时，shape和数据类型与`sin`一致；否则为`None`。

## 约束说明

- 该接口支持训练场景下使用。
- 该接口支持单算子模式调用。
- 不支持空Tensor，各输入Tensor的每个维度必须大于0。
- 输入张量`grad_query_embed`、`grad_key_embed`、`cos`、`sin`以及非空的`query`、`key`数据类型必须相同。
- `rotary_mode`仅支持`"half"`。
- `layout`仅支持1、2、4，分别对应BSND、SBND、TND。
- 输入Tensor的D维度必须相同，D小于等于1024且能被2整除。
- `grad_query_embed`和`grad_key_embed`除N维度外的其他维度必须相同。
- `cos`和`sin`的shape必须相同，且N维度必须等于1。
- `layout=1`（BSND）时，`cos`、`sin`的shape为(cosB, S, 1, D)，其中cosB可以为1或等于`grad_query_embed`的B。
- `layout=2`（SBND）时，`cos`、`sin`的shape为(S, cosB, 1, D)，其中cosB可以为1或等于`grad_query_embed`的B。
- `layout=4`（TND）时，`cos`、`sin`的shape为(T, 1, D)，T必须与`grad_query_embed`的T一致。
- `query`的shape必须与`grad_query_embed`一致，`key`的shape必须与`grad_key_embed`一致，且`query`和`key`必须同时传入或同时不传入。

## 确定性计算

默认支持确定性计算。

## 配套接口

该算子为[apply_rotary_pos_emb](./apply_rotary_pos_emb.md)的反向算子。正向接口使用`rotary_mode="half"`时，对loss执行`.backward()`会自动触发本算子；仅在需要显式控制梯度时手动调用本接口。

## 调用示例

- 单算子模式调用

    ```python
    import torch
    import torch_npu
    from cann_ops_transformer.ops import apply_rotary_pos_emb_grad

    torch_npu.npu.set_device(0)

    B = 1
    S = 64
    N = 8
    D = 128

    grad_query_embed = torch.randn(B, S, N, D, device="npu", dtype=torch.float16)
    grad_key_embed = torch.randn(B, S, N, D, device="npu", dtype=torch.float16)
    cos = torch.randn(B, S, 1, D, device="npu", dtype=torch.float16)
    sin = torch.randn(B, S, 1, D, device="npu", dtype=torch.float16)
    query = torch.randn(B, S, N, D, device="npu", dtype=torch.float16)
    key = torch.randn(B, S, N, D, device="npu", dtype=torch.float16)

    grad_query, grad_key, grad_cos, grad_sin = apply_rotary_pos_emb_grad(
        grad_query_embed,
        grad_key_embed,
        cos,
        sin,
        query=query,
        key=key,
        rotary_mode="half",
        layout=1,  # BSND
    )

    print(f"grad_query shape: {grad_query.shape}")
    print(f"grad_key shape: {grad_key.shape}")
    print(f"grad_cos shape: {grad_cos.shape}")
    print(f"grad_sin shape: {grad_sin.shape}")
    ```
