# causal_conv1d_fn

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

    因果一维卷积前向计算（prefill/chunk-prefill），封装aclnnCausalConv1dFn。沿序列维度使用缓存数据（长度为卷积核宽减1）对各序列头部进行padding，确保输出依赖当前及历史输入；卷积完成后可选施加SiLU激活；计算完成后将当前序列部分数据更新到缓存。支持缓存索引（cacheIndices）、初始状态模式（initialStateMode）等特性。

- **计算公式**：

    在每个时间步 $t$，根据当前输入 $x_t$、卷积权重 $w$ 和历史状态，计算卷积输出 $y_t$：

    $$
    y_t = \text{Activation}\left(\sum_{j=0}^{W-1} w_j \cdot x_{t-j} + b\right)
    $$

    其中，$W$ 为卷积核宽度（支持2、3、4），$w_j$ 为卷积权重，$b$ 为偏置，$\text{Activation}$ 为激活函数（SiLU或无激活）。

- **说明**：

  - 算子同时维护卷积状态 `conv_states`，用于在增量推理时缓存历史输入，实现高效的状态更新。
  - 模式由输入形状自动推断：x为3D且seq_len > 1 时为prefill模式。

## 函数原型

```python
cann_ops_transformer.causal_conv1d_fn(
    x, weight, bias, conv_states, query_start_loc,
    *,
    cache_indices=None, has_initial_state=None,
    activation="silu", pad_slot_id=-1, null_block_id=0,
    block_idx_first_scheduled_token=None,
    block_idx_last_scheduled_token=None,
    initial_state_idx=None, num_computed_tokens=None,
    block_size_to_align=0,
) -> Tensor
```

## 参数说明

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
|--------|----------|-----------|------|----------|-------------|
| x | Tensor | 必选 | 输入序列。 | float16、bfloat16 | [batch, seq_len, dim]（固定batch）或 [cu_seq_len, dim]（变长） |
| weight | Tensor | 必选 | 卷积权重。 | 同x | [kW, dim]，kW∈{2, 3, 4} |
| bias | Tensor | 必选 | 卷积偏置，None表示不使用。 | 同x | [dim] |
| conv_states | Tensor | 必选 | 卷积状态缓存，计算后原地更新。 | 同x | [num_cache_lines, state_len, dim]，state_len ≥ kW-1 |
| query_start_loc | Tensor | 必选 | 变长序列起始位置索引。变长场景必须提供，固定batch场景可传空Tensor。 | int32 | [batch+1] |
| cache_indices | Tensor | 可选 | 缓存索引，指定每个序列对应的缓存状态在conv_states中的索引。默认None使用恒等映射。 | int32 | [batch] |
| has_initial_state | Tensor | 可选 | 初始状态标志。1=使用缓存历史，0=零初始化（冷启动）。默认None全部零初始化。 | int32 | [batch] |
| activation | str | 可选 | 激活函数类型，"silu"或"none"。默认值为"silu"。 | - | - |
| pad_slot_id | int | 可选 | padding slot id。默认值为-1。 | - | - |
| null_block_id | int | 可选 | 无效缓存槽位标记ID，cache_indices[i]==null_block_id时跳过该序列并输出填零。默认值为0。 | - | - |
| block_idx_first_scheduled_token | Tensor | 可选 | 首个调度token块索引。 | int32 | [batch] |
| block_idx_last_scheduled_token | Tensor | 可选 | 最后调度token块索引。 | int32 | [batch] |
| initial_state_idx | Tensor | 可选 | 初始状态索引。 | int32 | [batch] |
| num_computed_tokens | Tensor | 可选 | 各序列已完成的token数。 | int32 | [batch] |
| block_size_to_align | int | 可选 | 缓存状态对齐的块大小。默认值为0。 | - | - |

## 返回值说明

返回卷积输出Tensor y，shape与x一致，dtype与x一致。

## 约束说明

- 该接口仅支持推理场景下使用。
- 该接口支持单算子模式和图模式调用。
- 不支持非连续Tensor。
- `weight`的kW仅支持2、3、4。
- `conv_states`的state_len必须 ≥ kW-1，num_cache_lines必须 ≥ batch（未提供cache_indices时）。
- `query_start_loc`[0]必须为0，`query_start_loc`[-1]必须等于cu_seq_len，值必须非递减。
- `cache_indices`的值∈[0, num_cache_lines)，或等于null_block_id表示跳过该序列。
- 算子入参与中间计算结果，在对应数据类型（float16/bfloat16）下，数值均不会超出该类型值域范围。
- 算子输入不支持有±inf和nan的情况。

## 调用说明

- 单算子模式调用

    ```python
    import torch
    import torch_npu
    from cann_ops_transformer.ops import causal_conv1d_fn

    torch_npu.npu.set_device(0)

    B = 2
    S = 16
    D = 512
    kW = 4

    x = torch.randn(B, S, D, device="npu", dtype=torch.float16)
    weight = torch.randn(kW, D, device="npu", dtype=torch.float16)
    bias = torch.randn(D, device="npu", dtype=torch.float16)
    conv_states = torch.zeros(B, kW - 1, D, device="npu", dtype=torch.float16)
    query_start_loc = torch.tensor([0, S, 2 * S], device="npu", dtype=torch.int32)

    y = causal_conv1d_fn(
        x, weight, bias, conv_states, query_start_loc,
        activation="silu",
    )
    ```

- 图模式调用

    通过`torch.compile(backend="atc")`或`torchair`自动将算子转换为GE图算子，无需额外配置。

    ```python
    import torch
    import torch_npu
    import torchair
    from cann_ops_transformer.ops import causal_conv1d_fn

    torch_npu.npu.set_device(0)

    B = 2
    S = 16
    D = 512
    kW = 4

    class CausalConv1dFnModel(torch.nn.Module):
        def forward(self, x, weight, bias, conv_states, query_start_loc):
            y = causal_conv1d_fn(
                x, weight, bias, conv_states, query_start_loc,
                activation="silu",
            )
            return y

    model = CausalConv1dFnModel().npu()
    npu_backend = torchair.get_npu_backend()
    model = torch.compile(model, backend=npu_backend, dynamic=False)

    x = torch.randn(B, S, D, device="npu", dtype=torch.float16)
    weight = torch.randn(kW, D, device="npu", dtype=torch.float16)
    bias = torch.randn(D, device="npu", dtype=torch.float16)
    conv_states = torch.zeros(B, kW - 1, D, device="npu", dtype=torch.float16)
    query_start_loc = torch.tensor([0, S, 2 * S], device="npu", dtype=torch.int32)

    output = model(x, weight, bias, conv_states, query_start_loc)
    ```
