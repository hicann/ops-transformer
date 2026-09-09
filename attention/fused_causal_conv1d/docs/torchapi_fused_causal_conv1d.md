# fused_causal_conv1d

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

    对序列执行因果一维卷积，沿序列维度使用缓存数据（长度为卷积核宽减1）对各序列头部进行padding，确保输出依赖当前及历史输入；卷积完成后，将当前序列部分数据更新到缓存；在因果一维卷积输出的基础上，将原始输入加到输出上以实现残差连接。支持APC（Automatic Prefix Caching）、MTP（投机解码）、残差连接等特性。相较于标准causal_conv1d算子，本算子新增APC缓存复用、PD混部、残差连接可选等功能。相比于fused_causal_conv1d_，本接口不支持输入x的原地更新。<br>


- **计算公式**：

    K是卷积核宽度（固定为3），L是原始序列长度，dim是特征维度。
    1. 缓存读取

        缓存行索引：

        $$
        readCacheLine = \begin{cases}
        cacheIndices[batchId, \; initialStateIdx[batchId]], & \text{APC模式} \\
        cacheIndices[batchId], & \text{非APC且cacheIndices存在} \\
        batchId, & \text{其他}
        \end{cases}
        $$

        Case 1：首次计算（numComputedTokens[batchId] == 0）

        $$
        cachedState[i, dim] = 0, \quad 0 \leq i < K-1
        $$

        $$
        offset = 0
        $$

        Case 2：投机解码模式（numAcceptedTokens存在）

        $$
        offset = numAcceptedTokens[batchId] - 1
        $$

        $$
        cachedState[i, dim] = convStates[readCacheLine][i, dim], \quad 0 \leq i <   offset + K - 1
        $$

        Case 3：默认模式

        $$
        offset = C - (K - 1)
        $$

        $$
        cachedState[i, dim] = convStates[readCacheLine][i, dim], \quad 0 \leq i < offset + K - 1
        $$

    2. 缓存拼接

        $$
        paddedInput[i, dim] =
        \begin{cases}
        cachedState[i, dim], & 0 \leq i < offset + K - 1 \\
        x[i - (offset + K - 1), dim], & offset + K - 1 \leq i < offset + K - 1 + L
        \end{cases}
        $$

    3. 缓存更新

        $$
        Len = offset + K - 1 + L
        $$

        $$
        M = \min(C, \; Len)
        $$

        $$
        writeCacheLine = \begin{cases}
        cacheIndices[batchId, \; idxLast], & \text{APC模式} \\
        cacheIndices[batchId], & \text{非APC且cacheIndices存在} \\
        batchId, & \text{其他}
        \end{cases}
        $$

        $$
        convStates[writeCacheLine][C - M + i, dim] = paddedInput[Len - M + i, dim], \quad i = 0, 1, \dots, M-1
        $$

    4. Offset裁剪

        $$
        x'[i, dim] = paddedInput[i + offset, dim], \quad 0 \leq i < K - 1 + L
        $$

    5. APC缓存填充（可选，APC模式下）

        $$
        seqCompletedOffsetToken = numComputedTokens[batchId] \mod B
        $$

        $$
        seqCompletedOffset = B - seqCompletedOffsetToken
        $$

        $$
        seqEndOffset = (L - seqCompletedOffset) \mod B
        $$

        $$
        lastFullBlockTokenIndex = \begin{cases}
        L - seqEndOffset - B, & seqEndOffset = 0 \\
        L - seqEndOffset, & \text{otherwise}
        \end{cases}
        $$

        $$
        nBlockToFill = idxLast - idxFirst
        $$

        对每个chunk = 0, 1, ..., nBlockToFill - 1：

        $$
        boundaryIdx = lastFullBlockTokenIndex - (nBlockToFill - chunk - 1) \times B
        $$

        $$
        convStates[cacheIndices[batchId, \; idxFirst + chunk]][C-(K-1)+j, \; dim] = x'[boundaryIdx + j, \; dim], \quad j = 0, \dots, K-2
        $$

    6. 因果1维卷积

        $$
        y[i, dim] = \sum_{k=0}^{K-1} w[k, dim] \cdot x'[i + k, dim], \quad i = 0, 1, \dots, L-1
        $$

    7. 零填充重置（可选，当convMode == 1并且numComputedTokens不为空时）

        $$
        resetIdx = \min\!\Big(\max\!\big(K - 1 - numComputedTokens[batchId], \; 0\big), \; L\Big)
        $$

        $$
        y[i, dim] = 0, \quad 0 \leq i < resetIdx
        $$

    8. 残差连接（可选）

        $$
        y[i, dim] = x[i, dim] + y[i, dim]
        $$

## 函数原型

```python
cann_ops_transformer.fused_causal_conv1d(
    x, weight, conv_states, *, query_start_loc=None, cache_indices=None, initial_state_mode=None, bias=None, num_accepted_tokens=None, num_computed_tokens=None, block_idx_first_scheduled_token=None, block_idx_last_scheduled_token=None, initial_state_idx=None, activation="None", pad_slot_id=-1, max_query_len=-1, residual_connection=1, block_size=128, conv_mode=1, max_draft_tokens=7
) -> Tensor
```

## 参数说明

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
|--------|----------|-----------|------|----------|-------------|
| x | Tensor | 必选 | 输入序列。 | float16、bfloat16 | [batch, seq_len, dim]（固定batch）或 [cu_seq_len, dim]（变长） |
| weight | Tensor | 必选 | 卷积权重。 | 同x | [kW, dim]，kW∈{2, 3, 4} |
| conv_states | Tensor | 必选 | 卷积状态缓存，计算后原地更新。 | 同x | [num_cache_lines, state_len, dim]，state_len ≥ kW-1 |
| query_start_loc | Tensor | 可选 | 变长序列起始位置索引。变长场景必须提供，固定batch场景可传空Tensor。 | int32 | [batch+1] |
| cache_indices | Tensor | 可选 | 缓存索引，指定每个序列对应的缓存状态在conv_states中的索引。默认None使用恒等映射。 | int32 | [batch] |
| initial_state_mode | Tensor | 可选 | 初始状态标志。1=使用缓存历史，0=零初始化（冷启动）。默认None全部零初始化。 | int32 | [batch] |
| bias | Tensor | 必选 | 卷积偏置，None表示不使用。 | 同x | [dim] |
| num_accepted_tokens | Tensor | 可选 | 当前batch的随机投机数。 | int32 | [batch] |
| num_computed_tokens | Tensor | 可选 | 当前batch已经处理的token总数，用于判断初始状态。 | int32 | [batch] |
| block_idx_first_scheduled_token | Tensor | 可选 | 当前batch的起始位置对应的block索引。 | int32 | [batch] |
| block_idx_last_scheduled_token | Tensor | 可选 | 当前batch的seq_len - 1处对应的block索引。 | int32 | [batch] |
| initial_state_idx | Tensor | 可选 | 初始索引块的索引。 | int32 | [batch] |
| activation | str | 可选 | 激活函数类型，目前不支持此字段，默认值为"None"。 | - | - |
| pad_slot_id | int | 可选 | padding slot id。默认值为-1。 | - | - |
| max_query_len | int | 可选 | 所有batch中最大的seq_len，仅decode场景（固定batch）支持为-1。 | - | - |
| residual_connection | int | 可选 | 是否做残差连接，1=做残差连接，0=不做残差连接。 | - | - |
| block_size | int | 可选 | block块的大小。取值范围大于等于2，典型值128、256。 | - | - |
| conv_mode | int | 可选 | 卷积模式，支持0和1，0=Qwen3-Next社区版本实现，1=Pangu V2实现。 | - | - |
| max_draft_tokens | int | 可选 | 最大投机个数，支持范围[0, 16]。默认值为7。 | - | - |

## 返回值说明

返回卷积输出Tensor y，shape与x一致，dtype与x一致。
## 约束说明

- 输入shape限制：
  - prefill场景：
    - x支持2维[cu_seq_len, dim]。
    - weight必须是2维[K, dim]，其中K固定为3。
    - conv_states必须是3维[..., K-1, dim]，第0维大小不固定且大于等于参与计算的batch个数（即cache_indices不等于pad_slot_id的batch个数）。
    - query_start_loc必须存在。
    - cache_indices为1维[batch, ]或2维[batch, max_num_blocks]，其中1维表示未开启APC，2维表示开启APC。
    - cu_seq_len范围[batch, 1024 * 1024]，dim范围[64, 16384]且是16的倍数，且两者乘积需满足[64 * batch, 4G], batch范围[1, 256]。
    - max_num_blocks >= ceiv(max_query_len, block_size)。
    - max_query_len > max_draft_tokens + 1。
  - prefill和decode混合场景：
    - x支持2维[cu_seq_len, dim]。
    - weight必须是2维[K, dim]，其中K固定为3。
    - conv_states必须是3维[..., K-1+m, dim]，第0维大小不固定且大于等于参与计算的batch个数（即cache_indices不等于pad_slot_id的batch个数）。
    - query_start_loc必须存在。
    - cache_indices为1维[batch, ]或2维[batch, max_num_blocks]，其中1维表示未开启APC，2维表示开启APC。
    - cu_seq_len范围[batch, 1024 * 1024]，dim范围[64, 16384]且是16的倍数，且两者乘积需满足[64 * batch, 4G], batch范围[1, 256]。
    - max_num_blocks >= ceiv(max_query_len, block_size)。
    - max_query_len > max_draft_tokens + 1。
  - decode场景（变长序列）：
    - x支持2维[cu_seq_len, dim]。
    - weight必须是2维[K, dim]，其中K固定为3。
    - conv_states必须是3维[..., k-1+m, dim]，第0维大小不固定且大于等于参与计算的batch个数（即cache_indices不等于pad_slot_id的batch个数）。
    - query_start_loc必须存在。
    - cache_indices为1维[batch, ]或2维[batch, max_num_blocks]，其中1维表示未开启APC，2维表示开启APC。
    - cu_seq_len范围[batch, batch * (max_draft_tokens + 1)]，每个batch的seq_len范围为[1, max_draft_tokens + 1]。dim范围[64, 16384]且是16的倍数，batch范围[1, 256]。
    - max_num_blocks >= ceiv(max_query_len, block_size)。
    - max_query_len范围[1, max_draft_tokens + 1]。
  - decode场景（固定batch）：
    - x支持3维[batch, m+1, dim]。
    - weight必须是2维[K, dim]，其中K固定为3。
    - conv_states必须是3维[..., K-1+m, dim]，第0维大小不固定且大于等于参与计算的batch个数（即cache_indices不等于pad_slot_id的batch个数）。
    - cache_indices为1维[batch, ]或2维[batch, max_num_blocks]，其中1维表示未开启APC，2维表示开启APC。
    - m范围[0, max_draft_tokens]，dim范围[64, 16384]且是16的倍数，batch范围[1, 256]。
    - max_num_blocks >= ceiv(max_query_len, block_size)。
    - max_query_len范围[1, max_draft_tokens + 1]，可为-1。

- 输入值域限制：
  - query_start_loc是累计偏移量，取值范围[0, cu_seq_len]，长度为batch+1，query_start_loc[i]表示第i个序列的起始偏移，query_start_loc[batch+1]表示最后一个序列的结束位置。
  - block_size 为0或者大于等于2，apc开启时不为0。
  - block_idx_first_scheduled_token、block_idx_last_scheduled_token、initial_state_idx、num_computed_tokens和cache_indices均存在时表示APC开启，且满足以下条件（i为batch的索引）：
    - cache_indices为2维
    - initial_state_idx[i] <= block_idx_first_scheduled_token[i] + 1
    - initial_state_idx[i] <= block_idx_last_scheduled_token[i]
    - block_idx_first_scheduled_token[i] <= block_idx_last_scheduled_token[i]
    - block_idx_last_scheduled_token[i] < max_num_blocks
  - num_accepted_tokens分为None和非None，非None情况下长度为batch，prefile对应的元素值为0，decode对应的元素值大于0且小于等于当前batch的seq_len-1。
  - num_computed_tokens中每个元素取值大于等于0。
  - cache_indices的取值范围为[0, conv_states.dim[0]-1],且值均不能相等（除非等于pad_slot_id）。
  - max_query_len = batch中的最大seq_len。
  - max_draft_tokens的取值范围为[0, 16]，默认值为7。
  - Pangu V2 模式（conv_mode = 1）下，num_computed_tokens不能为 None。
  - 算子入参与中间计算结果，在对应数据类型（float16/bfloat16）下，数值均不会超出该类型值域范围。
  - 算子输入不支持有±inf和nan的情况。

## 确定性计算

默认支持确定性计算。

## 调用示例

- 单算子模式调用

    ```python
    import torch
    import torch_npu
    from cann_ops_transformer.ops import fused_causal_conv1d

    torch_npu.npu.set_device(0)

    B = 2
    S = 16
    D = 512
    kW = 3

    x = torch.randn(B, S, D, device="npu", dtype=torch.float16)
    weight = torch.randn(kW, D, device="npu", dtype=torch.float16)
    conv_states = torch.zeros(B, kW - 1 + S, D, device="npu", dtype=torch.float16)
    query_start_loc = None
    cache_indices = None
    initial_state_mode = None
    bias = None
    num_accepted_tokens = None
    num_computed_tokens = None
    block_idx_first_scheduled_token = None
    block_idx_last_scheduled_token = None
    initial_state_idx = None
    activation = "None"
    pad_slot_id = -1
    max_query_len = 16
    residual_connection = 1
    block_size = 128
    conv_mode = 0
    max_draft_tokens = 7

    y = fused_causal_conv1d(
        x, weight, conv_states, query_start_loc=query_start_loc, cache_indices=cache_indices, initial_state_mode=initial_state_mode, bias=bias, num_accepted_tokens=num_accepted_tokens, num_computed_tokens=num_computed_tokens, block_idx_first_scheduled_token=block_idx_first_scheduled_token, block_idx_last_scheduled_token=block_idx_last_scheduled_token, initial_state_idx=initial_state_idx, activation=activation, pad_slot_id=pad_slot_id, max_query_len=max_query_len, residual_connection=residual_connection, block_size=block_size, conv_mode=conv_mode, max_draft_tokens=max_draft_tokens
    )
    ```

- 图模式调用

    通过`torch.compile(backend="atc")`或`torchair`自动将算子转换为GE图算子，无需额外配置。

    ```python
    import torch
    import torch_npu
    import torchair
    from cann_ops_transformer.ops import fused_causal_conv1d

    torch_npu.npu.set_device(0)

    B = 2
    S = 16
    D = 512
    kW = 3

    class FusedCausalConv1dModel(torch.nn.Module):
        def forward(self, x, weight, conv_states, query_start_loc, cache_indices, initial_state_mode, bias,
                    num_accepted_tokens, num_computed_tokens, block_idx_first_scheduled_token, block_idx_last_scheduled_token, initial_state_idx, activation, pad_slot_id, max_query_len, residual_connection, block_size, conv_mode, max_draft_tokens):
            y = fused_causal_conv1d(
                x, weight, conv_states, query_start_loc=query_start_loc, cache_indices=cache_indices, initial_state_mode=initial_state_mode, bias=bias, num_accepted_tokens=num_accepted_tokens, num_computed_tokens=num_computed_tokens, block_idx_first_scheduled_token=block_idx_first_scheduled_token, block_idx_last_scheduled_token=block_idx_last_scheduled_token, initial_state_idx=initial_state_idx, activation=activation, pad_slot_id=pad_slot_id, max_query_len=max_query_len, residual_connection=residual_connection, block_size=block_size, conv_mode=conv_mode, max_draft_tokens=max_draft_tokens
            )
            return y

    model = FusedCausalConv1dModel().npu()
    npu_backend = torchair.get_npu_backend()
    model = torch.compile(model, backend=npu_backend, dynamic=False)

    x = torch.randn(B, S, D, device="npu", dtype=torch.float16)
    weight = torch.randn(kW, D, device="npu", dtype=torch.float16)
    conv_states = torch.zeros(B, kW - 1 + S, D, device="npu", dtype=torch.float16)
    query_start_loc = None
    cache_indices = None
    initial_state_mode = None
    bias = None
    num_accepted_tokens = None
    num_computed_tokens = None
    block_idx_first_scheduled_token = None
    block_idx_last_scheduled_token = None
    initial_state_idx = None
    activation = "None"
    pad_slot_id = -1
    max_query_len = S
    residual_connection = 1
    block_size = 128
    conv_mode = 0
    max_draft_tokens = 7

    output = model(x, weight, conv_states, query_start_loc, cache_indices, initial_state_mode, bias,
                    num_accepted_tokens, num_computed_tokens, block_idx_first_scheduled_token, block_idx_last_scheduled_token, initial_state_idx, activation, pad_slot_id, max_query_len,residual_connection, block_size, conv_mode, max_draft_tokens)
    print(f"Finish")
    ```
