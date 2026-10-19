# attention_worker_combine

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT系列产品</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3系列产品</term>：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2系列产品</term>：支持
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

在Attention与FFN分离部署场景下，对FFN侧返回的专家token数据进行加权融合，并输出下一层ID。

## 函数原型

```python
cann_ops_transformer.attention_worker_combine(schedule_context, expert_scales, layer_id, hidden_size, *, token_dtype=0, need_schedule=0) -> (Tensor, Tensor)
```

## 参数说明

设R为token数，K为路由专家数（不含共享专家），H为hidden_size，其中R > 0，1 ≤ K ≤ 64，H > 0；S为Ceil(H/32)向上取整到偶数。

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
| --- | --- | --- | --- | --- | --- |
| schedule_context | Tensor | 必选 | 调度上下文，保存有效的ScheduleContext结构体。 | int8 | (1024,) |
| expert_scales | Tensor | 必选 | 非量化场景为专家权重；MXFP反量化场景为E8M0缩放因子，每32个H方向的元素共享一个缩放因子。uint8用于承载E8M0原始编码。 | 非量化：float32；MXFP反量化：uint8、float8_e8m0fnu | 非量化：(R, K)；MXFP反量化：(R, K+1, S)，K+1包含共享专家 |
| layer_id | Tensor | 必选 | 当前层ID。 | int32 | (1,) |
| hidden_size | int | 必选 | 每行token的元素数H，必须大于0。 | int | - |
| token_dtype | int | 可选 | 默认值为0。0表示FP16，1表示BF16，2表示MXFP8_E5M2，3表示MXFP8_E4M3FN，4表示MXFP4_E2M1。 | int | - |
| need_schedule | int | 可选 | 默认值为0。取值为0或1，0表示不等待token数据就绪，1表示等待token数据就绪。 | int | - |

## 返回值说明

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
| --- | --- | --- | --- | --- | --- |
| y | Tensor | 必选 | 融合结果。token_dtype为0时为float16，其余为bfloat16。 | float16、bfloat16 | (R, H) |
| next_layer_id | Tensor | 必选 | 下一层ID，值为layer_id+1。 | int32 | (1,) |

## 约束说明

- 该接口支持推理场景下使用。
- 该接口支持单算子模式和TorchAir图模式调用，图模式支持静态shape和动态shape。
- 所有输入须为同一NPU设备上的连续Tensor，数据格式为ND，不支持空Tensor。
- `schedule_context`及其关联的token和就绪标志须由上游调度流程准备，关联缓冲区须保持有效直至执行完成。开启调度时会消费就绪标志，重复调用前须准备下一批数据。

<!-- npu="A3,910b" id7 -->
- <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：`token_dtype`仅支持0、1，不支持MXFP反量化。
<!-- end id7 -->
<!-- npu="950" id8 -->
- <term>Ascend 950PR&950DT系列产品</term>：`token_dtype`支持0～4，取值为2、3或4时，表示MXFP反量化场景。
<!-- end id8 -->

## 确定性计算

- 确定性说明：默认支持确定性计算。

## 调用示例

- 单算子模式调用：

    ```python
    import torch
    import torch_npu
    import cann_ops_transformer


    def align_up(value, alignment):
        return (value + alignment - 1) // alignment * alignment


    torch.npu.set_device(0)
    R, K, H = 256, 6, 4096

    # Attention窗口包含就绪标志和K+1行专家数据，BF16每个元素占2字节。
    info_bytes = align_up(4 * R * (K + 1), 512)
    data_bytes = 2 * R * (K + 1) * H
    window_bytes = align_up(info_bytes + data_bytes, 2 * 1024 * 1024)
    window = torch.zeros(window_bytes, dtype=torch.uint8, device="npu")

    holder = torch_npu._afd.create_schedule_context_holder(
        schedule_mode=1,
        session_num=4,
        micro_batch_num=1,
        micro_batch_size=R,
        selected_expert_num=K + 1,
        expert_num=257,
        attn_to_ffn_token_size=2 * H,
        ffn_to_attn_token_size=2 * H,
        attention_window=window.data_ptr(),
        attention_window_size=window.numel(),
    )

    try:
        expert_scales = torch.ones((R, K), dtype=torch.float32, device="npu")
        layer_id = torch.zeros((1,), dtype=torch.int32, device="npu")
        y, next_layer_id = cann_ops_transformer.attention_worker_combine(
            holder.get_schedule_context_tensor(),
            expert_scales,
            layer_id,
            H,
            token_dtype=1,
            need_schedule=0,
        )
        torch.npu.synchronize()
        # 本例窗口中的token数据全为0，因此融合结果为0，下一层ID为1。
        print(y.shape, y.dtype)  # torch.Size([256, 4096]) torch.bfloat16
        print(torch.count_nonzero(y).item(), next_layer_id.item())  # 0 1
    finally:
        holder.stop_schedule()
    ```

- TorchAir图模式调用：

    ```python
    import torch
    import torch_npu
    import torchair
    from torchair.configs.compiler_config import CompilerConfig
    import cann_ops_transformer


    def align_up(value, alignment):
        return (value + alignment - 1) // alignment * alignment


    torch.npu.set_device(0)
    R, K, H = 256, 6, 4096

    # Attention窗口包含就绪标志和K+1行专家数据，BF16每个元素占2字节。
    info_bytes = align_up(4 * R * (K + 1), 512)
    data_bytes = 2 * R * (K + 1) * H
    window_bytes = align_up(info_bytes + data_bytes, 2 * 1024 * 1024)
    window = torch.zeros(window_bytes, dtype=torch.uint8, device="npu")

    holder = torch_npu._afd.create_schedule_context_holder(
        schedule_mode=1,
        session_num=4,
        micro_batch_num=1,
        micro_batch_size=R,
        selected_expert_num=K + 1,
        expert_num=257,
        attn_to_ffn_token_size=2 * H,
        ffn_to_attn_token_size=2 * H,
        attention_window=window.data_ptr(),
        attention_window_size=window.numel(),
    )

    def combine(schedule_context, expert_scales, layer_id):
        return cann_ops_transformer.attention_worker_combine(
            schedule_context, expert_scales, layer_id, H,
            token_dtype=1, need_schedule=0,
        )


    backend = torchair.get_npu_backend(compiler_config=CompilerConfig())
    # dynamic=True为动态图；改为dynamic=False则为静态图。
    compiled_combine = torch.compile(
        combine, backend=backend, fullgraph=True, dynamic=True,
    )

    try:
        expert_scales = torch.ones((R, K), dtype=torch.float32, device="npu")
        layer_id = torch.zeros((1,), dtype=torch.int32, device="npu")
        y, next_layer_id = compiled_combine(
            holder.get_schedule_context_tensor(), expert_scales, layer_id,
        )
        torch.npu.synchronize()
        # 本例窗口中的token数据全为0，因此融合结果为0，下一层ID为1。
        print(y.shape, y.dtype)  # torch.Size([256, 4096]) torch.bfloat16
        print(torch.count_nonzero(y).item(), next_layer_id.item())  # 0 1
    finally:
        holder.stop_schedule()
    ```
