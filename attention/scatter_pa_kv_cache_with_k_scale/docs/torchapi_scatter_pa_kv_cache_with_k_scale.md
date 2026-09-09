# scatter\_pa\_kv\_cache\_with\_k\_scale

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
<!-- npu="310p" id8 -->
- <term>Atlas 推理系列产品</term>：不支持
<!-- end id8 -->
<!-- npu="910" id9 -->
- <term>Atlas 训练系列产品</term>：不支持
<!-- end id9 -->

## 功能说明

- **接口功能**：

  `scatter_pa_kv_cache_with_k_scale`是基于`torch_npu`的`cann_ops_transformer`扩展接口，用于调用`ScatterPaKvCacheWithKScale`算子完成PagedAttention场景下FP8格式的key/value及其对应key_scale的KV Cache更新。

- **计算公式**：

  对于每个token（i ∈ [0, num_tokens)）和每个头（j ∈ [0, num_head)）：

  $$
  block\_idx = \lfloor slot\_mapping[i] / block\_size \rfloor
  $$

  $$
  block\_offset = slot\_mapping[i] \bmod block\_size
  $$

  $$
  key\_cache[block\_idx][j][block\_offset][:] = key[i][j][:]
  $$

  $$
  value\_cache[block\_idx][j][block\_offset][:] = value[i][j][:]
  $$

  $$
  key\_scale\_cache[block\_idx][j][block\_offset][0] = key\_scale[i][j]
  $$

  其中：
  - num_tokens = batch * seq_len
  - block_idx：slot_mapping映射到的block索引
  - block_offset：block内的偏移量

> [!NOTE]
>
> B（Batch）表示输入样本批量大小、S（Seq-Length）表示输入样本序列长度、N（Head-Num）表示多头数、D（Head-Dim）表示隐藏层最小的单元尺寸head_dim，num_tokens = B × S。
> num_blocks表示KV cache分块总数，block_size表示每个分块包含的token数，num_slots = num_blocks × block_size表示cache可容纳的总token数。

## 函数原型

```python
cann_ops_transformer.scatter_pa_kv_cache_with_k_scale(
    key,
    value,
    key_cache,
    value_cache,
    slot_mapping,
    key_scale,
    key_scale_cache,
    *,
    cache_layout='BNBD'
) -> None
```

## 参数说明

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| key | Tensor | 必选 | 待更新的key值，当前step多个token的key。不支持空Tensor。 | float8_e5m2、float8_e4m3fn | (num_tokens, num_head, k_head_size) |
| value | Tensor | 必选 | 待更新的value值，当前step多个token的value。不支持空Tensor。 | float8_e5m2、float8_e4m3fn | (num_tokens, num_head, v_head_size) |
| key_cache | Tensor | 必选 | 需要更新的key cache，当前layer的key cache。不支持空Tensor。 | 与key保持一致 | (num_blocks, num_head, block_size, k_head_size) |
| value_cache | Tensor | 必选 | 需要更新的value cache，当前layer的value cache。不支持空Tensor。 | 与value保持一致 | (num_blocks, num_head, block_size, v_head_size) |
| slot_mapping | Tensor | 必选 | 每个token key或value在cache中的存储偏移。不支持空Tensor。 | int32、int64 | (num_tokens,) |
| key_scale | Tensor | 必选 | 待更新的key scale值，当前step多个token的key scale，尾轴可以不连续。不支持空Tensor。 | float | (num_tokens, num_head) |
| key_scale_cache | Tensor | 必选 | 需要更新的key scale cache，当前layer的key scale cache，最后一维为1，尾轴必须连续。不支持空Tensor。 | float | (num_blocks, num_head, block_size, 1) |
| cache_layout | str | 可选 | 表示key_cache和value_cache的内存排布格式。当传"BNBD"时，表示格式为[num_blocks, num_head, block_size, head_size]。默认值为"BNBD"。 | STRING | - |

## 返回值说明

无返回值。该接口为原地更新接口，调用后 `key_cache`、`value_cache`、`key_scale_cache` 会被原地更新，无需接收返回值。

## 约束说明

- 声明：参数slot_mapping属于tensor。由于算子在Tiling阶段无法获取tensor的具体数值，tiling侧不对值进行校验，正确性需要用户自行保证。若传入非法值，会触发未定义行为（精度问题、非法内存访问导致的程序崩溃等）。
- 该接口支持推理场景下使用。
- 该接口支持单算子模式和图模式（torchair）调用。
- key、value、key_cache、value_cache的数据类型必须一致，且必须为float8_e5m2或float8_e4m3fn。
- key_scale和key_scale_cache的数据类型必须为float。
- key和value的前两维shape必须相同。
- slot_mapping的取值范围为\[0, num_blocks\*block_size-1\]，且slot_mapping内的元素值保证不重复，重复时不保证正确性。
- key_scale是两维tensor，shape为\[num_tokens, num_head\]，尾轴可以不连续。
- key_scale_cache是四维tensor，shape为\[num_blocks, num_head, block_size, 1\]，最后一维必须为1，尾轴必须连续。
- num_tokens表示当前需要更新到cache中的token数量，num_tokens = batch * seq_len。
- num_blocks表示KV cache分块的总数，block_size表示每个分块包含的token数。
- num_head表示注意力头数，k_head_size和v_head_size分别表示key和value的头维度大小。

## 确定性计算

默认支持确定性计算。

## 调用示例

- 单算子模式调用

    ```python
    import torch
    import torch_npu
    import cann_ops_transformer

    torch_npu.npu.set_device(0)

    # 形状定义
    num_tokens = 4       # 本次需要写入的token数量
    num_head = 8         # 注意力头数
    k_head_size = 128    # key头维度
    v_head_size = 128    # value头维度
    num_blocks = 2       # KV cache分块总数
    block_size = 16      # 每个分块包含的token数

    # FP8 dtype（float8_e5m2 与float8_e4m3fn均支持，此处以e4m3fn为例）
    kv_dtype = torch.float8_e4m3fn

    # 构造输入：key/value为待写入的新数据
    key = torch.randn(num_tokens, num_head, k_head_size, dtype=torch.float32, device="npu").to(kv_dtype)
    value = torch.randn(num_tokens, num_head, v_head_size, dtype=torch.float32, device="npu").to(kv_dtype)

    # 构造KV cache（被inplace更新的目标，初始置0便于校验）
    key_cache = torch.zeros(num_blocks, num_head, block_size, k_head_size, dtype=kv_dtype, device="npu")
    value_cache = torch.zeros(num_blocks, num_head, block_size, v_head_size, dtype=kv_dtype, device="npu")

    # 构造slot_mapping：每个token在cache中的偏移，取值范围 [0, num_blocks*block_size-1]
    slot_mapping = torch.tensor([0, 1, 16, 17], dtype=torch.int32, device="npu")

    # 构造key_scale及其cache（per-token-head的FP8反量化scale）
    key_scale = torch.randn(num_tokens, num_head, dtype=torch.float32, device="npu")
    key_scale_cache = torch.zeros(num_blocks, num_head, block_size, 1, dtype=torch.float32, device="npu")

    # 调用算子，将key/value/key_scale按slot_mapping写入cache（原地更新，无返回值）
    cann_ops_transformer.scatter_pa_kv_cache_with_k_scale(
        key,
        value,
        key_cache,
        value_cache,
        slot_mapping,
        key_scale,
        key_scale_cache,
        cache_layout='BNBD',
    )

    torch_npu.npu.synchronize()
    print(key_cache.shape, key_cache.dtype)
    print(value_cache.shape, value_cache.dtype)
    print(key_scale_cache.shape, key_scale_cache.dtype)
    ```

- 图模式（torchair）调用：

    ```python
    import torch
    import torch_npu
    import torch.nn as nn
    import torchair
    from torchair.configs.compiler_config import CompilerConfig
    import cann_ops_transformer

    torch_npu.npu.set_device(0)

    # 形状定义
    num_tokens = 4       # 本次需要写入的token数量
    num_head = 8         # 注意力头数
    k_head_size = 128    # key头维度
    v_head_size = 128    # value头维度
    num_blocks = 2       # KV cache分块总数
    block_size = 16      # 每个分块包含的token数

    # FP8 dtype（float8_e5m2 与float8_e4m3fn均支持，此处以e4m3fn为例）
    kv_dtype = torch.float8_e4m3fn

    # 构造输入：key/value为待写入的新数据
    key = torch.randn(num_tokens, num_head, k_head_size, dtype=torch.float32, device="npu").to(kv_dtype)
    value = torch.randn(num_tokens, num_head, v_head_size, dtype=torch.float32, device="npu").to(kv_dtype)

    # 构造KV cache（被inplace更新的目标，初始置0便于校验）
    key_cache = torch.zeros(num_blocks, num_head, block_size, k_head_size, dtype=kv_dtype, device="npu")
    value_cache = torch.zeros(num_blocks, num_head, block_size, v_head_size, dtype=kv_dtype, device="npu")

    # 构造slot_mapping：每个token在cache中的偏移，取值范围 [0, num_blocks*block_size-1]
    slot_mapping = torch.tensor([0, 1, 16, 17], dtype=torch.int32, device="npu")

    # 构造key_scale及其cache（per-token-head的FP8反量化scale）
    key_scale = torch.randn(num_tokens, num_head, dtype=torch.float32, device="npu")
    key_scale_cache = torch.zeros(num_blocks, num_head, block_size, 1, dtype=torch.float32, device="npu")

    class ScatterPaKvCacheWithKScaleNetwork(nn.Module):
        def __init__(self):
            super(ScatterPaKvCacheWithKScaleNetwork, self).__init__()
        @torch._dynamo.disable
        def forward(self, key, value, key_cache, value_cache, slot_mapping, key_scale, key_scale_cache,
                    cache_layout='BNBD'):
            torch.ops.cann_ops_transformer.scatter_pa_kv_cache_with_k_scale(
                key,
                value,
                key_cache,
                value_cache,
                slot_mapping,
                key_scale,
                key_scale_cache,
                cache_layout=cache_layout,
            )
            return key_cache, value_cache, key_scale_cache

    config = CompilerConfig()
    config.mode = "reduce-overhead"
    npu_backend = torchair.get_npu_backend(compiler_config=config)
    torch._dynamo.reset()
    npu_mode = torch.compile(ScatterPaKvCacheWithKScaleNetwork(), backend=npu_backend, dynamic=False)
    key_cache, value_cache, key_scale_cache = npu_mode(
        key, value, key_cache, value_cache, slot_mapping, key_scale, key_scale_cache, cache_layout='BNBD')

    print(key_cache.shape, key_cache.dtype)
    print(value_cache.shape, value_cache.dtype)
    print(key_scale_cache.shape, key_scale_cache.dtype)
    ```
