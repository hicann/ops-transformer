# stem\_oam\_prep\_paged\_kv

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

  `stem_oam_prep_paged_kv`是基于`torch_npu`的`cann_ops_transformer`扩展接口，用于调用`StemOamPrepPagedKv`算子完成Stem OAM (Output-Aware Metric) 动态稀疏注意力机制的前置评分计算。从Paged KV Cache中提取K/V数据，经K Processing（per-token K-scale × group sum + anti-diagonal flip）和V Processing（per-head vScale × L2 Norm → Log → Global Normalize → ReLU → Block Average）计算，输出kFlat和vBias供Stem OAM score computation使用。

- **输入输出支持以下数据场景**：

    ```
    k_cache: [total_blocks, kv_block_size, H_kv, 128] 或 [total_blocks, H_kv, kv_block_size, 128]
    v_cache: [total_blocks, kv_block_size, H_kv, 128] 或 [total_blocks, H_kv, kv_block_size, 128]
    kv_indices: [batch, max_kv_blocks]
    kv_seq_lens: [batch]
    k_scale_cache: [total_blocks, kv_block_size, H_kv, 1] 或 [total_blocks, H_kv, kv_block_size, 1]
    v_scale: [H_kv]
    k_flat: [batch, H_kv, max_Kb, stem_stride * 128]
    v_bias: [batch, H_kv, max_Kb]
    ```

- k_cache/v_cache支持两种layout，由`kv_layout`属性指定："BBND"表示[total_blocks, kv_block_size, H_kv, 128]，"BNBD"表示[total_blocks, H_kv, kv_block_size, 128]。当前仅支持"BNBD"。
- k_scale_cache布局随`kv_layout`变化，与k_cache/v_cache布局一致（仅最后一维D=1代替D=128）。
- 仅支持FP8_E4M3FN输入路径。

## 函数原型

```python
cann_ops_transformer.stem_oam_prep_paged_kv(
    k_cache,
    v_cache,
    kv_indices,
    kv_seq_lens,
    k_scale_cache,
    v_scale,
    lambda_mag=0.3,
    kv_layout="BNBD",
    stem_block_size=128,
    stem_stride=16
) -> Tuple[Tensor, Tensor]
```

## 参数说明

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| k_cache | Tensor | 必选 | Paged K cache。不支持空Tensor。kv_layout="BBND"时shape为[total_blocks, kv_block_size, H_kv, 128]，kv_layout="BNBD"时shape为[total_blocks, H_kv, kv_block_size, 128]。支持前三维非连续（stride>shape），最后一维必须连续，H_kv最大值为8。 | float8_e4m3fn | 4 |
| v_cache | Tensor | 必选 | Paged V cache。不支持空Tensor。shape与k_cache一致。 | 与k_cache保持一致 | 4 |
| kv_indices | Tensor | 必选 | 每batch KV Block index数组。不支持空Tensor。shape[1]=max_kv_blocks，batch最大值为16，max_kv_blocks最大值2048。 | int32 | 2 |
| kv_seq_lens | list[int] | 必选 | 每batch KV序列长度。不支持空列表。kv序列长度最大值262144。该值用于派生kv_indices第二维max_kv_blocks及输出shape中max_Kb。 | int32 | 1 |
| k_scale_cache | Tensor | 可选 | Per-token per-head K scale。k_cache数据类型为FP8时必填，其他类型可省略。随kv_layout变化：kv_layout="BBND": [total_blocks, kv_block_size, H_kv, 1]，kv_layout="BNBD": [total_blocks, H_kv, kv_block_size, 1]。支持前三维非连续（stride>shape），最后一维必须连续。 | float32 | 4 |
| v_scale | Tensor | 可选 | Per-head V scale。k_cache数据类型为FP8时必填，其他类型可省略。 | float32 | 1（[H_kv]） |
| lambda_mag | float | 可选 | V bias乘数，取值范围[0,1]，默认0.3。 | float | - |
| kv_layout | str | 可选 | KV Cache布局，"BBND"表示[total_blocks, kv_block_size, H_kv, 128]，"BNBD"表示[total_blocks, H_kv, kv_block_size, 128]。当前仅支持"BNBD"，默认"BNBD"。 | str | - |
| stem_block_size | int | 可选 | Stem block大小，%32==0且≤256，默认128。 | int | - |
| stem_stride | int | 可选 | Stride大小，%16==0，≤64，≤stem_block_size，且stem_block_size必须是stem_stride的整数倍，默认16。 | int | - |

## 返回值说明

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| k_flat | Tensor | 必选 | K group sum + anti-diag flip结果。不支持空Tensor。shape为[batch, H_kv, max_Kb, kflat_dim]，其中kflat_dim=stem_stride×D, max_Kb依赖kv_seq_lens输入计算获得。 | bfloat16 | 4 |
| v_bias | Tensor | 必选 | V block bias结果。不支持空Tensor。shape为[batch, H_kv, max_Kb], max_Kb依赖kv_seq_lens输入计算获得。 | float32 | 3 |

## 约束说明

- k_cache/v_cache的数据类型必须为float8_e4m3fn。
- k_scale_cache/v_scale的数据类型必须为float32。在k_cache数据类型为FP8时必填，其他类型可省略。
- k_scale_cache shape随kv_layout变化："BBND"为`[total_blocks, kv_block_size, H_kv, 1]`，"BNBD"为`[total_blocks, H_kv, kv_block_size, 1]`。当前仅支持"BNBD"。
- v_scale shape：`[H_kv]`。
- kv_block_size 从k_cache shape推导：BBND布局时为k_cache shape[1]，BNBD布局时为k_cache shape[2]，取值{64, 128}。
- kv_indices shape[1]=max_kv_blocks，max_kv_blocks最大值2048。
- kv_seq_lens kv序列长度最大值262144。
- stem_block_size % 32 == 0，≤256；stem_stride % 16 == 0，≤64，且stem_stride ≤ stem_block_size，stem_block_size必须是stem_stride的整数倍。
- 派生值：R = stem_block_size / stem_stride，kflat_dim = stem_stride × 128，k_down_len = num_Kb × R。
- 边界：kv_seq_lens[b]=0时该batch对应的k_flat/v_bias输出全零；k_scale_cache padding rows（beyond kv_len）→ zero。
- 仅支持arch35架构（Ascend 950PR/Ascend 950DT）。

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
    batch = 1
    total_blocks = 8
    kv_block_size = 64
    num_kv_heads = 4
    dim_qk = 128
    max_kv_blocks = 2
    stem_block_size = 128
    stem_stride = 16
    kv_layout = "BNBD"

    # 构造k_cache/v_cache（FP8_E4M3FN）
    k_cache_shape = (total_blocks, num_kv_heads, kv_block_size, dim_qk)
    k_cache = torch.randn(k_cache_shape, dtype=torch.float32, device="npu").to(torch.float8_e4m3fn)
    v_cache = torch.randn(k_cache_shape, dtype=torch.float32, device="npu").to(torch.float8_e4m3fn)

    # 构造kv_indices和kv_seq_lens
    kv_indices = torch.randint(0, total_blocks, (batch, max_kv_blocks), dtype=torch.int32, device="npu")
    kv_seq_lens = (128,)

    # 构造k_scale_cache和v_scale
    k_scale_cache_shape = (total_blocks, num_kv_heads, kv_block_size, 1)
    k_scale_cache = torch.randn(k_scale_cache_shape, dtype=torch.float32, device="npu")
    v_scale = torch.randn(num_kv_heads, dtype=torch.float32, device="npu")

    # 调用算子
    k_flat, v_bias = cann_ops_transformer.stem_oam_prep_paged_kv(
        k_cache,
        v_cache,
        kv_indices,
        kv_seq_lens,
        k_scale_cache=k_scale_cache,
        v_scale=v_scale,
        lambda_mag=0.3,
        kv_layout=kv_layout,
        stem_block_size=stem_block_size,
        stem_stride=stem_stride,
    )

    torch_npu.npu.synchronize()
    print(k_flat.shape, k_flat.dtype)
    print(v_bias.shape, v_bias.dtype)
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
    batch = 1
    total_blocks = 8
    kv_block_size = 64
    num_kv_heads = 4
    dim_qk = 128
    max_kv_blocks = 2
    stem_block_size = 128
    stem_stride = 16
    kv_layout = "BNBD"

    # 构造k_cache/v_cache（FP8_E4M3FN）
    k_cache_shape = (total_blocks, num_kv_heads, kv_block_size, dim_qk)
    k_cache = torch.randn(k_cache_shape, dtype=torch.float32, device="npu").to(torch.float8_e4m3fn)
    v_cache = torch.randn(k_cache_shape, dtype=torch.float32, device="npu").to(torch.float8_e4m3fn)

    # 构造kv_indices和kv_seq_lens
    kv_indices = torch.randint(0, total_blocks, (batch, max_kv_blocks), dtype=torch.int32, device="npu")
    kv_seq_lens = (128,)

    # 构造k_scale_cache和v_scale
    k_scale_cache_shape = (total_blocks, num_kv_heads, kv_block_size, 1)
    k_scale_cache = torch.randn(k_scale_cache_shape, dtype=torch.float32, device="npu")
    v_scale = torch.randn(num_kv_heads, dtype=torch.float32, device="npu")

    class StemOamPrepPagedKvNetwork(nn.Module):
        def __init__(self):
            super(StemOamPrepPagedKvNetwork, self).__init__()

        @torch._dynamo.disable
        def forward(self, k_cache, v_cache, kv_indices, k_scale_cache, v_scale,
                    kv_seq_lens, lambda_mag, kv_layout,
                    stem_block_size, stem_stride):
            return torch.ops.cann_ops_transformer.stem_oam_prep_paged_kv(
                k_cache, v_cache, kv_indices, kv_seq_lens,
                k_scale_cache=k_scale_cache, v_scale=v_scale,
                lambda_mag=lambda_mag, kv_layout=kv_layout,
                stem_block_size=stem_block_size, stem_stride=stem_stride)

    config = CompilerConfig()
    config.mode = "reduce-overhead"
    npu_backend = torchair.get_npu_backend(compiler_config=config)
    torch._dynamo.reset()
    npu_mode = torch.compile(StemOamPrepPagedKvNetwork(), backend=npu_backend, dynamic=False)

    k_flat, v_bias = npu_mode(
        k_cache, v_cache, kv_indices, k_scale_cache, v_scale,
        kv_seq_lens, 0.3, kv_layout,
        stem_block_size, stem_stride,
    )

    print(k_flat.shape, k_flat.dtype)
    print(v_bias.shape, v_bias.dtype)
    ```
