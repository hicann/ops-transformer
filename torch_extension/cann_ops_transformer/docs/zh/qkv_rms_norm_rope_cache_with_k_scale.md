# qkv_rms_norm_rope_cache_with_k_scale

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

  本文档包含`qkv_rms_norm_rope_cache_with_k_scale_`和`qkv_rms_norm_rope_cache_with_k_scale`两个torch_extension接口，均封装`aclnnQkvRmsNormRopeCacheWithKScale`，用于大语言模型推理场景下的Q/K/V预处理与PagedAttention KV Cache更新。两个接口都会从融合输入`qkv`中拆分Q、K、V分量，对Q/K执行RMSNorm、RoPE和共享`rotation`矩阵乘，随后将Q/K动态量化为FP8 E4M3FN；Q分支输出`q_out`和`q_scale`，K分支按`slot_mapping`写入K Cache和K scale cache，V分支按`v_scale`缩放后量化为FP8 E4M3FN，并按`slot_mapping`写入V Cache。
  - `qkv_rms_norm_rope_cache_with_k_scale_`：原地更新调用方传入的`k_cache`、`v_cache`和`k_scale_cache`，返回`q_out`和`q_scale`。
  - `qkv_rms_norm_rope_cache_with_k_scale`：内部先拷贝`k_cache`、`v_cache`和`k_scale_cache`，再对副本执行更新，返回`q_out`、`q_scale`和更新后的三个cache；调用方传入的cache保持不变。

- **计算公式**：

  按`head_nums=[Nq,Nk,Nv]`从`qkv`拆分Q、K、V：

  $$
  q, k, v = split(qkv, [Nq, Nk, Nv])
  $$

  Q/K分支分别使用`q_gamma`和`k_gamma`做RMSNorm：

  $$
  y = \frac{x}{\sqrt{mean(x^2) + epsilon}} * gamma
  $$

  第`b`个batch中第`i`个token的RoPE位置由`query_start_loc`和`seq_lens`确定：

  $$
  position = seq\_lens[b] - (query\_start\_loc[b + 1] - query\_start\_loc[b]) + i
  $$

  Q/K分支执行RoPE，`cos_sin[..., :D/2]`为cos，`cos_sin[..., D/2:]`为sin；V分支不执行RoPE：

  $$
  y_{rope} = concat(y_{low} * cos - y_{high} * sin,\ y_{high} * cos + y_{low} * sin)
  $$

  Q/K共享`rotation`矩阵：

  $$
  q_{rot} = q_{rope} @ rotation,\quad k_{rot} = k_{rope} @ rotation
  $$

  Q/K按每个token和head做动态量化，FP8 E4M3FN最大有限值使用`FP8_E4M3FN_MAX`，该值为448：

  $$
  scale = max(abs(x)) / FP8\_E4M3FN\_MAX,\quad x_{fp8} = cast(x / scale)
  $$

  V分支按`v_scale`缩放后量化：

  $$
  v_{fp8} = cast(v * v\_scale)
  $$

  Cache写回位置由`slot_mapping`决定：

  $$
  blockId = slot\_mapping[t] / BlockSize,\quad blockOffset = slot\_mapping[t]\ \%\ BlockSize
  $$

  $$
  k\_cache[blockId, nk, blockOffset, :] = k_{fp8}[t, nk, :]
  $$

  $$
  v\_cache[blockId, nv, blockOffset, :] = v_{fp8}[t, nv, :]
  $$

  $$
  k\_scale\_cache[blockId, nk, blockOffset, 0] = k_{scale}[t, nk]
  $$

## 函数原型

`qkv_rms_norm_rope_cache_with_k_scale_`为原地cache更新接口；`qkv_rms_norm_rope_cache_with_k_scale`为函数式变体。两个接口输入参数一致。

```python
cann_ops_transformer.qkv_rms_norm_rope_cache_with_k_scale_(
    qkv,
    q_gamma,
    k_gamma,
    cos_sin,
    slot_mapping,
    k_cache,
    v_cache,
    k_scale_cache,
    query_start_loc,
    seq_lens,
    head_nums,
    *,
    rotation=None,
    v_scale=None,
    layout_qkv="TND",
    layout_q_out="NTD",
    epsilon=0.000001,
) -> (Tensor, Tensor)
```

```python
cann_ops_transformer.qkv_rms_norm_rope_cache_with_k_scale(
    qkv,
    q_gamma,
    k_gamma,
    cos_sin,
    slot_mapping,
    k_cache,
    v_cache,
    k_scale_cache,
    query_start_loc,
    seq_lens,
    head_nums,
    *,
    rotation=None,
    v_scale=None,
    layout_qkv="TND",
    layout_q_out="NTD",
    epsilon=0.000001,
) -> (Tensor, Tensor, Tensor, Tensor, Tensor)
```

## 参数说明

以下两个接口输入参数一致。

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
|---|---|---|---|---|---|
| qkv | Tensor | 必选 | Q/K/V融合输入，对应公式中的`qkv`。`layout_qkv="TND"`时shape为`[T,Nq+Nk+Nv,D]`，`layout_qkv="NTD"`时shape为`[Nq+Nk+Nv,T,D]`。 | torch.bfloat16 | 3维 |
| q_gamma | Tensor | 必选 | Q分支RMSNorm权重，对应Q分支公式中的`gamma`。 | torch.float32 | `[D]` |
| k_gamma | Tensor | 必选 | K分支RMSNorm权重，对应K分支公式中的`gamma`。 | torch.float32 | `[D]` |
| cos_sin | Tensor | 必选 | RoPE位置编码表，对应公式中的`cos_sin`、`cos`和`sin`。前`D/2`列为cos，后`D/2`列为sin。 | torch.float32 | `[MaxSeqLen,D]` |
| slot_mapping | Tensor | 必选 | 每个token写入cache的slot索引，对应公式中的`slot_mapping`。 | torch.int32 | `[T]` |
| k_cache | Tensor | 必选 | K Cache写回Tensor，对应公式中的`k_cache`。原地接口会直接更新该Tensor；functional接口会更新其副本并返回。支持符合约束的非连续Tensor。 | torch.float8_e4m3fn | `[BlockNum,Nk,BlockSize,D]` |
| v_cache | Tensor | 必选 | V Cache写回Tensor，对应公式中的`v_cache`。原地接口会直接更新该Tensor；functional接口会更新其副本并返回。支持符合约束的非连续Tensor。 | torch.float8_e4m3fn | `[BlockNum,Nv,BlockSize,D]` |
| k_scale_cache | Tensor | 必选 | K动态量化scale cache写回Tensor，对应公式中的`k_scale_cache`。原地接口会直接更新该Tensor；functional接口会更新其副本并返回。支持符合约束的非连续Tensor。 | torch.float32 | `[BlockNum,Nk,BlockSize,1]` |
| query_start_loc | Tensor | 必选 | 当前调用内各batch token数的前缀和，对应公式中的`query_start_loc`。 | torch.int32 | `[Batch+1]` |
| seq_lens | Tensor | 必选 | 每个batch追加本次token后的实际序列长度，对应公式中的`seq_lens`。 | torch.int32 | `[Batch]` |
| head_nums | List[int] | 必选 | 必选属性，Q/K/V头数数组，依次映射为公式中的`Nq`、`Nk`、`Nv`。必须按`[Nq,Nk,Nv]`传入。 | int | 长度为3 |
| rotation | Tensor | 可选 | Q/K共享矩阵乘权重，对应公式中的`rotation`。该参数默认值为`None`，但当前实现不支持不传、传入`None`或传入空Tensor。 | torch.bfloat16 | `[D,D]` |
| v_scale | Tensor | 可选 | V分支量化缩放因子，对应公式中的`v_scale`。该参数默认值为`None`，但当前实现不支持不传、传入`None`或传入空Tensor。 | torch.float32 | `[Nv]` |
| layout_qkv | str | 可选 | 可选属性，`qkv`的N/T轴布局标识，对应公式中`T`、`Nq`、`Nk`、`Nv`所在轴。默认值为`"TND"`，传入`None`或空字符串时按默认值处理；大小写敏感，仅支持`"TND"`和`"NTD"`。 | str | - |
| layout_q_out | str | 可选 | 可选属性，`q_out`和`q_scale`的N/T轴布局标识。默认值为`"NTD"`，传入`None`或空字符串时按默认值处理；大小写敏感，仅支持`"TND"`和`"NTD"`。 | str | - |
| epsilon | float | 可选 | RMSNorm防除零参数，对应公式中的`epsilon`，默认值为`1e-6`。 | float | - |

## 返回值说明

### qkv_rms_norm_rope_cache_with_k_scale_

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
|---|---|---|---|---|---|
| q_out | Tensor | 必选 | Q分支FP8 E4M3FN量化输出，对应Q分支动态量化公式中的`x_{fp8}`。`layout_q_out="TND"`时shape为`[T,Nq,D]`，`layout_q_out="NTD"`时shape为`[Nq,T,D]`。 | torch.float8_e4m3fn | 3维 |
| q_scale | Tensor | 必选 | Q分支每个token/head对应的动态量化scale，对应Q分支动态量化公式中的`scale`。`layout_q_out="TND"`时shape为`[T,Nq]`，`layout_q_out="NTD"`时shape为`[Nq,T]`。 | torch.float32 | 2维 |

### qkv_rms_norm_rope_cache_with_k_scale

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
|---|---|---|---|---|---|
| q_out | Tensor | 必选 | Q分支FP8 E4M3FN量化输出，对应Q分支动态量化公式中的`x_{fp8}`。`layout_q_out="TND"`时shape为`[T,Nq,D]`，`layout_q_out="NTD"`时shape为`[Nq,T,D]`。 | torch.float8_e4m3fn | 3维 |
| q_scale | Tensor | 必选 | Q分支每个token/head对应的动态量化scale，对应Q分支动态量化公式中的`scale`。`layout_q_out="TND"`时shape为`[T,Nq]`，`layout_q_out="NTD"`时shape为`[Nq,T]`。 | torch.float32 | 2维 |
| k_cache_out | Tensor | 必选 | 更新后的K Cache，shape和dtype与输入`k_cache`一致。 | torch.float8_e4m3fn | `[BlockNum,Nk,BlockSize,D]` |
| v_cache_out | Tensor | 必选 | 更新后的V Cache，shape和dtype与输入`v_cache`一致。 | torch.float8_e4m3fn | `[BlockNum,Nv,BlockSize,D]` |
| k_scale_cache_out | Tensor | 必选 | 更新后的K动态量化scale cache，shape和dtype与输入`k_scale_cache`一致。 | torch.float32 | `[BlockNum,Nk,BlockSize,1]` |

## 约束说明

- 该接口支持推理场景下使用。
- 该接口支持单算子模式调用；当前不支持TorchAir图模式调用。
- `qkv_rms_norm_rope_cache_with_k_scale_`为原地接口，会修改调用方传入的`k_cache`、`v_cache`和`k_scale_cache`。
- `qkv_rms_norm_rope_cache_with_k_scale`会克隆输入cache，并返回更新后的cache副本。
- 当前实现仅支持`D=128`，且`Nv`必须等于`Nk`。
- `layout_qkv`控制`qkv`的N/T轴布局，默认值为`"TND"`；`layout_q_out`控制`q_out`和`q_scale`的N/T轴布局，默认值为`"NTD"`：
  - `layout_qkv="TND"`，`layout_q_out="TND"`：`qkv=[T,Nq+Nk+Nv,D]`，`q_out=[T,Nq,D]`，`q_scale=[T,Nq]`。
  - `layout_qkv="TND"`，`layout_q_out="NTD"`：`qkv=[T,Nq+Nk+Nv,D]`，`q_out=[Nq,T,D]`，`q_scale=[Nq,T]`。
  - `layout_qkv="NTD"`，`layout_q_out="NTD"`：`qkv=[Nq+Nk+Nv,T,D]`，`q_out=[Nq,T,D]`，`q_scale=[Nq,T]`。
- 当前不支持`layout_qkv="NTD"`、`layout_q_out="TND"`。
- `k_cache`、`v_cache`和`k_scale_cache`的`BlockNum`和`BlockSize`必须一致。
- `k_cache`、`v_cache`和`k_scale_cache`均为4维正stride，最后一维stride为1；`k_cache`和`v_cache`前三维stride必须一致。
- `query_start_loc[0]`应为0，`query_start_loc[-1]`应等于`T`。`seq_lens`长度应等于`query_start_loc.shape[0]-1`。
- `seq_lens[b]`必须满足`seq_lens[b] >= query_start_loc[b+1] - query_start_loc[b]`。若`seq_lens[b]`小于该batch本次调用的token数，行为未定义。
- `cos_sin`第一维需覆盖本次调用会访问的RoPE位置。
- `slot_mapping`取值范围应为`[0,BlockNum*BlockSize-1]`。同一次调用内多个token写入同一slot时，最终写入顺序和结果未定义。
- 资源边界约束：`Nq+Nk <= 128`，`Nq+Nk+Nv <= 160`，`Nv <= 80`，`ceil_align(Nq,16)+ceil_align(Nk,16) <= 256`。

## 确定性计算

默认支持确定性计算。

## 调用说明

- 单算子模式调用：

  ```python
  import torch
  import torch_npu
  import cann_ops_transformer

  torch_npu.npu.set_device(0)

  T, Nq, Nk, Nv, D = 4, 16, 2, 2, 128
  batch, max_seq_len = 1, 16
  block_num, block_size = 1, 16
  head_nums = [Nq, Nk, Nv]

  qkv = torch.randn(T, Nq + Nk + Nv, D, device="npu", dtype=torch.bfloat16)
  q_gamma = torch.ones(D, device="npu", dtype=torch.float32)
  k_gamma = torch.ones(D, device="npu", dtype=torch.float32)
  cos_sin = torch.zeros(max_seq_len, D, device="npu", dtype=torch.float32)
  cos_sin[:, : D // 2] = 1.0
  slot_mapping = torch.arange(T, device="npu", dtype=torch.int32)
  k_cache = torch.empty(block_num, Nk, block_size, D, device="npu", dtype=torch.float8_e4m3fn)
  v_cache = torch.empty(block_num, Nv, block_size, D, device="npu", dtype=torch.float8_e4m3fn)
  k_scale_cache = torch.empty(block_num, Nk, block_size, 1, device="npu", dtype=torch.float32)
  query_start_loc = torch.tensor([0, T], device="npu", dtype=torch.int32)
  seq_lens = torch.tensor([T], device="npu", dtype=torch.int32)
  rotation = torch.eye(D, device="npu", dtype=torch.float32).to(torch.bfloat16)
  v_scale = torch.ones(Nv, device="npu", dtype=torch.float32)

  q_out, q_scale = cann_ops_transformer.qkv_rms_norm_rope_cache_with_k_scale_(
      qkv,
      q_gamma,
      k_gamma,
      cos_sin,
      slot_mapping,
      k_cache,
      v_cache,
      k_scale_cache,
      query_start_loc,
      seq_lens,
      head_nums,
      rotation=rotation,
      v_scale=v_scale,
      epsilon=1e-6,
  )

  print(q_out.shape, q_out.dtype, q_scale.shape, q_scale.dtype)
  ```

- 函数式调用：

  ```python
  import torch
  import torch_npu
  import cann_ops_transformer

  torch_npu.npu.set_device(0)

  T, Nq, Nk, Nv, D = 4, 16, 2, 2, 128
  block_num, block_size = 1, 16
  head_nums = [Nq, Nk, Nv]

  qkv = torch.randn(T, Nq + Nk + Nv, D, device="npu", dtype=torch.bfloat16)
  q_gamma = torch.ones(D, device="npu", dtype=torch.float32)
  k_gamma = torch.ones(D, device="npu", dtype=torch.float32)
  cos_sin = torch.zeros(16, D, device="npu", dtype=torch.float32)
  cos_sin[:, : D // 2] = 1.0
  slot_mapping = torch.arange(T, device="npu", dtype=torch.int32)
  k_cache = torch.empty(block_num, Nk, block_size, D, device="npu", dtype=torch.float8_e4m3fn)
  v_cache = torch.empty(block_num, Nv, block_size, D, device="npu", dtype=torch.float8_e4m3fn)
  k_scale_cache = torch.empty(block_num, Nk, block_size, 1, device="npu", dtype=torch.float32)
  query_start_loc = torch.tensor([0, T], device="npu", dtype=torch.int32)
  seq_lens = torch.tensor([T], device="npu", dtype=torch.int32)
  rotation = torch.eye(D, device="npu", dtype=torch.float32).to(torch.bfloat16)
  v_scale = torch.ones(Nv, device="npu", dtype=torch.float32)

  q_out, q_scale, k_cache_out, v_cache_out, k_scale_cache_out = (
      cann_ops_transformer.qkv_rms_norm_rope_cache_with_k_scale(
          qkv,
          q_gamma,
          k_gamma,
          cos_sin,
          slot_mapping,
          k_cache,
          v_cache,
          k_scale_cache,
          query_start_loc,
          seq_lens,
          head_nums,
          rotation=rotation,
          v_scale=v_scale,
      )
  )

  print(q_out.shape, q_scale.shape, k_cache_out.shape, v_cache_out.shape, k_scale_cache_out.shape)
  ```

- TorchAir图模式调用：

  暂不支持TorchAir图模式调用。
