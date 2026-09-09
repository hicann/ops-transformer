# und_gen_qkv_rms_norm_rope_cache

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

  本接口封装`aclnnUndGenQkvRmsNormRopeCache`，用于多模态大模型推理场景下的Q/K/V预处理与PagedAttention KV Cache更新。接口把理解阶段（undecoded）与生成阶段（generated）两段QKV输入按`cat_indices`间接寻址拼成一条输出序列，逐token沿头维度拆分出Q/K/V，对Q和K逐头执行RMSNorm归一化与MRoPE（多模态旋转位置编码），V不参与归一化与旋转；Q作为返回值输出，K和V按`slot_mapping`写入分页KV Cache。RMSNorm权重按源token落在理解段还是生成段在两套权重间选择。计算全程float32，返回值与KV Cache均为bfloat16。

- **原地更新**：

  `k_cache`和`v_cache`是原地更新的入参（对应schema中的`Tensor(a!)`和`Tensor(b!)`），调用后直接读取这两个Tensor即可拿到更新结果，接口不额外返回它们。

## 函数原型

```python
cann_ops_transformer.und_gen_qkv_rms_norm_rope_cache(
    und_qkv,
    und_weights_q,
    und_weights_k,
    cos_sin_cache,
    k_cache,
    v_cache,
    slot_mapping,
    positions,
    gen_qkv=None,
    gen_weights_q=None,
    gen_weights_k=None,
    cat_indices=None,
    *,
    num_heads_q=8,
    num_heads_k=1,
    num_heads_v=1,
    norm_eps=1e-6,
    mrope_section=(),
)
```

## 参数说明

记`T = und_len + gen_len`为输出token总数，`N = Hq + Hk + Hv`，`D`为头维度（固定为128），`Bn`/`Bs`为KV Cache的页数与页内行数。

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
|---|---|---|---|---|---|
| und_qkv | Tensor | 必选 | 理解阶段的Q/K/V融合输入，头维度上按`[Hq,Hk,Hv]`依次排布。 | torch.bfloat16 | `[und_len,N,D]` |
| und_weights_q | Tensor | 必选 | 理解阶段Q分支的RMSNorm权重。 | torch.bfloat16 | `[D]` |
| und_weights_k | Tensor | 必选 | 理解阶段K分支的RMSNorm权重。 | torch.bfloat16 | `[D]` |
| cos_sin_cache | Tensor | 必选 | 位置编码表，前`D/2`列为cos，后`D/2`列为sin。 | torch.float32 | `[max_pos,D]` |
| k_cache | Tensor | 必选 | K Cache，原地更新。必须内存连续。 | torch.bfloat16 | `[Bn,Bs,Hk,D]` |
| v_cache | Tensor | 必选 | V Cache，原地更新。必须内存连续。 | torch.bfloat16 | `[Bn,Bs,Hv,D]` |
| slot_mapping | Tensor | 必选 | 每个输出token写入cache的slot索引，`slot = block_idx * Bs + row_idx`。 | torch.int64 | `[T]` |
| positions | Tensor | 必选 | MRoPE的T/H/W三轴位置索引，逐行对应一个轴。 | torch.int64 | `[3,T]` |
| gen_qkv | Optional[Tensor] | 可选 | 生成阶段的Q/K/V融合输入，N和D维必须与`und_qkv`一致。当前版本必须传入且`gen_len`为正数，传入`None`（纯prefill）暂不支持。 | torch.bfloat16 | `[gen_len,N,D]` |
| gen_weights_q | Optional[Tensor] | 可选 | 生成阶段Q分支的RMSNorm权重。当前版本必须传入，且需与`gen_weights_k`成对传入。 | torch.bfloat16 | `[D]` |
| gen_weights_k | Optional[Tensor] | 可选 | 生成阶段K分支的RMSNorm权重。当前版本必须传入，且需与`gen_weights_q`成对传入。 | torch.bfloat16 | `[D]` |
| cat_indices | Optional[Tensor] | 可选 | 输出token到源token的映射`out_t -> src_t`，取值小于`und_len`时取理解段，否则取生成段。当前版本必须传入，传入`None`（单序列恒等映射）暂不支持。 | torch.int64 | `[T]` |
| num_heads_q | int | 可选 | Q头数`Hq`，默认值为`8`。 | int | - |
| num_heads_k | int | 可选 | K头数`Hk`，默认值为`1`。 | int | - |
| num_heads_v | int | 可选 | V头数`Hv`，默认值为`1`。 | int | - |
| norm_eps | float | 可选 | RMSNorm防除零参数，默认值为`1e-6`，必须为正数。 | float | - |
| mrope_section | List[int] | 可选 | MRoPE的T/H/W section参数`[t,h,w]`，默认值为`()`。传入空列表时退化为标准RoPE（三轴同源）。 | int | 长度为0或3 |

## 返回值说明

`q`：Tensor，理解段与生成段拼接后的Q输出，数据类型为torch.bfloat16，shape为`[T,Hq,D]`。

`k_cache`和`v_cache`为原地更新，不在返回值中，调用后直接读取入参Tensor即可。

## 约束说明

- 该接口支持推理场景下的单算子模式调用，同时在`ops/graph_convert/`下提供了TorchAir图模式converter。
- 仅支持`D=128`；`(num_heads_q, num_heads_k, num_heads_v)`仅支持`(8,1,1)`和`(16,2,2)`两种组合，且三者之和必须等于`und_qkv`的N维。
- 当前版本要求`gen_qkv`、`gen_weights_q`、`gen_weights_k`、`cat_indices`四个参数全部传入，且`gen_len`为正数。传入`None`对应的退化路径（纯prefill、单序列恒等映射）暂不支持，调用时会返回失败。后续版本放开。
- `Bs`不设限，它不参与寻址与切分，唯一约束是容量`Bn * Bs >= T`。
- `T`不设上限，只要求为正；真实上限由KV Cache容量`Bn * Bs >= T`决定。上板已验证到`T = 64K`。
- `mrope_section`的三项必须非负，且三项之和不超过`D/2`。
- `k_cache`和`v_cache`必须内存连续；两者的`Bn`和`Bs`必须一致。
- 以下取值属运行期数据，Host侧无法校验，由调用方保证：`slot_mapping`取值须落在`[0, Bn*Bs)`且互不重复，`positions`取值须落在`[0, max_pos)`，`cat_indices`取值须落在`[0, T)`。

## 确定性计算

默认支持确定性计算，前提是`slot_mapping`取值互不重复。同一次调用内多个token写入同一slot时，多核下的写入顺序和结果未定义。

## 调用说明

- 单算子模式调用：

  ```python
  import torch
  import torch_npu
  import cann_ops_transformer

  torch_npu.npu.set_device(0)

  D, Hq, Hk, Hv = 128, 8, 1, 1
  N = Hq + Hk + Hv
  und_len, gen_len = 5, 3
  total = und_len + gen_len
  block_num, block_size = 2, 128

  und_qkv = torch.randn(und_len, N, D, dtype=torch.bfloat16).npu()
  gen_qkv = torch.randn(gen_len, N, D, dtype=torch.bfloat16).npu()
  w = [torch.randn(D, dtype=torch.bfloat16).npu() for _ in range(4)]
  cos_sin_cache = torch.randn(32, D, dtype=torch.float32).npu()
  k_cache = torch.zeros(block_num, block_size, Hk, D, dtype=torch.bfloat16).npu()
  v_cache = torch.zeros(block_num, block_size, Hv, D, dtype=torch.bfloat16).npu()
  slot_mapping = torch.arange(total, dtype=torch.int64).npu()
  positions = torch.randint(0, 32, (3, total), dtype=torch.int64).npu()
  cat_indices = torch.tensor([0, 5, 1, 6, 2, 7, 3, 4], dtype=torch.int64).npu()

  q = cann_ops_transformer.und_gen_qkv_rms_norm_rope_cache(
      und_qkv, w[0], w[1], cos_sin_cache, k_cache, v_cache, slot_mapping, positions,
      gen_qkv, w[2], w[3], cat_indices,
      num_heads_q=Hq, num_heads_k=Hk, num_heads_v=Hv,
      norm_eps=1e-6, mrope_section=[16, 16, 16])
  print(q.shape, k_cache.shape)
  ```
