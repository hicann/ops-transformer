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

  本文档包含`qkv_rms_norm_rope_cache_with_k_scale_`和`qkv_rms_norm_rope_cache_with_k_scale`两个torch_extension接口，均封装`aclnnQkvRmsNormRopeCacheWithKScale`，用于大语言模型推理场景下的Q/K/V预处理与PagedAttention KV Cache更新。接口支持以下三种完整场景：
  - RoPE：使用`query_start_loc/seq_lens`生成普通RoPE位置；Q/K执行`RMSNorm -> RoPE -> rotation -> PerTokenPerHead量化`，数据为FP8 E4M3FN，scale为FP32。
  - M-RoPE：使用`mrope_position/mrope_section`生成T/H/W交织位置；Q/K执行`RMSNorm -> M-RoPE -> rotation`，Q返回BF16且`q_scale=None`，K执行PerTokenPerHead INT8量化。
  - M-RoPE MX：Q/K执行`RMSNorm -> M-RoPE -> Dynamic MX Quant`，不执行`rotation`；每个D32 block使用cuBLAS MX FP8的FLOAT8_E8M0二次幂scale量化为FP8 E4M3FN。V按`v_scale[Nv,D]`执行per-head-per-channel静态量化并写入FP8 Cache。
  - `qkv_rms_norm_rope_cache_with_k_scale_`：原地更新调用方传入的`k_cache`、`v_cache`和`k_scale_cache`，返回`q_out`和`q_scale`。
  - `qkv_rms_norm_rope_cache_with_k_scale`：内部先拷贝三个cache，再对副本执行更新，返回`q_out`、`q_scale`和更新后的三个cache；调用方传入的cache保持不变。

- **计算公式**：

  按`head_nums=[Nq,Nk,Nv]`从`qkv`拆分Q、K、V：

  $$
  q, k, v = split(qkv, [Nq, Nk, Nv])
  $$

  Q/K分支分别使用`q_gamma`和`k_gamma`做RMSNorm：

  $$
  y = \frac{x}{\sqrt{mean(x^2) + epsilon}} * gamma
  $$

  Q/K分支执行RoPE，V分支不执行RoPE：

  $$
  y_{rope} = concat(y_{low} * cos - y_{high} * sin,\ y_{high} * cos + y_{low} * sin)
  $$

  RoPE和M-RoPE中，Q/K共享`rotation`矩阵：

  $$
  q_{rot} = q_{rope} @ rotation,\quad k_{rot} = k_{rope} @ rotation
  $$

  M-RoPE MX要求`rotation=None`，M-RoPE结果直接进入MX量化。

  对有限输入的每个D32 block $x_b$，在FP32中取$a_b=\max_i|x_{b,i}|$，令
  $e_b=\max(-127,\lceil\log_2(a_b/448)\rceil)$、$s_b=2^{e_b}$，再计算
  $q_{b,i}=\operatorname{cast}^{\mathrm{rint,sat}}_{\mathrm{FP8\ E4M3FN}}(x_{b,i}/s_b)$。
  $s_b$以FLOAT8_E8M0存储，有限scale的原始编码为$e_b+127$；$a_b=0$时取$e_b=-127$，block中存在Inf或NaN时scale原始编码为`0xFF`。该规则即cuBLAS MX FP8 scale语义（对应`DynamicMxQuantV3`的`scaleAlg=1`）。

- RoPE场景的位置：

  第`b`个batch中第`i`个token的RoPE位置由`query_start_loc`和`seq_lens`确定：

  $$
  position = seq\_lens[b] - (query\_start\_loc[b + 1] - query\_start\_loc[b]) + i
  $$

  `cos_sin[..., :D/2]`为cos，`cos_sin[..., D/2:]`为sin。

- M-RoPE场景的cos/sin交织：

  令 $C=\mathrm{cos\_sin}$、$P=\mathrm{mrope\_position}$、$D_{\mathrm{half}}=D/2$，并定义
  $\iota(\mathrm T)=0,\ \iota(\mathrm H)=1,\ \iota(\mathrm W)=2$。
  对token $u$、轴 $a\in\{\mathrm T,\mathrm H,\mathrm W\}$ 和列 $d$，三路原始位置编码定义如下。
  `mrope_position` 的逻辑shape为 $[T,3]$，每一行对应一个token，三列依次对应T、H、W轴。

  $$
  R_{a,u,d}=C_{P_{u,\iota(a)},d},\qquad 0\leq u<T,\quad 0\leq d<D.
  $$

  因此，`mrope_position` 的三列分别提供 T、H、W 三路位置，每一行对应一个token，而不是先拼接三份输出。
  将 `mrope_section=[t,h,w]` 记为
  $\boldsymbol{s}=(s_{\mathrm T},s_{\mathrm H},s_{\mathrm W})=(t,h,w)$；
  其中 $s_{\mathrm T}$ 不参与 lane 选源，也不做独立 lane 容量上限校验；它仍须非负并参与三项总和校验。对
  $0\leq\ell<D_{\mathrm{half}}$，定义

  $$
  r(\ell)=\left\lfloor\frac{\ell}{3}\right\rfloor,\qquad
  \rho(\ell)=\ell\bmod 3,
  $$

  $$
  \sigma(\ell)=
  \begin{cases}
  \mathrm H, & \rho(\ell)=1\ \land\ r(\ell)<s_{\mathrm H},\\
  \mathrm W, & \rho(\ell)=2\ \land\ r(\ell)<s_{\mathrm W},\\
  \mathrm T, & \text{其他情况}.
  \end{cases}
  $$

  有效 cos/sin 按列定义为

  $$
  \widehat c_{u,\ell}=R_{\sigma(\ell),u,\ell},\qquad
  \widehat s_{u,\ell}=R_{\sigma(\ell),u,D_{\mathrm{half}}+\ell}.
  $$

  其中未被 H/W 覆盖的 lane 回退到同一列的 T 路；将
  $\widehat c$ 和 $\widehat s$ 代入前述 RoPE 旋转公式即可。输出维度仍为
  $D$，不会扩展为 $3D$。

- RoPE和M-RoPE的Q/K输出与PerTokenPerHead动态量化：

  令 $X^Q=q_{rot}$、$X^K=k_{rot}$，目标类型为
  $\tau_A\in\{\mathrm{FP8\ E4M3FN},\mathrm{INT8}\}$。两种目标类型只在正向量化上限上不同：

  $$
  M_{\tau_A}=
  \begin{cases}
  448, & \tau_A=\mathrm{FP8\ E4M3FN},\\
  127, & \tau_A=\mathrm{INT8}.
  \end{cases}
  $$

  对 $A\in\{Q,K\}$、token $u$ 和head $n$，启用动态量化时统一计算

  $$
  \begin{aligned}
  m^A_{u,n}&=\max_{0\leq d<D}\left|X^A_{u,n,d}\right|,\\
  \alpha^A_{u,n}&=\frac{m^A_{u,n}}{M_{\tau_A}},\\
  \widehat X^A_{u,n,d}
  &=\operatorname{cast}_{\tau_A}
    \!\left(\frac{X^A_{u,n,d}}{\alpha^A_{u,n}}\right).
  \end{aligned}
  $$

  `cast`按目标类型执行舍入与饱和，INT8目标范围为$[-127,127]$。未启用PerTokenPerHead动态量化时，输出为

  $$
  O^A_{u,n,d}=\operatorname{cast}_{\tau_A}\!\left(X^A_{u,n,d}\right),
  $$

  且不产生scale。各场景对应的目标类型和量化开关统一列在“约束说明”。

- M-RoPE MX量化：

  对M-RoPE后的Q/K，每个D32 block独立执行上述cuBLAS MX FP8二次幂scale量化，生成FP8 E4M3FN数据和FLOAT8_E8M0 scale。有效scale按D32 block线性存储在长度为`ceil(D/32)`的末轴。当前实现仍固定`D=128`，因此每个head产生4个scale；公式使用符号D不表示已支持任意D。

  V分支统一在FP8 E4M3FN转换前乘以`v_scale`；`v_scale`按其输入shape广播到V的逻辑shape：

  $$
  v_{fp8} = cast_{FP8\ E4M3FN}(v * v\_scale)
  $$

  Cache写回位置由`slot_mapping`决定：

  $$
  blockId = slot\_mapping[t] / BlockSize,\quad blockOffset = slot\_mapping[t]\ \%\ BlockSize
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
    mrope_position=None,
    mrope_section=None,
    q_quant_mode="PerTokenPerHead",
    k_quant_mode="PerTokenPerHead",
    q_out_dtype=torch.float8_e4m3fn,
) -> (Tensor, Optional[Tensor])
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
    mrope_position=None,
    mrope_section=None,
    q_quant_mode="PerTokenPerHead",
    k_quant_mode="PerTokenPerHead",
    q_out_dtype=torch.float8_e4m3fn,
) -> (Tensor, Optional[Tensor], Tensor, Tensor, Tensor)
```

`query_start_loc`和`seq_lens`是不可省略的Optional位置参数：RoPE传有效Tensor，两个M-RoPE场景显式传`None`。`k_quant_mode`是keyword-only参数，默认使用`"PerTokenPerHead"`。底层同名ACLNN C函数包含该参数，OPP、ACLNN调用方和`cann_ops_transformer`扩展必须由一致版本成套构建和安装，不能混用不同ABI的二进制。

## 参数说明

以下两个接口输入参数一致。

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
|---|---|---|---|---|---|
| qkv | Tensor | 必选 | Q/K/V融合输入。`layout_qkv="TND"`时shape为`[T,Nq+Nk+Nv,D]`，`layout_qkv="NTD"`时shape为`[Nq+Nk+Nv,T,D]`。 | torch.bfloat16 | 3维 |
| q_gamma | Tensor | 必选 | Q分支RMSNorm权重。 | torch.float32 | `[D]` |
| k_gamma | Tensor | 必选 | K分支RMSNorm权重。 | torch.float32 | `[D]` |
| cos_sin | Tensor | 必选 | RoPE位置编码表，前`D/2`列为cos，后`D/2`列为sin。 | torch.float32 | `[MaxSeqLen,D]` |
| slot_mapping | Tensor | 必选 | 每个token写入cache的slot索引。 | torch.int32 | `[T]` |
| k_cache | Tensor | 必选 | K Cache。原地接口直接更新该Tensor；functional接口更新其副本并返回。RoPE和M-RoPE MX为FP8 E4M3FN；M-RoPE为INT8。 | torch.float8_e4m3fn或torch.int8 | `[BlockNum,Nk,BlockSize,D]` |
| v_cache | Tensor | 必选 | V Cache。原地接口直接更新该Tensor；functional接口更新其副本并返回。 | torch.float8_e4m3fn | `[BlockNum,Nv,BlockSize,D]` |
| k_scale_cache | Tensor | 必选 | K动态量化scale cache。RoPE/M-RoPE为FP32 `[BlockNum,Nk,BlockSize,1]`；M-RoPE MX为FLOAT8_E8M0 `[BlockNum,Nk,BlockSize,ceil(D/32)]`。原地接口更新该Tensor；functional接口更新其副本并返回。 | torch.float32或torch.float8_e8m0fnu | 4维 |
| query_start_loc | Optional[Tensor] | 必选（位置参数） | RoPE的位置源之一，必须与`seq_lens`同时传有效Tensor；两个M-RoPE场景必须显式传`None`。 | torch.int32或None | `[Batch+1]`或None |
| seq_lens | Optional[Tensor] | 必选（位置参数） | RoPE的位置源之一，必须与`query_start_loc`同时传有效Tensor；两个M-RoPE场景必须显式传`None`。 | torch.int32或None | `[Batch]`或None |
| head_nums | List[int] | 必选 | Q/K/V头数数组，必须按`[Nq,Nk,Nv]`传入。 | int | 长度为3 |
| rotation | Optional[Tensor] | 可选 | RoPE/M-RoPE的Q/K共享矩阵乘权重，必须传BF16 `[D,D]`；M-RoPE MX必须为`None`。 | torch.bfloat16或None | `[D,D]`或None |
| v_scale | Optional[Tensor] | 可选 | V分支量化缩放因子，三个场景均必须传有效Tensor。M-RoPE MX按每个head、每个channel静态量化V。 | torch.float32 | RoPE：`[Nv]`；M-RoPE/M-RoPE MX：`[Nv,D]` |
| layout_qkv | Optional[str] | 可选 | `qkv`的N/T轴布局标识，默认值为`"TND"`；传入`None`或空字符串时按默认值处理。大小写敏感，仅支持`"TND"`和`"NTD"`。 | str | - |
| layout_q_out | Optional[str] | 可选 | `q_out`和存在时的`q_scale`的N/T轴布局标识，默认值为`"NTD"`；传入`None`或空字符串时按默认值处理。大小写敏感，仅支持`"TND"`和`"NTD"`。 | str | - |
| epsilon | float | 可选 | RMSNorm防除零参数，默认值为`1e-6`。 | float | - |
| mrope_position | Optional[Tensor] | 可选 | M-RoPE位置索引。每行对应一个token，三列依次为T/H/W位置；RoPE场景必须为`None`。 | torch.int32 | `[T,3]` |
| mrope_section | Optional[List[int]] | 可选 | M-RoPE的T/H/W section参数`[t,h,w]`；RoPE场景必须为`None`或空列表。 | int | 长度为0或3 |
| q_quant_mode | str | 可选 | Q分支量化模式，默认值为`"PerTokenPerHead"`。支持`"PerTokenPerHead"`、`"NoQuant"`和`"Mx"`。 | str | - |
| k_quant_mode | str | 可选 | K分支量化算法，默认值为`"PerTokenPerHead"`，支持`"PerTokenPerHead"`和`"Mx"`；它不选择dtype，K存储类型仍由`k_cache` dtype和场景合同决定。 | str | - |
| q_out_dtype | torch.dtype | 可选 | `q_out`的数据类型。RoPE/M-RoPE MX为`torch.float8_e4m3fn`，M-RoPE为`torch.bfloat16`。 | torch.dtype | - |

## 返回值说明

### qkv_rms_norm_rope_cache_with_k_scale_

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
|---|---|---|---|---|---|
| q_out | Tensor | 必选 | Q分支输出。RoPE为FP8 E4M3FN，shape按`layout_q_out`为`[T,Nq,D]`或`[Nq,T,D]`；M-RoPE为BF16 `[T,Nq,D]`；M-RoPE MX为FP8 E4M3FN `[T,Nq,D]`。 | torch.float8_e4m3fn或torch.bfloat16 | 3维 |
| q_scale | Optional[Tensor] | 可选 | RoPE为FP32 rank2；M-RoPE为`None`；M-RoPE MX为FLOAT8_E8M0 `[T,Nq,ceil(D/32)]`。 | torch.float32、torch.float8_e8m0fnu或None | 2维、3维或None |

### qkv_rms_norm_rope_cache_with_k_scale

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
|---|---|---|---|---|---|
| q_out | Tensor | 必选 | Q分支输出，shape和dtype规则与原地接口一致。 | torch.float8_e4m3fn或torch.bfloat16 | 3维 |
| q_scale | Optional[Tensor] | 可选 | 与原地接口相同：RoPE为FP32、M-RoPE为None、M-RoPE MX为FLOAT8_E8M0 rank3。 | torch.float32、torch.float8_e8m0fnu或None | 2维、3维或None |
| k_cache_out | Tensor | 必选 | 更新后的K Cache，shape和dtype与输入`k_cache`一致。 | torch.float8_e4m3fn或torch.int8 | `[BlockNum,Nk,BlockSize,D]` |
| v_cache_out | Tensor | 必选 | 更新后的V Cache，shape和dtype与输入`v_cache`一致。 | torch.float8_e4m3fn | `[BlockNum,Nv,BlockSize,D]` |
| k_scale_cache_out | Tensor | 必选 | 更新后的K动态量化scale cache，shape和dtype与输入`k_scale_cache`一致。 | torch.float32或torch.float8_e8m0fnu | 4维 |

## 约束说明

- 该接口支持推理场景下的单算子模式调用，当前不支持TorchAir图模式调用。
- 三种场景的参数组合如下，其他交叉组合不支持：

  | 条件 | RoPE | M-RoPE | M-RoPE MX |
  |---|---|---|---|
  | query_start_loc / seq_lens | 两者都为有效Tensor | 两者都为None | 两者都为None |
  | mrope_position / mrope_section | 前者为None，后者为None或空列表 | 前者存在，后者为非空三项列表 | 前者存在，后者为非空三项列表 |
  | rotation | BF16 `[D,D]` | BF16 `[D,D]` | None |
  | q_quant_mode | `"PerTokenPerHead"` | `"NoQuant"` | `"Mx"` |
  | k_quant_mode | `"PerTokenPerHead"` | `"PerTokenPerHead"` | `"Mx"` |
  | q_out / q_scale | FP8 E4M3FN；FP32 rank2 | BF16；None | FP8 E4M3FN `[T,Nq,D]`；FLOAT8_E8M0 `[T,Nq,ceil(D/32)]` |
  | k_cache / k_scale_cache | FP8 E4M3FN；FP32 `[BlockNum,Nk,BlockSize,1]` | INT8；FP32 `[BlockNum,Nk,BlockSize,1]` | FP8 E4M3FN；FLOAT8_E8M0 `[BlockNum,Nk,BlockSize,ceil(D/32)]` |
  | v_cache / v_scale | FP8 E4M3FN；`[Nv]` | FP8 E4M3FN；`[Nv,D]` | FP8 E4M3FN；`[Nv,D]` per-head-per-channel静态量化 |
  | layout_qkv -> layout_q_out | NTD->NTD、TND->TND、TND->NTD | 仅TND->TND | 仅TND->TND |

- 仅支持`D=128`，`head_nums=[Nq,Nk,Nv]`必须满足`0<Nq<=64`、`Nq=8*Nk`、`Nk=Nv`。
- M-RoPE场景下，`mrope_section=[t,h,w]`的三项必须非负，H/W范围均为`[0,21]`，T没有独立lane上限，且`t+h+w<=64`；`mrope_position`的shape必须为`[T,3]`，每个位置索引必须满足`0 <= value < MaxSeqLen`。
- `k_cache`、`v_cache`和`k_scale_cache`的`BlockNum`和`BlockSize`必须一致；`k_cache`和`v_cache`均为4维正stride、最后一维stride为1，且前三维stride必须一致。RoPE/M-RoPE的`k_scale_cache`为4维正stride；M-RoPE MX同样为4维，末轴`ceil(D/32)`连续且stride为1。
- RoPE场景中，`query_start_loc[0]`应为0，`query_start_loc[-1]`应等于`T`，`seq_lens`长度应等于`query_start_loc.shape[0]-1`，且`seq_lens[b] >= query_start_loc[b+1] - query_start_loc[b]`；`cos_sin`第一维需覆盖本次调用访问的所有位置。
- `slot_mapping`取值范围应为`[0,BlockNum*BlockSize-1]`。M-RoPE MX要求同一次调用内的slot互不重复；RoPE和M-RoPE存在重复slot时最终写入顺序和结果未定义。
- M-RoPE MX要求`1<=T<=262144`。

## 确定性计算

默认支持确定性计算。

## 调用说明

- RoPE场景原地调用：

  可选参数均有默认值。

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
  rotation = torch.eye(D, device="npu", dtype=torch.bfloat16)
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
  )

  print(q_out.shape, q_out.dtype, q_scale.shape, q_scale.dtype)
  ```

- M-RoPE场景调用：

  M-RoPE在`query_start_loc`和`seq_lens`的固定位置显式传`None`，并在keyword-only参数中传入完整M-RoPE合同。

  ```python
  import torch
  import torch_npu
  import cann_ops_transformer

  torch_npu.npu.set_device(0)

  T, Nq, Nk, Nv, D = 5, 16, 2, 2, 128
  block_num, block_size = 1, 16
  head_nums = [Nq, Nk, Nv]

  qkv = torch.randn(T, Nq + Nk + Nv, D, device="npu", dtype=torch.bfloat16)
  q_gamma = torch.ones(D, device="npu", dtype=torch.float32)
  k_gamma = torch.ones(D, device="npu", dtype=torch.float32)
  cos_sin = torch.randn(32, D, device="npu", dtype=torch.float32)
  slot_mapping = torch.arange(T, device="npu", dtype=torch.int32)
  k_cache = torch.empty(block_num, Nk, block_size, D, device="npu", dtype=torch.int8)
  v_cache = torch.empty(block_num, Nv, block_size, D, device="npu", dtype=torch.float8_e4m3fn)
  k_scale_cache = torch.empty(block_num, Nk, block_size, 1, device="npu", dtype=torch.float32)
  rotation = torch.eye(D, device="npu", dtype=torch.bfloat16)
  v_scale = torch.ones(Nv, D, device="npu", dtype=torch.float32)
  mrope_position = torch.tensor(
      [[0, 1, 2], [3, 5, 8], [7, 4, 3], [7, 9, 6], [2, 6, 1]],
      device="npu",
      dtype=torch.int32,
  )

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
          None,
          None,
          head_nums,
          rotation=rotation,
          v_scale=v_scale,
          layout_qkv="TND",
          layout_q_out="TND",
          mrope_position=mrope_position,
          mrope_section=[22, 12, 10],
          q_quant_mode="NoQuant",
          q_out_dtype=torch.bfloat16,
      )
  )

  assert q_scale is None
  print(q_out.shape, q_out.dtype, k_cache_out.dtype)
  ```

- M-RoPE MX场景调用：

  M-RoPE MX不传`rotation`，Q/K均使用`Mx`；Q/K数据为FP8 E4M3FN，每个D32 block的scale为`torch.float8_e8m0fnu`。

  ```python
  import torch
  import torch_npu
  import cann_ops_transformer

  torch_npu.npu.set_device(0)

  T, Nq, Nk, Nv, D = 5, 16, 2, 2, 128
  block_num, block_size = 1, 16
  head_nums = [Nq, Nk, Nv]

  qkv = torch.randn(T, Nq + Nk + Nv, D, device="npu", dtype=torch.bfloat16)
  q_gamma = torch.ones(D, device="npu", dtype=torch.float32)
  k_gamma = torch.ones(D, device="npu", dtype=torch.float32)
  cos_sin = torch.randn(32, D, device="npu", dtype=torch.float32)
  slot_mapping = torch.arange(T, device="npu", dtype=torch.int32)
  k_cache = torch.empty(block_num, Nk, block_size, D, device="npu", dtype=torch.float8_e4m3fn)
  v_cache = torch.empty(block_num, Nv, block_size, D, device="npu", dtype=torch.float8_e4m3fn)
  k_scale_cache = torch.empty(
      block_num, Nk, block_size, D // 32, device="npu", dtype=torch.float8_e8m0fnu
  )
  v_scale = torch.ones(Nv, D, device="npu", dtype=torch.float32)
  mrope_position = torch.tensor(
      [[0, 1, 2], [3, 5, 8], [7, 4, 3], [7, 9, 6], [2, 6, 1]],
      device="npu",
      dtype=torch.int32,
  )

  q_out, q_scale = cann_ops_transformer.qkv_rms_norm_rope_cache_with_k_scale_(
      qkv,
      q_gamma,
      k_gamma,
      cos_sin,
      slot_mapping,
      k_cache,
      v_cache,
      k_scale_cache,
      None,
      None,
      head_nums,
      v_scale=v_scale,
      layout_qkv="TND",
      layout_q_out="TND",
      mrope_position=mrope_position,
      mrope_section=[22, 12, 10],
      q_quant_mode="Mx",
      k_quant_mode="Mx",
      q_out_dtype=torch.float8_e4m3fn,
  )

  print(q_out.shape, q_out.dtype, q_scale.shape, q_scale.dtype)
  ```

- TorchAir图模式调用：

  暂不支持TorchAir图模式调用。
