# QkvRmsNormRopeCacheWithKScale

## 产品支持情况

| 产品 | 是否支持 |
|:---|:---:|
| <term>Ascend 950PR/Ascend 950DT</term> | √ |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | × |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | × |
| <term>Atlas 200I/500 A2 推理产品</term> | × |
| <term>Atlas 推理系列产品</term> | × |
| <term>Atlas 训练系列产品</term> | × |

## 功能说明

- 算子功能：输入Q/K/V融合张量`qkv`，按`head_nums=[Nq, Nk, Nv]`拆分出Q、K、V，并根据`slot_mapping`更新K/V Cache和K scale cache。当前支持三种完整语义场景：
  - RoPE：Q/K执行`RMSNorm -> RoPE -> rotation`，Q/K做PerTokenPerHead动态量化；
  - M-RoPE：Q/K执行`RMSNorm -> M-RoPE -> rotation`，Q不量化，K做PerTokenPerHead INT8动态量化；
  - M-RoPE MX：Q/K执行`RMSNorm -> M-RoPE -> Dynamic MX Quant`，不执行`rotation`；每个D32 block使用cuBLAS MX FP8的FLOAT8_E8M0二次幂scale量化为FP8 E4M3FN。
  - 三种场景的V分支均不做RMSNorm、位置编码或`rotation`；RoPE使用`v_scale[Nv]`，M-RoPE和M-RoPE MX使用每个head、每个channel的`v_scale[Nv,D]`量化为FP8 E4M3FN。
- 使用场景：适用于推理场景下的PagedAttention KV Cache更新，当前仅支持<term>Ascend 950PR/Ascend 950DT</term>。
- 计算公式：

  按`head_nums=[Nq, Nk, Nv]`从融合输入中拆分Q、K、V：

  $$
  q, k, v = split(qkv, [Nq, Nk, Nv])
  $$

  Q/K分支分别使用`q_gamma`和`k_gamma`做RMSNorm：

  $$
  y = \frac{x}{\sqrt{mean(x^2) + epsilon}} * gamma
  $$

  Q/K分支执行位置编码，`cos_sin[..., :D/2]`为cos，`cos_sin[..., D/2:]`为sin；V分支不执行位置编码：

  $$
  y_{pos} = concat(y_{low} * cos - y_{high} * sin,\ y_{high} * cos + y_{low} * sin)
  $$

  RoPE和M-RoPE中，Q/K共享`rotation`矩阵：

  $$
  q_{rot} = q_{pos} @ rotation,\quad k_{rot} = k_{pos} @ rotation
  $$

  RoPE和M-RoPE场景启用PerTokenPerHead动态量化时，按每个token和head计算一个FP32 scale：

  $$
  scale = max(abs(x)) / quant\_max,\quad x_{quant} = cast(x / scale)
  $$

  其中`quant_max`为目标数据类型对应的量化上限。M-RoPE MX不执行上述整D量化，而是将每个Q/K head的D维划分为D32 block。对有限输入的每个block $x_b$，在FP32中计算：

  $$
  a_b=\max_i|x_{b,i}|,\qquad
  e_b=\max\left(-127,\left\lceil\log_2\frac{a_b}{448}\right\rceil\right),\qquad
  s_b=2^{e_b},\qquad
  q_{b,i}=\operatorname{cast}^{\mathrm{rint,sat}}_{\mathrm{FP8\ E4M3FN}}\left(\frac{x_{b,i}}{s_b}\right).
  $$

  当$a_b=0$时取$e_b=-127$。$s_b$以FLOAT8_E8M0存储，有限scale的原始编码为$e_b+127$；block中存在Inf或NaN时scale原始编码为`0xFF`。该规则即cuBLAS MX FP8 scale语义（对应`DynamicMxQuantV3`的`scaleAlg=1`）。有效scale按D32 block线性存储，数量为`ceil(D/32)`。

  V分支在FP8 E4M3FN转换前统一乘以`v_scale`；`v_scale`按其输入shape广播到V的逻辑shape：

  $$
  v_{fp8} = cast_{FP8\ E4M3FN}(v * v\_scale)
  $$

  Cache写回位置由`slot_mapping`决定：

  $$
  blockId = slot\_mapping[t] / BlockSize,\quad blockOffset = slot\_mapping[t]\ \%\ BlockSize
  $$

  $$
  k\_cache[blockId, nk, blockOffset, :] = k_{quant}[t, nk, :]
  $$

  $$
  v\_cache[blockId, nv, blockOffset, :] = v_{fp8}[t, nv, :]
  $$

  $$
  k\_scale\_cache[blockId, nk, blockOffset, ...] = k\_scale[t, nk, ...]
  $$

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
|---|---|---|---|---|
| qkv | 输入 | Q/K/V融合输入。`layout_qkv="TND"`时shape为`[T, Nq+Nk+Nv, D]`，`layout_qkv="NTD"`时shape为`[Nq+Nk+Nv, T, D]`。 | BFLOAT16 | ND |
| q_gamma | 输入 | Q分支RMSNorm权重，shape为`[D]`。 | FLOAT | ND |
| k_gamma | 输入 | K分支RMSNorm权重，shape为`[D]`。 | FLOAT | ND |
| cos_sin | 输入 | RoPE/M-RoPE位置编码表，shape为`[MaxSeqLen, D]`。前`D/2`列为cos，后`D/2`列为sin。 | FLOAT | ND |
| slot_mapping | 输入 | 每个token写入cache的slot索引，shape为`[T]`。 | INT32 | ND |
| k_cache | 输入/输出 | K Cache，输入输出同地址复用，shape为`[BlockNum,Nk,BlockSize,D]`，支持非连续Tensor。RoPE和M-RoPE MX为FP8 E4M3FN；M-RoPE为INT8。 | FLOAT8_E4M3FN / INT8 | ND |
| v_cache | 输入/输出 | V Cache，输入输出同地址复用，shape为`[BlockNum, Nv, BlockSize, D]`，支持非连续Tensor。 | FLOAT8_E4M3FN | ND |
| k_scale_cache | 输入/输出 | K动态量化scale cache，输入输出同地址复用。RoPE/M-RoPE为FP32 `[BlockNum,Nk,BlockSize,1]`；M-RoPE MX为FLOAT8_E8M0 `[BlockNum,Nk,BlockSize,ceil(D/32)]`。 | FLOAT / FLOAT8_E8M0 | ND |
| query_start_loc | 可选输入 | RoPE场景当前调用内各batch token数的前缀和，shape为`[Batch+1]`；M-RoPE场景不传。 | INT32 | ND |
| seq_lens | 可选输入 | RoPE场景每个batch追加当前token后的实际序列长度，shape为`[Batch]`；M-RoPE场景不传。 | INT32 | ND |
| rotation | 可选输入 | Q/K共享矩阵乘权重。RoPE/M-RoPE必须传BF16 `[D,D]`；M-RoPE MX必须不传。 | BFLOAT16 | ND |
| v_scale | 可选输入 | V分支量化缩放因子。RoPE为`[Nv]`；M-RoPE和M-RoPE MX为每head、每channel的`[Nv,D]`。三个场景均必须传有效Tensor。 | FLOAT | ND |
| mrope_position | 可选输入 | M-RoPE场景token-major位置索引，shape为`[T,3]`，三列依次为T/H/W；RoPE场景不传。 | INT32 | ND |
| q_out | 输出 | Q输出。RoPE为FP8 E4M3FN，shape由`layout_q_out`决定；M-RoPE为BF16 `[T,Nq,D]`；M-RoPE MX为FP8 E4M3FN `[T,Nq,D]`。 | FLOAT8_E4M3FN / BFLOAT16 | ND |
| q_scale | 输出 | Q动态量化scale。RoPE为FP32 `[T,Nq]`或`[Nq,T]`；M-RoPE公开输出为`nullptr`/`None`；M-RoPE MX为FLOAT8_E8M0 `[T,Nq,ceil(D/32)]`。 | FLOAT / FLOAT8_E8M0 | ND |
| head_nums | 属性 | Q/K/V头数，按`[Nq,Nk,Nv]`传入；三个场景均要求`0<Nq<=64`、`Nq=8*Nk`、`Nk=Nv`。 | INT64数组 | - |
| layout_qkv | 可选属性 | `qkv`的N/T轴布局，默认值为`"TND"`。仅支持大小写敏感的`"TND"`和`"NTD"`。 | STRING | - |
| layout_q_out | 可选属性 | `q_out`和`q_scale`的N/T轴布局，默认值为`"NTD"`。仅支持大小写敏感的`"TND"`和`"NTD"`。 | STRING | - |
| epsilon | 可选属性 | RMSNorm防除零参数，默认值为`1e-6`。 | FLOAT | - |
| mrope_section | 可选属性 | M-RoPE场景`[t,h,w]`，非空时长度必须为3；空列表等价于未传。 | INT64数组 | - |
| q_quant_mode | 可选属性 | Q量化模式。RoPE为`PerTokenPerHead`，M-RoPE为`NoQuant`，M-RoPE MX为`Mx`。 | STRING | - |
| q_out_dtype | 可选属性 | Q输出类型。RoPE和M-RoPE MX为FP8 E4M3FN；M-RoPE为BF16。 | INT64 | - |
| k_quant_mode | 可选属性 | K量化算法模式，默认`PerTokenPerHead`，支持`PerTokenPerHead`和`Mx`；该属性不选择dtype，K存储dtype由`k_cache`和场景合同确定。 | STRING | - |

## 约束说明

- 三种场景的参数组合如下，其他交叉组合不支持：

  | 场景 | 位置输入 | rotation | Q / q_scale | K / k_scale_cache | V |
  |---|---|---|---|---|---|
  | RoPE | 传`query_start_loc/seq_lens`；不传`mrope_position/mrope_section` | 必须传`[D,D]` BF16 | `q_quant_mode=PerTokenPerHead`；Q为FP8 E4M3FN，scale为FP32 rank2 | `k_quant_mode=PerTokenPerHead`；K为FP8 E4M3FN，scale为FP32 rank4 | `v_scale=[Nv]`，输出FP8 E4M3FN |
  | M-RoPE | 传`mrope_position/mrope_section`；不传`query_start_loc/seq_lens` | 必须传`[D,D]` BF16 | `q_quant_mode=NoQuant`；Q为BF16，无公开scale | `k_quant_mode=PerTokenPerHead`；K为INT8，scale为FP32 rank4 | `v_scale=[Nv,D]`，输出FP8 E4M3FN |
  | M-RoPE MX | 传`mrope_position/mrope_section`；不传`query_start_loc/seq_lens` | 必须不传 | `q_quant_mode=Mx`；Q为FP8 E4M3FN，scale为FLOAT8_E8M0 rank3 | `k_quant_mode=Mx`；K为FP8 E4M3FN，scale为FLOAT8_E8M0 rank4 | `v_scale=[Nv,D]`，按head/channel静态量化为FP8 E4M3FN |

- 输入shape限制：
  - 当前实现仅支持`D=128`，三个场景均要求`1<=T<=262144`、`0<Nq<=64`、`Nq=8*Nk`、`Nk=Nv`。
  - RoPE场景必须同时传`query_start_loc/seq_lens`，不传`mrope_position`且`mrope_section`为空或未传；
    `q_quant_mode=k_quant_mode=PerTokenPerHead`、`q_out_dtype=FP8 E4M3FN`、`q_scale`非空，K Cache为FP8。
  - M-RoPE场景必须同时传`mrope_position[T,3]`和非空`mrope_section=[t,h,w]`，不传`query_start_loc/seq_lens`；
    M-RoPE要求`q_quant_mode=NoQuant`、`k_quant_mode=PerTokenPerHead`、`q_out_dtype=BF16`、公开`q_scale=None`，K Cache为INT8，且仅支持
    `layout_qkv="TND"`、`layout_q_out="TND"`。两组位置输入不能同时存在或同时缺失。
  - M-RoPE MX场景固定`layout_qkv=layout_q_out="TND"`、`q_quant_mode=k_quant_mode="Mx"`且不传`rotation`；
    Q/K输出为FP8 E4M3FN，Q scale为FLOAT8_E8M0 `[T,Nq,ceil(D/32)]`，K scale Cache为FLOAT8_E8M0
    `[BlockNum,Nk,BlockSize,ceil(D/32)]`。
  - `v_scale`在RoPE场景的shape必须为`[Nv]`，在M-RoPE和M-RoPE MX场景的shape必须为`[Nv,D]`。
  - M-RoPE的`mrope_section`长度必须为3且三项非负；D128时H/W两项的范围均为`[0,21]`，第0项无独立lane上限，三项之和不得超过64；`mrope_position`必须为INT32、rank 2、shape `[T,3]`。
  - `layout_qkv`控制`qkv`的N/T轴布局，默认值为`"TND"`；`layout_q_out`控制`q_out`和`q_scale`的N/T轴布局，默认值为`"NTD"`：
    - `layout_qkv="TND"`，`layout_q_out="TND"`：`qkv=[T, Nq+Nk+Nv, D]`，`q_out=[T, Nq, D]`，`q_scale=[T, Nq]`。
    - `layout_qkv="TND"`，`layout_q_out="NTD"`：`qkv=[T, Nq+Nk+Nv, D]`，`q_out=[Nq, T, D]`，`q_scale=[Nq, T]`。
    - `layout_qkv="NTD"`，`layout_q_out="NTD"`：`qkv=[Nq+Nk+Nv, T, D]`，`q_out=[Nq, T, D]`，`q_scale=[Nq, T]`。
  - `cos_sin`第二维必须为`D`，第一维必须覆盖本次调用会访问的RoPE/M-RoPE位置。
  - `k_cache`、`v_cache`和`k_scale_cache`支持非连续Tensor，需符合以下约束：
    - `k_cache`和`v_cache`均为4维正stride，最后一维stride为1，head维和token维stride均不小于`D=128`；
      K Cache dtype 按场景分别为FP8或INT8，V Cache始终为FP8。
    - RoPE/M-RoPE的`k_scale_cache`为4维正stride，最后一维stride为1；M-RoPE MX同样为4维，尾部`ceil(D/32)` scale轴连续，最后一维stride必须为1。
    - `k_cache`和`v_cache`前三维stride必须一致。
- 输入值域限制：
  - RoPE场景中，`query_start_loc`表示当前调用内token的batch前缀和，`query_start_loc[0]`应为0，`query_start_loc[Batch]`应等于`T`。
  - RoPE场景中，`seq_lens[b]`表示第`b`个batch追加本次token后的实际序列长度。对该batch内第`i`个token，RoPE位置由`seq_lens[b] - (query_start_loc[b+1] - query_start_loc[b]) + i`得到；调用方需保证`seq_lens[b] >= query_start_loc[b+1] - query_start_loc[b]`。若`seq_lens[b]`小于该batch本次调用的token数，行为未定义。
  - 两个M-RoPE场景中，`mrope_position`的每个位置索引必须满足`0 <= value < MaxSeqLen`。
  - `slot_mapping`取值范围应为`[0, BlockNum * BlockSize - 1]`。M-RoPE MX要求同一次调用内的slot互不重复；RoPE和M-RoPE场景存在重复slot时最终写入顺序和结果未定义。
- 输入属性限制：
  - `head_nums`必须包含3个正整数，顺序为`[Nq, Nk, Nv]`，并满足`Nq<=64`、`Nq=8*Nk`、
    `Nk=Nv`。
  - `layout_qkv`和`layout_q_out`大小写敏感，仅支持`"TND"`和`"NTD"`，且当前不支持`layout_qkv="NTD"`、`layout_q_out="TND"`。
- 输入数据类型限制：
  - 各输入的数据类型和数据格式需符合参数说明，不支持隐式类型转换。
  - 输入均为ND格式，不支持私有格式。
- 其他限制：
  - `v_scale`在三个场景中均必须传有效Tensor。`rotation`在RoPE/M-RoPE中必须传有效Tensor，在M-RoPE MX中必须不传。
  - 非可选输入和输出不支持空Tensor；可选位置输入按场景合同使用`None`，空`mrope_section=[]`等价于未传。

## 调用说明

<term>Ascend 950PR/Ascend 950DT</term>

| 调用方式 | 调用样例 | 说明 |
|---|---|---|
| aclnn API | [test_aclnn_qkv_rms_norm_rope_cache_with_k_scale](examples/test_aclnn_qkv_rms_norm_rope_cache_with_k_scale.cpp) | 通过[aclnnQkvRmsNormRopeCacheWithKScale](docs/aclnnQkvRmsNormRopeCacheWithKScale.md)接口方式调用QkvRmsNormRopeCacheWithKScale算子。 |
| PyTorch API | - | 通过[qkv_rms_norm_rope_cache_with_k_scale](../../torch_extension/cann_ops_transformer/docs/zh/qkv_rms_norm_rope_cache_with_k_scale.md)接口方式调用QkvRmsNormRopeCacheWithKScale算子。 |
