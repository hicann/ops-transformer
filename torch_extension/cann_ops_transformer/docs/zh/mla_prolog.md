# mla\_prolog

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR/Ascend 950DT</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：支持
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

- 接口功能：

  `mla_prolog`是`cann_ops_transformer`扩展接口，用于调用`MlaPrologV4WeightNz`算子完成 MLA（Multi-head Latent Attention）Decoder 前向过程中的 Query、KV Cache、Kr Cache 计算与缓存更新，并支持通过`rope_sin`/`rope_cos`是否成对传入控制是否执行 RoPE（Rotary Positional Embedding）。

  在 PagedAttention（PA）场景下，该接口以`kv_cache`/`kr_cache`作为缓存输入并在算子内部原地更新。接口支持以下场景：

  - `rope_sin`/`rope_cos`同时非空：对 Query 执行 RoPE，`query_rope`输出 RoPE 结果。
  - `rope_sin`/`rope_cos`同时为空（或`None`）：不执行 RoPE。
  - `rope_sin`/`rope_cos`一空一非空：非法输入，接口直接报错拦截。

- 计算公式：

    RmsNorm公式

    $$
    \text{RmsNorm}(x) = \gamma \cdot \frac{x_i}{\text{RMS}(x)}
    $$

    $$
    \text{RMS}(x) = \sqrt{\frac{1}{N} \sum_{i=1}^{N} x_i^2 + \epsilon}
    $$

    Query的计算公式，包括下采样、RmsNorm和两次上采样

    $$
    c^Q = \alpha_q\cdot\mathrm{RmsNorm}(x \cdot W^{DQ})
    $$

    $$
    q^C = c^Q \cdot W^{UQ}
    $$

    $$
    q^N = q^C \cdot W^{UK}
    $$

    对Query进行ROPE旋转位置编码

    $$
    q^R = \mathrm{ROPE}(c^Q \cdot W^{QR})
    $$

    不执行ROPE计算时

    $$
    q^R = c^Q \cdot W^{QR}
    $$

    Key的计算公式，包括下采样和RmsNorm，将计算结果存入cache

    $$
    c^{KV} = \alpha_{kv}\cdot\mathrm{RmsNorm}(x \cdot W^{DKV})
    $$

    $$
    k^C = \mathrm{Cache}(c^{KV})
    $$

    对Key进行ROPE旋转位置编码，并将结果存入cache

    $$
    k^R = \mathrm{Cache}(\mathrm{ROPE}(x \cdot W^{KR}))
    $$

    不执行ROPE计算时

    $$
    k^R = \mathrm{Cache}(x \cdot W^{KR})
    $$

    Dequant Scale Query Nope计算公式

    $$
    \mathrm{dequantScaleQNope} = {\mathrm{RowMax}(\mathrm{abs}(q^{N})) / 127}
    $$

    $$
    q^{N} = {\mathrm{round}(q^{N} / \mathrm{dequantScaleQNope})}
    $$


## 函数原型

`mla_prolog`为原地 cache 更新接口，直接更新调用方传入的`kv_cache`/`kr_cache`。

```python
cann_ops_transformer.mla_prolog(
    token_x,
    weight_dq,
    weight_uq_qr,
    weight_uk,
    weight_dkv_kr,
    rmsnorm_gamma_cq,
    rmsnorm_gamma_ckv,
    kv_cache,
    kr_cache,
    *,
    rope_sin=None,
    rope_cos=None,
    cache_index=None,
    dequant_scale_x=None,
    dequant_scale_w_dq=None,
    dequant_scale_w_uq_qr=None,
    dequant_scale_w_dkv_kr=None,
    quant_scale_ckv=None,
    quant_scale_ckr=None,
    smooth_scales_cq=None,
    actual_seq_len=None,
    k_nope_clip_alpha=None,
    rmsnorm_epsilon_cq=1e-05,
    rmsnorm_epsilon_ckv=1e-05,
    cache_mode='PA_BSND',
    query_norm_flag=False,
    weight_quant_mode=0,
    kv_cache_quant_mode=0,
    query_quant_mode=0,
    ckvkr_repo_mode=0,
    quant_scale_repo_mode=0,
    tile_size=128,
    qc_qr_scale=1.0,
    kc_scale=1.0,
    token_x_dtype=None,
    weight_dq_dtype=None,
    weight_uq_qr_dtype=None,
    weight_dkv_kr_dtype=None,
    kv_cache_dtype=None,
) -> (Tensor, Tensor, Tensor, Tensor, Tensor)
```

## 参数说明

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
|---|---|---|---|---|---|
| token_x | Tensor | 必选 | 公式中用于计算 Query 和 Key 的输入 tensor，不支持非连续，数据格式支持 ND。`weight_quant_mode`非全量化场景支持 2 维 `[T, He]`（BS 合轴）或 3 维 `[B, S, He]`（BS 非合轴）；`weight_quant_mode=5`（hifloat8 全量化）场景为 `torch.uint8` 存储，需配合`token_x_dtype`指定底层 ACL 数据类型。 | torch.float16/torch.bfloat16；hifloat8 场景为 torch.uint8 | 2维或3维，ND格式 |
| weight_dq | Tensor | 必选 | 公式中用于计算 Query 的下采样权重矩阵 $W^{DQ}$，不支持非连续，数据格式支持 FRACTAL_NZ，逻辑 shape 为`[He, Hcq]`，dtype 约束同`token_x`（hifloat8 场景为 torch.uint8）。 | 同`token_x` | 必须为 FRACTAL_NZ（4维）格式 |
| weight_uq_qr | Tensor | 必选 | 公式中用于计算 Query 的上采样权重矩阵 $W^{UQ}$ 和位置编码权重矩阵 $W^{QR}$，不支持非连续，数据格式支持 FRACTAL_NZ，逻辑 shape 为`[Hcq, N*(D+Dr)]`，`weight_quant_mode=0`场景 dtype 与`token_x`一致；`weight_quant_mode=1`（半量化）场景为 int8。 | 与`token_x`一致；mode=1 为 torch.int8 | 必须为 FRACTAL_NZ（4维）格式 |
| weight_uk | Tensor | 必选 | 公式中用于计算 Key 的上采样权重 $W^{UK}$，不支持非连续，数据格式支持 ND，shape 为`[N, D, Hckv]`。 | torch.bfloat16 | 3维`[N, D, Hckv]`，ND格式 |
| weight_dkv_kr | Tensor | 必选 | 公式中用于计算 Key 的下采样权重矩阵 $W^{DKV}$ 和位置编码权重矩阵 $W^{KR}$，不支持非连续，数据格式支持 FRACTAL_NZ，逻辑 shape 为`[He, Hckv+Dr]`，dtype 与`token_x`一致。 | 同`token_x` | 必须为 FRACTAL_NZ（4维）格式 |
| rmsnorm_gamma_cq | Tensor | 必选 | 计算 $c^{Q}$ 的 RMSNorm 公式中的 $\gamma$ 参数，不支持非连续，数据格式支持 ND，shape 为`[Hcq]`。 | 与`token_x`一致 | `[Hcq]` |
| rmsnorm_gamma_ckv | Tensor | 必选 | 计算 $c^{KV}$ 的 RMSNorm 公式中的 $\gamma$ 参数，不支持非连续，数据格式支持 ND，shape 为`[Hckv]`。 | 与`token_x`一致 | `[Hckv]` |
| kv_cache | Tensor | 必选 | 表示 cache 的索引，计算结果原地更新（对应公式中的 $k^{C}$），仅支持首轴非连续，除首轴外的其余轴必须连续，数据格式支持 ND。PA 模式（`cache_mode`为"PA_BSND"/"PA_NZ"/"PA_BLK_BSND"/"PA_BLK_NZ"）下支持空 Tensor；"BSND"/"TND" 模式下不支持空 Tensor。Nkv 与 N 关联，N 是超参，故不支持 Nkv=0。 | torch.bfloat16；hifloat8 全量化 KV 场景为 torch.uint8 | PA 模式：4维`[BlockNum, BlockSize, Nkv, Dtile]`；"BSND"：4维`[B, S, Nkv, Dtile]`；"TND"：3维`[T, Nkv, Dtile]`，ND格式 |
| kr_cache | Tensor | 必选 | 用于 key 位置编码的 cache，计算结果原地更新（对应公式中的 $k^{R}$），仅支持首轴非连续，除首轴外的其余轴必须连续，数据格式支持 ND。PA 模式下支持空 Tensor；"BSND"/"TND" 模式下不支持空 Tensor。`rope_sin`/`rope_cos`均为空（或`None`）时不执行 RoPE。 | torch.bfloat16 | 与`kv_cache`布局一致：PA 模式`[BlockNum, BlockSize, Nkv, Dr]`；"BSND"`[B, S, Nkv, Dr]`；"TND"`[T, Nkv, Dr]`，ND格式 |
| rope_sin | Tensor | 可选 | 用于计算旋转位置编码的正弦参数矩阵，不支持非连续，数据格式支持 ND。与`rope_cos`同时非空时执行 RoPE，维度与`token_x`一致；同时为空（或`None`）时不执行 RoPE；一空一非空为非法输入。默认值为`None`，支持 B=0、S=0、T=0 的空 Tensor。 | torch.bfloat16 | 2维`[T, Dr]`或3维`[B, S, Dr]` |
| rope_cos | Tensor | 可选 | 用于计算旋转位置编码的余弦参数矩阵，不支持非连续，数据格式支持 ND，约束同`rope_sin`。默认值为`None`。 | torch.bfloat16 | 与`rope_sin`一致 |
| cache_index | Tensor | PA 模式必选，BSND/TND 可选 | 用于存储`kv_cache`和`kr_cache`的索引，不支持非连续，数据格式支持 ND。`cache_mode`为"PA_BSND"/"PA_NZ"：BS 合轴时 shape 为`[T]`，BS 非合轴时 shape 为`[B, S]`，取值范围需在 `[0, BlockNum*BlockSize)` 内；`cache_mode`为"PA_BLK_BSND"/"PA_BLK_NZ"：BS 合轴时 shape 为`[Sum(Ceil(S_i/BlockSize))]`（S_i 表示第 i 个 batch 的序列长度），BS 非合轴时 shape 为`[B, Ceil(S/BlockSize)]`，取值范围需在 `[0, BlockNum)` 内；`cache_mode`为"BSND"/"TND"：无需传入。PagedAttention 模式下不传该参数会被接口报错拦截。 | torch.int32/torch.int64 | 1维`[T]`或2维`[B, S]` |
| dequant_scale_x | Tensor | 可选 | `token_x`的反量化参数，不支持非连续，数据格式支持 ND。`weight_quant_mode=2/4/5`时 shape 为`[T,1]`或`[B*S,1]`，`weight_quant_mode=3`时 shape 为`[T, He/32]`或`[B*S, He/32]`；mode=3 时 dtype 为`torch.float8_e8m0fnu`，mode=2/4/5 时 dtype 为 float32。默认值为`None`，支持 B=0、S=0、T=0 的空 Tensor。 | torch.float8_e8m0fnu（mode=3）/torch.float32（mode=2/4/5） | mode=2/4/5：`[T, 1]`或`[B*S, 1]`；mode=3：`[T, He/32]`或`[B*S, He/32]` |
| dequant_scale_w_dq | Tensor | 可选 | `weight_dq`的反量化参数，不支持非连续，数据格式支持 ND。`weight_quant_mode=2/4/5`时 shape 为`[1, Hcq]`，`weight_quant_mode=3`时 shape 为`[Hcq, He/32]`，dtype 约束同`dequant_scale_x`。默认值为`None`。 | 同`dequant_scale_x` | mode=2/4/5：`[1, Hcq]`；mode=3：`[Hcq, He/32]` |
| dequant_scale_w_uq_qr | Tensor | 可选 | `weight_uq_qr`的反量化参数，用于 MatmulQcQr 矩阵乘后反量化操作的 perchannel 参数，不支持非连续，数据格式支持 ND。`weight_quant_mode=1/2/4/5`时 shape 为`[1, N*(D+Dr)]`，`weight_quant_mode=3`时 shape 为`[N*(D+Dr), Hcq/32]`，dtype 约束同`dequant_scale_x`。默认值为`None`。 | 同`dequant_scale_x` | mode=1/2/4/5：`[1, N*(D+Dr)]`；mode=3：`[N*(D+Dr), Hcq/32]` |
| dequant_scale_w_dkv_kr | Tensor | 可选 | `weight_dkv_kr`的反量化参数，不支持非连续，数据格式支持 ND。`weight_quant_mode=2/4/5`时 shape 为`[1, Hckv+Dr]`，`weight_quant_mode=3`时 shape 为`[Hckv+Dr, He/32]`，dtype 约束同`dequant_scale_x`。默认值为`None`。 | 同`dequant_scale_x` | mode=2/4/5：`[1, Hckv+Dr]`；mode=3：`[Hckv+Dr, He/32]` |
| quant_scale_ckv | Tensor | 可选 | 用于对 kv_cache 输出数据做量化操作的参数，不支持非连续，数据格式支持 ND。`kv_cache_quant_mode=1`时 shape 为`[1]`；`kv_cache_quant_mode=2`时 shape 为`[1, Hckv]`；`kv_cache_quant_mode=3`时无需赋值。全量化 KV per-tensor 场景（`kv_cache_quant_mode=1`且`weight_quant_mode`为 2/3/4/5）时须传入。默认值为`None`。 | torch.float32 | `kv_cache_quant_mode=1`：`[1]`；mode=2：`[1, Hckv]` |
| quant_scale_ckr | Tensor | 可选 | 用于对 kr_cache 输出数据做量化操作的参数，不支持非连续，数据格式支持 ND。`kv_cache_quant_mode=2`时 shape 为`[1, Dr]`；`kv_cache_quant_mode=1/3`时无需赋值。默认值为`None`。 | torch.float32 | `[1, Dr]` |
| smooth_scales_cq | Tensor | 可选 | 用于对 RMSNorm_cq 输出做动态量化操作的参数，不支持非连续，数据格式支持 ND。`weight_quant_mode=1/2/4/5`时 shape 为`[1, Hcq]`。默认值为`None`。 | torch.float32 | `[1, Hcq]` |
| actual_seq_len | Tensor | 可选 | 表示每个 batch 中的序列长度，以前缀和的形式储存，不支持非连续，数据格式支持 ND，仅 BS 合轴且`cache_mode`为"PA_BLK_BSND"/"PA_BLK_NZ"时需要传入。默认值为`None`。 | torch.int32 | `[B]` |
| k_nope_clip_alpha | Tensor | 可选 | 表示 kv_cache 做 clip 操作时的缩放因子，在部分量化 pertoken-pergroup 场景和 int8 全量化 pertoken-pergroup 场景下使用，其他场景无需赋值，不支持非连续，数据格式支持 ND。默认值为`None`。 | torch.float32 | `[1]` |
| rmsnorm_epsilon_cq | float | 可选 | 计算 $c^{Q}$ 的 RMSNorm 公式中的 $\epsilon$ 参数，默认值为`1e-5`。 | float | - |
| rmsnorm_epsilon_ckv | float | 可选 | 计算 $c^{KV}$ 的 RMSNorm 公式中的 $\epsilon$ 参数，默认值为`1e-5`。 | float | - |
| cache_mode | str | 可选 | kv_cache 的模式，可选值为"PA_BSND"、"PA_NZ"、"PA_BLK_BSND"、"PA_BLK_NZ"（对应 PagedAttention）、"TND"（对应 BS 合轴）和"BSND"（对应 BS 非合轴），默认为"PA_BSND"。 | str | - |
| query_norm_flag | bool | 可选 | 是否输出 query_norm。False 表示不输出，True 表示输出（量化场景下伴随输出 dequant_scale_q_norm），默认值为 False。 | bool | - |
| weight_quant_mode | int | 可选 | weight_dq、weight_uq_qr、weight_uk、weight_dkv_kr 的量化模式。0：非量化；1：weight_uq_qr 量化（半量化）；2：weight_dq、weight_uq_qr、weight_dkv_kr int8 全量化；3：mxfp8 全量化；4：fp8 全量化；5：hif8 全量化。默认值为 0。 | int | - |
| kv_cache_quant_mode | int | 可选 | kv_cache 的量化模式。0：非量化；1：per-tensor 量化；2：per-channel 量化；3：per-token-per-group 量化。默认值为 0。 | int | - |
| query_quant_mode | int | 可选 | query 的量化模式。0：非量化；1：per-token-head 量化。默认值为 0。 | int | - |
| ckvkr_repo_mode | int | 可选 | kv_cache 和 kr_cache 的存储模式。0：分别存储；1：合并存储。默认值为 0。 | int | - |
| quant_scale_repo_mode | int | 可选 | 量化 scale 的存储模式。0：scale 和数据分别存储；1：scale 和数据合并存储。默认值为 0。 | int | - |
| tile_size | int | 可选 | per-token-per-group 量化时每个 tile 的大小，仅在`kv_cache_quant_mode=3`时有效，默认值为 128；`kv_cache_quant_mode=1/2`时无需赋值。 | int | - |
| qc_qr_scale | float | 可选 | Query 的尺度矫正系数，默认值为 1.0。 | float | - |
| kc_scale | float | 可选 | Key 的尺度矫正系数，默认值为 1.0。 | float | - |
| token_x_dtype | int | 可选 | 参数 token_x 的传入 dtype，在 hif8 全量化场景为`torch_npu.hifloat8`（枚举值 290），其他场景为 None。 | int | - |
| weight_dq_dtype | int | 可选 | 参数 weight_dq 的传入 dtype，在 hif8 全量化场景为`torch_npu.hifloat8`，其他场景为 None。 | int | - |
| weight_uq_qr_dtype | int | 可选 | 参数 weight_uq_qr 的传入 dtype，在 hif8 全量化场景为`torch_npu.hifloat8`，其他场景为 None。 | int | - |
| weight_dkv_kr_dtype | int | 可选 | 参数 weight_dkv_kr 的传入 dtype，在 hif8 全量化场景为`torch_npu.hifloat8`，其他场景为 None。 | int | - |
| kv_cache_dtype | int | 可选 | 参数 kv_cache 的传入 dtype，在 hif8 kv_cache per-tensor 量化和 hif8 kv_cache per-token-per-group 量化场景为`torch_npu.hifloat8`，其他场景为 None。 | int | - |

## 返回值说明

`mla_prolog`返回 5 个 Tensor，且原地更新调用方传入的`kv_cache`/`kr_cache`。

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
|---|---|---|---|---|---|
| query | Tensor | 必选 | Q 侧输出。全量化 KV 场景（`weight_quant_mode∈{2,3,4,5}`且`kv_cache_quant_mode=1`）dtype 与`token_x`一致，其余场景为 bfloat16。 | torch.float16/torch.bfloat16；hifloat8 场景为 torch.uint8 | `[T, N, Hckv]`或`[B, S, N, Hckv]` |
| query_rope | Tensor | 必选 | RoPE 输出，rope_sin`/`rope_cos`均为空（或`None`）时不做RoPE计算直接输出。 | torch.bfloat16 | `[T, N, Dr]`或`[B, S, N, Dr]` |
| dequant_scale_q_nope | Tensor | 必选 | Q 的 nope 部分反量化 scale，仅在`weight_quant_mode∈{2,3,4,5}`且`kv_cache_quant_mode=1`（全量化 KV per-tensor 场景）时有效，否则为空 Tensor。 | torch.float32 | `[T, N, 1]`或`[B*S, N, 1]`；否则为空 Tensor |
| query_norm | Tensor | 必选 | Query 归一化输出，`query_norm_flag=True`时有效，否则为空 Tensor。 | 与`weight_uq_qr`一致 | `[T, Hcq]`或`[B, S, Hcq]`；否则为空 Tensor |
| dequant_scale_q_norm | Tensor | 必选 | Query 归一化的反量化 scale，`query_norm_flag=True`且`weight_quant_mode!=0`时有效，否则为空 Tensor。 | torch.float32；`weight_quant_mode=3`时与`dequant_scale_x`一致 | `[T, 1]`或`[T, Hcq/32]`；否则为空 Tensor |

## 约束说明

- 该接口支持推理场景下使用。

- 该接口支持单算子模式和图模式。

- shape 格式字段含义说明

    | 字段名 | 英文全称/含义 | 取值规则与说明 |
    |--------------|--------------------------------|------------------------------------------------------------------------------|
    | B | Batch（输入样本批量大小） | 取值范围：0~65536 |
    | S | Seq-Length（输入样本序列长度） | 取值范围：不限制 |
    | He | Head-Size（隐藏层大小） | 取值固定为：1024、2048、3072、4096、5120、6144、7168、7680、8192 |
    | Hcq | q 低秩矩阵维度 | 取值固定为：1536、2048 |
    | N | Head-Num（多头数） | 取值范围：1-128 |
    | Hckv | kv 低秩矩阵维度 | 取值固定为：512 |
    | Dtile | kv_cache 的 D 轴维度 | 取值固定为：pertoken-pergroup 场景（`kv_cache_quant_mode=3`）下为 656，非 pertoken-pergroup 场景下为 512 |
    | D | qk 不含位置编码维度 | 取值固定为：128、192 |
    | Dr | qk 位置编码维度 | 取值固定为：64 |
    | Nkv | kv 的 head 数 | 取值固定为：1 |
    | BlockNum | PagedAttention 场景下的块数 | 取值为计算 `B*Skv/BlockSize` 的结果后向上取整（Skv 表示 kv 的序列长度，允许取 0） |
    | BlockSize | PagedAttention 场景下的块大小 | 取值范围：16-1024，且为 16 的倍数 |
    | T | BS 合轴后的大小 | 取值范围：不限制；注：若采用 BS 合轴，此时 token_x、query_norm 均为 2 维，query_out、query_rope_out 为 3 维，cache_index 为 1 维，rope_sin、rope_cos同为None（关闭RoPE）或同为2维（开启RoPE）|

- 接口内置校验：上述 shape 格式字段约束（He/Hcq/N/D/Hckv/Dr/Nkv/Dtile/BlockSize）在接口内部统一校验，不满足时报错拦截。其中 N（Head-Num 多头数）允许 `[1, 128]` 之间的任意整型值。

- `token_x`仅支持 2 维或 3 维；`weight_uk`必须为 3 维。
- `weight_dq`/`weight_uq_qr`/`weight_dkv_kr`必须为 FRACTAL_NZ 格式（通过 `torch_npu.npu_format_cast(t, 29)` 转换），ND 格式会被接口报错拦截。
- `rope_sin`/`rope_cos`必须成对传入：同时非空时执行 RoPE；同时为空（或`None`）时不执行 RoPE；一空一非空为非法输入，接口报错拦截。
- `cache_mode`为 PagedAttention 模式（`"PA_BSND"`/`"PA_NZ"`/`"PA_BLK_BSND"`/`"PA_BLK_NZ"`）时，`cache_index`必须传入（非空），否则接口报错拦截；`"BSND"`/`"TND"` 模式可选。
- `weight_quant_mode=3`（mxfp8）或`weight_quant_mode=5`（hifloat8）时，`dequant_scale_x`/`dequant_scale_w_dq`/`dequant_scale_w_uq_qr`/`dequant_scale_w_dkv_kr`必须同时非空；mode=3 时 dtype 均为 `torch.float8_e8m0fnu`。
- `weight_quant_mode=5` 且输入为 `torch.uint8` 时，必须显式指定各 `*_dtype` 为 `torch_npu.hifloat8`（枚举值 290）。
- `kv_cache_quant_mode=1`（per-tensor）且`weight_quant_mode`为全量化（2/3/4/5）时，`quant_scale_ckv`必须传入。
- `kv_cache_quant_mode`取值须与`weight_quant_mode`匹配：如 `weight_quant_mode=0` 时仅支持 `kv_cache_quant_mode=0`。
- `query_quant_mode`仅支持 0 或 1。
- 当前仅支持 NPU 设备（PrivateUse1 dispatch key），CPU 不支持。

## 确定性计算

默认支持确定性计算。

## 调用说明

- 基本调用（执行 RoPE）：

  ```python
  import torch
  import torch_npu
  import cann_ops_transformer

  # 满足 shape 格式字段约束的入参示例：
  # He=1024，Hcq=1536，N=8，D=128，Dr=64，Hckv=512，Nkv=1，BlockNum=4，BlockSize=128，Dtile=512
  T = 128
  He, Hcq, N, D, Dr = 1024, 1536, 8, 128, 64
  Hckv, Nkv = 512, 1
  BlockNum, BlockSize, Dtile = 4, 128, 512

  dtype = torch.bfloat16
  token_x = torch.randn(T, He, dtype=dtype).npu()
  weight_dq = torch.randn(He, Hcq, dtype=dtype).npu()
  weight_dq_cast = torch_npu.npu_format_cast(weight_dq.contiguous(), 29)
  weight_uq_qr = torch.randn(Hcq, N * (D + Dr), dtype=dtype).npu()
  weight_uq_qr_cast = torch_npu.npu_format_cast(weight_uq_qr.contiguous(), 29)
  weight_uk = torch.randn(N, D, Hckv, dtype=dtype).npu()
  weight_dkv_kr = torch.randn(He, Hckv + Dr, dtype=dtype).npu()
  weight_dkv_kr_cast = torch_npu.npu_format_cast(weight_dkv_kr.contiguous(), 29)
  rmsnorm_gamma_cq = torch.randn(Hcq, dtype=dtype).npu()
  rmsnorm_gamma_ckv = torch.randn(Hckv, dtype=dtype).npu()
  rope_sin = torch.randn(T, Dr, dtype=dtype).npu()
  rope_cos = torch.randn(T, Dr, dtype=dtype).npu()
  kv_cache = torch.zeros(BlockNum, BlockSize, Nkv, Dtile, dtype=dtype).npu()
  kr_cache = torch.zeros(BlockNum, BlockSize, Nkv, Dr, dtype=dtype).npu()
  cache_index = torch.arange(T, dtype=torch.int64).npu()  # 默认 PA_BSND 模式下必传

  # 原地更新 kv_cache / kr_cache
  query, query_rope, dequant_scale_q_nope, query_norm, dequant_scale_q_norm = (
      cann_ops_transformer.mla_prolog(
          token_x, weight_dq_cast, weight_uq_qr_cast, weight_uk, weight_dkv_kr_cast,
          rmsnorm_gamma_cq, rmsnorm_gamma_ckv, kv_cache, kr_cache,
          rope_sin=rope_sin,
          rope_cos=rope_cos,
          cache_index=cache_index,
          query_norm_flag=True,
      )
  )
  ```

- 不执行 RoPE 的调用：

  不传`rope_sin`/`rope_cos`（或同时显式传`None`）即不执行 RoPE，底层以`doRope=false`调用 aclnn 接口。

  ```python
  # rope_sin/rope_cos 同时省略（或同时为 None），不执行 RoPE
  query, query_rope, dequant_scale_q_nope, query_norm, dequant_scale_q_norm = (
      cann_ops_transformer.mla_prolog(
          token_x, weight_dq_cast, weight_uq_qr_cast, weight_uk, weight_dkv_kr_cast,
          rmsnorm_gamma_cq, rmsnorm_gamma_ckv, kv_cache, kr_cache,
          cache_index=cache_index,  # 默认 PA_BSND 模式下必传
      )
  )
  ```
