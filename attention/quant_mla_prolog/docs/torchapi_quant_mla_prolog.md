# quant_mla_prolog

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT系列产品</term>：支持
<!-- end id1 -->

<!-- npu="A3" id2 -->
- <term>Atlas A3系列产品</term>：不支持
<!-- end id2 -->

<!-- npu="910b" id3 -->
- <term>Atlas A2系列产品</term>：不支持
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

多头潜在注意力（Multi-Head Latent Attention，MLA）前处理使用均方根归一化（Root Mean Square Normalization，RMSNorm）和旋转位置编码（Rotary Positional Embedding，RoPE）。Q表示Query，KV表示共享Key/Value低秩向量，KR表示Key的旋转子段；QA/QB/KVA对应三路投影，Wkb将Q的非位置编码（NoPE）部分吸收到512维。ND为线性密排存储，NZ为FRACTAL_NZ分块格式；BF16/FP32分别表示bfloat16/float32，HiF8使用uint8原始编码。

完成多头潜在注意力（Multi-Head Latent Attention，MLA）推理前处理，融合投影、均方根归一化（Root Mean Square Normalization，RMSNorm）、旋转位置编码（Rotary Positional Embedding，RoPE）、可选量化和Cache原地更新。输入/投影权重使用MXFP8或HiF8。公式及转换点见[算子说明](../README.md#功能说明)。

## 函数原型

`quant_mla_prolog`原地更新调用方传入的`kv_cache`，返回`q`、`descale_q`、`qr`、`descale_qr`。

```python
cann_ops_transformer.ops.quant_mla_prolog(
    x,
    wqa,
    wqb,
    wkva,
    wkb,
    descale_x,
    descale_wqa,
    descale_wqb,
    descale_wkva,
    norm_weight_qa,
    norm_weight_kva,
    kv_cache,
    cache_index,
    w_qr_quant_mode,
    qkv_quant_mode,
    *,
    quant_scale_kv=None,
    rope_sin=None,
    rope_cos=None,
    norm_eps=1e-05,
    cache_mode=CacheMode.PA_BSND,
    qr_scale=1.0,
    kvr_scale=1.0,
    rope_mode=RopeMode.ROPE_INTERLEAVE_HALF,
    return_qr=False,
) -> (Tensor, Tensor, Tensor, Tensor)
```

## 参数说明

`*`之前的参数为必选，可按位置或关键字传入；`*`之后的参数为可选，仅接受关键字传入。`quant_scale_kv`及RoPE参数在特定模式下的必填条件见下表。

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
| --- | --- | --- | --- | --- | --- |
| `x` | Tensor | 必选 | 输入隐状态。格式为ND。 | MXFP8:float8_e4m3fn; HiF8:uint8 | `(T,dim)` |
| `wqa` | Tensor | 必选 | Query低秩投影权重，将dim维输入投影到q_lora维。格式为ND/FRACTAL_NZ。 | MXFP8:float8_e4m3fn; HiF8:uint8 | `ND:(q_lora,dim); NZ:(dim,q_lora)` |
| `wqb` | Tensor | 必选 | Q上投影权重。格式为ND/FRACTAL_NZ。 | MXFP8:float8_e4m3fn; HiF8:uint8 | `ND:(N*(Dn+Dr),q_lora); NZ:(q_lora,N*(Dn+Dr))` |
| `wkva` | Tensor | 必选 | KVA投影权重。格式为ND/FRACTAL_NZ。 | MXFP8:float8_e4m3fn; HiF8:uint8 | `ND:(512+Dr,dim); NZ:(dim,512+Dr)` |
| `wkb` | Tensor | 必选 | 每头Q-NoPE吸收权重，不参与Cache生成。格式为ND。 | bfloat16 | `(N,Dn,512)` |
| `descale_x` | Tensor | 必选 | 激活反量化乘子。格式为ND。 | MXFP8:float8_e8m0fnu; HiF8:float32 | `MXFP8:(T,dim/32); HiF8:(1,)` |
| `descale_wqa` | Tensor | 必选 | Wqa反量化乘子；O=q_lora，K=dim。格式为ND/FRACTAL_NZ。 | MXFP8:float8_e8m0fnu/uint8(NZ); HiF8:float32 | `ND:(O,K/64,2); NZ:(K/64,O,2); HiF8:(1,O)` |
| `descale_wqb` | Tensor | 必选 | Wqb反量化乘子；O=N*(Dn+Dr)，K=q_lora。格式为ND/FRACTAL_NZ。 | MXFP8:float8_e8m0fnu/uint8(NZ); HiF8:float32 | `ND:(O,K/64,2); NZ:(K/64,O,2); HiF8:(1,O)` |
| `descale_wkva` | Tensor | 必选 | Wkva反量化乘子；O=512+Dr，K=dim。格式为ND/FRACTAL_NZ。 | MXFP8:float8_e8m0fnu/uint8(NZ); HiF8:float32 | `ND:(O,K/64,2); NZ:(K/64,O,2); HiF8:(1,O)` |
| `norm_weight_qa` | Tensor | 必选 | QA均方根归一化的gamma。格式为ND。 | float32 | `(q_lora,)` |
| `norm_weight_kva` | Tensor | 必选 | KV均方根归一化的gamma。格式为ND。 | float32 | `(512,)` |
| `kv_cache` | Tensor | 必选 | 原地更新的KV/KR合并Cache；仅写入有效索引对应的槽位。格式为ND载体。 | mode 0:bfloat16; mode 1/2:float8_e4m3fn; mode 3:uint8 | `(B,S,1,Dcache)` |
| `cache_index` | Tensor | 必选 | token对应的线性Cache槽位；负值跳过Cache写入。格式为ND。 | int64 | `(T,)` |
| `w_qr_quant_mode` | WQrQuantMode | 必选 | 输入/权重/QR量化模式，使用WQrQuantMode；必填。 | - | - |
| `qkv_quant_mode` | QkvQuantMode | 必选 | Q/Cache输出模式，取本算子QkvQuantMode枚举，见模式表。 | - | - |
| `quant_scale_kv` | Tensor \| None | 可选 | 默认None；模式2/3必填，须为有限值，允许0和负数；模式0/1必须为None。Cache按value*quant_scale_kv量化。格式为ND。 | float32 | `(1,)` |
| `rope_sin` | Tensor \| None | 可选 | 默认None；启用RoPE时必填，与rope_cos成对传入。格式为ND。 | bfloat16 | `(T,Dr)` |
| `rope_cos` | Tensor \| None | 可选 | 默认None；RoPE余弦参数，与rope_sin成对传入。格式为ND。 | bfloat16 | `(T,Dr)` |
| `norm_eps` | float | 可选 | 默认1e-5；两路RMSNorm共用，转FP32后有限且大于0，拒绝bool。 | - | - |
| `cache_mode` | CacheMode | 可选 | CacheMode.PA_BSND或CacheMode.PA_NZ；模式1仅支持PA_BSND。可选，默认PA_BSND。 | - | - |
| `qr_scale` | float | 可选 | 默认1.0；QA归一化后的乘子，FP32有限，允许0和负数，拒绝bool。 | - | - |
| `kvr_scale` | float | 可选 | 默认1.0；KV归一化后的乘子，不缩放KR；数值要求与qr_scale一致。 | - | - |
| `rope_mode` | RopeMode | 可选 | 默认RopeMode.ROPE_INTERLEAVE_HALF；支持NO_ROPE、ROPE_INTERLEAVE_HALF和ROPE_HALF。 | - | - |
| `return_qr` | bool | 可选 | 默认False；为True时填充qr输出槽位。 | - | - |

表中`wqa`、`wqb`、`wkva`的Shape均指Python可见的逻辑形状（`Tensor.shape`），三者须全部使用ND或全部使用FRACTAL_NZ。ND按`(输出维度,输入维度)`传入；NZ按`(输入维度,输出维度)`传入转置后的同一组权重，并采用实际的FRACTAL_NZ存储。仅转置或reshape不能完成NZ格式转换。

T为token数，N为head数，B/S为Cache页数/页大小。Dcache在模式1为512+2*Dr+16，其余为512+Dr。HiF8使用uint8原始编码载体。

## 返回值说明

| 返回项 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
| --- | --- | --- | --- | --- | --- |
| q | Tensor | 必选 | Q-NoPE与Q-RoPE拼接，每头512+Dr个值。 | mode 0/1:bfloat16; mode 2:float8_e4m3fn; mode 3:uint8 | `(T,N,512+Dr)` |
| descale_q | Tensor | 必选 | Q反量化乘子；模式0/1及空输入模式3返回空Tensor。 | float32 | mode 2:(T,N,1); mode 3,T>0:(1,); otherwise:(0,) |
| qr | Tensor | 必选 | 仅return_qr=True时填充，否则Shape为(0,)。 | MXFP8:float8_e4m3fn; HiF8:uint8 | return_qr=True:(T,q_lora); otherwise:(0,) |
| descale_qr | Tensor | 必选 | QR反量化乘子；未输出QR或HiF8空输入时为空。 | MXFP8:float8_e8m0fnu; HiF8:float32 | QR+MXFP8:(T,q_lora/32); QR+HiF8,T>0:(1,); otherwise:(0,) |

## 约束说明

- `T >= 0`；`dim`为1024～8192内的256倍数；`q_lora`为640、1024、1536或2048；`N`为1～128；`(Dn,Dr)`为`(128,64)`、`(192,64)`或`(256,0)`；KV低秩维度固定为512。
- `B > 0`；`S`为16～1024内的16倍数。模式1仅支持PA_BSND，其余模式支持PA_BSND/PA_NZ。输入只接受合轴的`(T,dim)`，不接受`(B,S,dim)`或PA_BLK布局。
- 所有输入位于同一NPU。除Cache外均须连续；Cache内侧stride为`(Dcache,Dcache,1)`，且`stride(0) >= S*Dcache`，支持首轴间隙及非零storage offset。未命中槽位和页间空隙的字节保持不变。
- 调用方保证非负Cache索引小于`B*S`且唯一；算子不回读或检查索引值。负索引仅跳过Cache写入，对应token仍参与Q/QR及全局量化统计。
- `Dr=0`必须指定`NO_ROPE`；执行RoPE时sin/cos必须成对传入。禁用时可同时为None，若传入仍检查Shape及类型。
- `norm_eps`转FP32后须有限且大于0；`qr_scale/kvr_scale`须为有限FP32，允许0和负数，均拒绝bool。模式2/3的`quant_scale_kv`须为有限值，允许0和负数，模式0/1必须为None。
- `T=0`仍校验元信息，按接口返回空输出，不启动计算Kernel，Cache不变。模式0、2、3的Cache宽度为`512+Dr`，模式1为`512+2*Dr+16`。
- 支持Decode/Prefill推理，不提供自动求导或功能化Cache更新。输出和按阶段复用的临时空间由内部申请，无外部out/workspace参数；非空调用要求当前Stream具有有效MIX核资源。
- 非空输入使用quant_scale_kv时，在执行前及修改scale后调用公开入口的precheck；该操作在图外回读scale，执行入口不重复检查其值。ND激活/权重descale必须为E8M0，UINT8仅用于合法分配的NZ权重descale载体。

## 确定性计算

默认支持确定性计算。

## 调用示例

以下三个示例均使用ND布局的MXFP8输入，各自包含导入、输入准备和调用，可分别复制运行。

### Eager

```python
import torch
import torch_npu
from cann_ops_transformer.ops import quant_mla_prolog as prolog

torch.npu.set_device(0)
device = "npu:0"
T, dim, q_lora, N, Dn, Dr = 3, 1024, 1024, 4, 128, 64
bf16 = torch.bfloat16

# 准备ND输入；E8M0 scale的编码127表示缩放值1。
inputs = dict(
    x=torch.full((T, dim), 0.125).to(torch.float8_e4m3fn).to(device),
    wqa=torch.full((q_lora, dim), 0.125).to(torch.float8_e4m3fn).to(device),
    wqb=torch.full((N * (Dn + Dr), q_lora), 0.125).to(torch.float8_e4m3fn).to(device),
    wkva=torch.full((512 + Dr, dim), 0.125).to(torch.float8_e4m3fn).to(device),
    wkb=torch.full((N, Dn, 512), 0.125, dtype=bf16, device=device),
    descale_x=torch.full((T, dim // 32), 127, dtype=torch.uint8).view(torch.float8_e8m0fnu).to(device),
    descale_wqa=torch.full((q_lora, dim // 64, 2), 127, dtype=torch.uint8).view(torch.float8_e8m0fnu).to(device),
    descale_wqb=torch.full((N * (Dn + Dr), q_lora // 64, 2), 127, dtype=torch.uint8).view(torch.float8_e8m0fnu).to(device),
    descale_wkva=torch.full((512 + Dr, dim // 64, 2), 127, dtype=torch.uint8).view(torch.float8_e8m0fnu).to(device),
    norm_weight_qa=torch.ones(q_lora, dtype=torch.float32, device=device),
    norm_weight_kva=torch.ones(512, dtype=torch.float32, device=device),
    kv_cache=torch.zeros((2, 128, 1, 512 + Dr), dtype=bf16, device=device),
    cache_index=torch.arange(T, dtype=torch.int64, device=device),
    rope_sin=torch.zeros((T, Dr), dtype=bf16, device=device),
    rope_cos=torch.ones((T, Dr), dtype=bf16, device=device),
)
attrs = {
    "w_qr_quant_mode": prolog.WQrQuantMode.A8W8_MXFP8,
    "qkv_quant_mode": prolog.QkvQuantMode.A16C16,
}
prolog.precheck(**inputs, **attrs)
q, descale_q, qr, descale_qr = prolog(**inputs, **attrs)
torch.npu.synchronize()
print(q.shape, q.dtype)
```

### 图模式（ACLGraph）

```python
import torch
import torch_npu
from cann_ops_transformer.ops import quant_mla_prolog as prolog

torch.npu.set_device(0)
device = "npu:0"
T, dim, q_lora, N, Dn, Dr = 3, 1024, 1024, 4, 128, 64
bf16 = torch.bfloat16

# 准备ND输入；E8M0 scale的编码127表示缩放值1。
inputs = dict(
    x=torch.full((T, dim), 0.125).to(torch.float8_e4m3fn).to(device),
    wqa=torch.full((q_lora, dim), 0.125).to(torch.float8_e4m3fn).to(device),
    wqb=torch.full((N * (Dn + Dr), q_lora), 0.125).to(torch.float8_e4m3fn).to(device),
    wkva=torch.full((512 + Dr, dim), 0.125).to(torch.float8_e4m3fn).to(device),
    wkb=torch.full((N, Dn, 512), 0.125, dtype=bf16, device=device),
    descale_x=torch.full((T, dim // 32), 127, dtype=torch.uint8).view(torch.float8_e8m0fnu).to(device),
    descale_wqa=torch.full((q_lora, dim // 64, 2), 127, dtype=torch.uint8).view(torch.float8_e8m0fnu).to(device),
    descale_wqb=torch.full((N * (Dn + Dr), q_lora // 64, 2), 127, dtype=torch.uint8).view(torch.float8_e8m0fnu).to(device),
    descale_wkva=torch.full((512 + Dr, dim // 64, 2), 127, dtype=torch.uint8).view(torch.float8_e8m0fnu).to(device),
    norm_weight_qa=torch.ones(q_lora, dtype=torch.float32, device=device),
    norm_weight_kva=torch.ones(512, dtype=torch.float32, device=device),
    kv_cache=torch.zeros((2, 128, 1, 512 + Dr), dtype=bf16, device=device),
    cache_index=torch.arange(T, dtype=torch.int64, device=device),
    rope_sin=torch.zeros((T, Dr), dtype=bf16, device=device),
    rope_cos=torch.ones((T, Dr), dtype=bf16, device=device),
)
attrs = {
    "w_qr_quant_mode": prolog.WQrQuantMode.A8W8_MXFP8,
    "qkv_quant_mode": prolog.QkvQuantMode.A16C16,
}
prolog.precheck(**inputs, **attrs)

# 在图捕获前完成首次调用。
prepared_outputs = prolog(**inputs, **attrs)
torch.npu.synchronize()

graph = torch.npu.NPUGraph()
with torch.npu.graph(graph):
    q, descale_q, qr, descale_qr = prolog(**inputs, **attrs)
graph.replay()
torch.npu.synchronize()
print(q.shape, q.dtype)
```

### 图模式（NPUGraph-EX）

```python
import torch
import torch_npu
from cann_ops_transformer.ops import quant_mla_prolog as prolog

torch.npu.set_device(0)
device = "npu:0"
T, dim, q_lora, N, Dn, Dr = 3, 1024, 1024, 4, 128, 64
bf16 = torch.bfloat16

# 准备ND输入；E8M0 scale的编码127表示缩放值1。
inputs = dict(
    x=torch.full((T, dim), 0.125).to(torch.float8_e4m3fn).to(device),
    wqa=torch.full((q_lora, dim), 0.125).to(torch.float8_e4m3fn).to(device),
    wqb=torch.full((N * (Dn + Dr), q_lora), 0.125).to(torch.float8_e4m3fn).to(device),
    wkva=torch.full((512 + Dr, dim), 0.125).to(torch.float8_e4m3fn).to(device),
    wkb=torch.full((N, Dn, 512), 0.125, dtype=bf16, device=device),
    descale_x=torch.full((T, dim // 32), 127, dtype=torch.uint8).view(torch.float8_e8m0fnu).to(device),
    descale_wqa=torch.full((q_lora, dim // 64, 2), 127, dtype=torch.uint8).view(torch.float8_e8m0fnu).to(device),
    descale_wqb=torch.full((N * (Dn + Dr), q_lora // 64, 2), 127, dtype=torch.uint8).view(torch.float8_e8m0fnu).to(device),
    descale_wkva=torch.full((512 + Dr, dim // 64, 2), 127, dtype=torch.uint8).view(torch.float8_e8m0fnu).to(device),
    norm_weight_qa=torch.ones(q_lora, dtype=torch.float32, device=device),
    norm_weight_kva=torch.ones(512, dtype=torch.float32, device=device),
    kv_cache=torch.zeros((2, 128, 1, 512 + Dr), dtype=bf16, device=device),
    cache_index=torch.arange(T, dtype=torch.int64, device=device),
    rope_sin=torch.zeros((T, Dr), dtype=bf16, device=device),
    rope_cos=torch.ones((T, Dr), dtype=bf16, device=device),
)
attrs = {
    "w_qr_quant_mode": prolog.WQrQuantMode.A8W8_MXFP8,
    "qkv_quant_mode": prolog.QkvQuantMode.A16C16,
}
prolog.precheck(**inputs, **attrs)

# 在图捕获前完成首次调用。
prepared_outputs = prolog(**inputs, **attrs)
torch.npu.synchronize()

compiled = torch.compile(
    prolog,
    backend="npugraph_ex",
    fullgraph=True,
    dynamic=False,
    options={"force_eager": False, "clone_input": False},
)
compiled(**inputs, **attrs)  # 首次调用完成图捕获
q, descale_q, qr, descale_qr = compiled(**inputs, **attrs)  # 图回放
torch.npu.synchronize()
print(q.shape, q.dtype)
```

## 异常说明

Python接口及`precheck`校验的常见异常如下。

| 异常 | 原因 |
| --- | --- |
| TypeError | 模式参数未使用规定的枚举类型、Tensor类型或dtype不符合要求，或标量、布尔属性的类型错误。 |
| ValueError | Shape、设备、连续性、权重ND/NZ布局、模式组合或标量取值不满足接口约束。 |
| RuntimeError | 在图捕获期间调用`precheck`回读`quant_scale_kv`。 |
| NotImplementedError | 使用尚未实现的`WQrQuantMode.A4W4_HIFLOAT4`。 |
