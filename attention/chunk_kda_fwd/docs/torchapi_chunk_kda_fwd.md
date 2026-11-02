# chunk_kda_fwd

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

`cann_ops_transformer.ops.chunk_kda_fwd`基于`aclnnChunkKdaFwd`完成KDA分块正向计算，返回注意力输出、最终状态以及可选的反向计算中间量。计算公式、布局定义和完整参数约束见[ChunkKdaFwd](../README.md)。

## 函数原型

```python
from cann_ops_transformer.ops import chunk_kda_fwd

chunk_kda_fwd(
    q,
    k,
    v,
    g,
    beta,
    scale,
    *,
    initial_state=None,
    output_final_state=False,
    cu_seqlens=None,
    cu_seqlens_cpu=None,
    chunk_indices=None,
    safe_gate=False,
    lower_bound=None,
    use_gate_in_kernel=False,
    A_log=None,
    dt_bias=None,
    chunk_size=64,
    disable_recompute=False,
    return_intermediate_states=False,
    state_v_first=False,
    layout="BSND",
) -> Tuple[Optional[Tensor], ...]
```

## 参数说明

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度（shape） |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `q` | Tensor | 必选 | Query输入。 | bfloat16、float16 | BSND：[B, S, H, K]<br>BNSD：[B, H, S, K]<br>TND：[T, H, K]<br>NTD：[H, T, K] |
| `k` | Tensor | 必选 | Key输入，shape和数据类型必须与`q`相同。 | bfloat16、float16 | 同`q` |
| `v` | Tensor | 必选 | Value输入，数据类型必须与`q`相同。 | bfloat16、float16 | BSND：[B, S, HV, V]<br>BNSD：[B, HV, S, V]<br>TND：[T, HV, V]<br>NTD：[HV, T, V] |
| `g` | Tensor | 必选 | `use_gate_in_kernel=true`时为raw gate，否则为已激活的自然对数gate。 | float32、bfloat16 | BSND：[B, S, HV, K]<br>BNSD：[B, HV, S, K]<br>TND：[T, HV, K]<br>NTD：[HV, T, K] |
| `beta` | Tensor | 必选 | Delta更新系数。 | float32、bfloat16 | BSND：[B, S, HV]<br>BNSD：[B, HV, S]<br>TND：[T, HV]<br>NTD：[HV, T] |
| `scale` | float | 必选 | Query缩放系数，通常取`K ** -0.5`。 | - | - |
| `initial_state` | Tensor | 可选 | 每条逻辑序列的初始状态。 | float32 | `state_v_first=false`：[N, HV, K, V]<br>`state_v_first=true`：[N, HV, V, K] |
| `output_final_state` | bool | 可选 | 是否返回最终状态，默认`False`。 | - | - |
| `cu_seqlens` | Sequence[int]、Tensor | 可选 | 变长序列累计长度，首元素为0，末元素为T或S，元素单调不减。 | int64 | [N+1] |
| `cu_seqlens_cpu` | Sequence[int] | 可选 | `cu_seqlens`的host侧表示；传入时覆盖device侧数据。 | - | [N+1] |
| `chunk_indices` | Sequence[int] | 可选 | 按`(seq_id, chunk_id)`展平的chunk索引，必须采用sequence-major canonical顺序。传入时必须同时传入`cu_seqlens`。 | int64 | [2×NC] |
| `safe_gate` | bool | 可选 | 是否使用有界gate，默认`False`。 | - | - |
| `lower_bound` | float | 可选 | 有界gate下界，默认`-5.0`。 | - | - |
| `use_gate_in_kernel` | bool | 可选 | 是否在kernel内由raw gate计算激活后的gate，默认`False`。 | - | - |
| `A_log` | Tensor | 可选 | gate衰减参数，`use_gate_in_kernel=true`时必选。 | float32 | [HV] |
| `dt_bias` | Tensor | 可选 | gate偏置。 | float32 | [HV×K] |
| `chunk_size` | int | 可选 | chunk大小，支持64和128，默认`64`。 | - | - |
| `disable_recompute` | bool | 可选 | 是否关闭反向重计算并导出全部反向中间量，默认`False`。 | - | - |
| `return_intermediate_states` | bool | 可选 | 是否返回每个chunk的输入状态，默认`False`。 | - | - |
| `state_v_first` | bool | 可选 | 状态最后两维是否使用`(V, K)`顺序，默认`False`。 | - | - |
| `layout` | str | 可选 | 输入布局，支持`BSND`、`BNSD`、`TND`、`NTD`，默认`BSND`。 | - | - |

## 返回值说明

接口固定返回12项。

| 序号 | 返回项 | 返回条件 | 数据类型 | 维度（shape） |
| :---: | :--- | :--- | :--- | :--- |
| 0 | `attn_out` | 始终返回 | 与`q`相同 | BSND：[B, S, HV, V]<br>TND：[T, HV, V] |
| 1 | `final_state` | `output_final_state=true` | float32 | `state_v_first=false`：[N, HV, K, V]<br>`state_v_first=true`：[N, HV, V, K] |
| 2 | `gk` | `use_gate_in_kernel=false`或`disable_recompute=true` | float32 | BSND：[B, HV, S, K]<br>TND：[HV, T, K] |
| 3 | `Aqk` | 始终返回 | 与`q`相同 | BSND：[B, HV, S, C]<br>TND：[HV, T, C] |
| 4 | `Akk` | 始终返回 | 与`q`相同 | BSND：[B, HV, S, C]<br>TND：[HV, T, C] |
| 5 | `w` | `disable_recompute=true` | 与`q`相同 | BSND：[B, HV, S, K]<br>TND：[HV, T, K] |
| 6 | `u` | `disable_recompute=true` | 与`q`相同 | BSND：[B, HV, S, V]<br>TND：[HV, T, V] |
| 7 | `qg` | `disable_recompute=true` | 与`q`相同 | BSND：[B, HV, S, K]<br>TND：[HV, T, K] |
| 8 | `kg` | `disable_recompute=true` | 与`q`相同 | BSND：[B, HV, S, K]<br>TND：[HV, T, K] |
| 9 | `v_new` | `disable_recompute=true` | 与`q`相同 | BSND：[B, HV, S, V]<br>TND：[HV, T, V] |
| 10 | `h` | `disable_recompute=true`或`return_intermediate_states=true` | 与`q`相同 | BSND：[B, NC, HV, K, V]<br>TND：[NC, HV, K, V]<br>`state_v_first=true`时交换末两维 |
| 11 | `initial_state` | 始终返回输入对象，未传入时为`None` | 与输入相同 | 与输入相同 |

不满足返回条件时，对应项为`None`。可选输出仅影响返回值，不改变算法语义。

## 约束说明

- `H`和`HV`必须为正整数，满足`HV >= H`、`HV <= 128`且`HV % H == 0`。
- `K`和`V`必须在[16, 256]范围内，且均为16的倍数。
- `q`、`k`、`v`的数据类型必须相同。
- `use_gate_in_kernel=true`时`A_log`必须传入，`dt_bias`可选。
- `safe_gate=true`且`use_gate_in_kernel=true`时，`lower_bound`必须在[-5, 0)范围内。
- rank-4变长输入要求`B=1`，变长输入最多支持1024条逻辑序列。
- `layout`只描述输入布局；`attn_out`固定为sequence-major，反向中间量固定为head-major。
- 默认支持确定性计算。

## 调用说明

```python
import math

import torch
import torch_npu

from cann_ops_transformer.ops import chunk_kda_fwd

torch_npu.npu.set_device(0)

B, S, H, HV, K, V = 1, 64, 2, 2, 128, 128
q = torch.randn(B, S, H, K, device="npu", dtype=torch.bfloat16)
k = torch.randn(B, S, H, K, device="npu", dtype=torch.bfloat16)
v = torch.randn(B, S, HV, V, device="npu", dtype=torch.bfloat16)
g = torch.randn(B, S, HV, K, device="npu", dtype=torch.float32)
beta = torch.rand(B, S, HV, device="npu", dtype=torch.float32)
a_log = torch.zeros(HV, device="npu", dtype=torch.float32)

outputs = chunk_kda_fwd(
    q,
    k,
    v,
    g,
    beta,
    1.0 / math.sqrt(K),
    output_final_state=True,
    safe_gate=True,
    lower_bound=-5.0,
    use_gate_in_kernel=True,
    A_log=a_log,
    chunk_size=64,
    layout="BSND",
)

attn_out, final_state, _, aqk, akk, *_ = outputs
```
