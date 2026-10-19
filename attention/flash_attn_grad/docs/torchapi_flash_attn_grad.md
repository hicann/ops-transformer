# flash_attn_grad

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

- **接口功能**：

  `flash_attn_grad`是基于`TorchNPU`的`cann_ops_transformer`扩展接口，用于计算Flash Attention的反向结果。接口根据前向计算的`softmax_lse`、`attn_out`和上游梯度`dout`，计算Query、Key、Value对应的梯度`dq`、`dk`、`dv`；当传入`sinks`时，同时计算sink参数的梯度`dsink`。

  调用本接口前，需要调用`flash_attn_metadata`生成反向计算所需的`metadata`。生成时必须设置`is_grad_enabled=True`，并保证两个接口的shape、layout、mask和序列长度相关参数一致。

  本接口显式返回梯度，不会自动为前向调用建立PyTorch自动求导关系；`is_grad_enabled=True`也不等同于启用`.backward()`。详见「与PyTorch自动求导的关系」。

- **计算公式**：

  前向计算为：

  $$
  S=scale\cdot QK^T
  $$

  以下公式仅对`mask_mode`、窗口及有效序列长度允许参与注意力计算的位置求值，被屏蔽位置的$P_{ij}$为0。

  $$
  P_{ij}=\exp(S_{ij}-softmax\_lse_i)
  $$

  $$
  Y=PV
  $$

  反向计算为：

  $$
  dV=P^TdY
  $$

  $$
  dP=dYV^T
  $$

  $$
  dS=P\odot\left(dP-rowsum(dY\odot Y)\right)
  $$

  $$
  dQ=scale\cdot(dS\cdot K)
  $$

  $$
  dK=scale\cdot(dS^T\cdot Q)
  $$

  当传入`sinks`时，前向softmax的分母额外包含sink项（`softmax_lse`已包含该贡献），sink的梯度为：

  $$
  dSink=-\sum_{b,s1,s2}P\cdot dP\cdot\exp(sink-softmax\_lse)
  $$

  其中，$Q$、$K$、$V$分别对应输入`q`、`k`、`v`，$Y$对应`attn_out`，$dY$对应`dout`。`softmax_scale`非0时，$scale$取`softmax_scale`；为0时，$scale=1/\sqrt{D}$。

> [!NOTE]
>
> B表示batch大小，S1表示Query序列长度，S2表示Key/Value序列长度，N1表示Query head数，N2表示Key/Value head数，D表示Query/Key的head dim，Dv表示Value/输出的head dim。T1表示所有batch中Query序列长度的累加和，T2表示所有batch中Key/Value序列长度的累加和。GQA场景满足N1是N2的整数倍。

## 函数原型

```python
cann_ops_transformer.flash_attn_grad(
    q,
    k,
    v,
    dout,
    attn_out,
    softmax_lse,
    cu_seqlens_q=None,
    cu_seqlens_kv=None,
    seqused_q=None,
    seqused_kv=None,
    sinks=None,
    attn_mask=None,
    metadata=None,
    softmax_scale=0.0,
    mask_mode=0,
    win_left=-1,
    win_right=-1,
    max_seqlen_q=-1,
    max_seqlen_kv=-1,
    layout_q="BSND",
    layout_kv="BSND",
    layout_out="BSND"
) -> (Tensor, Tensor, Tensor, Tensor)
```

`cu_seqlens_q`及之后的参数既可以按位置传入，也可以按关键字传入。建议使用关键字传入可选参数。

## 掩码模式

`mask_mode`为整数，支持以下取值。不支持传入`"CAUSAL"`等字符串；本接口不提供`MaskMode`枚举。

| 值 | 含义 | `win_left` / `win_right` | `attn_mask`要求 |
| :---: | :--- | :--- | :--- |
| 0 | Dense，全计算模式（默认值） | 均为-1 | 必须为None |
| 3 | Causal，右下对齐的因果掩码 | 均为-1 | 必须提供下述压缩掩码模板 |
| 4 | Sliding Window，右下对齐的滑动窗口 | 每侧为-1或非负整数；-1表示该侧不限 | 任一侧为有限窗口时必须提供下述模板；两侧均为-1时等价于Dense，可不传，传入时不参与掩码计算 |

### 压缩掩码输入

需要`attn_mask`时，应传入与其他输入位于同一NPU设备的int8、shape为(2048, 2048)的严格上三角模板，其中1表示屏蔽，0表示保留：

```python
attn_mask = torch.triu(
    torch.ones((2048, 2048), dtype=torch.int8, device=q.device),
    diagonal=1,
)
```

该模板适用于`mask_mode=3`和需要掩码的`mask_mode=4`，不是按实际S1/S2构造的任意业务掩码；窗口范围由`win_left`和`win_right`指定。

### 有效位置与对齐规则

设某个batch的有效Q/KV长度为`Lq`/`Lkv`，有效序列内的位置索引为`i`/`j`，均从0开始。未提供`seqused`的一侧使用该侧完整序列长度；TND的完整长度为相邻`cu_seqlens`元素之差。

- `mask_mode=0`：保留所有有效Q/KV配对位置。
- `mask_mode=3`：保留`j <= i + Lkv - Lq`的位置。
- `mask_mode=4`：以`i + Lkv - Lq`为窗口中心；有限左侧要求`j >= i + Lkv - Lq - win_left`，有限右侧要求`j <= i + Lkv - Lq + win_right`，边界均包含在内。值为-1的一侧不施加窗口限制。

以上条件均与`0 <= i < Lq`、`0 <= j < Lkv`取交集。使用`seqused`时，掩码按有效长度对齐，不按包含未使用尾部的物理长度对齐。

## 参数说明

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 数据格式 | 维度(shape) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| q | Tensor | 必选 | 公式中的Q | bfloat16/float16 | ND | <ul><li>BSND：(B, S1, N1, D)</li><li>BNSD：(B, N1, S1, D)</li><li>TND：(T1, N1, D)</li></ul> |
| k | Tensor | 必选 | 公式中的K | bfloat16/float16 | ND | <ul><li>BSND：(B, S2, N2, D)</li><li>BNSD：(B, N2, S2, D)</li><li>TND：(T2, N2, D)</li></ul> |
| v | Tensor | 必选 | 公式中的V | bfloat16/float16 | ND | <ul><li>BSND：(B, S2, N2, Dv)</li><li>BNSD：(B, N2, S2, Dv)</li><li>TND：(T2, N2, Dv)</li></ul> |
| dout | Tensor | 必选 | 公式中的dY，前向输出的上游梯度 | bfloat16/float16 | ND | <ul><li>BSND：(B, S1, N1, Dv)</li><li>BNSD：(B, N1, S1, Dv)</li><li>TND：(T1, N1, Dv)</li></ul> |
| attn_out | Tensor | 必选 | 公式中的Y，即前向接口返回的注意力输出 | bfloat16/float16 | ND | shape与`dout`相同 |
| softmax_lse | Tensor | 必选 | 前向接口在`return_softmax_lse=True`时返回的log-sum-exp结果 | float32 | ND | <ul><li>BSND/BNSD：(B, N1, S1)</li><li>TND：(N1, T1)</li></ul> |
| cu_seqlens_q | Tensor | 可选 | TND布局下Q的非递减累积序列长度，第一个元素必须为0，最后一个元素等于T1；相邻元素相等表示空序列。`layout_q`为TND时必须传入，非TND时不支持传入。默认值为None | int32 | ND | (B+1,) |
| cu_seqlens_kv | Tensor | 可选 | TND布局下KV的非递减累积序列长度，第一个元素必须为0，最后一个元素等于T2；相邻元素相等表示空序列。`layout_kv`为TND时必须传入，非TND时不支持传入。默认值为None | int32 | ND | (B+1,) |
| seqused_q | Tensor | 可选 | 各布局每个batch实际使用的Q前缀长度；不改变物理存储跨度。默认值为None | int32 | ND | (B,) |
| seqused_kv | Tensor | 可选 | 各布局每个batch实际使用的KV前缀长度；不改变物理存储跨度。默认值为None | int32 | ND | (B,) |
| sinks | Tensor | 可选 | Sink参数，用于改善自注意力计算的数值稳定性，shape为(N1,)。默认值为None，此时`dsink`内容未定义（不应读取） | float32 | ND | (N1,) |
| attn_mask | Tensor | 可选 | 压缩掩码模板，默认值为None。是否必传及模板内容见「掩码模式」 | int8 | ND | (2048, 2048) |
| metadata | Tensor | 实际必选 | 必须传入`flash_attn_metadata(..., is_grad_enabled=True)`的返回值；不能省略或传None | int32 | ND | 一维Tensor，长度由配套接口确定 |
| softmax_scale | float | 可选 | Softmax缩放系数。默认值为0.0，表示使用$1/\sqrt{D}$ | float32 | - | - |
| mask_mode | int | 可选 | 掩码模式，支持0、3、4，具体规则见「掩码模式」。默认值为0 | int32 | - | - |
| win_left | int | 可选 | Window mask左窗口，值需大于等于-1，-1表示正无穷。默认值为-1 | int32 | - | - |
| win_right | int | 可选 | Window mask右窗口，值需大于等于-1，-1表示正无穷。默认值为-1 | int32 | - | - |
| max_seqlen_q | int | 可选 | 指定查询q序列的长度上限，必须大于等于-1；传入时必须等于实际的最大序列长度，否则行为未定义。须与`flash_attn_metadata`（`is_grad_enabled=True`）调用中的`max_seqlen_q`保持一致 | int32 | - | - |
| max_seqlen_kv | int | 可选 | 指定键k和值v序列的长度上限，必须大于等于-1；传入时必须等于实际的最大序列长度，否则行为未定义。须与`flash_attn_metadata`（`is_grad_enabled=True`）调用中的`max_seqlen_kv`保持一致 | int32 | - | - |
| layout_q | string | 可选 | q的布局。当前支持BSND、BNSD、TND。默认值为"BSND" | string | - | - |
| layout_kv | string | 可选 | k和v的布局，必须与`layout_q`相同。默认值为"BSND" | string | - | - |
| layout_out | string | 可选 | `dout`和`attn_out`的布局，必须与`layout_q`相同。默认值为"BSND" | string | - | - |

> [!NOTE]
>
> `q`、`k`、`v`、`dout`、`attn_out`及输出`dq`、`dk`、`dv`支持float16和bfloat16，数据类型必须一致。

## 返回值说明

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 数据格式 | 维度(shape) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| dq | Tensor | 必选 | 公式中的dQ，Query的梯度 | bfloat16/float16 | ND | shape与`q`相同 |
| dk | Tensor | 必选 | 公式中的dK，Key的梯度 | bfloat16/float16 | ND | shape与`k`相同 |
| dv | Tensor | 必选 | 公式中的dV，Value的梯度 | bfloat16/float16 | ND | shape与`v`相同 |
| dsink | Tensor | 必选 | 公式中的dSink，sink参数的梯度；仅在传入`sinks`时有定义 | float32 | ND | (N1,)；`sinks`为None时内容未定义 |

## 约束说明

- 公共约束：
  - 接口支持以下组合：
    - 数据类型为float16或bfloat16。
    - `layout_q`、`layout_kv`、`layout_out`支持BSND、BNSD、TND，且必须相同。
    - Dense、Causal及Sliding Window Attention，即`mask_mode=0/3/4`；窗口和`attn_mask`必须满足「掩码模式」中的对应条件。
    - TND布局下`cu_seqlens_q`、`cu_seqlens_kv`必须传入；非TND布局下不支持传入。
- `q`、`k`、`v`、`dout`、`attn_out`的数据类型必须一致。
- B、N1、N2、D和Dv必须为正数，其中B的取值范围为(0, 65536)。非TND的S1/S2必须为正；TND允许空batch及某侧物理token总数为0。
- N1必须能被N2整除，支持MHA（N1=N2）和GQA（N1>N2）。
- Query和Key的head dim均为D；Value、`dout`和`attn_out`的head dim均为Dv。各布局均支持`1 <= D <= 768`，且要求`Dv == D`或特殊组合`D=192, Dv=128`。
- 与当前仓库中的`flash_attn`联合调用时，前向接口支持的(D, Dv)为(64,64)、(72,72)、(128,128)、(256,256)、(192,128)；本接口覆盖上述组合。
- `softmax_lse`必须为float32，shape为(B, N1, S1)（BSND/BNSD场景）或(N1, T1)（TND场景）。
- 所有输入的数据格式均为ND，本接口支持非连续Tensor输入。与`flash_attn`联合调用时，还需满足前向接口的输入约束；通过`expand()`等方式得到的零stride K/V输入，请先调用`.contiguous()`再传入前向接口。
- BSND、BNSD和TND均支持可选的`seqused_q`和`seqused_kv`（可以只传其中一侧），类型为int32、shape为(B,)。BSND/BNSD每侧满足`0 <= seqused[b] <= S`，未传的一侧使用完整物理S；TND每侧满足`0 <= seqused[b] <= cu_seqlens[b+1] - cu_seqlens[b] <= max_seqlen`，未传的一侧使用cu差分。
- 生成FAG metadata时须明确传入正数`batch_size`。`max_seqlen_q/max_seqlen_kv`按布局区分：TND布局下可省略（默认-1），由cu_seqlens末位（物理packed总量）推导；BSND/BNSD布局下，每侧满足"提供该侧seqused"或"传入非负`max_seqlen`"至少其一——未提供seqused的侧必须传入，且该值必须等于输入shape中的物理S1/S2，不能只取seqused的最大值。反向接口的非TND最大长度可保持-1由shape推导；如显式指定正数，应与物理S一致。
- `seqused`只控制有效计算范围和mask几何：BSND/BNSD仍使用完整S的存储跨度，TND仍使用cu描述的物理偏移。输出shape和LSE的物理跨度不变。未使用的dq/dk/dv尾部输出为零；所有batch均无有效Q/KV配对任务时仍清零输出，提供sinks时dsink也为零。
- `metadata`必须由`flash_attn_metadata`生成，并设置`is_grad_enabled=True`。生成metadata和调用本接口时，N1、N2、D、Dv、B、S1、S2、layout、cu_seqlens、seqused及mask相关参数必须一致，否则行为未定义。特别是`D=192, Dv=128`时，生成metadata必须显式传入`head_dim_v=128`。
- 在配置一致的条件下，前向`flash_attn`和反向`flash_attn_grad`可使用同一份`is_grad_enabled=True`生成的metadata，无需分别生成。直接使用接口返回值，不要自行构造或修改metadata。
- `softmax_lse`、`attn_out`必须来自与本次反向计算配置一致的前向调用，尤其是`softmax_scale`和layout必须一致。
- `sinks`为float32、shape为(N1,)，N1为Query head数；提供时必须与前向`flash_attn`所用sinks一致。
- `dsink`为float32、shape为(N1,)，仅在传入`sinks`时有定义；未传入时不得读取其数值。重复调用的可复现性设置见「确定性计算」。
- 当前仅支持单算子模式。

## 配套接口说明

调用`flash_attn_grad`之前，需要通过`flash_attn_metadata`生成配套metadata。以下参数要求针对`is_grad_enabled=True`的前后向联合使用场景。

```python
cann_ops_transformer.flash_attn_metadata(
    num_heads_q,
    num_heads_kv,
    head_dim,
    *,
    head_dim_v=None,
    cu_seqlens_q=None,
    cu_seqlens_kv=None,
    seqused_q=None,
    seqused_kv=None,
    batch_size=None,
    max_seqlen_q=None,
    max_seqlen_kv=None,
    mask_mode=None,
    win_left=None,
    win_right=None,
    layout_q=None,
    layout_kv=None,
    layout_out=None,
    is_grad_enabled=False
) -> Tensor
```


| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 数据格式 | 维度(shape) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| num_heads_q | int | 必选 | Query head数，即N1 | int32 | - | - |
| num_heads_kv | int | 必选 | Key/Value head数，即N2 | int32 | - | - |
| head_dim | int | 必选 | Query/Key的head dim，即D | int32 | - | - |
| head_dim_v | int | 可选 | Value的head dim，即Dv。默认值None表示与D相同；D与Dv不同时必须显式传入Dv，当前仅支持特殊组合D=192、Dv=128 | int32 | - | - |
| cu_seqlens_q | Tensor | 可选 | TND布局下Q的累积序列长度。`layout_q`为TND时必须传入，非TND时不支持传入。默认值为None | int32 | ND | (B+1,) |
| cu_seqlens_kv | Tensor | 可选 | TND布局下KV的累积序列长度。`layout_kv`为TND时必须传入，非TND时不支持传入。默认值为None | int32 | ND | (B+1,) |
| seqused_q | Tensor | 可选 | 各布局每个batch实际使用的Q前缀长度；不改变物理存储跨度。默认值为None | int32 | ND | (B,) |
| seqused_kv | Tensor | 可选 | 各布局每个batch实际使用的KV前缀长度；不改变物理存储跨度。默认值为None | int32 | ND | (B,) |
| batch_size | int | FAG场景必选 | 所有布局均须显式传入实际B，取值范围为(0, 65536)；TND的两侧cu_seqlens长度均须为B+1 | int32 | - | - |
| max_seqlen_q | int | 可选 | 指定查询q序列的长度上限 | int32 | - | - |
| max_seqlen_kv | int | 可选 | 指定键k和值v序列的长度上限 | int32 | - | - |
| mask_mode | int | FAG场景必选 | 显式传入0、3或4，必须与`flash_attn_grad`一致；不要依赖原型中的None默认值 | int32 | - | - |
| win_left | int | 可选 | 必须与`flash_attn_grad`一致 | int32 | - | - |
| win_right | int | 可选 | 必须与`flash_attn_grad`一致 | int32 | - | - |
| layout_q | string | 可选 | 必须与`flash_attn_grad`一致 | string | - | - |
| layout_kv | string | 可选 | 必须与`flash_attn_grad`一致 | string | - | - |
| layout_out | string | 可选 | 必须与`flash_attn_grad`一致 | string | - | - |
| is_grad_enabled | bool | 可选 | 是否生成反向算子所需的metadata。调用`flash_attn_grad`前必须设置为True。默认值为False | bool | - | - |

返回的`metadata`为int32、ND格式的一维Tensor。其长度由接口确定，调用方应直接传递返回的Tensor。

## 确定性计算

BSND、BNSD和TND布局均支持确定性计算，包括单侧或双侧传入`seqused`的场景；仍须满足前述dtype、shape和mask约束。

在支持该API的TorchNPU版本中，于调用前设置：

```python
import torch_npu

torch_npu.npu.set_deterministic_level(1)
```

- 该设置影响当前进程后续调用，不是`flash_attn_grad`的参数；`is_grad_enabled=True`不能代替此设置。
- 确定性指同一软硬件环境下，输入Tensor数值、参数及相关配置完全一致时，重复调用的有效输出逐位一致。比较范围为`dq`、`dk`、`dv`，以及提供`sinks`时的`dsink`；不包含未定义输出。
- 不保证确定性模式与非确定性模式之间，或不同设备型号、软件版本之间的结果逐位一致。设置随机种子与开启算子确定性是不同事项；比较重复调用时也需保持输入不变。
- 使用`set_deterministic_level`后，不要混用`torch.use_deterministic_algorithms`切换状态。如后续不需要确定性，可调用`torch_npu.npu.set_deterministic_level(0)`关闭。

## 与PyTorch自动求导的关系

`flash_attn_grad`是显式反向接口：调用者提供`dout`、前向输出和LSE，接口返回梯度Tensor。它不会自动把返回值写入`q.grad`、`k.grad`或`v.grad`。

`flash_attn_metadata(..., is_grad_enabled=True)`仅使metadata可用于反向接口，不会自动为`flash_attn`建立PyTorch自动求导关系。仅设置`q.requires_grad=True`或`is_grad_enabled=True`，不能将下述手动反向示例变为可直接调用`.backward()`的示例。

若需通过`torch.autograd.grad()`或`.backward()`使用这组接口，调用方需要提供`torch.autograd.Function`封装：前向调用`flash_attn`并保存`q/k/v/attn_out/softmax_lse`、metadata及相关参数；反向接收上游梯度并调用`flash_attn_grad`，将返回梯度对应到前向Tensor输入。提供`sinks`时，还需正确传递其`dsink`。本文示例仅演示显式反向调用，不承诺高阶梯度支持。

## 调用示例

- `flash_attn_metadata`、`flash_attn`和`flash_attn_grad`联合调用示例（BSND）

  前向和反向配置一致时，可以共用`is_grad_enabled=True`生成的metadata。下面显式调用反向接口，不使用PyTorch自动求导。

    ```python
    import math
    import torch
    import torch_npu
    import cann_ops_transformer

    torch_npu.npu.set_device(0)

    dtype = torch.float16
    B = 2
    S1 = 128
    S2 = 128
    N1 = 8
    N2 = 2
    D = 128
    Dv = 128
    scale = 1.0 / math.sqrt(D)

    q = torch.randn(B, S1, N1, D, dtype=dtype, device="npu")
    k = torch.randn(B, S2, N2, D, dtype=dtype, device="npu")
    v = torch.randn(B, S2, N2, Dv, dtype=dtype, device="npu")

    metadata = cann_ops_transformer.flash_attn_metadata(
        N1,
        N2,
        D,
        head_dim_v=Dv,
        batch_size=B,
        max_seqlen_q=S1,
        max_seqlen_kv=S2,
        mask_mode=0,
        win_left=-1,
        win_right=-1,
        layout_q="BSND",
        layout_kv="BSND",
        layout_out="BSND",
        is_grad_enabled=True,
    )

    attn_out, softmax_lse = cann_ops_transformer.flash_attn(
        q,
        k,
        v,
        metadata=metadata,
        softmax_scale=scale,
        mask_mode=0,
        win_left=-1,
        win_right=-1,
        max_seqlen_q=S1,
        max_seqlen_kv=S2,
        layout_q="BSND",
        layout_kv="BSND",
        layout_out="BSND",
        return_softmax_lse=True,
    )
    dout = torch.randn_like(attn_out)

    dq, dk, dv, dsink = cann_ops_transformer.flash_attn_grad(
        q,
        k,
        v,
        dout,
        attn_out,
        softmax_lse,
        metadata=metadata,
        softmax_scale=scale,
        mask_mode=0,
        win_left=-1,
        win_right=-1,
        max_seqlen_q=S1,
        max_seqlen_kv=S2,
        layout_q="BSND",
        layout_kv="BSND",
        layout_out="BSND",
    )
    torch_npu.npu.synchronize()

    assert dq.shape == q.shape
    assert dk.shape == k.shape
    assert dv.shape == v.shape
    # 未传 sinks 时 dsink 内容未定义，仅在传入 sinks 时才有意义
    assert dsink.shape == (N1,)
    assert dsink.dtype == torch.float32
    ```
