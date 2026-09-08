# flash\_attn\_grad

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

  `flash_attn_grad`是基于`TorchNPU`的`cann_ops_transformer`扩展接口，用于计算Flash Attention的反向结果。接口根据前向计算的`softmax_lse`、`attn_out`和上游梯度`dout`，计算Query、Key、Value对应的梯度`dq`、`dk`、`dv`。

  调用本接口前，需要调用`flash_attn_metadata`生成反向计算所需的`metadata`。生成时必须设置`is_grad_enabled=True`，并保证两个接口的shape、layout、mask和序列长度相关参数一致。

- **计算公式**：

  前向计算为：

  $$
  S=scale\cdot QK^T
  $$

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
  sfmg=rowsum(dY\odot Y)
  $$

  $$
  dS=P\odot(dP-sfmg)
  $$

  $$
  dQ=scale\cdot(dS\cdot K)
  $$

  $$
  dK=scale\cdot(dS^T\cdot Q)
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
) -> (Tensor, Tensor, Tensor)
```

当前注册的PyTorch schema未使用`*`分隔符，因此`cu_seqlens_q`及之后的参数既可以按位置传入，也可以按关键字传入。建议使用关键字传入可选参数。

## 枚举说明

`mask_mode` 在 Python 接口中支持传入 `IntEnum` 枚举或对应 int 值，枚举定义于 `cann_ops_transformer.ops.attention.flash_attn_grad`：

### mask_mode 枚举

| 枚举名 | 值 | 含义 |
| :--- | :---: | :--- |
| `NO_MASK` | 0 | 全计算模式（默认值） |
| `CAUSAL` | 3 | Causal 模式 |
| `SLIDING_WINDOW` | 4 | Sliding Window 模式 |

> [!NOTE]
>
> 枚举为 `IntEnum`，可直接作为 int 传入底层算子；接口同时兼容传入枚举名对应的字符串（不区分大小写）与 int 值。

## 参数说明

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 数据格式 | 维度 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| q | Tensor | 必选 | 公式中的Q | bfloat16/float16 | ND | <ul><li>BSND：(B, S1, N1, D)</li><li>BNSD：(B, N1, S1, D)</li><li>TND：(T1, N1, D)</li></ul> |
| k | Tensor | 必选 | 公式中的K | bfloat16/float16 | ND | <ul><li>BSND：(B, S2, N2, D)</li><li>BNSD：(B, N2, S2, D)</li><li>TND：(T2, N2, D)</li></ul> |
| v | Tensor | 必选 | 公式中的V | bfloat16/float16 | ND | <ul><li>BSND：(B, S2, N2, Dv)</li><li>BNSD：(B, N2, S2, Dv)</li><li>TND：(T2, N2, Dv)</li></ul> |
| dout | Tensor | 必选 | 公式中的dY，前向输出的上游梯度 | bfloat16/float16 | ND | <ul><li>BSND：(B, S1, N1, Dv)</li><li>BNSD：(B, N1, S1, Dv)</li><li>TND：(T1, N1, Dv)</li></ul> |
| attn_out | Tensor | 必选 | 公式中的Y，即前向接口返回的注意力输出 | bfloat16/float16 | ND | shape与`dout`相同 |
| softmax_lse | Tensor | 必选 | 前向接口在`return_softmax_lse=True`时返回的log-sum-exp结果 | float32 | ND | <ul><li>BSND/BNSD：(B, N1, S1)</li><li>TND：(N1, T1)</li></ul> |
| cu_seqlens_q | Tensor | 可选 | TND布局下Q的累积序列长度，第一个元素必须为0，最后一个元素等于T1。`layout_q`为TND时必须传入，非TND时不支持传入。默认值为None | int32 | ND | (B+1,) |
| cu_seqlens_kv | Tensor | 可选 | TND布局下KV的累积序列长度，第一个元素必须为0，最后一个元素等于T2。`layout_kv`为TND时必须传入，非TND时不支持传入。默认值为None | int32 | ND | (B+1,) |
| seqused_q | Tensor | 可选 | 每个batch实际使用的Q序列长度。默认值为None | int32 | ND | (B,) |
| seqused_kv | Tensor | 可选 | 每个batch实际使用的KV序列长度。默认值为None | int32 | ND | (B,) |
| sinks | Tensor | 可选 | Sink参数，用于改善自注意力计算的数值稳定性。默认值为None | float32 | ND | (N1,) |
| attn_mask | Tensor | 可选 | 掩码矩阵。默认值为None | int8 | ND | (2048, 2048) |
| metadata | Tensor | 可选 | `flash_attn_metadata`生成的FAG任务切分数据。schema中为可选参数，但实际调用时必须传入 | int32 | ND | shape根据batch大小和N2动态计算 |
| softmax_scale | float | 可选 | Softmax缩放系数。默认值为0.0，表示使用$1/\sqrt{D}$ | float32 | - | - |
| mask_mode | int/MaskMode | 可选 | 掩码模式，支持传入枚举或对应 int 值，枚举定义见「mask_mode 枚举」。默认值为0 | int32 | - | - |
| win_left | int | 可选 | Window mask左窗口，值需大于等于-1，-1表示正无穷。默认值为-1 | int32 | - | - |
| win_right | int | 可选 | Window mask右窗口，值需大于等于-1，-1表示正无穷。默认值为-1 | int32 | - | - |
| max_seqlen_q | int | 可选 | Q最大序列长度，必须大于等于-1。BSND/BNSD场景可保持默认值-1，由输入shape推导S1 | int32 | - | - |
| max_seqlen_kv | int | 可选 | KV最大序列长度，必须大于等于-1。BSND/BNSD场景可保持默认值-1，由输入shape推导S2 | int32 | - | - |
| layout_q | string | 可选 | q的布局。当前支持BSND、BNSD、TND。默认值为"BSND" | string | - | - |
| layout_kv | string | 可选 | k和v的布局，必须与`layout_q`相同。默认值为"BSND" | string | - | - |
| layout_out | string | 可选 | `dout`和`attn_out`的布局，必须与`layout_q`相同。默认值为"BSND" | string | - | - |

> [!NOTE]
>
> `q`、`k`、`v`、`dout`、`attn_out`及输出`dq`、`dk`、`dv`支持float16和bfloat16，数据类型必须一致。

## 返回值说明

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 数据格式 | 维度 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| dq | Tensor | 必选 | 公式中的dQ，Query的梯度 | bfloat16/float16 | ND | shape与`q`相同 |
| dk | Tensor | 必选 | 公式中的dK，Key的梯度 | bfloat16/float16 | ND | shape与`k`相同 |
| dv | Tensor | 必选 | 公式中的dV，Value的梯度 | bfloat16/float16 | ND | shape与`v`相同 |

## 约束说明

- 接口支持以下组合：
  - 数据类型为float16或bfloat16。
  - `layout_q`、`layout_kv`、`layout_out`支持BSND、BNSD、TND，且必须相同。
  - Dense Attention，`mask_mode=0`、`attn_mask=None`、`win_left=-1`、`win_right=-1`。
  - TND布局下`cu_seqlens_q`、`cu_seqlens_kv`必须传入；非TND布局下不支持传入。
- `q`、`k`、`v`、`dout`、`attn_out`的数据类型必须一致。
- B、S1、S2、N1、N2、D和Dv必须为正数，其中B的取值范围为(0, 65536)。
- N1必须能被N2整除，支持MHA（N1=N2）和GQA（N1>N2）。
- Query和Key的head dim均为D；Value、`dout`和`attn_out`的head dim均为Dv，并满足`0 < Dv <= D <= 192`。
- 与当前仓库中的`flash_attn`联合调用时，前向接口还要求D=Dv且D取64、128或256；结合本接口D不超过192的约束，联合调用当前支持D=Dv=64或128。
- `softmax_lse`必须为float32，shape为(B, N1, S1)（BSND/BNSD场景）或(N1, T1)（TND场景）。
- 所有输入的数据格式均为ND。算子注册了`AutoContiguous`，传入非连续Tensor时由框架转换为连续Tensor。
- `metadata`必须由`flash_attn_metadata`生成，并设置`is_grad_enabled=True`。生成metadata和调用本接口时，N1、N2、D、B、S1、S2、layout及mask相关参数必须一致，否则行为未定义。
- `is_grad_enabled=True`生成的metadata同时包含正向和反向任务切分数据，前向`flash_attn`和反向`flash_attn_grad`均可使用同一份metadata，无需分别生成。
- `softmax_lse`、`attn_out`必须来自与本次反向计算配置一致的前向调用，尤其是`softmax_scale`和layout必须一致。
- 当前仅支持单算子模式。

## 配套接口说明

调用`flash_attn_grad`之前，需要通过`flash_attn_metadata`生成反向任务切分数据。

```python
cann_ops_transformer.flash_attn_metadata(
    num_heads_q,
    num_heads_kv,
    head_dim,
    *,
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

与反向接口直接相关的参数如下：

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 数据格式 | 维度 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| num_heads_q | int | 必选 | Query head数，即N1 | int32 | - | - |
| num_heads_kv | int | 必选 | Key/Value head数，即N2 | int32 | - | - |
| head_dim | int | 必选 | Query/Key的head dim，即D | int32 | - | - |
| cu_seqlens_q | Tensor | 可选 | TND布局下Q的累积序列长度。`layout_q`为TND时必须传入，非TND时不支持传入。默认值为None | int32 | ND | (B+1,) |
| cu_seqlens_kv | Tensor | 可选 | TND布局下KV的累积序列长度。`layout_kv`为TND时必须传入，非TND时不支持传入。默认值为None | int32 | ND | (B+1,) |
| seqused_q | Tensor | 可选 | 每个batch实际使用的Q序列长度。默认值为None | int32 | ND | (B,) |
| seqused_kv | Tensor | 可选 | 每个batch实际使用的KV序列长度。默认值为None | int32 | ND | (B,) |
| batch_size | int | 可选 | batch大小。BSND/BNSD场景必须传入实际B，取值范围为(0, 65536) | int32 | - | - |
| max_seqlen_q | int | 可选 | Q最大序列长度。BSND/BNSD场景必须传入实际S1，且大于0 | int32 | - | - |
| max_seqlen_kv | int | 可选 | KV最大序列长度。BSND/BNSD场景必须传入实际S2，且大于0 | int32 | - | - |
| mask_mode | int/MaskMode | 可选 | 必须与`flash_attn_grad`一致 | int32 | - | - |
| win_left | int | 可选 | 必须与`flash_attn_grad`一致 | int32 | - | - |
| win_right | int | 可选 | 必须与`flash_attn_grad`一致 | int32 | - | - |
| layout_q | string | 可选 | 必须与`flash_attn_grad`一致 | string | - | - |
| layout_kv | string | 可选 | 必须与`flash_attn_grad`一致 | string | - | - |
| layout_out | string | 可选 | 必须与`flash_attn_grad`一致 | string | - | - |
| is_grad_enabled | bool | 可选 | 是否生成反向算子所需的metadata。调用`flash_attn_grad`前必须设置为True。默认值为False | bool | - | - |

返回的`metadata`为int32、ND格式的一维Tensor，长度根据batch大小和N2动态计算。

## 调用示例

- `flash_attn_metadata`、`flash_attn`和`flash_attn_grad`联合调用示例（BSND）

  `is_grad_enabled=True`的`flash_attn_metadata`会同时生成正向和反向的任务切分数据，前向和反向均可使用该metadata，无需分别生成。

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

    dq, dk, dv = cann_ops_transformer.flash_attn_grad(
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
    ```
